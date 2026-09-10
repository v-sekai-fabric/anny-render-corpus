"""RFD 2196/2197: emit the OmniGen2 azimuth ladder as an EditScore-shaped dataset.

The EditScore ladder shape is one row per (source, edited, instruction) with the
per-pair measurement beside the images. This run measured recovered azimuth
rather than EditScore's 0-25 axes, so the measurement columns are the ones the
run actually wrote: no pf/sc/pq/overall column exists here to be filled in.

Two configs rather than one wide table with holes. The recovery pass fitted all
eight asked azimuths; the generation log kept prompt, wall time and peak VRAM
for four of them. A single table would carry four rows of nulls, which the
normal form forbids, so `runs` is a satellite of `views` keyed on pair_idx.

    python omnigen2_azimuth_ladder_hf_publish.py --source <corpus> --out <dir>
    python omnigen2_azimuth_ladder_hf_publish.py --self-test
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# HF renders a column as an image when it is a struct of raw bytes and a path.
IMAGE = pa.struct([("bytes", pa.binary()), ("path", pa.string())])

VIEWS_SCHEMA = pa.schema([
    ("pair_idx", pa.int32()),
    ("condition", pa.string()),
    ("asked_deg", pa.float32()),
    ("recovered_deg", pa.float32()),
    ("error_deg", pa.float32()),
    ("source_img", IMAGE),
    ("edited_img", IMAGE),
])

RUNS_SCHEMA = pa.schema([
    ("pair_idx", pa.int32()),
    ("view_phrase", pa.string()),
    ("instruction", pa.string()),
    ("wall_s", pa.float32()),
    ("peak_vram_gib", pa.float32()),
    ("n_input_images", pa.int32()),
])


def image_cell(path: Path) -> dict:
    return {"bytes": path.read_bytes(), "path": path.name}


def asset_name(run_file: str) -> str:
    """az180_A.png as the run wrote it, T_Az180_A.png as the corpus stores it."""
    stem = run_file[:-4]
    az, cond = stem.split("_")
    return f"T_Az{az[2:]}_{cond}.png"


def build(source: Path):
    ladder = json.loads((source / "ladder/DA_Ladder.json").read_text())
    recovery = json.loads((source / "ladder/DA_AzimuthRecoveryA.json").read_text())

    src_png = source.parent / "anny-render-corpus-constructed/renders" / ladder["source_frame"]
    if not src_png.exists():
        raise SystemExit(f"source frame missing: {src_png}")
    src_cell = image_cell(src_png)

    order = {row["file"]: i for i, row in enumerate(recovery["rows"])}
    views, runs = [], []

    for row in recovery["rows"]:
        edited = source / "ladder" / asset_name(row["file"])
        if not edited.exists():
            raise SystemExit(f"edited image missing: {edited}")
        views.append({
            "pair_idx": order[row["file"]],
            "condition": recovery["condition"],
            "asked_deg": row["asked_deg"],
            "recovered_deg": row["recovered_deg"],
            "error_deg": row["error_deg"],
            "source_img": src_cell,
            "edited_img": image_cell(edited),
        })

    for view in ladder["views"]:
        if view["file"] not in order:
            raise SystemExit(f"generation log names {view['file']}, absent from the recovery fit")
        runs.append({
            "pair_idx": order[view["file"]],
            "view_phrase": view["view_phrase"],
            "instruction": view["prompt"],
            "wall_s": view["seconds"],
            "peak_vram_gib": view["peak_vram_gib"],
            "n_input_images": view["n_input_images"],
        })

    return (
        pa.Table.from_pylist(views, schema=VIEWS_SCHEMA),
        pa.Table.from_pylist(runs, schema=RUNS_SCHEMA),
        ladder,
        recovery,
    )


def verify(views: pa.Table, runs: pa.Table) -> None:
    """Every satellite row joins, and no column is null."""
    v_keys = set(views.column("pair_idx").to_pylist())
    r_keys = set(runs.column("pair_idx").to_pylist())
    orphans = r_keys - v_keys
    if orphans:
        raise SystemExit(f"{len(orphans)} run rows key to no view: {sorted(orphans)}")
    for table, name in ((views, "views"), (runs, "runs")):
        for col in table.schema.names:
            if table.column(col).null_count:
                raise SystemExit(f"{name}.{col} carries {table.column(col).null_count} nulls")


def emit(views: pa.Table, runs: pa.Table, out: Path) -> list[Path]:
    written = []
    for table, config in ((views, "views"), (runs, "runs")):
        d = out / "data" / config
        d.mkdir(parents=True, exist_ok=True)
        p = d / "train-00000-of-00001.parquet"
        # row_group_size bounds what the HF viewer has to pull to render a page.
        pq.write_table(table, p, compression="zstd", row_group_size=100)
        written.append(p)
    return written


def _fabricate(root: Path) -> Path:
    """A toy corpus: two asked azimuths, one of which logged a generation."""
    png = bytes.fromhex("89504e470d0a1a0a") + b"toy"
    (root / "anny-render-corpus-constructed/renders").mkdir(parents=True)
    (root / "anny-render-corpus-constructed/renders/src.png").write_bytes(png)
    gen = root / "generated"
    (gen / "ladder").mkdir(parents=True)
    for n in ("T_Az000_A.png", "T_Az180_A.png"):
        (gen / "ladder" / n).write_bytes(png)
    (gen / "ladder/DA_Ladder.json").write_text(json.dumps({
        "source_frame": "src.png",
        "views": [{"file": "az180_A.png", "view_phrase": "back", "prompt": "turn",
                   "seconds": 1.5, "peak_vram_gib": 2.0, "n_input_images": 1}],
    }))
    (gen / "ladder/DA_AzimuthRecoveryA.json").write_text(json.dumps({
        "condition": "A",
        "rows": [{"file": "az000_A.png", "asked_deg": 0.0, "recovered_deg": -1.0, "error_deg": 1.0},
                 {"file": "az180_A.png", "asked_deg": 180.0, "recovered_deg": 170.0, "error_deg": 10.0}],
    }))
    return gen


def self_test() -> int:
    ok = True

    def check(label, cond):
        nonlocal ok
        print(f"  {'behaved' if cond else 'CONTROL FAILED'} -- {label}")
        ok = ok and cond

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        gen = _fabricate(root)
        views, runs, _, _ = build(gen)
        verify(views, runs)

        check("both asked azimuths become view rows", views.num_rows == 2)
        check("only the logged generation becomes a run row", runs.num_rows == 1)
        check("the satellite keys to a view", runs.column("pair_idx").to_pylist() == [1])
        check("images arrive as bytes, not paths",
              views.column("edited_img")[0]["bytes"].as_py().startswith(b"\x89PNG"))
        check("the source frame is the same cell on every row",
              len({v["bytes"].as_py() for v in views.column("source_img")}) == 1)
        check("asset names map az180_A.png to T_Az180_A.png",
              asset_name("az180_A.png") == "T_Az180_A.png")
        check("no column carries a null",
              all(views.column(c).null_count == 0 for c in views.schema.names))

        paths = emit(views, runs, root / "out")
        back = pq.read_table(paths[0])
        check("the emit round-trips", back.num_rows == 2)
        check("parquet is ZStandard",
              pq.ParquetFile(paths[0]).metadata.row_group(0).column(0).compression == "ZSTD")

        # negative control: a generation log naming a view the fit never saw
        log = json.loads((gen / "ladder/DA_Ladder.json").read_text())
        log["views"][0]["file"] = "az999_A.png"
        (gen / "ladder/DA_Ladder.json").write_text(json.dumps(log))
        try:
            build(gen)
            caught = False
        except SystemExit:
            caught = True
        check("an unjoinable run row is refused", caught)

        # negative control: a missing image is refused rather than emitted empty
        (gen / "ladder/T_Az000_A.png").unlink()
        log["views"][0]["file"] = "az180_A.png"
        (gen / "ladder/DA_Ladder.json").write_text(json.dumps(log))
        try:
            build(gen)
            caught = False
        except SystemExit:
            caught = True
        check("a missing edited image is refused", caught)

    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, help="the anny-render-corpus-generated checkout")
    ap.add_argument("--out", type=Path, help="where to write data/<config>/*.parquet")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if not args.source or not args.out:
        ap.error("--source and --out are required unless --self-test")

    views, runs, ladder, recovery = build(args.source)
    verify(views, runs)
    for p in emit(views, runs, args.out):
        print(f"  wrote {p} ({p.stat().st_size} bytes)")
    print(f"  views={views.num_rows} runs={runs.num_rows} "
          f"slope={recovery['slope']:.4f} revision={ladder['revision'][:9]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
