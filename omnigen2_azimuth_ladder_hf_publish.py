"""RFD 2196/2197: emit the OmniGen2 azimuth ladder as an EditScore-shaped dataset.

The EditScore ladder shape is a candidate measured against a baseline on the
same prompts. Here the baseline is the base model and the candidate is the same
model after LoRA, both asked for eight camera azimuths. The measurement is
recovered azimuth rather than EditScore's 0-25 axes, so no pf/sc/pq/overall
column exists to be filled in.

Five relations rather than one wide table, because what each pair has differs.
Every pair was asked; four LoRA images were dropped as regeneratable; two LoRA
views had no person to fit; the base generation log kept prompt and cost for
four of its eight. A wide table carries that as nulls, which the normal form
forbids, so `pairs` interns the identity and the rest are satellites on
pair_idx.

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

SCHEMAS = {
    "pairs": pa.schema([
        ("pair_idx", pa.int32()),
        ("arm", pa.string()),
        ("condition", pa.string()),
        ("asked_deg", pa.float32()),
    ]),
    "images": pa.schema([
        ("pair_idx", pa.int32()),
        ("source_img", IMAGE),
        ("edited_img", IMAGE),
    ]),
    "recovery": pa.schema([
        ("pair_idx", pa.int32()),
        ("recovered_deg", pa.float32()),
        ("error_deg", pa.float32()),
    ]),
    "not_fitted": pa.schema([
        ("pair_idx", pa.int32()),
        ("reason", pa.string()),
    ]),
    "runs": pa.schema([
        ("pair_idx", pa.int32()),
        ("view_phrase", pa.string()),
        ("instruction", pa.string()),
        ("wall_s", pa.float32()),
        ("peak_vram_gib", pa.float32()),
        ("n_input_images", pa.int32()),
    ]),
}

ARMS = {"base": "ladder", "lora": "lora"}


def image_cell(path: Path) -> dict:
    return {"bytes": path.read_bytes(), "path": path.name}


def asset_name(run_file: str) -> str:
    """az180_A.png as the run wrote it, T_Az180_A.png as the corpus stores it."""
    stem = run_file[:-4]
    az, cond = stem.split("_")
    return f"T_Az{az[2:]}_{cond}.png"


def build(source: Path):
    """Every relation, keyed on a pair_idx interned across both arms."""
    tables = {name: [] for name in SCHEMAS}
    src_cell = None
    meta = {}
    idx = 0

    for arm, directory in ARMS.items():
        d = source / directory
        ladder = json.loads((d / "DA_Ladder.json").read_text())
        recovery = json.loads((d / "DA_AzimuthRecoveryA.json").read_text())
        meta[arm] = {"ladder": ladder, "recovery": recovery}

        if src_cell is None:
            src_png = source.parent / "anny-render-corpus-constructed/renders" / ladder["source_frame"]
            if not src_png.exists():
                raise SystemExit(f"source frame missing: {src_png}")
            src_cell = image_cell(src_png)

        # Every asked azimuth the run logged, whatever survived of it afterwards.
        asked = {v["file"]: v["azimuth_deg"] for v in ladder["views"]}
        for row in recovery["rows"]:
            asked.setdefault(row["file"], row.get("asked_deg"))
        for miss in recovery["not_fitted"]:
            asked.setdefault(miss["file"], None)
        unknown = sorted(f for f, a in asked.items() if a is None)
        if unknown:
            raise SystemExit(f"{arm}: no asked azimuth recorded for {unknown}")

        pair_of = {}
        for file in sorted(asked, key=lambda f: asked[f]):
            pair_of[file] = idx
            tables["pairs"].append({
                "pair_idx": idx, "arm": arm,
                "condition": recovery["condition"], "asked_deg": asked[file],
            })
            edited = d / asset_name(file)
            if edited.exists():
                tables["images"].append({
                    "pair_idx": idx, "source_img": src_cell,
                    "edited_img": image_cell(edited),
                })
            idx += 1

        for row in recovery["rows"]:
            tables["recovery"].append({
                "pair_idx": pair_of[row["file"]],
                "recovered_deg": row["recovered_deg"], "error_deg": row["error_deg"],
            })
        for miss in recovery["not_fitted"]:
            tables["not_fitted"].append({
                "pair_idx": pair_of[miss["file"]], "reason": miss["error"],
            })
        for view in ladder["views"]:
            tables["runs"].append({
                "pair_idx": pair_of[view["file"]],
                "view_phrase": view["view_phrase"], "instruction": view["prompt"],
                "wall_s": view["seconds"], "peak_vram_gib": view["peak_vram_gib"],
                "n_input_images": view["n_input_images"],
            })

    built = {n: pa.Table.from_pylist(rows, schema=SCHEMAS[n]) for n, rows in tables.items()}
    return built, meta


def verify(tables: dict) -> None:
    """Every satellite row keys to a pair, and no column is null."""
    keys = set(tables["pairs"].column("pair_idx").to_pylist())
    if len(keys) != tables["pairs"].num_rows:
        raise SystemExit("pair_idx is not unique in pairs")
    for name, table in tables.items():
        if name != "pairs":
            orphans = set(table.column("pair_idx").to_pylist()) - keys
            if orphans:
                raise SystemExit(f"{name}: {len(orphans)} rows key to no pair: {sorted(orphans)}")
        for col in table.schema.names:
            if table.column(col).null_count:
                raise SystemExit(f"{name}.{col} carries {table.column(col).null_count} nulls")
    fitted = set(tables["recovery"].column("pair_idx").to_pylist())
    unfitted = set(tables["not_fitted"].column("pair_idx").to_pylist())
    both = fitted & unfitted
    if both:
        raise SystemExit(f"pairs both fitted and not fitted: {sorted(both)}")


def emit(tables: dict, out: Path) -> list[Path]:
    written = []
    for name, table in tables.items():
        d = out / "data" / name
        d.mkdir(parents=True, exist_ok=True)
        p = d / "train-00000-of-00001.parquet"
        # row_group_size bounds what the HF viewer has to pull to render a page.
        pq.write_table(table, p, compression="zstd", row_group_size=100)
        written.append(p)
    return written


def _fabricate(root: Path) -> Path:
    """A toy corpus with both arms: one logged generation, one dropped image,
    one view with no person to fit."""
    png = bytes.fromhex("89504e470d0a1a0a") + b"toy"
    (root / "anny-render-corpus-constructed/renders").mkdir(parents=True)
    (root / "anny-render-corpus-constructed/renders/src.png").write_bytes(png)
    gen = root / "generated"

    def arm(name, images, views, rows, not_fitted):
        d = gen / name
        d.mkdir(parents=True)
        for n in images:
            (d / n).write_bytes(png)
        (d / "DA_Ladder.json").write_text(json.dumps({"source_frame": "src.png", "views": views}))
        (d / "DA_AzimuthRecoveryA.json").write_text(json.dumps(
            {"condition": "A", "rows": rows, "not_fitted": not_fitted}))

    logged = {"view_phrase": "back", "prompt": "turn", "seconds": 1.5,
              "peak_vram_gib": 2.0, "n_input_images": 1}
    arm("ladder", ["T_Az000_A.png", "T_Az180_A.png"],
        [dict(file="az180_A.png", azimuth_deg=180.0, **logged)],
        [{"file": "az000_A.png", "asked_deg": 0.0, "recovered_deg": -1.0, "error_deg": 1.0},
         {"file": "az180_A.png", "asked_deg": 180.0, "recovered_deg": 170.0, "error_deg": 10.0}],
        [])
    # the lora arm keeps az000's image, dropped az180's, and could not fit az045
    arm("lora", ["T_Az000_A.png", "T_Az045_A.png"],
        [dict(file="az000_A.png", azimuth_deg=0.0, **logged),
         dict(file="az045_A.png", azimuth_deg=45.0, **logged),
         dict(file="az180_A.png", azimuth_deg=180.0, **logged)],
        [{"file": "az000_A.png", "asked_deg": 0.0, "recovered_deg": 3.0, "error_deg": 3.0},
         {"file": "az180_A.png", "asked_deg": 180.0, "recovered_deg": 175.0, "error_deg": 5.0}],
        [{"file": "az045_A.png", "error": "no person detected"}])
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
        t, meta = build(gen)
        verify(t)

        def col(name, c):
            return t[name].column(c).to_pylist()

        check("every asked azimuth in both arms becomes a pair", t["pairs"].num_rows == 5)
        check("both arms are named", set(col("pairs", "arm")) == {"base", "lora"})
        check("pair_idx is interned across arms, not restarted per arm",
              sorted(col("pairs", "pair_idx")) == [0, 1, 2, 3, 4])
        check("a dropped image yields no image row, and no null",
              t["images"].num_rows == 4)
        check("a pair that could not be fitted is named, not omitted",
              t["not_fitted"].num_rows == 1)
        check("a fit and a non-fit never describe the same pair",
              not (set(col("recovery", "pair_idx")) & set(col("not_fitted", "pair_idx"))))
        check("every logged generation becomes a run row", t["runs"].num_rows == 4)
        check("images arrive as bytes, not paths",
              t["images"].column("edited_img")[0]["bytes"].as_py().startswith(b"\x89PNG"))
        check("asset names map az180_A.png to T_Az180_A.png",
              asset_name("az180_A.png") == "T_Az180_A.png")
        check("no column in any relation carries a null",
              all(tb.column(c).null_count == 0 for tb in t.values() for c in tb.schema.names))

        paths = emit(t, root / "out")
        back = {p.parent.name: pq.read_table(p) for p in paths}
        check("the emit round-trips every relation",
              {n: tb.num_rows for n, tb in back.items()} == {n: tb.num_rows for n, tb in t.items()})
        check("parquet is ZStandard",
              all(pq.ParquetFile(p).metadata.row_group(0).column(0).compression == "ZSTD"
                  for p in paths))

        # negative control: a fit for a view no run ever asked for
        rec = json.loads((gen / "lora/DA_AzimuthRecoveryA.json").read_text())
        rec["rows"].append({"file": "az999_A.png", "recovered_deg": 0.0, "error_deg": 0.0})
        (gen / "lora/DA_AzimuthRecoveryA.json").write_text(json.dumps(rec))
        try:
            build(gen)
            caught = False
        except SystemExit:
            caught = True
        check("a fit with no recorded asked azimuth is refused", caught)

        # negative control: the same pair fitted and declared unfittable
        rec["rows"] = rec["rows"][:-1]
        rec["not_fitted"].append({"file": "az000_A.png", "error": "planted"})
        (gen / "lora/DA_AzimuthRecoveryA.json").write_text(json.dumps(rec))
        t2, _ = build(gen)
        try:
            verify(t2)
            caught = False
        except SystemExit:
            caught = True
        check("a pair both fitted and unfittable is refused", caught)

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

    tables, meta = build(args.source)
    verify(tables)
    for p in emit(tables, args.out):
        print(f"  wrote {p} ({p.stat().st_size} bytes)")
    print("  " + " ".join(f"{n}={tb.num_rows}" for n, tb in tables.items()))
    for arm, m in meta.items():
        print(f"  {arm}: slope={m['recovery']['slope']:.4f} "
              f"fitted={len(m['recovery']['rows'])} "
              f"not_fitted={len(m['recovery']['not_fitted'])} "
              f"revision={m['ladder']['revision'][:9]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
