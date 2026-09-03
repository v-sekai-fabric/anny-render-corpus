"""RFD 2183 rung 0: composite + per-layer renders over sphere_hammersley_sequence.

Bootstraps from ANNY (face groups partition the layers) until VRMs land from
the atelier-workshop pipeline. Emits root/candidates/root_extras/candidate_extras
parquet per `maskscore_stub_schema`; the eval script writes scores later.

    pixi run -e anny-mac python render_layer_decomp_corpus.py \\
        --pose-dir build/bootstrap --out build/rfd2183/multiview \\
        --views 24 --spp 16 --variant metal_ad_rgb
    python render_layer_decomp_corpus.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from maskscore_stub_schema import (
    CANDIDATE_EXTRAS, CANDIDATES, ROOT, ROOT_EXTRAS, SCORES,
)


LAYER_AXIS = "view"
LAYER_KIND_BODY = "anny_body"
LAYER_KIND_EYES = "anny_eyes"
LAYER_KIND_TONGUE = "anny_tongue"


def _mesh_key(name: str, view: int) -> str:
    return f"rfd2183/multiview/{name}/view_{view:03d}"


def _write_parquet(rows: list[dict], schema: pa.Schema, out: pathlib.Path) -> None:
    if not rows:
        table = schema.empty_table()
    else:
        cols = {f.name: [r[f.name] for r in rows] for f in schema}
        table = pa.table(cols, schema=schema)
    pq.write_table(table, out, compression="zstd")


def render_one_mesh(mesh_npz: pathlib.Path, out_dir: pathlib.Path, views: int,
                    fov: float, spp: int, threads: int, variant: str,
                    layer_face_groups: dict[str, tuple[int, int]]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    import render_view
    out_dir.mkdir(parents=True, exist_ok=True)
    name = mesh_npz.stem
    full = np.load(mesh_npz)
    full_faces = np.asarray(full["faces"], dtype=np.int64)

    root_rows, cand_rows, root_extras, cand_extras = [], [], [], []

    for v in range(views):
        composite_png = out_dir / name / f"view_{v:03d}_composite.png"
        side = render_view.render(
            mesh_npz=str(mesh_npz), out_png=composite_png,
            index=v, views=views, fov_deg=fov, offset=(0.0, 0.0),
            spp=spp, threads=threads, variant=variant, distance=1.0,
            direction=None, aov=True,
        )
        rk = _mesh_key(name, v)
        root_rows.append({
            "key": rk, "task_type": "instruction_edit",
            "dimension": "instruction_following",
            "input_column": "composite_image",
            "input_asset": str(composite_png.relative_to(out_dir)),
            "input_asset_kind": "png",
        })
        root_extras.append({
            "row_key": rk, "extra_name": "aov_npz",
            "extra_value": str(composite_png.with_suffix(".aov.npz").relative_to(out_dir)),
            "extra_kind": "npz_pointer",
        })
        root_extras.append({
            "row_key": rk, "extra_name": "camera_json",
            "extra_value": str(composite_png.with_suffix(".json").relative_to(out_dir)),
            "extra_kind": "json_path",
        })

        for rank, (layer_name, (f0, f1)) in enumerate(sorted(layer_face_groups.items()), start=1):
            layer_faces = full_faces[f0:f1]
            layer_npz = out_dir / name / f"view_{v:03d}_layer_{layer_name}.npz"
            layer_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez(layer_npz, verts=full["verts"], faces=layer_faces)
            layer_png = out_dir / name / f"view_{v:03d}_layer_{layer_name}.png"
            render_view.render(
                mesh_npz=str(layer_npz), out_png=layer_png,
                index=v, views=views, fov_deg=fov, offset=(0.0, 0.0),
                spp=spp, threads=threads, variant=variant, distance=1.0,
                direction=None, aov=False,
            )
            kind = {"body": LAYER_KIND_BODY, "eyes": LAYER_KIND_EYES,
                    "tongue": LAYER_KIND_TONGUE}.get(layer_name, f"anny_{layer_name}")
            cand_rows.append({
                "row_key": rk, "candidate_axis": LAYER_AXIS,
                "rank": int(rank),
                "candidate_asset": str(layer_png.relative_to(out_dir)),
                "candidate_asset_kind": "png",
                "candidate_kind": kind,
            })
            cand_extras.append({
                "row_key": rk, "candidate_axis": LAYER_AXIS,
                "candidate_rank": int(rank),
                "extra_name": "face_range",
                "extra_value": f"{f0}:{f1}",
                "extra_kind": "frame_span",
            })

        print(f"  [{name}] view {v+1:3d}/{views}  composite + {len(layer_face_groups)} layers"
              f"  yaw {side['yaw_deg']:7.2f}  pitch {side['pitch_deg']:6.2f}")

    return root_rows, cand_rows, root_extras, cand_extras


def anny_face_groups(mesh_npz: pathlib.Path) -> dict[str, tuple[int, int]]:
    import anny
    from anny.models.model_data import TopologyConfig
    model = anny.Anny(topology=TopologyConfig(base_mesh="makehuman", remove_unattached_vertices=False))
    groups = getattr(model, "face_groups", None)
    if groups is None:
        n = np.load(mesh_npz)["faces"].shape[0]
        return {"body": (0, n)}
    return {name: (int(lo), int(hi)) for name, (lo, hi) in groups.items()}


def _self_test() -> int:
    import tempfile
    from maskscore_stub_schema import validate

    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp)
        rk = _mesh_key("rest", 0)
        root_rows = [{
            "key": rk, "task_type": "instruction_edit",
            "dimension": "instruction_following",
            "input_column": "composite_image",
            "input_asset": "rest/view_000_composite.png",
            "input_asset_kind": "png",
        }]
        cand_rows = [{
            "row_key": rk, "candidate_axis": LAYER_AXIS, "rank": 1,
            "candidate_asset": "rest/view_000_layer_body.png",
            "candidate_asset_kind": "png",
            "candidate_kind": LAYER_KIND_BODY,
        }]
        root_extras = [{
            "row_key": rk, "extra_name": "aov_npz",
            "extra_value": "rest/view_000_composite.aov.npz",
            "extra_kind": "npz_pointer",
        }]
        cand_extras = [{
            "row_key": rk, "candidate_axis": LAYER_AXIS, "candidate_rank": 1,
            "extra_name": "face_range", "extra_value": "0:13718",
            "extra_kind": "frame_span",
        }]

        _write_parquet(root_rows, ROOT, out / "root.parquet")
        _write_parquet(cand_rows, CANDIDATES, out / "cands.parquet")
        _write_parquet([], SCORES, out / "scores.parquet")
        _write_parquet(root_extras, ROOT_EXTRAS, out / "root_extras.parquet")
        _write_parquet(cand_extras, CANDIDATE_EXTRAS, out / "cand_extras.parquet")

        root_t = pq.read_table(out / "root.parquet")
        cands_t = pq.read_table(out / "cands.parquet")
        scores_t = pq.read_table(out / "scores.parquet")
        root_extras_t = pq.read_table(out / "root_extras.parquet")
        cand_extras_t = pq.read_table(out / "cand_extras.parquet")

        try:
            validate(root_t, cands_t, scores_t, root_extras_t, cand_extras_t)
        except ValueError as e:
            print(f"FAIL (positive control): {e}")
            return 1

        broken_root = pa.table({
            "key": [rk], "task_type": ["FAKE_TASK"],
            "dimension": ["instruction_following"],
            "input_column": ["composite_image"],
            "input_asset": ["x.png"], "input_asset_kind": ["png"],
        }, schema=ROOT)
        try:
            validate(broken_root, cands_t, scores_t, root_extras_t, cand_extras_t)
        except ValueError:
            pass
        else:
            print("FAIL (negative control escaped): out-of-vocab task_type accepted")
            return 1

        orphan_cands = pa.table({
            "row_key": ["not_a_real_key"],
            "candidate_axis": [LAYER_AXIS],
            "rank": pa.array([1], type=pa.int32()),
            "candidate_asset": ["x.png"], "candidate_asset_kind": ["png"],
            "candidate_kind": [LAYER_KIND_BODY],
        }, schema=CANDIDATES)
        try:
            validate(root_t, orphan_cands, scores_t, root_extras_t, cand_extras_t)
        except ValueError:
            pass
        else:
            print("FAIL (negative control escaped): orphan candidate row_key accepted")
            return 1

    print("ok: layer-decomp corpus schema round-trips and rejects each planted defect")
    return 0


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose-dir", type=pathlib.Path, default=pathlib.Path("build/bootstrap"))
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("rfd2183-layer-decomp/multiview"))
    ap.add_argument("--views", type=int, default=24)
    ap.add_argument("--fov", type=float, default=40.0)
    ap.add_argument("--spp", type=int, default=16)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--variant", default="metal_ad_rgb",
                    help="metal_ad_rgb for throughput; llvm_ad_rgb for reproducibility")
    ap.add_argument("--meshes", nargs="+",
                    help="explicit list of .npz meshes; overrides --pose-dir default")
    ap.add_argument("--layer-groups",
                    help="JSON dict of {layer_name: [face_lo, face_hi]}; "
                         "default probes ANNY.face_groups then falls back to {'body': [0, N]}")
    ap.add_argument("--sanity", action="store_true",
                    help="render 1 mesh at 2 views to prove the pipeline works")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv[1:])

    if a.self_test:
        return _self_test()

    meshes = ([pathlib.Path(m) for m in a.meshes]
              if a.meshes else sorted(a.pose_dir.glob("*.npz")))
    if not meshes:
        raise SystemExit(f"no .npz meshes found under {a.pose_dir}")
    if a.sanity:
        meshes = meshes[:1]
        a.views = 2

    if a.layer_groups:
        groups = {k: (int(v[0]), int(v[1])) for k, v in json.loads(a.layer_groups).items()}
    else:
        groups = anny_face_groups(meshes[0])
        print(f"layer groups: {groups}")

    a.out.mkdir(parents=True, exist_ok=True)
    all_root, all_cands, all_root_extras, all_cand_extras = [], [], [], []
    t0 = time.time()
    for m in meshes:
        r, c, re, ce = render_one_mesh(m, a.out, a.views, a.fov, a.spp,
                                       a.threads, a.variant, groups)
        all_root += r
        all_cands += c
        all_root_extras += re
        all_cand_extras += ce

    _write_parquet(all_root, ROOT, a.out / "layer_decomp_root.parquet")
    _write_parquet(all_cands, CANDIDATES, a.out / "layer_decomp_candidates.parquet")
    _write_parquet([], SCORES, a.out / "layer_decomp_scores.parquet")
    _write_parquet(all_root_extras, ROOT_EXTRAS, a.out / "layer_decomp_root_extras.parquet")
    _write_parquet(all_cand_extras, CANDIDATE_EXTRAS, a.out / "layer_decomp_candidate_extras.parquet")

    provenance = {
        "generator": "sphere_hammersley_sequence, TencentARC/Pixal3D @ cdbb2bb",
        "variant": a.variant, "views_per_mesh": a.views, "spp": a.spp,
        "meshes": [str(m) for m in meshes],
        "layer_groups": {k: list(v) for k, v in groups.items()},
        "elapsed_s": round(time.time() - t0, 2),
        "rfd": "urn:oid:1.3.6.1.4.1.66606.1.2.2183",
    }
    (a.out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"done. {len(all_root)} composites, {len(all_cands)} layer renders -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
