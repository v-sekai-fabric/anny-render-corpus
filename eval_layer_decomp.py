"""RFD 2183 baseline eval: per-layer IoU + SSIM + LPIPS, depth abs-rel + Pearson.

Expects `<root>/composite/<id>.png`, `<root>/masks/<id>/layer_<name>.png`,
`<root>/layers/<id>/layer_<name>.png`, `<root>/depth/<id>.npy` on both sides.
Cribs metric shapes from `voxhammer-upstream/Edit3D-Bench/eval_modules/image_metrics.py`.
LPIPS needs `lpips` + torch; the code raises rather than silently skips.

    pixi run -e omnigen2 python eval_layer_decomp.py \\
        --candidate <cand>/ --baseline <base>/ \\
        --floors floors.json --out logs/rfd2183-baseline.md
    python eval_layer_decomp.py --self-test
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass


# Household-object anchors for mm-scale numbers, per CLAUDE.md.
_ANCHORS: tuple[tuple[str, float], ...] = (
    ("credit card thickness", 0.76),
    ("penny", 1.52),
    ("pencil", 7.0),
    ("AAA battery", 10.5),
    ("AA battery", 14.5),
    ("nickel", 21.2),
    ("golf ball", 42.7),
    ("adult wrist", 57.0),
    ("soda can", 66.0),
)


def household_mm(mm: float) -> str:
    if mm <= 0:
        return "below one credit card thickness"
    name, size = min(_ANCHORS, key=lambda ns: abs(mm - ns[1]))
    n = mm / size
    if n < 0.5:
        return f"under half a {name}"
    if n < 1.5:
        return f"about one {name}"
    return f"about {n:.1f} stacked {name}s"


def _load_image(path: pathlib.Path):
    from PIL import Image
    import numpy as np
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _load_mask(path: pathlib.Path):
    from PIL import Image
    import numpy as np
    m = np.asarray(Image.open(path).convert("L"))
    return (m > 127).astype(bool)


def _load_depth(path: pathlib.Path):
    import numpy as np
    if path.suffix == ".npy":
        return np.load(path).astype("float32")
    if path.suffix == ".npz":
        d = np.load(path)
        return d["depth"].astype("float32") if "depth" in d.files else d[d.files[0]].astype("float32")
    raise SystemExit(f"depth file {path} is neither .npy nor .npz")


def iou(a, b) -> float:
    import numpy as np
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter / union)


def ssim(a, b) -> float:
    try:
        from skimage.metrics import structural_similarity as sk_ssim
        return float(sk_ssim(a, b, channel_axis=2, data_range=1.0))
    except ImportError:
        import numpy as np
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        vals = []
        for c in range(a.shape[-1]):
            x, y = a[..., c], b[..., c]
            mx, my = x.mean(), y.mean()
            vx, vy = x.var(), y.var()
            cov = ((x - mx) * (y - my)).mean()
            vals.append(((2 * mx * my + c1) * (2 * cov + c2)) /
                        ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2)))
        return float(sum(vals) / len(vals))


def lpips_distance(a, b) -> float:
    try:
        import torch
        import lpips as lpips_pkg
    except ImportError as e:
        raise SystemExit(
            f"lpips import failed: {e}. This is a FAIL per CLAUDE.md rule 3, not a skip. "
            "Install lpips or run without --lpips."
        )
    model = lpips_distance._model  # type: ignore[attr-defined]
    if model is None:
        model = lpips_pkg.LPIPS(net="alex", verbose=False)
        model.eval()
        lpips_distance._model = model  # type: ignore[attr-defined]
    t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    u = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    with torch.no_grad():
        return float(model(t.float(), u.float()).item())


lpips_distance._model = None  # type: ignore[attr-defined]


def depth_abs_rel(pred, gt) -> float:
    import numpy as np
    m = (gt > 1e-6) & np.isfinite(gt) & np.isfinite(pred)
    if not m.any():
        return float("nan")
    return float(np.abs(pred[m] - gt[m]).mean() / gt[m].mean())


def depth_pearson(pred, gt) -> float:
    import numpy as np
    m = np.isfinite(pred) & np.isfinite(gt)
    if m.sum() < 2:
        return float("nan")
    p, g = pred[m].astype("float64"), gt[m].astype("float64")
    p -= p.mean(); g -= g.mean()
    denom = (np.sqrt((p * p).sum()) * np.sqrt((g * g).sum()))
    return float((p * g).sum() / denom) if denom > 0 else float("nan")


@dataclass
class LayerResult:
    image_id: str
    layer: str
    iou: float
    ssim: float
    lpips: float | None


@dataclass
class DepthResult:
    image_id: str
    abs_rel: float
    pearson: float
    scene_span_mm: float


def _enumerate_pairs(cand_root: pathlib.Path, base_root: pathlib.Path) -> list[str]:
    cand = {p.stem for p in (cand_root / "composite").glob("*.png")}
    base = {p.stem for p in (base_root / "composite").glob("*.png")}
    both = sorted(cand & base)
    only_cand = sorted(cand - base)
    only_base = sorted(base - cand)
    if only_cand or only_base:
        print(f"UNPAIRED (counted, not skipped): only in candidate={only_cand}  only in baseline={only_base}",
              file=sys.stderr)
    return both


def _layers_for(root: pathlib.Path, image_id: str) -> set[str]:
    return {p.stem.removeprefix("layer_")
            for p in (root / "masks" / image_id).glob("layer_*.png")}


def score_pair(cand_root: pathlib.Path, base_root: pathlib.Path,
               use_lpips: bool) -> tuple[list[LayerResult], list[DepthResult], dict[str, int]]:
    layer_results: list[LayerResult] = []
    depth_results: list[DepthResult] = []
    stats = {"pairs": 0, "layers": 0, "unpaired_layers": 0, "missing_depth": 0}

    for image_id in _enumerate_pairs(cand_root, base_root):
        stats["pairs"] += 1
        cand_layers = _layers_for(cand_root, image_id)
        base_layers = _layers_for(base_root, image_id)
        shared = sorted(cand_layers & base_layers)
        stats["unpaired_layers"] += len((cand_layers | base_layers) - set(shared))

        for layer in shared:
            m_c = _load_mask(cand_root / "masks" / image_id / f"layer_{layer}.png")
            m_b = _load_mask(base_root / "masks" / image_id / f"layer_{layer}.png")
            iou_v = iou(m_c, m_b)
            l_c = _load_image(cand_root / "layers" / image_id / f"layer_{layer}.png")
            l_b = _load_image(base_root / "layers" / image_id / f"layer_{layer}.png")
            if l_c.shape != l_b.shape:
                raise SystemExit(
                    f"layer image shape mismatch on {image_id}/{layer}: "
                    f"candidate {l_c.shape} vs baseline {l_b.shape}. Rule 7: name the interface."
                )
            ssim_v = ssim(l_c, l_b)
            lp_v = lpips_distance(l_c, l_b) if use_lpips else None
            layer_results.append(LayerResult(image_id, layer, iou_v, ssim_v, lp_v))
            stats["layers"] += 1

        cand_depth = cand_root / "depth" / f"{image_id}.npy"
        base_depth = base_root / "depth" / f"{image_id}.npy"
        if not cand_depth.exists() or not base_depth.exists():
            stats["missing_depth"] += 1
            continue
        d_c = _load_depth(cand_depth)
        d_b = _load_depth(base_depth)
        depth_results.append(DepthResult(
            image_id=image_id,
            abs_rel=depth_abs_rel(d_c, d_b),
            pearson=depth_pearson(d_c, d_b),
            scene_span_mm=float((d_b[d_b > 0].max() - d_b[d_b > 0].min()) * 1000.0)
                if (d_b > 0).any() else 0.0,
        ))

    return layer_results, depth_results, stats


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None and x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def render_markdown(layer: list[LayerResult], depth: list[DepthResult],
                    stats: dict[str, int], cand_root: pathlib.Path,
                    base_root: pathlib.Path, floor: dict[str, float]) -> str:
    lines = [
        "# RFD 2183 baseline: OmniGen2 out-of-the-box vs See-Through",
        "",
        f"Candidate: `{cand_root}`  Baseline: `{base_root}`",
        f"Pairs: {stats['pairs']}, layers scored: {stats['layers']}, "
        f"unpaired layers: {stats['unpaired_layers']}, missing depth: {stats['missing_depth']}",
        "",
        "## Per-layer averages (candidate vs baseline)",
        "",
        "| layer | n | IoU  | IoU floor | SSIM | SSIM floor | LPIPS | LPIPS ceiling |",
        "|-------|---|------|-----------|------|------------|-------|---------------|",
    ]
    layers = sorted({r.layer for r in layer})
    for lname in layers:
        rs = [r for r in layer if r.layer == lname]
        lines.append(
            f"| {lname} | {len(rs)} | {_mean([r.iou for r in rs]):.3f} | {floor['iou']:.3f} "
            f"| {_mean([r.ssim for r in rs]):.3f} | {floor['ssim']:.3f} "
            f"| {_mean([r.lpips for r in rs]):.3f} | {floor['lpips']:.3f} |"
        )
    lines += ["", "## Depth (candidate vs baseline)", "",
              "| n | abs-rel | abs-rel floor | Pearson | Pearson floor | mean scene span |",
              "|---|---------|---------------|---------|---------------|-----------------|"]
    span_mm = _mean([d.scene_span_mm for d in depth])
    lines.append(
        f"| {len(depth)} | {_mean([d.abs_rel for d in depth]):.4f} | {floor['abs_rel']:.4f} "
        f"| {_mean([d.pearson for d in depth]):.4f} | {floor['pearson']:.4f} "
        f"| {span_mm:.1f} mm ({household_mm(span_mm)}) |"
    )
    lines += [
        "",
        "Floors: IoU/SSIM/Pearson floor is the score two DIFFERENT images score against each "
        "other (random baseline); LPIPS ceiling is the same. A candidate at the floor has not "
        "beaten \"pick any other layer's mask\". abs-rel floor is the abs-rel of a constant "
        "depth map at the baseline's mean.",
    ]
    return "\n".join(lines) + "\n"


def _self_test() -> int:
    import numpy as np
    rng = np.random.default_rng(0)
    a = rng.random((32, 32, 3), dtype=np.float32)
    b = a.copy()
    if iou(a[..., 0] > 0.5, b[..., 0] > 0.5) != 1.0:
        print("FAIL (positive control): identical masks did not score IoU 1.0")
        return 1
    if ssim(a, b) < 0.99:
        print(f"FAIL (positive control): identical SSIM {ssim(a, b)} < 0.99")
        return 1
    c = rng.random((32, 32, 3), dtype=np.float32)
    if iou(a[..., 0] > 0.5, c[..., 0] > 0.5) >= 0.99:
        print("FAIL (negative control): random masks scored IoU as if identical")
        return 1
    if ssim(a, c) > 0.5:
        print(f"FAIL (negative control): random SSIM {ssim(a, c)} > 0.5, decoration")
        return 1
    d1 = np.linspace(0.1, 1.0, 1024, dtype=np.float32)
    d2 = d1 * 1.2 + 0.01
    ar = depth_abs_rel(d2, d1)
    if not (0.15 < ar < 0.35):
        print(f"FAIL (positive control): abs-rel {ar} outside [0.15, 0.35] for scale+bias 1.2/0.01")
        return 1
    if depth_pearson(d2, d1) < 0.999:
        print("FAIL (positive control): scaled+biased depth did not score Pearson ~1")
        return 1
    d3 = rng.random(1024, dtype=np.float32)
    if depth_pearson(d3, d1) > 0.5:
        print("FAIL (negative control): random depth scored Pearson > 0.5")
        return 1
    anchor = household_mm(4.3)
    if "pencil" not in anchor:
        print(f"FAIL (positive control): household_mm(4.3) missing 'pencil', got {anchor!r}")
        return 1
    print("ok: iou/ssim/depth positive+negative controls pass, household anchor lands")
    return 0


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", type=pathlib.Path)
    ap.add_argument("--baseline", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--no-lpips", action="store_true",
                    help="skip LPIPS. Reported as a null column, not silently dropped.")
    ap.add_argument("--floors", type=pathlib.Path,
                    help="JSON: {iou, ssim, lpips, abs_rel, pearson}. Missing => a FAIL, "
                         "rule 4: a number without a baseline is not a measurement.")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv[1:])

    if a.self_test:
        return _self_test()
    for name, val in [("--candidate", a.candidate), ("--baseline", a.baseline),
                      ("--out", a.out), ("--floors", a.floors)]:
        if val is None:
            raise SystemExit(f"{name} is required (unless --self-test). Rule 3: no silent skips.")

    floor = json.loads(a.floors.read_text())
    for k in ("iou", "ssim", "lpips", "abs_rel", "pearson"):
        if k not in floor:
            raise SystemExit(f"floors JSON missing key {k!r}. Rule 4.")

    layer, depth, stats = score_pair(a.candidate, a.baseline, use_lpips=not a.no_lpips)
    md = render_markdown(layer, depth, stats, a.candidate, a.baseline, floor)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
