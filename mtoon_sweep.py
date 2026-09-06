# SPDX-License-Identifier: Apache-2.0 OR MIT
"""The MToon sweep, rendered through the one MToon renderer (mtoon_forward).

MToon is not a BSDF: its shade colour is painted where dot(N, L) < 0, where a physically
based integrator contributes nothing, so a BSDF renders lit-to-black and the shade plateau
never appears. This module used to drive exactly that broken BSDF and pin its breakage --
its flat-material control asserted contrast > 1.0 where a correct renderer must produce
none. Retired 2026-09-06 in favour of mtoon_forward; the control now asserts flatness.

    python mtoon_sweep.py [--self-test]
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

import mtoon

MST = ["f6ede4", "f3e7db", "f7ead0", "eadaba", "d7bd96",
       "a07e56", "825c43", "604134", "3a312a", "292420"]
TOONY = (0.0, 0.5, 0.9, 1.0)
SHIFT = (-0.3, 0.0, 0.3)


def srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def linear_of(hexs):
    h = hexs.lstrip("#")
    return [srgb_to_linear(int(h[i:i + 2], 16) / 255) for i in (0, 2, 4)]


def solve_multiplier(lit, target):
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if mtoon.delta_e(lit, [c * mid for c in lit]) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def render_sphere(lit, shade, res=128, **params):
    """One sphere, one key light, through the shared MToon renderer.

    `direction=[-1,-0.2,-0.6]` on the old directional emitter is the direction the light
    travels, so the direction *to* the light is [1, 0.2, 0.6] -- mtoon_forward's default.
    No occluder: the sweep reads plateaus off an unshadowed body.
    """
    import mtoon_forward as forward
    img = forward.render(lit, shade, width=res, height=res, spp=16,
                         shape=None, occluder=False, **params)
    return np.asarray(img)


def plateaus(image):
    """Brightest and darkest tenth of the body.

    Selects fully covered pixels when the film carries alpha. Two traps, both of which
    manufacture contrast that a flat material does not have:

      * selecting by luma admits the antialiased silhouette outright;
      * the film is PREMULTIPLIED, so a half-covered edge pixel carries half the body
        colour. `alpha > 0.5` therefore still admits pixels darkened by coverage alone --
        at spp=16 that put the darkest tenth at 0.38 of a 0.68 body. Only `alpha ~= 1`
        is safe. (mtoon_forward's own flat control escapes this because it renders at
        spp=1, where coverage is binary; the bug is invisible there.)
    """
    image = np.asarray(image)
    if image.shape[-1] >= 4:
        flat = image.reshape(-1, image.shape[-1])
        body = flat[flat[:, 3] > 0.999][:, :3]
    else:
        flat = image.reshape(-1, 3)
        body = flat[flat @ np.array([0.2126, 0.7152, 0.0722]) > 1e-6]
    if len(body) < 20:
        return None, None
    order = np.argsort(body @ np.array([0.2126, 0.7152, 0.0722]))
    k = max(1, len(order) // 10)
    return body[order[-k:]].mean(axis=0), body[order[:k]].mean(axis=0)


def run(out, res=128, target=12.0, verbose=True):
    rows = []
    for i, hexs in enumerate(MST, 1):
        lit = linear_of(hexs)
        mult = solve_multiplier(lit, target)
        shade = [c * mult for c in lit]
        material_de = mtoon.delta_e(lit, shade)
        img = render_sphere(lit, shade, res=res, shading_toony_factor=0.9, shading_shift_factor=0.0)
        hi, _ = plateaus(img)
        rendered_de = mtoon.rendered_contrast(lit, shade, shading_toony_factor=0.9, shading_shift_factor=0.0)
        rows.append((i, hexs, mult, material_de, rendered_de))
        if verbose:
            print("  MST %2d  x%.4f   material dE %5.2f   rendered dE %5.2f"
                  % (i, mult, material_de, rendered_de))
    if out:
        pathlib.Path(out).mkdir(parents=True, exist_ok=True)
        ramp = np.array([[mtoon.shading(x, shading_shift_factor=s,
                                          shading_toony_factor=t)
                          for x in np.linspace(-1, 1, 64)]
                         for t in TOONY for s in SHIFT])
        np.save(pathlib.Path(out) / "ramps.npy", ramp)
    return rows


def self_test():
    """Eight controls. Five must reject a render that is not showing the material."""
    r = []
    lit = linear_of("d7bd96")
    shade = [c * 0.4 for c in lit]

    img = render_sphere(lit, shade, res=64, shading_toony_factor=1.0, shading_shift_factor=0.0)
    hi, lo = plateaus(img)
    r.append(("the sphere renders something", hi is not None))
    r.append(("a hard ramp shows two plateaus, not a gradient",
              hi is not None and mtoon.delta_e(list(hi), list(lo)) > 5.0))

    flat = render_sphere(lit, lit, res=64, shading_toony_factor=1.0)
    fhi, flo = plateaus(flat)
    r.append(("base == shade renders flat, which the retired BSDF could not do",
              fhi is not None and mtoon.delta_e(list(fhi), list(flo)) < 1.0))

    dark = render_sphere([c * 0.25 for c in lit], [c * 0.1 for c in lit], res=64)
    dhi, _ = plateaus(dark)
    r.append(("a darker material renders darker",
              dhi is not None and dhi.mean() < hi.mean()))

    r.append(("the ramp reaches both ends",
              mtoon.shading(1.0) == 1.0 and mtoon.shading(-1.0) == 0.0))
    r.append(("two different lit colours render differently",
              not np.allclose(render_sphere((1., 0, 0), (.2, 0, 0), res=32, shading_toony_factor=1.0),
                              render_sphere((0, 0, 1.), (0, 0, .2), res=32, shading_toony_factor=1.0))))
    r.append(("the model itself does separate lit from shade",
              mtoon.rendered_contrast(lit, shade) > 5.0
              and mtoon.rendered_contrast(lit, lit) < 1e-9))
    r.append(("the plateau finder rejects an empty image",
              plateaus(np.zeros((16, 16, 3)))[0] is None))

    bad = sum(1 for _, ok in r if not ok)
    for name, ok in r:
        print("  %-4s control: %s" % ("ok" if ok else "FAIL", name))
    print("  %d of %d controls fired." % (len(r) - bad, len(r)))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    import mtoon_forward as forward
    forward.set_wide_variant()
    if args.self_test:
        return self_test()

    rows = run(args.out, res=args.res)
    rendered = [r[4] for r in rows if not math.isnan(r[4])]
    print()
    print("  material dE spread  %.2f to %.2f" % (min(r[3] for r in rows),
                                                  max(r[3] for r in rows)))
    print("  rendered dE spread  %.2f to %.2f  (%.1fx)"
          % (min(rendered), max(rendered), max(rendered) / min(rendered)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
