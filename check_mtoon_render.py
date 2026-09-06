# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Gate: the one MToon renderer agrees with the one MToon model.

Three MToon render paths existed until 2026-09-06 and two of them disagreed with the model
and with each other, each with a green self-test:

  * mtoon.py registered a Mitsuba BSDF. MToon paints its shade colour where dot(N,L) < 0,
    where a physically based integrator contributes nothing, so the shade plateau never
    reached film. mtoon_sweep.py drove it and *pinned the breakage*: its flat-material
    control asserted contrast > 1.0 on a base == shade material, which is precisely what a
    correct MToon renderer must never produce.
  * mtoon_integrator.py was a correct scalar integrator with no callers.
  * mtoon_forward.py was a correct-shaped wide integrator that had silently dropped two
    spec terms -- `lightColor` and `rimLightingMixFactor` -- because it re-implemented the
    shading inline instead of calling the model.

Nothing held any renderer against mtoon.evaluate, so that drift was invisible. This gate is
that join. mtoon_forward is now the only render path and calls mtoon.evaluate_wide;
check_mtoon_slang.py holds mtoon.evaluate against the Slang kernel, so the chain is

    mtoon.slang  ==  mtoon.evaluate  ==  mtoon.evaluate_wide  ==  rendered pixels

with a gate on every join.

    python check_mtoon_render.py --self-test
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mtoon  # noqa: E402
import mtoon_forward as forward  # noqa: E402

# A spread that moves every term, including the two mtoon_forward used to drop.
CASES = [
    dict(),
    dict(shading_toony_factor=0.0),
    dict(shading_toony_factor=1.0),
    dict(shading_toony_factor=0.5, shading_shift_factor=0.3),
    dict(shading_toony_factor=0.5, shading_shift_factor=-0.3),
    dict(shading_shift_texture=0.2),
    dict(parametric_rim_color_factor=(1.0, 0.5, 0.25),
         parametric_rim_fresnel_power_factor=3.0),
    dict(parametric_rim_color_factor=(1.0, 1.0, 1.0),
         parametric_rim_lift_factor=0.3, rim_lighting_mix_factor=1.0),
    dict(parametric_rim_color_factor=(1.0, 1.0, 1.0), rim_lighting_mix_factor=0.0),
]
LIGHTS = [(1.0, 1.0, 1.0), (0.4, 0.7, 1.0)]
BASE, SHADE = (0.72, 0.52, 0.32), (0.30, 0.21, 0.13)


def wide_vs_scalar(dot_nl, dot_nv, light_color, **kw):
    """evaluate_wide over an array, against evaluate elementwise."""
    import drjit as dr  # noqa: F401
    import mitsuba as mi
    nl, nv = mi.Float(list(map(float, dot_nl))), mi.Float(list(map(float, dot_nv)))
    got = np.array(mtoon.evaluate_wide(nl, nv, BASE, SHADE, light_color, **kw)).T
    want = np.array([mtoon.evaluate(float(a), float(b), BASE, SHADE, light_color, **kw)[0]
                     for a, b in zip(dot_nl, dot_nv)])
    return got.reshape(want.shape), want


def analytic_plateaus(light_color, **kw):
    """The model's fully lit and fully shaded colours -- what the render must reach."""
    hi, _ = mtoon.shade_mix(1.0, BASE, SHADE, **kw)
    lo, _ = mtoon.shade_mix(-1.0, BASE, SHADE, **kw)
    return hi * np.asarray(light_color, float), lo * np.asarray(light_color, float)


def rendered_plateaus(res=96, light_color=(1.0, 1.0, 1.0), **kw):
    """Brightest and darkest fully covered pixels of an unshadowed sphere."""
    img = np.asarray(forward.render(BASE, SHADE, width=res, height=res, spp=16,
                                    shape=None, occluder=False, shadows=False,
                                    light_color=light_color, **kw))
    body = img.reshape(-1, img.shape[-1])
    body = body[body[:, 3] > 0.999][:, :3]  # premultiplied film; only full coverage
    if len(body) < 50:
        return None, None
    luma = body @ np.array([0.2126, 0.7152, 0.0722])
    order = np.argsort(luma)
    k = max(1, len(order) // 20)
    return body[order[-k:]].mean(axis=0), body[order[:k]].mean(axis=0)


def self_test():
    """Nine controls. Six reject a renderer that drifted from the model."""
    r = []
    variant = forward.set_wide_variant()
    print("  variant: %s" % variant)

    rng = np.random.default_rng(20260906)
    nl = np.concatenate([np.linspace(-1, 1, 129), rng.uniform(-1, 1, 128)])
    nv = np.concatenate([np.linspace(0, 1, 129), rng.uniform(0, 1, 128)])

    worst = 0.0
    for light in LIGHTS:
        for kw in CASES:
            got, want = wide_vs_scalar(nl, nv, light, **kw)
            worst = max(worst, float(np.abs(got - want).max()))
    r.append(("evaluate_wide equals evaluate over %d cases (worst %.2e)"
              % (len(CASES) * len(LIGHTS), worst), worst <= 1e-5))

    # Negative control: the comparison must be capable of failing.
    got, _ = wide_vs_scalar(nl, nv, (1.0, 1.0, 1.0), shading_toony_factor=0.9)
    _, want_other = wide_vs_scalar(nl, nv, (1.0, 1.0, 1.0), shading_toony_factor=0.1)
    r.append(("a different parameter disagrees, so the agreement is not vacuous",
              float(np.abs(got - want_other).max()) > 0.01))

    # Renderer against the model, at both plateaus, with a hard ramp.
    for light in LIGHTS:
        hi_w, lo_w = analytic_plateaus(light, shading_toony_factor=1.0)
        hi_g, lo_g = rendered_plateaus(light_color=light, shading_toony_factor=1.0)
        ok = hi_g is not None and lo_g is not None
        r.append(("rendered lit plateau is the model's, lightColor %s" % (light,),
                  ok and mtoon.delta_e(list(hi_g), list(hi_w)) < 1.5))
        r.append(("rendered shade plateau is the model's, lightColor %s" % (light,),
                  ok and mtoon.delta_e(list(lo_g), list(lo_w)) < 1.5))

    # The invariant that separated the correct renderers from the retired BSDF.
    hi_f, lo_f = rendered_plateaus(shading_toony_factor=1.0)
    flat = np.asarray(forward.render(BASE, BASE, width=96, height=96, spp=16,
                                     shape=None, occluder=False, shadows=False,
                                     shading_toony_factor=1.0))
    fb = flat.reshape(-1, flat.shape[-1])
    fb = fb[fb[:, 3] > 0.999][:, :3]
    r.append(("base == shade renders flat, which the retired BSDF could not do",
              len(fb) > 50
              and mtoon.delta_e(list(fb.max(axis=0)), list(fb.min(axis=0))) < 1.0))

    # The two terms the retired mtoon_forward dropped, at renderer level.
    dim, _ = rendered_plateaus(light_color=(0.5, 0.5, 0.5), shading_toony_factor=1.0)
    r.append(("lightColor reaches the film, which the retired forward ignored",
              dim is not None and hi_f is not None
              and abs(float(np.max(dim)) - float(np.max(hi_f)) * 0.5) < 0.02))

    mix0, _ = rendered_plateaus(light_color=(0.2, 0.2, 0.2),
                                parametric_rim_color_factor=(1.0, 1.0, 1.0),
                                rim_lighting_mix_factor=0.0)
    mix1, _ = rendered_plateaus(light_color=(0.2, 0.2, 0.2),
                                parametric_rim_color_factor=(1.0, 1.0, 1.0),
                                rim_lighting_mix_factor=1.0)
    r.append(("rimLightingMixFactor reaches the film, which the retired forward ignored",
              mix0 is not None and mix1 is not None
              and float(np.abs(np.asarray(mix0) - np.asarray(mix1)).max()) > 0.01))

    bad = sum(1 for _, ok in r if not ok)
    for name, ok in r:
        print("  %-4s control: %s" % ("ok" if ok else "FAIL", name))
    print("  %d of %d controls fired." % (len(r) - bad, len(r)))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    ap.error("pass --self-test")


if __name__ == "__main__":
    # If the llvm_ad_rgb backend was used, drjit 1.5.0 + mitsuba 3.9.1 corrupt process
    # exit on this box: sys.exit(3) returns 139 and os._exit(3) returns 127, on a green
    # run and a red one alike. mtoon_forward prefers cuda_ad_rgb partly for that reason.
    # Where only llvm is available, read the "N of M controls fired" line, not $?.
    sys.exit(main())
