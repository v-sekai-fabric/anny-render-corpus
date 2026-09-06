# SPDX-License-Identifier: Apache-2.0 OR MIT
"""MToon 1.0 (VRMC_materials_mtoon-1.0), as the spec's pseudocode writes it.

    python mtoon.py --self-test
    python mtoon.py --ramp
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np

EPS = 1e-6

DEFAULTS = {
    "shading_shift_factor": 0.0,
    "shading_toony_factor": 0.9,
    "shading_shift_texture": 0.0,
    "parametric_rim_color_factor": (0.0, 0.0, 0.0),
    "parametric_rim_lift_factor": 0.0,
    "parametric_rim_fresnel_power_factor": 1.0,
    "rim_lighting_mix_factor": 0.0,
}

PROP_KEYS = ("shading_shift_factor", "shading_toony_factor", "shading_shift_texture",
             "parametric_rim_lift_factor", "parametric_rim_fresnel_power_factor",
             "rim_lighting_mix_factor")


def linearstep(a, b, t):
    if abs(b - a) <= EPS:
        return 1.0 if t >= a else 0.0
    return float(np.clip((t - a) / (b - a), 0.0, 1.0))


def shading(dot_nl, **kw):
    """linearstep(-1 + toony, 1 - toony, dot(N,L) + shift + shiftTexture)."""
    p = dict(DEFAULTS, **kw)
    toony = float(np.clip(p["shading_toony_factor"], 0.0, 1.0))
    s = dot_nl + p["shading_shift_factor"] + p["shading_shift_texture"]
    return linearstep(-1.0 + toony, 1.0 - toony, s)


def shade_mix(dot_nl, base, shade, **kw):
    t = shading(dot_nl, **kw)
    base, shade = np.asarray(base, float), np.asarray(shade, float)
    return shade + (base - shade) * t, t


def rim_term(dot_nv, lighting=(1.0, 1.0, 1.0), **kw):
    p = dict(DEFAULTS, **kw)
    r = np.clip(1.0 - dot_nv + p["parametric_rim_lift_factor"], 0.0, 1.0)
    r = r ** max(p["parametric_rim_fresnel_power_factor"], EPS)
    rim = np.asarray(p["parametric_rim_color_factor"], float) * r
    white = np.ones(3)
    return rim * (white + (np.asarray(lighting, float) - white)
                  * p["rim_lighting_mix_factor"])


def evaluate(dot_nl, dot_nv, base, shade, light_color=(1.0, 1.0, 1.0), **kw):
    """lerp(shade, base, shading) * lightColor, plus rim."""
    col, t = shade_mix(dot_nl, base, shade, **kw)
    col = col * np.asarray(light_color, float)
    return col + rim_term(dot_nv, light_color, **kw), t


def evaluate_wide(dot_nl, dot_nv, base, shade, light_color=(1.0, 1.0, 1.0), **kw):
    """`evaluate`, in Dr.Jit ops, so a Mitsuba integrator widens it over a whole film.

    Same model as `evaluate` and as mtoon.slang's `mtoonEvaluate`; check_mtoon_render.py
    holds this against `evaluate`, and check_mtoon_slang.py holds `evaluate` against the
    Slang kernel. One model, three evaluation widths, both joins gated.

    `dot_nl` and `dot_nv` are Dr.Jit floats. Returns an mi.Color3f. drjit and mitsuba are
    imported here, not at module scope, so mtoon.py stays importable without them --
    check_mtoon_slang.py imports this module with neither installed.
    """
    import drjit as dr
    import mitsuba as mi

    p = dict(DEFAULTS, **kw)
    toony = min(max(float(p["shading_toony_factor"]), 0.0), 1.0)
    lo, hi = -1.0 + toony, 1.0 - toony
    s = dot_nl + float(p["shading_shift_factor"]) + float(p["shading_shift_texture"])
    if abs(hi - lo) <= EPS:
        t = dr.select(s >= lo, mi.Float(1.0), mi.Float(0.0))
    else:
        t = dr.clip((s - lo) / (hi - lo), 0.0, 1.0)

    light = mi.Color3f(*[float(c) for c in light_color])
    col = (mi.Color3f(*[float(c) for c in shade])
           + (mi.Color3f(*[float(c) for c in base])
              - mi.Color3f(*[float(c) for c in shade])) * t) * light

    r = dr.clip(1.0 - dot_nv + float(p["parametric_rim_lift_factor"]), 0.0, 1.0)
    r = dr.power(r, max(float(p["parametric_rim_fresnel_power_factor"]), EPS))
    white = mi.Color3f(1.0, 1.0, 1.0)
    mix = float(p["rim_lighting_mix_factor"])
    rim = (mi.Color3f(*[float(c) for c in p["parametric_rim_color_factor"]]) * r
           * (white + (light - white) * mix))
    return col + rim


def sweep(samples=64, **kw):
    """The ramp over the hemisphere."""
    nl = np.linspace(-1.0, 1.0, samples)
    return nl, np.array([shading(x, **kw) for x in nl])


def _lab(rgb):
    m = ((.4124, .3576, .1805), (.2126, .7152, .0722), (.0193, .1192, .9505))
    x, y, z = (sum(m[i][j] * rgb[j] for j in range(3)) for i in range(3))
    w = (.95047, 1.0, 1.08883)
    f = lambda t: t ** (1 / 3) if t > 216 / 24389 else (841 / 108) * t + 4 / 29  # noqa: E731
    fx, fy, fz = f(x / w[0]), f(y / w[1]), f(z / w[2])
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def delta_e(a, b):
    return math.dist(_lab(a), _lab(b))


def rendered_contrast(base, shade, **kw):
    """dE between the fully lit and fully shaded plateaus."""
    hi, _ = shade_mix(1.0, base, shade, **kw)
    lo, _ = shade_mix(-1.0, base, shade, **kw)
    return delta_e(list(hi), list(lo))


def self_test():
    """Eleven controls. Six must reject a model that drifted from the spec."""
    r = []
    base, shade = (0.8, 0.7, 0.6), (0.4, 0.35, 0.3)
    ramp = [shading(v) for v in np.linspace(-1, 1, 50)]

    r.append(("full light saturates the ramp", shading(1.0) == 1.0))
    r.append(("full shadow bottoms the ramp", shading(-1.0) == 0.0))
    r.append(("the ramp is monotone in dot_nl",
              all(x <= y + 1e-12 for x, y in zip(ramp, ramp[1:]))))
    r.append(("toony 1 gives a hard step",
              [shading(v, shading_toony_factor=1.0) for v in (-0.01, 0.01)] == [0.0, 1.0]))
    r.append(("toony 0 is a full-width ramp",
              abs(shading(0.0, shading_toony_factor=0.0) - 0.5) < 1e-9
              and 0.0 < shading(-0.5, shading_toony_factor=0.0) < 0.5))
    r.append(("shift moves the terminator",
              shading(0.0, shading_shift_factor=0.5)
              > shading(0.0, shading_shift_factor=-0.5)))
    r.append(("the shift texture adds to the shift, as the spec says",
              abs(shading(0.0, shading_shift_factor=0.2)
                  - shading(0.0, shading_shift_texture=0.2)) < 1e-12))
    r.append(("the mix reaches base and shade exactly",
              np.allclose(shade_mix(1.0, base, shade)[0], base)
              and np.allclose(shade_mix(-1.0, base, shade)[0], shade)))
    r.append(("rim is zero without a rim colour",
              np.allclose(rim_term(0.5, rim_lighting_mix_factor=1.0), 0.0)))
    r.append(("rim rises toward the silhouette",
              rim_term(0.1, parametric_rim_color_factor=(1, 1, 1))[0]
              > rim_term(0.9, parametric_rim_color_factor=(1, 1, 1))[0]))
    r.append(("rim_lighting_mix 0 lerps toward white, not black",
              np.allclose(rim_term(0.0, lighting=(0, 0, 0),
                                   parametric_rim_color_factor=(1, 1, 1),
                                   rim_lighting_mix_factor=0.0), 1.0)))

    bad = sum(1 for _, ok in r if not ok)
    for name, ok in r:
        print("  %-4s control: %s" % ("ok" if ok else "FAIL", name))
    print("  %d of %d controls fired." % (len(r) - bad, len(r)))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--ramp", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.ramp:
        for toony in (0.0, 0.5, 0.9, 1.0):
            _, t = sweep(samples=11, shading_toony_factor=toony)
            print("  toony %.1f  " % toony + " ".join("%.2f" % v for v in t))
        return 0
    ap.error("pass --self-test or --ramp")


if __name__ == "__main__":
    sys.exit(main())
