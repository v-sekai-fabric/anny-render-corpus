# SPDX-License-Identifier: Apache-2.0 OR MIT
"""The MToon 1.0 renderer: a wide forward Mitsuba integrator over mtoon.evaluate_wide.

MToon is not a BSDF. Its shade colour is painted where dot(N, L) < 0, and a physically
based integrator contributes nothing there, so a BSDF renders lit-to-black and the shade
plateau never reaches film. This shades every hit against the key light directly, so both
plateaus land.

This is the only MToon render path. It replaced two others on 2026-09-06:

  * the BSDF that mtoon.py used to register, driven by mtoon_sweep.py. Broken as above,
    and its self-test pinned the breakage -- it asserted that base == shade renders with
    contrast > 1.0, which is exactly what a correct MToon renderer must not do.
  * mtoon_integrator.py, a scalar version of this integrator with no callers.

    python mtoon_forward.py --self-test [--bench]
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mtoon  # noqa: E402

SHAPE = pathlib.Path(__file__).resolve().parent / "mtoon-reference" / "testshape.obj"

# Widest variant this machine can actually initialise, chosen at run time. Hardcoding
# llvm_ad_rgb made every render here unrunnable on a box with two CUDA GPUs and no
# LLVM-C.dll.
#
# CUDA is tried first: it is the faster wide backend wherever a device exists, and on this
# box it is also the only wide backend that leaves the process able to exit. With drjit
# 1.5.0 + mitsuba 3.9.1, once llvm_ad_rgb has rendered, teardown segfaults -- `sys.exit(3)`
# returns 139 and even `os._exit(3)` returns 127 -- so a suite's exit code stops meaning
# anything. cuda_ad_rgb and scalar_rgb both return 3 from the same probe.
WIDE_VARIANTS = ("cuda_ad_rgb", "llvm_ad_rgb")

_REGISTERED = False


def set_wide_variant():
    """Select and return a working wide variant, or fall back to scalar_rgb."""
    import mitsuba as mi
    if mi.variant() in WIDE_VARIANTS:
        return mi.variant()
    for v in WIDE_VARIANTS:
        try:
            mi.set_variant(v)
            import drjit as dr
            dr.zeros(mi.Float, 1)  # force backend init; set_variant alone is lazy
            return v
        except Exception:  # noqa: BLE001 -- any backend failure means "try the next one"
            continue
    mi.set_variant("scalar_rgb")
    return "scalar_rgb"


def register():
    global _REGISTERED
    if _REGISTERED:
        return
    import drjit as dr
    import mitsuba as mi

    class _Forward(mi.SamplingIntegrator):
        def __init__(self, props):
            mi.SamplingIntegrator.__init__(self, props)
            g = lambda k, d=0.0: float(props.get(k, d))  # noqa: E731
            v = np.array([g("light_x", 1.0), g("light_y", 0.2), g("light_z", 0.6)])
            self.light = tuple(float(x) for x in v / np.linalg.norm(v))
            self.base = [g("base_" + c) for c in "rgb"]
            self.shade = [g("shade_" + c) for c in "rgb"]
            self.rim = [g("rim_" + c) for c in "rgb"]
            self.light_color = [g("light_" + c, 1.0) for c in "rgb"]
            self.shadows = g("shadows", 1.0) > 0.5
            self.p = {k: g(k, mtoon.DEFAULTS[k]) for k in mtoon.PROP_KEYS}

        def _kw(self):
            kw = dict(self.p)
            kw["parametric_rim_color_factor"] = tuple(self.rim)
            return kw

        def sample(self, scene, sampler, ray, medium=None, active=True):
            si = scene.ray_intersect(ray, active)
            hit = si.is_valid() & active
            n = si.sh_frame.n
            to_light = mi.Vector3f(*self.light)
            dot_nl = dr.dot(n, to_light)
            dot_nv = dr.dot(n, -ray.d)

            if self.shadows:
                lit = hit & (dot_nl > 0.0)
                occluded = scene.ray_test(si.spawn_ray(to_light), lit)
                dot_nl = dr.select(occluded, mi.Float(-1.0), dot_nl)

            col = mtoon.evaluate_wide(dot_nl, dot_nv, self.base, self.shade,
                                      self.light_color, **self._kw())
            return dr.select(hit, col, mi.Color3f(0.0)), hit, []

        def to_string(self):
            return "MToonForward[]"

    mi.register_integrator("mtoon_forward", lambda props: _Forward(props))
    _REGISTERED = True


def integrator_dict(base, shade, light=(1.0, 0.2, 0.6), light_color=(1.0, 1.0, 1.0),
                    shadows=True, **kw):
    p = dict(mtoon.DEFAULTS, **kw)
    d = {"type": "mtoon_forward", "shadows": 1.0 if shadows else 0.0}
    for i, name in enumerate("rgb"):
        d["base_" + name] = float(base[i])
        d["shade_" + name] = float(shade[i])
        d["rim_" + name] = float(p["parametric_rim_color_factor"][i])
        d["light_" + name] = float(light_color[i])
    for i, name in enumerate("xyz"):
        d["light_" + name] = float(light[i])
    for k in mtoon.PROP_KEYS:
        d[k] = float(p[k])
    return d


def _shape_entries(shape, light=(1.0, 0.2, 0.6), occluder=True):
    """The test OBJ when it has been generated, else the stand-in it replaces.

    mtoon-reference/testshape.obj is gitignored and derives from thebasemesh-stage, which
    is not in this manifest, so a clean checkout has no shape at all and every render here
    used to die in mi.load_dict.

    The stand-in is a sphere plus a small occluder placed toward the key light. A bare
    sphere is convex, so `ray_test` toward the light never hits and the shadow controls
    cannot fire -- they would pass vacuously on a shape that can't cast one. The occluder
    puts a real shadow on the body, so the same controls mean the same thing either way.
    """
    import mitsuba as mi
    if shape is not None and pathlib.Path(shape).exists():
        return {"obj": {"type": "obj", "filename": str(shape)}}
    if not occluder:
        return {"obj": {"type": "sphere", "radius": 1.0}}
    v = np.array(light, dtype=float)
    v = v / np.linalg.norm(v) * 1.7
    return {
        "obj": {"type": "sphere", "radius": 1.0},
        "occluder": {
            "type": "sphere", "radius": 0.55,
            "to_world": mi.ScalarTransform4f().translate([float(c) for c in v]),
        },
    }


def build(base, shade, width=256, height=256, spp=1, shape=SHAPE, occluder=True, **kw):
    import mitsuba as mi
    set_wide_variant()
    register()
    scene = {
        "type": "scene",
        "integrator": integrator_dict(base, shade, **kw),
        "sensor": {
            "type": "perspective", "fov": 40,
            "to_world": mi.ScalarTransform4f().look_at(
                origin=[0, 0, 3], target=[0, 0, 0], up=[0, 1, 0]),
            "film": {"type": "hdrfilm", "width": width, "height": height,
                     "rfilter": {"type": "box"}, "pixel_format": "rgba"},
            "sampler": {"type": "independent", "sample_count": spp},
        },
    }
    scene.update(_shape_entries(shape, kw.get("light", (1.0, 0.2, 0.6)), occluder))
    return mi.load_dict(scene)


def retune(scene, base, shade, light=None, light_color=None, **kw):
    """Change the material in place; the integrator is a Python object."""
    it = scene.integrator()
    it.base = [float(c) for c in base]
    it.shade = [float(c) for c in shade]
    if light is not None:
        v = np.array(light, dtype=float)
        it.light = tuple(float(x) for x in v / np.linalg.norm(v))
    if light_color is not None:
        it.light_color = [float(c) for c in light_color]
    if "parametric_rim_color_factor" in kw:
        it.rim = [float(c) for c in kw.pop("parametric_rim_color_factor")]
    for k, v in kw.items():
        if k in it.p:
            it.p[k] = float(v)
    return scene


def render(base, shade, width=256, height=256, spp=1, shape=SHAPE, occluder=True, **kw):
    import mitsuba as mi
    scene = build(base, shade, width, height, spp, shape, occluder, **kw)
    return np.array(mi.render(scene, spp=spp))


def self_test():
    """Ten controls; seven reject a render that lost the model, the light or the shadows."""
    r = []
    base, shade = (0.72, 0.52, 0.32), (0.30, 0.21, 0.13)
    variant = set_wide_variant()
    print("  variant: %s" % variant)

    img = render(base, shade, 192, 192, shading_toony_factor=1.0)
    rgb, alpha = img[..., :3], img[..., 3]
    hit = alpha > 0.5
    r.append(("the shape renders", 0.05 < hit.mean() < 0.95))
    r.append(("alpha is coverage", np.all(rgb[~hit] == 0)))

    body = rgb[hit]
    r.append(("the lit plateau is the base colour",
              mtoon.delta_e(list(body.max(axis=0)), list(base)) < 1.5))
    r.append(("the shade plateau reaches film",
              mtoon.delta_e(list(body.min(axis=0)), list(shade)) < 1.5))

    lit_only = render(base, shade, 192, 192, shading_toony_factor=1.0, shadows=False)
    shaded = (rgb[hit] @ np.ones(3)) < (body.max() * 0.9)
    lit_frac = ((lit_only[..., :3][hit] @ np.ones(3)) < (body.max() * 0.9))
    r.append(("shadows change the picture, so the ray is doing something",
              not np.allclose(img, lit_only)))
    r.append(("shadows put MORE of the body in shade",
              shaded.mean() > lit_frac.mean()))

    flat = render(base, base, 192, 192, shading_toony_factor=1.0)
    fb = flat[..., :3][flat[..., 3] > 0.5]
    r.append(("base == shade renders flat",
              mtoon.delta_e(list(fb.max(axis=0)), list(fb.min(axis=0))) < 1.0))

    other = render((0.2, 0.2, 0.9), (0.05, 0.05, 0.3), 192, 192)
    r.append(("a different material renders differently", not np.allclose(img, other)))

    # The two terms the retired forward integrator dropped. Both must move the picture.
    half = render(base, shade, 96, 96, shading_toony_factor=1.0,
                  light_color=(0.5, 0.5, 0.5))
    full = render(base, shade, 96, 96, shading_toony_factor=1.0)
    r.append(("lightColor scales the render, so it is not being ignored",
              half[..., :3].max() < full[..., :3].max() * 0.75))

    mix0 = render(base, shade, 96, 96, light_color=(0.2, 0.2, 0.2),
                  parametric_rim_color_factor=(1.0, 1.0, 1.0),
                  rim_lighting_mix_factor=0.0)
    mix1 = render(base, shade, 96, 96, light_color=(0.2, 0.2, 0.2),
                  parametric_rim_color_factor=(1.0, 1.0, 1.0),
                  rim_lighting_mix_factor=1.0)
    r.append(("rim_lighting_mix_factor changes the rim, so it is not being ignored",
              not np.allclose(mix0, mix1)))

    bad = sum(1 for _, ok in r if not ok)
    for name, ok in r:
        print("  %-4s control: %s" % ("ok" if ok else "FAIL", name))
    print("  %d of %d controls fired." % (len(r) - bad, len(r)))
    return 1 if bad else 0


def bench(width=3840, height=2160):
    set_wide_variant()
    base, shade = (0.72, 0.52, 0.32), (0.30, 0.21, 0.13)
    render(base, shade, 256, 256)
    t0 = time.perf_counter()
    render(base, shade, width, height)
    dt = time.perf_counter() - t0
    px = width * height
    print("  forward %dx%d with shadows: %.3f s  (%.1f ns/px)" % (width, height, dt,
                                                                  dt / px * 1e9))
    print("  60 frames: %.1f s" % (60 * dt))
    return dt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    if args.bench:
        bench()
        return 0
    if args.self_test:
        return self_test()
    ap.error("pass --self-test or --bench")


if __name__ == "__main__":
    sys.exit(main())
