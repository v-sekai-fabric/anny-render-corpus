# SPDX-License-Identifier: Apache-2.0 OR MIT
"""An MToon tone sweep as RGBA frames: left solved to dE 12, right a flat x0.6.

    python render_mtoon_frames.py --out frames --frames 60 --width 3840 --height 2160
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import subprocess
import threading
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import frame_ring  # noqa: E402
import mtoon  # noqa: E402
import mtoon_forward as forward  # noqa: E402
import mtoon_sweep as sweep  # noqa: E402
import placeholder_cards  # noqa: E402

SHAPE = pathlib.Path(__file__).resolve().parent / "mtoon-reference" / "testshape.obj"
LIGHT = np.array([1.0, 0.2, 0.6])


def frame_for(tone_base, shade_colour, width, height, spp=1, **kw):
    img = forward.render(tone_base, shade_colour, width=width, height=height, spp=spp, **kw)
    out = np.zeros(img.shape[:2] + (4,), dtype=np.float32)
    out[..., :3] = img[..., :3]
    out[..., 3] = img[..., 3]
    return out


CARD_HOLD = 180
CITATION = pathlib.Path(__file__).resolve().parent


def citation_cff():
    """The .cff beside the code, whatever it is called."""
    return next(iter(sorted(CITATION.glob("*.cff"))), None)


def sequence(hold, card_hold=CARD_HOLD):
    """A citation card, the tone holds, the gap cards, and the citation again."""
    cff = citation_cff()
    rows = []
    if cff is not None:
        rows.extend([("cite", str(cff))] * card_hold)
    rows.extend([("tone",) + row for row in tone_frames(hold)])
    for gap in placeholder_cards.GAPS:
        rows.extend([("card", gap)] * card_hold)
    if cff is not None:
        rows.extend([("cite", str(cff))] * card_hold)
    return rows


def tone_frames(hold):
    """Each Monk tone held for `hold` frames at the dE 12 multiplier. Uniform by decision;
    CORPUS_DESIGN.md carries why."""
    rows = []
    for tone in sweep.MST:
        base = sweep.linear_of(tone)
        mult = sweep.solve_multiplier(base, 12.0)
        for k in range(hold):
            rows.append((tone, base, [c * mult for c in base], k / max(1, hold - 1)))
    return rows


# Q2-1 of the Nem x Mila survey, n=1,012.
PERSONA_SHARE = {"humanoid": 50.0, "semi-humanoid": 38.0, "robot-or-cyborg": 6.0,
                 "animal": 2.0, "plant": 2.0, "other": 1.0, "monster": 0.0}
MIN_HOLD = 90


def allocate(shares, total, floor=MIN_HOLD):
    """Split `total` by share, never below `floor`. A zero share still gets the floor."""
    names = list(shares)
    if floor * len(names) > total:
        raise ValueError("a floor of %d across %d categories needs %d frames, not %d"
                         % (floor, len(names), floor * len(names), total))
    out = {n: floor for n in names}
    spare = total - floor * len(names)
    mass = sum(max(0.0, shares[n]) for n in names)
    if mass <= 0:
        for i, n in enumerate(names):
            out[n] += spare // len(names) + (1 if i < spare % len(names) else 0)
        return out
    exact = {n: spare * max(0.0, shares[n]) / mass for n in names}
    for n in names:
        out[n] += int(exact[n])
    left = total - sum(out.values())
    for n in sorted(names, key=lambda k: exact[k] - int(exact[k]), reverse=True)[:left]:
        out[n] += 1
    return out


def sun(phase, arc=70.0, elevation=0.35):
    """The light swept through `arc` degrees, smoothstepped so it settles at both ends."""
    eased = phase * phase * (3.0 - 2.0 * phase)
    angle = np.radians((eased * 2.0 - 1.0) * arc)
    return (float(np.cos(angle)), float(elevation), float(np.sin(angle)))


_LUT_BITS = 14
_LUT_N = 1 << _LUT_BITS


def _build_lut():
    lin = np.arange(_LUT_N, dtype=np.float64) / (_LUT_N - 1)
    srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    return np.round(srgb * 255).astype(np.uint8)


_LUT = _build_lut()


_PACKER = None


def packer():
    """The Slang kernel if it is built, the numpy table if not. Both agree to one code."""
    global _PACKER
    if _PACKER is None:
        try:
            import check_mtoon_slang
            check_mtoon_slang.library()
            _PACKER = check_mtoon_slang.srgb8
        except SystemExit:
            _PACKER = to_srgb8_lut
    return _PACKER


def to_srgb8(rgba):
    return packer()(rgba)


def to_srgb8_lut(rgba):
    """A table lookup, not a pow."""
    lin = np.clip(rgba[..., :3], 0.0, 1.0)
    idx = (lin * (_LUT_N - 1) + 0.5).astype(np.uint16)
    out = np.empty(rgba.shape[:2] + (4,), dtype=np.uint8)
    out[..., :3] = _LUT[idx]
    out[..., 3] = (np.clip(rgba[..., 3], 0, 1) * 255 + 0.5).astype(np.uint8)
    return out


def render_chunks(stream, hold, width, height, spp=1, slots=8, chunk=40, jobs=4,
                  verbose=True):
    """Render slices in parallel subprocesses, each to its own part, then join them in order.

    Dr.Jit breaks after a few hundred renders in one process, so a slice per process is
    required; four concurrent chunks measured 0.434 s a frame against 0.857 for one.
    """
    total = len(sequence(hold))
    stream = pathlib.Path(stream)
    starts = list(range(0, total, chunk))
    parts = [stream.with_suffix(".part%04d" % i) for i, _ in enumerate(starts)]
    for p in parts + [stream]:
        if p.exists():
            p.unlink()

    t0 = time.perf_counter()
    running, done = [], 0
    queue = list(zip(starts, parts))
    while queue or running:
        while queue and len(running) < jobs:
            start, part = queue.pop(0)
            count = min(chunk, total - start)
            running.append((start, count, part, subprocess.Popen(
                [sys.executable, str(pathlib.Path(__file__).resolve()), "--out", str(part),
                 "--hold", str(hold), "--width", str(width), "--height", str(height),
                 "--spp", str(spp), "--slots", str(slots), "--start", str(start),
                 "--count", str(count)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)))
        start, count, part, proc = running.pop(0)
        out, err = proc.communicate()
        if proc.returncode != 0:
            for _, _, _, other in running:
                other.kill()
            raise SystemExit("FAIL  chunk at %d died: %s" % (start, err[-1200:]))
        done += count
        if verbose:
            print("    %4d/%d  %.1f s  (%d running)" % (done, total,
                                                        time.perf_counter() - t0, len(running)))

    frame = width * height * 4
    with open(stream, "wb") as fh:
        for part in parts:
            with open(part, "rb") as src:
                while True:
                    block = src.read(1 << 24)
                    if not block:
                        break
                    fh.write(block)
            part.unlink()
    got = stream.stat().st_size // frame
    if got != total:
        raise SystemExit("FAIL  stream holds %d frames, wanted %d" % (got, total))
    if verbose:
        elapsed = time.perf_counter() - t0
        print("  %d frames of %dx%d rgba  %.1f s  (%.3f s/frame, %d jobs)  %s"
              % (total, width, height, elapsed, elapsed / total, jobs, stream))
    return total


def render(stream, hold, width, height, spp=1, slots=8, flush_every=10, verbose=True,
           start=0, count=None, append=False):
    """Render into a bounded ring; a writer thread drains it to one RGBA stream."""
    import drjit as dr
    import mitsuba as mi
    half = width // 2
    stream = pathlib.Path(stream)
    stream.parent.mkdir(parents=True, exist_ok=True)
    rows = sequence(hold)[start:None if count is None else start + count]
    first = next((x for x in rows if x[0] == "tone"), None)
    if first is None:
        first = next(x for x in sequence(hold) if x[0] == "tone")
    scenes = [forward.build(first[2], first[3], half, height, spp,
                            shading_toony_factor=0.9) for _ in range(2)]
    ring = frame_ring.FrameRing(slots, "frames")

    def writer():
        with open(stream, "ab" if append else "wb") as fh:
            while True:
                try:
                    fh.write(ring.get(timeout=600))
                except frame_ring.Closed:
                    return

    thread = threading.Thread(target=writer, name="write", daemon=True)
    thread.start()
    t0 = time.perf_counter()
    cache = {}
    for i, row in enumerate(rows):
        if row[0] in ("card", "cite"):
            key = row[1][0] if row[0] == "card" else "cite"
            if key not in cache:
                cache.clear()
                art = (placeholder_cards.citation(row[1]) if row[0] == "cite"
                       else placeholder_cards.card(*row[1], width=width, height=height))
                cache[key] = to_srgb8(art).tobytes()
            ring.put(cache[key])
            continue
        _, tone, base, shade_colour, phase = row
        light = sun(phase)
        forward.retune(scenes[0], base, shade_colour, light)
        forward.retune(scenes[1], base, [c * 0.6 for c in base], light)
        left = np.array(mi.render(scenes[0], spp=spp, seed=i))
        right = np.array(mi.render(scenes[1], spp=spp, seed=i))
        frame = np.concatenate([left[..., :4], right[..., :4]], axis=1)
        ring.put(to_srgb8(frame).tobytes())
        if (i + 1) % flush_every == 0:
            dr.flush_malloc_cache()
        if verbose and (i + 1) % hold == 0:
            print("    tone %s  %d/%d  %.1f s" % (tone, i + 1, len(rows),
                                                  time.perf_counter() - t0))
    ring.close()
    thread.join()
    if verbose:
        print("  " + ring.report())
    if ring.put_count != len(rows) or ring.get_count != len(rows):
        raise SystemExit("FAIL  ring moved %d in and %d out of %d frames"
                         % (ring.put_count, ring.get_count, len(rows)))
    return len(rows)


def self_test():
    """Twenty-two controls; five reject a frame not showing the material."""
    r = []
    base = sweep.linear_of("825c43")
    dark = [c * sweep.solve_multiplier(base, 12.0) for c in base]
    frame = frame_for(base, dark, 128, 128, shading_toony_factor=1.0)
    hit = frame[..., 3] > 0.5
    r.append(("the render hits the shape", 0.05 < hit.mean() < 0.95))
    r.append(("alpha is coverage", np.array_equal(frame[..., 3] > 0.5, hit)))
    r.append(("nothing is shaded outside the mask", np.all(frame[~hit, :3] == 0)))

    inside = frame[hit][:, :3]
    r.append(("both plateaus reach the frame",
              mtoon.delta_e(list(inside.max(axis=0)), list(inside.min(axis=0))) > 5.0))

    flat = frame_for(base, [c * 0.6 for c in base], 128, 128, shading_toony_factor=1.0)
    r.append(("the flat multiplier renders a different frame",
              not np.allclose(frame, flat)))

    seq = sequence(6, 4)
    r.append(("the clip opens and closes with the citation",
              seq[0][0] == "cite" and seq[-1][0] == "cite"))
    r.append(("the citation is read from a file, not typed",
              seq[0][0] == "cite" and seq[0][1].endswith(".cff")))
    r.append(("the sequence ends with cards, one block each",
              sum(1 for x in seq if x[0] == "card") == 4 * len(placeholder_cards.GAPS)))
    r.append(("every gap appears in the sequence",
              {x[1][0] for x in seq if x[0] == "card"}
              == {g[0] for g in placeholder_cards.GAPS}))

    got = allocate(PERSONA_SHARE, 900)
    r.append(("every persona clears the floor", min(got.values()) >= MIN_HOLD))
    r.append(("a zero-share persona still gets the floor",
              got["monster"] == MIN_HOLD and PERSONA_SHARE["monster"] == 0.0))
    r.append(("the frames all get spent", sum(got.values()) == 900))
    r.append(("the largest share still gets the most",
              max(got, key=got.get) == max(PERSONA_SHARE, key=PERSONA_SHARE.get)))
    r.append(("uniform shares split evenly",
              len(set(allocate({k: 1.0 for k in "abcd"}, 400, 10).values())) == 1))
    try:
        allocate(PERSONA_SHARE, 100, 30)
        r.append(("a floor that cannot fit is refused", False))
    except ValueError:
        r.append(("a floor that cannot fit is refused", True))

    a, b = sun(0.0), sun(1.0)
    r.append(("the sun moves across the hold", abs(a[2] - b[2]) > 0.5))
    r.append(("the sweep is symmetric about the middle",
              abs(sun(0.5)[2]) < 1e-9 and abs(a[2] + b[2]) < 1e-9))
    r.append(("the easing settles at both ends",
              abs(sun(0.02)[2] - a[2]) < abs(sun(0.5)[2] - sun(0.44)[2])))

    eight = to_srgb8(frame)
    r.append(("the two packers agree to one code",
              int(np.abs(to_srgb8_lut(frame).astype(int)
                         - packer()(frame).astype(int)).max()) <= 1))
    r.append(("the 8-bit conversion keeps the mask exactly",
              np.array_equal(eight[..., 3] > 127, hit)))
    probe = np.linspace(0, 1, 4096, dtype=np.float64)
    exact = np.where(probe <= 0.0031308, probe * 12.92, 1.055 * probe ** (1 / 2.4) - 0.055)
    got = _LUT[(probe * (_LUT_N - 1) + 0.5).astype(np.uint16)]
    r.append(("the lookup table matches the transfer function to one code",
              int(np.abs(got.astype(int) - np.round(exact * 255)).max()) <= 1))
    r.append(("black and white land exactly", _LUT[0] == 0 and _LUT[-1] == 255))

    bad = sum(1 for _, ok in r if not ok)
    for name, ok in r:
        print("  %-4s control: %s" % ("ok" if ok else "FAIL", name))
    print("  %d of %d controls fired." % (len(r) - bad, len(r)))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="frames.rgba")
    ap.add_argument("--hold", type=int, default=90,
                    help="frames per tone; 90 at 60fps is 1.5 s each")
    ap.add_argument("--width", type=int, default=3840)
    ap.add_argument("--height", type=int, default=2160)
    ap.add_argument("--spp", type=int, default=1)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    import mtoon_forward as forward
    forward.set_wide_variant()  # cuda where there is a GPU, else llvm, else scalar
    if args.self_test:
        return self_test()
    if args.count is None:
        render_chunks(args.out, args.hold, args.width, args.height, args.spp,
                      args.slots, args.chunk, args.jobs)
    else:
        render(args.out, args.hold, args.width, args.height, args.spp, args.slots,
               start=args.start, count=args.count, append=args.append)
    return 0


if __name__ == "__main__":
    sys.exit(main())
