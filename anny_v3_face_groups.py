"""Stopgap: assign ANNY mesh triangles to See-Through V3 body-part labels.

RFD 2183 (MaskScore-driven layer decomposition on OmniGen2) needs a
face_groups dict per ANNY mesh so the layer-decomposition renderer can
emit one image per part. ANNY has no face_groups attribute today.

This module is the stopgap: it clusters triangles by nearest joint, then
maps joint names to See-Through's VALID_BODY_PARTS_V3 (23 parts) via a
lookup table. Parts that no ANNY joint covers (front-hair, back-hair,
eyelash, iris, eyewhite, eyebrow, handwear) are named and counted per
CLAUDE.md rule 3 rather than silently skipped.

The full taxonomy work is Path C in a later RFD (hierarchical OID
registry under PEN); this file is what runs today.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np


# See-Through's V3 taxonomy: 23 anime-illustration body parts.
# Sourced from common/live2d/scrap_model.py in shitagaki-lab/see-through.
VALID_BODY_PARTS_V3: tuple[str, ...] = (
    "face", "front-hair", "back-hair", "eyebrow", "eyelash", "iris",
    "eyewhite", "mouth", "ear", "neck", "torso-front", "torso-back",
    "arm-upper", "arm-lower", "hand", "handwear", "hip", "leg-upper",
    "leg-lower", "foot", "footwear", "accessory", "background",
)

# ANNY joint name → V3 part. Only joints that map to a V3 part appear here.
# Facial-detail V3 parts (eyelash, iris, eyewhite, eyebrow, mouth-inner)
# have no joint in a humanoid rig; they surface as `uncovered_parts` in the
# returned dict so the caller can see what this stopgap does not deliver.
JOINT_TO_V3: dict[str, str] = {
    "head": "face",
    "neck": "neck",
    "spine": "torso-front",
    "chest": "torso-front",
    "upper_chest": "torso-front",
    "shoulder_l": "arm-upper", "shoulder_r": "arm-upper",
    "upper_arm_l": "arm-upper", "upper_arm_r": "arm-upper",
    "lower_arm_l": "arm-lower", "lower_arm_r": "arm-lower",
    "hand_l": "hand", "hand_r": "hand",
    "hips": "hip",
    "upper_leg_l": "leg-upper", "upper_leg_r": "leg-upper",
    "lower_leg_l": "leg-lower", "lower_leg_r": "leg-lower",
    "foot_l": "foot", "foot_r": "foot",
}


def uncovered_parts() -> tuple[str, ...]:
    """V3 parts no ANNY joint covers today; the stopgap emits no faces for these."""
    mapped = set(JOINT_TO_V3.values())
    return tuple(p for p in VALID_BODY_PARTS_V3 if p not in mapped)


def face_groups_from_mesh(vertices: np.ndarray,
                          faces: np.ndarray,
                          joint_positions: np.ndarray,
                          joint_names: list[str]) -> dict[str, tuple[int, int]]:
    """Return {part: (f0, f1)} slicing `faces` into contiguous per-part runs.

    Each triangle is assigned to the V3 part of its nearest joint. Faces are
    reordered so per-part triangles are contiguous, matching the (f0, f1)
    slice contract render_layer_decomp_corpus.py expects.
    """
    tri_centroids = vertices[faces].mean(axis=1)
    d2 = ((tri_centroids[:, None, :] - joint_positions[None, :, :]) ** 2).sum(axis=2)
    nearest_joint = d2.argmin(axis=1)
    part_of_tri = np.array(
        [JOINT_TO_V3.get(joint_names[j], "accessory") for j in nearest_joint])
    order = np.argsort(part_of_tri, kind="stable")
    part_sorted = part_of_tri[order]
    faces[:] = faces[order]
    groups: dict[str, tuple[int, int]] = {}
    if len(part_sorted) == 0:
        return groups
    start = 0
    for i in range(1, len(part_sorted)):
        if part_sorted[i] != part_sorted[i - 1]:
            groups[part_sorted[start]] = (start, i)
            start = i
    groups[part_sorted[start]] = (start, len(part_sorted))
    return groups


def coverage_report(groups: dict[str, tuple[int, int]]) -> dict:
    """Report which V3 parts got triangles and which were left uncovered."""
    covered = set(groups.keys())
    return {
        "covered": sorted(covered),
        "uncovered_by_joint_map": sorted(set(uncovered_parts()) - covered),
        "v3_parts_total": len(VALID_BODY_PARTS_V3),
        "v3_parts_covered": len(covered),
    }


def _self_test() -> int:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                      [1, 1, 0], [1, 0, 1]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [3, 4, 5], [0, 4, 5]], dtype=np.int32)
    joints = np.array([[0.1, 0.1, 0.1], [1.0, 0.5, 0.5]], dtype=np.float32)
    names = ["head", "hips"]

    groups = face_groups_from_mesh(verts, faces.copy(), joints, names)
    assert "face" in groups, f"positive control: 'face' missing from {groups}"
    assert "hip" in groups, f"positive control: 'hip' missing from {groups}"

    empty_groups = face_groups_from_mesh(verts, np.zeros((0, 3), dtype=np.int32),
                                         joints, names)
    assert empty_groups == {}, f"negative control: empty faces should give empty groups, got {empty_groups}"

    all_unknown = face_groups_from_mesh(verts, faces.copy(), joints,
                                        ["nose_tip", "left_earlobe"])
    assert list(all_unknown.keys()) == ["accessory"], \
        f"negative control: unknown joints should fall back to 'accessory', got {all_unknown}"

    covered = coverage_report({"face": (0, 1), "hip": (1, 3)})
    assert covered["v3_parts_covered"] == 2
    assert "front-hair" in covered["uncovered_by_joint_map"]

    print("self-test PASS: 4 controls")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--coverage-report", action="store_true",
                   help="print which V3 parts this stopgap can and cannot fill")
    args = p.parse_args()

    if args.self_test:
        return _self_test()
    if args.coverage_report:
        print(json.dumps({
            "v3_parts_total": len(VALID_BODY_PARTS_V3),
            "joints_mapped": len(JOINT_TO_V3),
            "v3_parts_reached_by_joints": sorted(set(JOINT_TO_V3.values())),
            "v3_parts_uncovered": sorted(uncovered_parts()),
        }, indent=2))
        return 0

    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
