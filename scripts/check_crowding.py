"""Check pairing invariants and analysis against known task outputs (no training)."""
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from measure_distinguish.data import CROWDING_VARIANTS, OrientationDataset, make_scene
from analyze_crowding import analyze


def main():
    for objects in (2, 3):
        for split in ("calibration", "validation"):
            dataset = OrientationDataset(split, 16, seed=20261018, objects=objects, protocol="crowding",
                                         target_anchored=objects == 3)
            rows = []
            for start in range(0, len(dataset), CROWDING_VARIANTS):
                scenes = [dataset.scene_at(start+i) for i in range(CROWDING_VARIANTS)]
                isolated = scenes[0]
                bases = [scenes[i] for i in (2, 8, 14)]
                if objects == 3:
                    family = start//dataset.variants
                    for base in bases:
                        iid = make_scene(20261018,split,family,objects=3,target_anchored=True)
                        np.testing.assert_allclose(iid.centers[iid.target], base.centers[base.target], atol=1e-12)
                for base in bases:
                    assert base.centers[base.target] == isolated.centers[0]
                    assert base.label == isolated.label
                    assert base.background == isolated.background
                    assert base.angles == bases[0].angles
                    offsets = np.asarray(base.centers)-np.asarray(isolated.centers[0])
                    reference = np.asarray(bases[0].centers)-np.asarray(isolated.centers[0])
                    np.testing.assert_allclose(offsets/base.spacing, reference/bases[0].spacing, atol=1e-12)
                # Perfect target output: all paired error and distractor effects must vanish.
                for i, scene in enumerate(scenes):
                    family, variant = divmod(start+i, dataset.variants)
                    rows.append({"family_id": scene.family_id, "protocol": "crowding",
                                 "delta_index": variant//CROWDING_VARIANTS, "condition": scene.condition,
                                 "spacing": scene.spacing, "target_deg": scene.label,
                                 "prediction_deg": scene.label, "angles_deg": scene.angles,
                                 "target_index": scene.target})
            for index in (0, 2, 8, 14):
                sample = dataset[index]
                assert sample["image"].shape == (3, 128, 128)
                assert int(sample["object_masks"].sum()) > 0
            temporary_root = Path(__file__).resolve().parents[1]/"outputs"
            temporary_root.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(dir=temporary_root, prefix="check_crowding_") as directory:
                path = Path(directory)/f"predictions_{split}_crowding.jsonl"
                path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
                result = analyze(path)
                for key, value in result["paired"].items():
                    expected = 1. if key.endswith("target_sensitivity") else 0.
                    assert np.isclose(value["mean"], expected), (key, value)
                # A deliberately contaminated near output must yield the paired MAE cost.
                for row in rows:
                    if row["condition"] == "base" and row["spacing"] == 1.4:
                        row["prediction_deg"] += 4.
                path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
                result = analyze(path)
                assert np.isclose(result["paired"]["near_minus_far_mae_deg"]["mean"], 4.)
                # Partial families must fail rather than produce a misleading estimate.
                path.write_text("\n".join(json.dumps(r) for r in rows[:-1]), encoding="utf-8")
                try:
                    analyze(path)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Incomplete family accepted")
            print(f"Passed: objects={objects}, split={split}, {len(dataset)} paired scenes")


if __name__ == "__main__":
    main()
