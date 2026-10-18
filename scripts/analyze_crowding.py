"""Analyze matched isolated/near/far families without assigning mechanism labels."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from analyze_predictions import interval


SPACINGS = (1.4, 2., 3.)
CONDITIONS = {"base", "target_plus", "distractor_plus", "both_plus", "swap", "cue_swap"}


def analyze(path):
    groups = defaultdict(dict)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        if row.get("protocol") != "crowding":
            raise ValueError("Expected crowding protocol metadata")
        key = (row["family_id"], row["delta_index"])
        name = row["condition"]
        condition = name if name.startswith("isolated") else (float(row["spacing"]), name)
        if condition in groups[key]:
            raise ValueError(f"Duplicate condition: {key}, {condition}")
        groups[key][condition] = row
    required = {"isolated", "isolated_target_plus"} | {(s, c) for s in SPACINGS for c in CONDITIONS}
    families = defaultdict(list)
    for (family, delta_index), group in groups.items():
        if set(group) != required:
            raise ValueError(f"Incomplete matched family: {family}, {delta_index}")
        isolated = group["isolated"]
        target = isolated["target_deg"]
        isolated_error = abs(isolated["prediction_deg"]-target)
        values = {"isolated_mae_deg": isolated_error,
                  "isolated_target_sensitivity": (group["isolated_target_plus"]["prediction_deg"]
                                                   -isolated["prediction_deg"])/2.}
        for spacing in SPACINGS:
            g = {c: group[spacing, c] for c in CONDITIONS}
            b = g["base"]
            if not np.isclose(b["target_deg"], target, atol=1e-5):
                raise ValueError("Target label changed across matched spacings")
            prediction = b["prediction_deg"]
            prefix = f"s{spacing:g}/"
            error = abs(prediction-target)
            values[prefix+"base_mae_deg"] = error
            values[prefix+"excess_over_isolated_deg"] = error-isolated_error
            values[prefix+"context_shift_deg"] = abs(prediction-isolated["prediction_deg"])
            values[prefix+"target_sensitivity"] = (g["target_plus"]["prediction_deg"]-prediction)/2.
            values[prefix+"distractor_sensitivity"] = (g["distractor_plus"]["prediction_deg"]-prediction)/2.
            values[prefix+"abs_distractor_shift_deg"] = abs(g["distractor_plus"]["prediction_deg"]-prediction)
            values[prefix+"error_gt3_fraction"] = float(error > 3.)
            # Descriptive nearest-label statistic; never a selection mechanism label.
            others = [a for i, a in enumerate(b["angles_deg"]) if i != b["target_index"]]
            if min(abs(a-target) for a in others) >= 10.:
                values[prefix+"closer_to_distractor_fraction"] = float(min(abs(prediction-a) for a in others) < error)
        values["near_minus_far_mae_deg"] = values["s1.4/base_mae_deg"]-values["s3/base_mae_deg"]
        families[family].append(values)
    if not families:
        raise ValueError("Empty matched predictions")
    expected = set(range(5 if "calibration" in path.name else 4))
    for family in families:
        if {d for f, d in groups if f == family} != expected:
            raise ValueError(f"Missing delta block: {family}")
    stats = []
    for family in sorted(families):
        values = families[family]
        keys = set.union(*(set(v) for v in values))
        stats.append({k: float(np.mean([v[k] for v in values if k in v])) for k in keys})
    rng = np.random.default_rng(20261018)
    keys = sorted(set.intersection(*(set(v) for v in stats)))
    return {"samples": len(rows), "families": len(families),
            "paired": {k: interval([v[k] for v in stats], rng) for k in keys},
            "note": "Three-object local intervention changes one designated distractor, not all distractors."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    results = {f"{p.parent.name}/{p.stem.removeprefix('predictions_')}": analyze(p)
               for p in sorted(args.root.glob("*/predictions_*_crowding.jsonl"))
               if "test" not in p.name}
    if not results:
        raise FileNotFoundError("No calibration/validation matched crowding predictions found")
    destination = args.root/"analysis"
    destination.mkdir(exist_ok=True)
    (destination/"crowding_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = ["# 同场景拥挤配对分析", "", "2000 次 scene-family bootstrap；区间不包含训练种子间变异。",
             "目标位置、方向、提示和背景固定，只移动或移除干扰物。三对象的局部干预仅改变指定的一个干扰物。",
             "最近属性比例仅为行为描述；不能推断内部选择或聚合。", "",
             "E2 继续条件：孤立 MAE≤3°；近减远 MAE≥2°且区间下界>0；还需可复现的配对干扰效应。",
             "均值门槛与置信区间分别报告，不自动赋予机制标签。", ""]
    for name, result in results.items():
        lines += [f"## {name}", "", f"{result['families']} families，{result['samples']} 张图。", "",
                  "| 指标 | 均值 | 95% CI |", "|---|---:|---:|"]
        for key, value in result["paired"].items():
            lo, hi = value["ci95"]
            lines.append(f"| {key} | {value['mean']:.5f} | [{lo:.5f}, {hi:.5f}] |")
        lines.append("")
    (destination/"CROWDING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(destination.resolve())


if __name__ == "__main__":
    main()
