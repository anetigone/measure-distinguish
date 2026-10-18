"""Summarize paired synthetic interventions; bootstrap independent scene families."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def interval(values, rng):
    values = np.asarray(values, dtype=float)
    means = values[rng.integers(len(values), size=(2000, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "ci95": np.quantile(means, [.025, .975]).tolist(),
            "families": len(values)}


def analyze(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    groups = defaultdict(dict)
    for row in rows:
        # Six conditions are emitted together for each base delta.
        variant = int(row["sample_id"].rsplit(":", 1)[1])
        key = (row["family_id"], variant//6)
        if row["condition"] in groups[key]:
            raise ValueError(f"Duplicate condition: {key}")
        groups[key][row["condition"]] = row
    required = {"base", "target_plus", "distractor_plus", "both_plus", "swap", "cue_swap"}
    families = defaultdict(list)
    conditions = defaultdict(list)
    for row in rows:
        conditions[row["condition"]].append(abs(row["prediction_deg"]-row["target_deg"]))
    for (family, _), group in groups.items():
        if set(group) != required:
            raise ValueError(f"Incomplete intervention group: {family}")
        b = group["base"]
        pred = lambda name: group[name]["prediction_deg"]
        step = group["target_plus"]["target_deg"]-b["target_deg"]
        if not np.isclose(step, 2., atol=1e-4):
            raise ValueError("Expected a two-degree intervention")
        dt = (pred("target_plus")-pred("base"))/step
        dd = (pred("distractor_plus")-pred("base"))/step
        values = {"base_mae_deg": abs(pred("base")-b["target_deg"]),
                  "target_sensitivity": dt, "distractor_sensitivity": dd,
                  "abs_distractor_shift_deg": abs(pred("distractor_plus")-pred("base")),
                  "abs_interaction_deg": abs(pred("both_plus")-pred("target_plus")
                                             -pred("distractor_plus")+pred("base"))}
        for kind in ("swap", "cue_swap"):
            change = group[kind]["target_deg"]-b["target_deg"]
            if abs(change) > 1e-4:
                values[kind+"_tracking"] = (pred(kind)-pred("base"))/change
                values[kind+"_change_error_deg"] = abs(pred(kind)-pred("base")-change)
        values["spacing"] = b["spacing"]
        values["delta"] = b["delta_deg"]
        values["error"] = pred("base")-b["target_deg"]
        families[family].append(values)
    rng = np.random.default_rng(20261018)
    family_stats = []
    for group in families.values():
        available = set.union(*(set(v) for v in group)) - {"delta", "error"}
        stats = {key: float(np.mean([v[key] for v in group if key in v]))
                 for key in available}
        # Through-origin error-versus-delta coefficient; descriptive, not a mechanism estimate.
        delta = np.array([v["delta"] for v in group])
        stats["error_delta_slope"] = float(delta @ np.array([v["error"] for v in group])/(delta @ delta))
        family_stats.append(stats)
    keys = sorted(set.intersection(*(set(v) for v in family_stats)) - {"spacing"})
    result = {"samples": len(rows), "families": len(families), "pairs": len(groups),
              "condition_mae_deg": {k: float(np.mean(v)) for k, v in conditions.items()},
              "paired": {k: interval([v[k] for v in family_stats], rng) for k in keys},
              "spacing": {}}
    for spacing in sorted({v["spacing"] for v in family_stats}):
        subset = [v for v in family_stats if v["spacing"] == spacing]
        result["spacing"][str(spacing)] = {
            k: interval([v[k] for v in subset], rng)
            for k in ("base_mae_deg", "target_sensitivity", "distractor_sensitivity", "abs_distractor_shift_deg")}
    near = [v["base_mae_deg"] for v in family_stats if v["spacing"] == 1.4]
    far = [v["base_mae_deg"] for v in family_stats if v["spacing"] == 3.]
    if near and far:
        diffs = np.asarray(near)[rng.integers(len(near), size=(2000, len(near)))].mean(axis=1)
        diffs -= np.asarray(far)[rng.integers(len(far), size=(2000, len(far)))].mean(axis=1)
        result["near_minus_far_mae_deg"] = {"mean": float(np.mean(near)-np.mean(far)),
                                           "ci95": np.quantile(diffs, [.025, .975]).tolist()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    results = {}
    for run in sorted(args.root.iterdir()):
        if run.is_dir():
            for split in ("calibration", "validation"):
                path = run/f"predictions_{split}.jsonl"
                if path.exists():
                    results[f"{run.name}/{split}"] = analyze(path)
    if not results:
        raise FileNotFoundError("No calibration/validation predictions found")
    destination = args.root/"analysis"
    destination.mkdir(exist_ok=True)
    (destination/"summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = ["# 合成实验配对干预分析", "", "2000 次 bootstrap，以 scene family 为抽样单位。区间只反映当前训练 seed 的场景抽样误差。",
             "", "理想目标跟随：目标敏感性=1，干扰敏感性=0，属性交换/蓝环交换跟随率=1。", "",
             "这些量描述输入输出依赖，不能直接等同内部聚合权重。近远距比较来自不同 family，并非同场景距离配对。", ""]
    for name, result in results.items():
        lines += [f"## {name}", "", f"{result['families']} families，{result['samples']} 张图。", "",
                  "| 指标 | 均值 | 95% CI |", "|---|---:|---:|"]
        for key, value in result["paired"].items():
            lo, hi = value["ci95"]
            lines.append(f"| {key} | {value['mean']:.5f} | [{lo:.5f}, {hi:.5f}] |")
        lines += ["", "| 间距/直径 | base MAE (°) | 干扰敏感性 |", "|---|---:|---:|"]
        for spacing, stats in result["spacing"].items():
            lines.append(f"| {spacing} | {stats['base_mae_deg']['mean']:.4f} | {stats['distractor_sensitivity']['mean']:.5f} |")
        lines += ["", f"近距减远距 MAE：{result.get('near_minus_far_mae_deg')}", ""]
    (destination/"REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(destination.resolve())


if __name__ == "__main__":
    main()
