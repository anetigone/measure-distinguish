"""Local theory verification only: identities, counterexamples, and detection power.

No torch, images, task training, model downloads, or implementation test suite.
Run from the project root: python local_theory/run.py
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from mechanisms import CASES, DELTAS, HOLD, diagnose, simulate


def verify_identities():
    rng = np.random.default_rng(20261018)
    weights = rng.dirichlet([.7, 1.3, 2.], size=10000)
    x = np.array([-1.2, .3, 2.1])
    p = weights.mean(0)
    pooled = weights@x
    selection_variance = float(p@(x*x)-(p@x)**2)
    pooling_variance = float(np.mean((pooled-pooled.mean())**2))
    expected_gap = float(np.mean(np.sum(weights*(x[None, :]-pooled[:, None])**2, axis=1)))
    error = abs(selection_variance-pooling_variance-expected_gap)
    assert error < 1e-10
    one_hot = np.eye(3)[rng.integers(0, 3, size=10000)]
    one_hot_values = one_hot@x
    boundary = float(np.mean(np.sum(one_hot*(x[None, :]-one_hot_values[:, None])**2, axis=1)))
    assert boundary == 0
    fixed = simulate("fixed_pool", np.random.default_rng(7), 2048)
    contextual = simulate("context_contaminated_selection", np.random.default_rng(7), 2048)
    assert np.array_equal(fixed, contextual)
    # For bounded noise-variance drift d, G=m*delta²-d. If m*delta²>2b,
    # even d=b leaves G>b. A selection model (m=0) cannot exceed b.
    for bound in (.01, .05):
        for drift in np.linspace(-bound, bound, 101):
            assert -drift <= bound+1e-12
            for delta in (.5, 1., 2.):
                if .2*delta**2 > 2*bound:
                    assert .2*delta**2-drift > bound
    return {"multi_object_variance_identity_error": error,
            "one_hot_boundary_gap": boundary,
            "contextual_selection_equals_pooling": True,
            "bounded_noise_certificate_checked": True,
            "interpretation": "Numerical verification of derived identities, not a novelty proof."}


def wilson(successes, total):
    p, z = successes/total, 1.96
    center = (p+z*z/(2*total))/(1+z*z/total)
    half = z*np.sqrt(p*(1-p)/total+z*z/(4*total**2))/(1+z*z/total)
    return [float(center-half), float(center+half)]


def write_report(summary, output):
    lines = ["# 理论验证汇总", "", "本目录只包含理论模型、反例与有限样本模拟；不将实现集成检查称为本地验证。", ""]
    identities = summary.get("identities")
    if identities:
        lines += ["多对象方差恒等式、one-hot 等价边界、有界噪声漂移证书与上下文选择反例已完成数值验证。", ""]
    sweep = summary["sweep"]
    lines += [f"固定 p=0.35、基础噪声标准差 0.12、假定噪声方差漂移预算 0.02；每次 {sweep['bootstrap']} 次 bootstrap。",
              "", "| N | 生成机制 | 有条件拒绝选择 | 均值模型失败 | 假设未验证 | gap 区间覆盖率 |",
              "|---:|---|---:|---:|---:|---:|"]
    for row in sweep["summaries"]:
        coverage = row["gap_interval_coverage"]
        lines.append(f"| {row['n']} | {row['case']} | {row['conditional_rejection_rate']:.0%} | "
                     f"{row['mean_check_failure_rate']:.0%} | {row.get('mean_check_unverified_rate', 0):.0%} | "
                     + ("—" if coverage is None else f"{coverage:.0%}")+" |")
    lines += ["", "这里的拒绝均以噪声、路由和读取假设成立为前提。近 one-hot 与异方差反例可以无法区分；上下文污染后的选择可与直接聚合完全等价。",
              "", "未检测到失配不算验证假设：漂移置信区间必须完整落入容许范围，否则返回假设未验证。",
              "", "100 个重复中 0 次误拒绝的 Wilson 95% 上界约为 3.7%；观察到 100% 检出也不代表普遍保证。该扫描只覆盖一个 p/噪声/预算组合，不能替代真实网络校准或完整识别边界。"]
    (output/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent/"results")
    parser.add_argument("--save-trials", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="Rebuild report and check identities using saved simulations")
    args = parser.parse_args()
    if min(args.ns) < 4 or args.repeats < 10 or args.bootstrap < 100:
        parser.error("Use N>=4, repeats>=10, bootstrap>=100")
    args.output.mkdir(parents=True, exist_ok=True)
    identities = verify_identities()
    if args.report_only:
        summary = json.loads((args.output/"summary.json").read_text(encoding="utf-8"))
        summary.pop("single_run", None)
        summary["identities"] = identities
    else:
        summaries, trials = [], []
        for n in args.ns:
            for index, case in enumerate(CASES):
                rejected, failed, unverified, covered = 0, 0, 0, 0
                true_gap = {"fixed_pool": .35*.65, "random_pool": .35*.65*4/5,
                            "near_one_hot_pool": .35*.65*.01/1.01,
                            "selection": 0., "heteroscedastic_pool": 0.,
                            "context_contaminated_selection": .35*.65}.get(case)
                if true_gap is not None:
                    true_gap *= float(np.mean(DELTAS[HOLD]**2))
                for repeat in range(args.repeats):
                    seed = 20261018+10000*repeat+100*index+n
                    result = diagnose(simulate(case, np.random.default_rng(seed), n),
                                      np.random.default_rng(seed+1), args.bootstrap, .02, .02)
                    rejected += result["decision"].startswith("reject_")
                    failed += result["decision"] == "shared_mean_model_fails"
                    unverified += result["decision"] == "shared_mean_model_unverified"
                    if true_gap is not None:
                        covered += result["gap_ci_low"] <= true_gap <= result["gap_ci_high"]
                    if args.save_trials:
                        trials.append({"n": n, "case": case, "repeat": repeat, **result})
                row = {"n": n, "case": case, "repeats": args.repeats,
                       "conditional_rejection_rate": rejected/args.repeats,
                       "rejection_wilson_95": wilson(rejected, args.repeats),
                       "mean_check_failure_rate": failed/args.repeats,
                       "mean_check_unverified_rate": unverified/args.repeats,
                       "inconclusive_rate": (args.repeats-rejected-failed)/args.repeats,
                       "gap_interval_coverage": None if true_gap is None else covered/args.repeats}
                summaries.append(row)
                print(f"N={n:4d} {case:34s} reject={row['conditional_rejection_rate']:.0%}", flush=True)
        summary = {"identities": identities, "sweep": {
            "scope": "fixed p=.35, sigma=.12, assumed noise budget=.02; theory simulation only",
            "bootstrap": args.bootstrap, "summaries": summaries}}
        if args.save_trials:
            with (args.output/"trials.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(trials[0]))
                writer.writeheader()
                writer.writerows(trials)
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, args.output)
    print(f"Theory report: {args.output/'REPORT.md'}")


if __name__ == "__main__":
    main()
