"""Orchestration that runs each baseline against a query set and produces a result table. Runs every baseline against every query, computes metrics, aggregates
per category and overall, runs paired significance tests, saves outputs.

Outputs to data/evaluation/:
- results.json = per-query per-baseline retrieved_ids + metrics
- results.csv = flat rows for easy pandas/Excel analysis
- summary.json = aggregated metrics per baseline (overall and by category)
- significance.json = paired t-tests between key baseline comparisons
"""
import csv
import json
import time
from pathlib import Path
from collections import defaultdict

from scipy import stats
from tqdm import tqdm

from src.evaluation.query import EvalQuery, load_queries, get_relevant_tracks
from src.evaluation.baselines import BASELINES, run_baseline
from src.evaluation.judge import judge_recommendations, JUDGE_MODEL
from src.evaluation.metrics import compute_all_metrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation"

DEFAULT_K = 10
DEFAULT_KS = (5, 10)  # metric k values

def evaluate_query_baseline(
    query: EvalQuery,
    baseline_name: str,
    k: int = DEFAULT_K,
    use_judge: bool = True,
) -> dict:
    """Run one baseline against one query. Returns retrieved_ids + metrics."""
    retrieved_ids = run_baseline(baseline_name, query.query, k)
    if query.ground_truth_strategy == "llm_judge":
        if not use_judge:
            return {
                "retrieved_ids": retrieved_ids,
                "metrics": None,
                "note": "skipped: llm_judge disabled",
            }
        relevance_scores = judge_recommendations(
            query.id, query.query, retrieved_ids
        )
        metrics = compute_all_metrics(
            retrieved_ids,
            relevance_scores=relevance_scores,
            ks=DEFAULT_KS,
        )
    else:
        relevant = get_relevant_tracks(query)
        if relevant is None:
            return {
                "retrieved_ids": retrieved_ids,
                "metrics": None,
                "note": "no ground truth available",
            }
        metrics = compute_all_metrics(
            retrieved_ids,
            relevant=relevant,
            ks=DEFAULT_KS,
        )
        metrics["_gold_set_size"] = len(relevant)

    return {
        "retrieved_ids": retrieved_ids,
        "metrics": metrics,
    }


def run_full_evaluation(
    queries: list[EvalQuery],
    baseline_names: list[str],
    k: int = DEFAULT_K,
    use_judge: bool = True,
) -> list[dict]:
    """Run every baseline against every query. Returns list of per-query results."""
    results = []
    total_calls = len(queries) * len(baseline_names)

    with tqdm(total=total_calls, desc="evaluating") as pbar:
        for query in queries:
            baseline_results = {}
            for baseline_name in baseline_names:
                r = evaluate_query_baseline(
                    query, baseline_name, k=k, use_judge=use_judge
                )
                baseline_results[baseline_name] = r
                pbar.set_postfix(query=query.id, baseline=baseline_name)
                pbar.update(1)
            results.append({
                "query_id": query.id,
                "category": query.category,
                "query": query.query,
                "baseline_results": baseline_results,
            })

    return results

def aggregate_metrics(results: list[dict]) -> dict:
    """Aggregate per-query metrics into per-baseline and per-category means."""
    overall: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_cat: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for row in results:
        for baseline_name, r in row["baseline_results"].items():
            if r.get("metrics") is None:
                continue
            for metric_name, value in r["metrics"].items():
                if metric_name.startswith("_"):
                    continue  # skip internal fields
                overall[baseline_name][metric_name].append(value)
                by_cat[row["category"]][baseline_name][metric_name].append(value)

    def mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    aggregates = {
        "overall": {
            baseline: {
                metric: round(mean(vals), 4)
                for metric, vals in metrics_by_baseline.items()
            }
            for baseline, metrics_by_baseline in overall.items()
        },
        "by_category": {
            cat: {
                baseline: {
                    metric: round(mean(vals), 4)
                    for metric, vals in metrics_by_baseline.items()
                }
                for baseline, metrics_by_baseline in cat_data.items()
            }
            for cat, cat_data in by_cat.items()
        },
    }
    return aggregates

def paired_t_test(
    results: list[dict],
    baseline_a: str,
    baseline_b: str,
    metric: str = "ndcg_at_10",
) -> dict:
    """Paired t-test comparing two baselines on the given metric.

    Pairs are per-query — for each query, we get metric[baseline_a] and
    metric[baseline_b], and compare across queries. Only queries where both
    baselines have a computed metric are included.
    """
    values_a = []
    values_b = []
    for row in results:
        r_a = row["baseline_results"].get(baseline_a, {})
        r_b = row["baseline_results"].get(baseline_b, {})
        m_a = r_a.get("metrics")
        m_b = r_b.get("metrics")
        if not m_a or not m_b:
            continue
        if metric not in m_a or metric not in m_b:
            continue
        values_a.append(m_a[metric])
        values_b.append(m_b[metric])

    if len(values_a) < 5:
        return {
            "comparison": f"{baseline_a} vs {baseline_b}",
            "metric": metric,
            "n_pairs": len(values_a),
            "note": "too few pairs for meaningful test",
        }

    t_stat, p_value = stats.ttest_rel(values_a, values_b)
    mean_a = sum(values_a) / len(values_a)
    mean_b = sum(values_b) / len(values_b)
    #cohens d
    diffs = [a - b for a, b in zip(values_a, values_b)]
    diff_mean = sum(diffs) / len(diffs)
    diff_std = (sum((d - diff_mean) ** 2 for d in diffs) / (len(diffs) - 1)) ** 0.5
    cohens_d = diff_mean / diff_std if diff_std > 0 else 0.0

    return {
        "comparison": f"{baseline_a} vs {baseline_b}",
        "metric": metric,
        "n_pairs": len(values_a),
        "mean_a": round(mean_a, 4),
        "mean_b": round(mean_b, 4),
        "t_stat": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "cohens_d": round(cohens_d, 4),
        "significant_at_0.05": bool(p_value < 0.05),
    }


def run_all_significance_tests(results: list[dict]) -> list[dict]:
    """Run the standard set of pairwise significance tests."""
    key_comparisons = [
        ("rag", "dense"), 
        ("rag", "bm25"), 
        ("rag", "popularity"), 
        ("rag", "random"), 
        ("dense", "bm25"), 
        ("dense", "popularity"),
    ]
    tests = []
    for a, b in key_comparisons:
        tests.append(paired_t_test(results, a, b, metric="ndcg_at_10"))
    return tests

def save_results(
    results: list[dict],
    aggregates: dict,
    significance: list[dict],
    meta: dict,
) -> None:
    """Save all four output files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)

    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"meta": meta, "aggregates": aggregates}, f, indent=2)

    sig_path = OUTPUT_DIR / "significance.json"
    with open(sig_path, "w") as f:
        json.dump({"meta": meta, "tests": significance}, f, indent=2)

    csv_path = OUTPUT_DIR / "results.csv"

    metric_names = set()
    for row in results:
        for r in row["baseline_results"].values():
            m = r.get("metrics") or {}
            for k in m.keys():
                if not k.startswith("_"):
                    metric_names.add(k)
    metric_names = sorted(metric_names)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "category", "baseline"] + metric_names)
        for row in results:
            for baseline_name, r in row["baseline_results"].items():
                m = r.get("metrics") or {}
                writer.writerow(
                    [row["query_id"], row["category"], baseline_name]
                    + [m.get(k, "") for k in metric_names]
                )

    print(f"\nResults saved to {OUTPUT_DIR}/")
    print(f"  results.json      {results_path.stat().st_size // 1024} KB")
    print(f"  summary.json      {summary_path.stat().st_size // 1024} KB")
    print(f"  significance.json {sig_path.stat().st_size // 1024} KB")
    print(f"  results.csv       {csv_path.stat().st_size // 1024} KB")


def print_summary(aggregates: dict) -> None:
    """Print a readable summary of the overall results to stdout."""
    print("\n" + "=" * 78)
    print("OVERALL RESULTS (mean across all evaluated queries)")
    print("=" * 78)

    baselines = list(aggregates["overall"].keys())
    if not baselines:
        print("No results computed.")
        return

    display_metrics = ["precision_at_5", "precision_at_10", "ndcg_at_5", "ndcg_at_10", "mrr"]

    print(f"{'baseline':<12}  " + "  ".join(f"{m:>14}" for m in display_metrics))
    print("-" * (14 + 16 * len(display_metrics)))
    for baseline in baselines:
        row = aggregates["overall"][baseline]
        values = [row.get(m, 0.0) for m in display_metrics]
        print(
            f"{baseline:<12}  "
            + "  ".join(f"{v:>14.4f}" for v in values)
        )

    print("\n" + "=" * 78)
    print("BY CATEGORY (mean ndcg_at_10 per baseline per category)")
    print("=" * 78)
    categories = sorted(aggregates["by_category"].keys())
    header = f"{'category':<12}  " + "  ".join(f"{b:>12}" for b in baselines)
    print(header)
    print("-" * len(header))
    for cat in categories:
        cat_row = aggregates["by_category"].get(cat, {})
        values = [
            cat_row.get(baseline, {}).get("ndcg_at_10", 0.0)
            for baseline in baselines
        ]
        print(f"{cat:<12}  " + "  ".join(f"{v:>12.4f}" for v in values))


def print_significance(tests: list[dict]) -> None:
    """Print significance tests as a readable table."""
    print("\n" + "=" * 78)
    print("STATISTICAL SIGNIFICANCE (NDCG@10, paired t-test)")
    print("=" * 78)
    print(f"{'comparison':<28}  {'n':>4}  {'mean_a':>8}  {'mean_b':>8}  {'t':>7}  {'p':>10}  {'d':>7}  sig")
    print("-" * 78)
    for t in tests:
        if "note" in t:
            print(f"{t['comparison']:<28}  {t['n_pairs']:>4}  {t['note']}")
            continue
        sig = "YES" if t["significant_at_0.05"] else "no"
        print(
            f"{t['comparison']:<28}  {t['n_pairs']:>4}  "
            f"{t['mean_a']:>8.4f}  {t['mean_b']:>8.4f}  "
            f"{t['t_stat']:>7.2f}  {t['p_value']:>10.4f}  "
            f"{t['cohens_d']:>7.2f}  {sig}"
        )
        
def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, help="Evaluate only N random queries (smoke test)")
    parser.add_argument("--category", help="Only queries in this category (genre/mood/compound/reference/contextual)")
    parser.add_argument("--skip-baselines", nargs="+", default=[], help="Baselines to skip")
    parser.add_argument("--skip-judge", action="store_true", help="Skip LLM-judged (subjective) queries — objective only")
    parser.add_argument("-k", type=int, default=DEFAULT_K, help=f"Top-k results per baseline (default {DEFAULT_K})")
    args = parser.parse_args()

    all_queries = load_queries()

    if args.category:
        all_queries = [q for q in all_queries if q.category == args.category]
    if args.sample:
        import random
        random.seed(42)
        all_queries = random.sample(all_queries, min(args.sample, len(all_queries)))
    if args.skip_judge:
        all_queries = [q for q in all_queries if q.ground_truth_strategy != "llm_judge"]

    baseline_names = [b for b in BASELINES.keys() if b not in args.skip_baselines]

    print(f"\nEvaluation configuration:")
    print(f"  Queries:   {len(all_queries)}")
    print(f"  Baselines: {baseline_names}")
    print(f"  k:         {args.k}")
    print(f"  Judge:     {JUDGE_MODEL if not args.skip_judge else 'skipped'}")

    start = time.time()
    results = run_full_evaluation(
        all_queries,
        baseline_names,
        k=args.k,
        use_judge=not args.skip_judge,
    )
    elapsed = time.time() - start

    print(f"\nEvaluation complete in {elapsed:.0f}s")

    aggregates = aggregate_metrics(results)
    significance = run_all_significance_tests(results)

    meta = {
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_queries": len(all_queries),
        "baselines": baseline_names,
        "k": args.k,
        "judge_model": JUDGE_MODEL if not args.skip_judge else None,
        "elapsed_seconds": round(elapsed, 1),
    }

    save_results(results, aggregates, significance, meta)
    print_summary(aggregates)
    print_significance(significance)


if __name__ == "__main__":
    main()