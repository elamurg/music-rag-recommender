"""Evaluation based on Precision@10, NDCG@10, MRR calculators. All functions take a ranked list of retrieved track_ids 
and either a set of "relevant" IDs (binary correctness) or a dict mapping track_ids to graded relevance scores (0.0 to 2.0)"""

import math

def precision_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """Fraction of the k retrieved elements that are in the relevant set."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for tid in top_k if tid in relevant)
    return hits/len(top_k)

def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """Fraction of relevant ids that appear in top_k"""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for tid in top_k if tid in relevant)
    return hits/len(relevant)

def _dcg(relevances: list[float], k: int) -> float:
    """Discounted cumulative gain over top-k relevance values"""
    dcg_val = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg_val += rel/math.log2(i+2)
        return dcg_val

def ndcg_at_k(retrieved: list[int], relevance_scores: dict[int, float], k: int) -> float:
    """Normalised Discounted Cumulative Gain over top_k"""
    actual_relevances = [relevance_scores.get(tid, 0.0) for tid in retrieved[:k]]
    actual_dcg = _dcg(actual_relevances, k)
    #rank all the relevance scores descending, use top-k as if perfectly ranked
    all_scores = sorted(relevance_scores.values(), reverse=True)
    ideal_relevances = all_scores[:k]
    ideal_dcg = _dcg(ideal_relevances, k)

    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg/ideal_dcg

def ndcg_at_k_binary(retrieved: list[int], relevant: dict[int], k: int) -> float:
    """NDCG@k for binary relevance. Treats every relevant track as score 1.0"""
    scores = {tid: 1.0 for tid in relevant}
    return ndcg_at_k(retrieved, scores, k)

def mrr(retrieved: list[int], relevant: set[int]) -> float:
    """Mean Reciprocal Rank of the position of the first relevant item.
    The first on the list is 1.0, seconf is 0.5, tenth 0.1..."""
    for i, tid in enumerate(retrieved, start = 1):
        if tid in relevant:
            return 1.0/i
    return 0.0
    
def hit_rate_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """Basically just describes if any of the hits have been captured"""
    if not relevant:
        return 0.0
    return 1.0 if any(tid in relevant for tid in retrieved[:k]) else 0.0

def compute_all_metrics(retrieved: list[int], relevant: set[int]| None = None, relevance_scores: dict[int, float]| None = None, ks: tuple[int, ...] = (5, 10)) -> dict[str, float]:
    """Compute the full metric, returns a flat dict for easy aggregation"""
    results = {}
    if relevance_scores is None and relevant is None:
        relevant = {tid for tid, score in relevance_scores.items() if score > 0}
    if relevant is None:
        relevant = set()
    for k in ks:
        results[f"precision_at_{k}"] = precision_at_k(retrieved, relevant, k)
        results[f"recall_at_{k}"] = recall_at_k(retrieved, relevant, k)
        results[f"hit_rate_at_{k}"] = hit_rate_at_k(retrieved, relevant, k)

        #NDCG graded scores when available, else use binary
        if relevance_scores is not None:
            results[f"ndcg_at_{k}"] = ndcg_at_k(retrieved, relevance_scores, k)
        else:
            results[f"ndcg_at_{k}"] = ndcg_at_k_binary(retrieved, relevant, k)
    results["mrr"] = mrr(retrieved, relevant)
    return results

if __name__ == "__main__":
    print("self-test: known metrics on toy data")
    
    retrieved = [1, 2, 3, 4, 5]
    relevant = {1, 2, 3}
    print("\nCase: relevant={1,2,3}, retrieved=[1,2,3,4,5]")
    print(f"P@5 = {precision_at_k(retrieved, relevant, 5):.3f} (expected 0.600)")
    print(f"P@3 = {precision_at_k(retrieved, relevant, 3):.3f} (expected 1.000)")
    print(f"R@5 = {recall_at_k(retrieved, relevant, 5):.3f} (expected 1.000)")
    print(f"MRR = {mrr(retrieved, relevant):.3f} (expected 1.000)")

    retrieved_2 = [10, 20, 1, 30, 2]
    relevant_2 = {1, 2,3}
    print("\nCase: relevant={1,2,3}, retrieved=[10,20,1,30,2]")
    print(f"P@5 = {precision_at_k(retrieved_2, relevant_2, 5):.3f} (expected 0.400)")
    print(f"MRR = {mrr(retrieved_2, relevant_2):.3f} (expected 0.333)")
    print(f"Hit@3 = {hit_rate_at_k(retrieved_2, relevant_2, 3):.3f} (expected 1.000)")

    retrieved_3 = [1, 2, 3]
    scores_3 = {1: 2.0, 2: 1.0, 3: 0.0}  # graded relevance NDCG
    print("\nCase: graded scores={1:2, 2:1, 3:0}, retrieved=[1,2,3]")
    print(f"NDCG@3 (graded) = {ndcg_at_k(retrieved_3, scores_3, 3):.3f}")

    results = compute_all_metrics(retrieved=[1, 2, 3, 4, 5], relevant={1, 2, 3})
    print("\nCase: compute_all_metrics(retrieved=[1,2,3,4,5], relevant={1,2,3})")
    for k, v in results.items():
        print(f"{k}: {v:.3f}")
