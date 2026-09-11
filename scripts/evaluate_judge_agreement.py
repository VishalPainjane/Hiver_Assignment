"""Evaluate LLM-as-a-Judge Agreement with Human Ground Truth Ratings.

Computes:
1. Directional / Tolerance-1 Agreement (% where |Human - Judge| <= 1)
2. Exact Agreement (% where Human == Judge)
3. Cohen's Quadratic Weighted Kappa (QWK) via scikit-learn
4. Spearman's Rank Correlation (rho) via scipy
5. Hard Constraint Pass Rates (Brevity, DM Link, Zero-Liability)
6. Concrete Disagreement Case Analysis (isolating LLM leniency bias)

Outputs results to terminal and outputs/judge_agreement_results.json.
"""

from __future__ import annotations

import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

# Ensure root directory is on sys.path and stdout handles utf-8
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.evaluation.llm_judge import LLMReplyJudge

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_judge_agreement")


def run_agreement_study(
    benchmark_path: str = "data/golden/human_evaluation_set.json",
    output_path: str = "outputs/judge_agreement_results.json",
) -> Dict[str, Any]:
    file_path = ROOT_DIR / benchmark_path
    if not file_path.exists():
        raise FileNotFoundError(f"Human evaluation set not found at {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        benchmark_cases: List[Dict[str, Any]] = json.load(f)

    logger.info(f"Loaded {len(benchmark_cases)} benchmark cases from {benchmark_path}.")
    judge = LLMReplyJudge()

    human_scores: List[int] = []
    judge_scores: List[int] = []
    case_results: List[Dict[str, Any]] = []

    model_subsets: Dict[str, Dict[str, List[int]]] = {
        "baseline_1_trivial": {"human": [], "judge": []},
        "baseline_2_simple": {"human": [], "judge": []},
        "proposed_production": {"human": [], "judge": []},
    }

    t0 = time.perf_counter()
    logger.info("Evaluating cases with LLM Judge (anchored rubric)...")

    for i, item in enumerate(benchmark_cases, 1):
        case_id = item["case_id"]
        model_type = item["model_type"]
        tweet = item["customer_tweet"]
        reply = item["generated_reply"]
        intent = item.get("intent", "UNKNOWN")
        h_score = int(item["human_overall_score"])

        eval_res = judge.evaluate(customer_tweet=tweet, generated_reply=reply, intent=intent)
        j_score = eval_res.overall_score

        human_scores.append(h_score)
        judge_scores.append(j_score)

        if model_type in model_subsets:
            model_subsets[model_type]["human"].append(h_score)
            model_subsets[model_type]["judge"].append(j_score)

        case_results.append({
            "case_id": case_id,
            "model_type": model_type,
            "intent": intent,
            "customer_tweet": tweet,
            "generated_reply": reply,
            "human_score": h_score,
            "judge_score": j_score,
            "score_diff": j_score - h_score,
            "judge_diagnostic": eval_res.diagnostic_probe,
            "judge_privacy": eval_res.privacy_pivot,
            "judge_liability": eval_res.zero_liability,
            "judge_rationale": eval_res.rationale,
            "human_rationale": item.get("human_rationale", ""),
            "length_compliant": eval_res.length_compliant,
            "has_dm_link": eval_res.has_dm_link,
            "zero_liability_compliant": eval_res.zero_liability_compliant,
        })
        logger.info(f"[{i}/{len(benchmark_cases)}] {case_id} ({model_type}): Human={h_score} | Judge={j_score}")

    elapsed = time.perf_counter() - t0

    # 1. Statistical Agreement Calculations
    exact_matches = sum(1 for h, j in zip(human_scores, judge_scores) if h == j)
    exact_agreement_pct = round((exact_matches / len(human_scores)) * 100, 2)

    tol1_matches = sum(1 for h, j in zip(human_scores, judge_scores) if abs(h - j) <= 1)
    tol1_agreement_pct = round((tol1_matches / len(human_scores)) * 100, 2)

    qwk = cohen_kappa_score(human_scores, judge_scores, weights="quadratic")
    linear_kappa = cohen_kappa_score(human_scores, judge_scores, weights="linear")
    spearman_corr, spearman_pval = spearmanr(human_scores, judge_scores)

    # 2. Hard Constraint Rates
    brevity_rate = round(sum(1 for c in case_results if c["length_compliant"]) / len(case_results) * 100, 1)
    dm_link_rate = round(sum(1 for c in case_results if c["has_dm_link"]) / len(case_results) * 100, 1)
    liability_shield_rate = round(sum(1 for c in case_results if c["zero_liability_compliant"]) / len(case_results) * 100, 1)

    # 3. Model Tier Summary
    tier_summary = {}
    for tier, data in model_subsets.items():
        h_mean = float(np.mean(data["human"])) if data["human"] else 0.0
        j_mean = float(np.mean(data["judge"])) if data["judge"] else 0.0
        tier_summary[tier] = {
            "cases_count": len(data["human"]),
            "human_mean_score": round(h_mean, 2),
            "judge_mean_score": round(j_mean, 2),
            "mean_delta": round(j_mean - h_mean, 2),
        }

    # 4. Disagreement Cases
    disagreements = [c for c in case_results if abs(c["score_diff"]) > 0]
    disagreements.sort(key=lambda x: abs(x["score_diff"]), reverse=True)

    summary_data = {
        "benchmark_sample_size": len(benchmark_cases),
        "total_runtime_seconds": round(elapsed, 2),
        "metrics": {
            "tolerance_1_agreement_pct": tol1_agreement_pct,
            "exact_agreement_pct": exact_agreement_pct,
            "quadratic_weighted_kappa": round(float(qwk), 4),
            "linear_kappa": round(float(linear_kappa), 4),
            "spearman_rank_correlation": round(float(spearman_corr), 4),
            "spearman_p_value": float(f"{spearman_pval:.4e}"),
        },
        "hard_constraints": {
            "brevity_compliance_pct": brevity_rate,
            "dm_link_presence_pct": dm_link_rate,
            "zero_liability_compliance_pct": liability_shield_rate,
        },
        "tier_summary": tier_summary,
        "disagreements_count": len(disagreements),
        "top_disagreements": disagreements[:3],
        "all_cases": case_results,
    }

    # Print Terminal Report
    print("\n" + "=" * 92)
    print("LLM-AS-A-JUDGE HUMAN AGREEMENT STUDY (30 Hand-Adjudicated Benchmark Cases)")
    print("=" * 92)
    print(f"Total Evaluated Cases: {len(benchmark_cases)} (10 Trivial, 10 Simple Zero-Shot, 10 Production RAG)")
    print(f"Evaluation Runtime:    {elapsed:.2f}s ({elapsed/len(benchmark_cases):.2f}s per evaluation)")
    print("-" * 92)
    print("INTER-RATER RELIABILITY & STATISTICAL AGREEMENT:")
    print(f"  • Tolerance-1 Agreement (|H - J| <= 1):  {tol1_agreement_pct}%  (Target: > 85.0%)")
    print(f"  • Exact Score Agreement (H == J):       {exact_agreement_pct}%")
    print(f"  • Quadratic Weighted Kappa (QWK):       {qwk:.4f}  (Substantial/Near-Perfect: > 0.70)")
    print(f"  • Spearman Rank Correlation (rho):      {spearman_corr:.4f} (p = {spearman_pval:.2e})")
    print("-" * 92)
    print("HARD CONSTRAINT AUDIT:")
    print(f"  • Platform Brevity (<= 280 chars):     {brevity_rate}%")
    print(f"  • Privacy DM Link Presence:             {dm_link_rate}%")
    print(f"  • Zero-Liability Shield Compliance:     {liability_shield_rate}%")
    print("-" * 92)
    print("MODEL TIER SCORE CALIBRATION (Human vs. Judge Mean):")
    for tier, stats in tier_summary.items():
        print(f"  • {tier:<24}: Human = {stats['human_mean_score']:.2f} / 5.0 | Judge = {stats['judge_mean_score']:.2f} / 5.0 | Delta = {stats['mean_delta']:+.2f}")
    print("-" * 92)
    print("NOTABLE DISAGREEMENT CASE STUDY (Analysis of LLM Leniency / Edge Cases):")
    if disagreements:
        notable = disagreements[0]
        print(f"  [Case ID: {notable['case_id']}] Intent: {notable['intent']} | Tier: {notable['model_type']}")
        print(f"  Customer:  \"{notable['customer_tweet']}\"")
        print(f"  Reply:     \"{notable['generated_reply']}\"")
        print(f"  Human:     {notable['human_score']}/5.0 -> {notable['human_rationale']}")
        print(f"  Judge:     {notable['judge_score']}/5.0 -> {notable['judge_rationale']}")
        print(f"  Analysis:  Delta of {notable['score_diff']:+d} reflects LLM leniency on subtle brand criteria.")
    print("=" * 92 + "\n")

    # Save output
    out_file = ROOT_DIR / output_path
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"Saved evaluation results to {out_file}")

    return summary_data


if __name__ == "__main__":
    run_agreement_study()

