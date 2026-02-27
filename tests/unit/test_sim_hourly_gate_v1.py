from __future__ import annotations

from careai.sim_hourly.qa_trust_v1 import evaluate_gate


def test_gate_pass_minimal() -> None:
    metrics = {
        "realism": {"global_out_of_range_rate": 0.01, "per_feature_out_of_range_rate": {"s_t_sofa": 0.01}},
        "ood": {"ood_exceedance_rate": 0.01},
        "done": {"done_prob_mean": 0.05, "termination_rate_mean": 0.1, "done_ece": 0.05},
        "readmission_valid": {"auroc": 0.6, "brier": 0.15, "ece": 0.05},
        "stability": {"mean_spearman": 0.9, "top_policy_agreement": 5},
    }
    thresholds = {
        "global_out_of_range_max": 0.05,
        "per_feature_out_of_range_max": 0.10,
        "ood_exceedance_max": 0.10,
        "done_mean_min": 0.01,
        "done_mean_max": 0.40,
        "termination_rate_min": 0.02,
        "termination_rate_max": 0.95,
        "done_ece_max": 0.10,
        "readmit_brier_delta_max": 0.01,
        "readmit_auroc_delta_min": -0.01,
        "rank_consistency_min": 0.70,
        "top_policy_agreement_min": 4,
    }
    gate = evaluate_gate(metrics, thresholds, baseline_readmit={"auroc": 0.59, "brier": 0.16})
    assert gate.status == "pass"

