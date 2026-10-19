"""Evidence boundaries for new comparisons, without relabeling saved results."""

import math


def comparison_evidence(target, control, *, min_effect, noise_sd, alpha=0.05):
    """Describe the existing independent-arm test; do not infer cause specificity.

    This records statistical settings, not verification that an external noise
    estimate transfers to these data or that the arms used equal resources.
    """
    target_seeds, control_seeds = len(target.metrics), len(control.metrics)
    assignments = math.comb(target_seeds + control_seeds, target_seeds)
    interval_method = (
        "external_sd_normal"
        if noise_sd is not None
        else "pooled_sd_t"
        if target_seeds + control_seeds - 2 <= 30
        else "pooled_sd_normal"
    )
    return {
        "version": "intervention-evidence-v1",
        "claim": "tested_intervention_vs_control",
        "unique_cause_established": False,
        "full_recovery_established": False,
        "resource_matching": "not_verified_by_statistical_check",
        "target_seeds": target_seeds,
        "control_seeds": control_seeds,
        "min_effect": min_effect,
        "interval": {
            "confidence": 0.95,
            "method": interval_method,
            "noise_sd": noise_sd,
            "noise_transfer": "not_assessed_by_comparison",
        },
        "permutation": {
            "alternative": "target_better",
            "design": "independent_arms",
            "alpha": alpha,
            "minimum_attainable_p": 1 / assignments,
            "gate": "applied" if assignments * alpha >= 1 else "skipped_unattainable",
        },
    }
