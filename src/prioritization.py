"""
Maintenance Prioritization Module.

Translates bearing fault predictions and vibration energy metrics into actionable
industrial maintenance priority scores and operational recommendations.
"""

from typing import Dict, Any


# Base severity weights based on bearing fault kinematics and mechanical risk
# Inner race faults rotate at shaft speed and experience highest cyclic stress concentration.
# Ball faults generate multi-frequency impacts across both races.
# Outer race faults are typically stationary and exhibit predictable wear progression.
BASE_FAULT_SEVERITY = {
    "Normal": 0.05,
    "Outer Race Fault": 0.50,
    "Ball Fault": 0.70,
    "Inner Race Fault": 0.90
}


def compute_maintenance_priority(
    predicted_class: str,
    confidence: float,
    vibration_rms: float,
    baseline_rms: float = 0.10,
    motor_load_hp: float = 2.0
) -> Dict[str, Any]:
    """
    Computes a maintenance priority score and operational recommendation.

    Score formulation:
        - Base Severity (0.0 to 1.0) according to fault type.
        - Vibration Energy Factor: Ratio of current RMS to baseline healthy RMS.
        - Operational Load Factor: Higher load increases mechanical fatigue.
        - Model Confidence: Weight assigned to the diagnostic prediction.

    Args:
        predicted_class: Diagnosed condition ("Normal", "Inner Race Fault", etc.).
        confidence: Classification probability (0.0 to 1.0).
        vibration_rms: Measured vibration RMS amplitude.
        baseline_rms: Nominal healthy bearing RMS amplitude.
        motor_load_hp: Motor operating load in horsepower (0 to 3 HP in CWRU).

    Returns:
        Dict[str, Any]: Priority score (0-100), urgency tier, and recommended action.
    """
    base_severity = BASE_FAULT_SEVERITY.get(predicted_class, 0.3)

    # Vibration amplitude ratio (capped to prevent extreme outliers)
    rms_ratio = max(1.0, vibration_rms / max(baseline_rms, 1e-6))
    rms_penalty = min(2.5, rms_ratio)

    # Load factor (higher horsepower causes higher contact stress)
    load_factor = 1.0 + (motor_load_hp / 10.0)

    # Combined composite priority score normalized to 0 - 100
    raw_score = base_severity * rms_penalty * load_factor * confidence * 100.0
    priority_score = float(min(100.0, max(0.0, raw_score)))

    # Determine Urgency Level and Recommended Operational Action
    if priority_score < 25.0:
        tier = "Low"
        action = "Equipment operating within acceptable parameters. Continue routine condition monitoring."
    elif priority_score < 55.0:
        tier = "Medium"
        action = "Early wear patterns observed. Increase sampling frequency; inspect during next scheduled maintenance."
    elif priority_score < 80.0:
        tier = "High"
        action = "Significant fault progression detected. Order replacement bearing; schedule intervention within 48 hours."
    else:
        tier = "Critical"
        action = "Severe bearing degradation detected with high dynamic risk. Immediate inspection and planned shutdown recommended."

    return {
        "fault_class": predicted_class,
        "confidence": confidence,
        "priority_score": round(priority_score, 2),
        "priority_tier": tier,
        "recommended_action": action,
        "vibration_rms": vibration_rms,
        "operating_load_hp": motor_load_hp
    }
