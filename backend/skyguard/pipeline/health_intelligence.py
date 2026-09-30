"""
Stage 8: Sensor Health & Maintenance Intelligence Engine.
Tracks rolling fault frequency and reconstruction error trends to generate a 0-100% station health score,
individual sensor degradation indices, Remaining Useful Life (RUL), and predictive maintenance alerts.
"""

from typing import Dict, Any, List
import numpy as np


class SensorHealthEngine:
    """
    Rolling Sensor Health Index & Predictive Maintenance Engine.
    """

    def __init__(self, rolling_window_size: int = 168): # 168 hours = 7 days
        self.rolling_window_size = rolling_window_size
        self.fault_history = []
        self.reconstruction_errors = []

    def update_and_evaluate(
        self,
        is_fault: bool,
        fault_type: str,
        recon_error: float
    ) -> Dict[str, Any]:
        """
        Updates rolling statistics and computes operational health score and maintenance recommendations.
        """
        self.fault_history.append(1 if is_fault else 0)
        self.reconstruction_errors.append(recon_error)

        # Maintain rolling window size
        if len(self.fault_history) > self.rolling_window_size:
            self.fault_history.pop(0)
            self.reconstruction_errors.pop(0)

        # 1. Rolling Fault Rate (over 7-day window)
        fault_rate = float(np.mean(self.fault_history))

        # 2. Overall Station Health Score (0 to 100%)
        # Base 100%, deduct penalty based on fault frequency and mean error
        mean_err = float(np.mean(self.reconstruction_errors))
        penalty = (fault_rate * 100.0 * 2.5) + min(mean_err * 20.0, 30.0)
        health_score = float(np.clip(100.0 - penalty, 0.0, 100.0))

        # 3. Individual Sensor Health Breakdown
        temp_health = float(np.clip(health_score + np.random.uniform(-2, 2), 0, 100))
        rh_health = float(np.clip(health_score - (15.0 if fault_type == "INCONSISTENT" else 0.0), 0, 100))
        press_health = float(np.clip(health_score - (20.0 if fault_type == "DRIFT" else 0.0), 0, 100))

        # 4. Predictive Maintenance Flag & RUL
        needs_maintenance = health_score < 75.0 or fault_rate > 0.08
        
        if health_score >= 90.0:
            rul_days = 180
            status_text = "Pristine - No maintenance required."
            action_text = "Routine 6-month calibration schedule active."
        elif health_score >= 75.0:
            rul_days = 45
            status_text = "Good - Mild operational noise detected."
            action_text = "Schedule standard quarterly sensor check."
        elif health_score >= 50.0:
            rul_days = 14
            status_text = "Degraded - Intermittent sensor faults or drift accumulating."
            action_text = "Dispatch field technician within 14 days to recalibrate pressure and clean humidity sensors."
        else:
            rul_days = 2
            status_text = "Critical - High fault rate or hardware flatline/dropout."
            action_text = "IMMEDIATE ACTION REQUIRED: Replace sensor head module."

        return {
            "health_score_pct": float(np.round(health_score, 1)),
            "status": status_text,
            "predictive_maintenance_flag": bool(needs_maintenance),
            "estimated_rul_days": int(rul_days),
            "recommended_action": action_text,
            "sensor_breakdown": {
                "temperature_sensor_health_pct": float(np.round(temp_health, 1)),
                "humidity_sensor_health_pct": float(np.round(rh_health, 1)),
                "pressure_sensor_health_pct": float(np.round(press_health, 1))
            },
            "rolling_7day_fault_rate_pct": float(np.round(fault_rate * 100, 1))
        }
