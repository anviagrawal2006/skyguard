"""
Stage 1: Real-Time Edge Quality Control (QC) Engine.
Fast, non-ML first line of defense executing range bounds, rate-of-change,
flatline/frozen, and missing-data checks.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


class EdgeQCEngine:
    """
    Lightweight rule-based edge QC checking physical thresholds,
    sudden jumps, stuck values, and NaN gaps.
    """

    def __init__(
        self,
        temp_range: tuple = (-10.0, 55.0),
        rh_range: tuple = (0.0, 100.0),
        press_range: tuple = (850.0, 1080.0),
        max_rate_temp: float = 5.0,     # °C / hour
        max_rate_rh: float = 25.0,     # % / hour
        max_rate_press: float = 6.0,    # hPa / hour
        flatline_window: int = 4
    ):
        self.temp_range = temp_range
        self.rh_range = rh_range
        self.press_range = press_range
        self.max_rate_temp = max_rate_temp
        self.max_rate_rh = max_rate_rh
        self.max_rate_press = max_rate_press
        self.flatline_window = flatline_window

    def process_reading(
        self,
        current_reading: Dict[str, Any],
        recent_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Processes single hourly telemetry reading against recent window history.
        Returns dict of flags and violation details.
        """
        flags = {
            "range_violation": False,
            "rate_of_change_violation": False,
            "flatline_violation": False,
            "missing_data": False,
            "details": []
        }

        temp = current_reading.get("temperature_2m")
        rh = current_reading.get("relative_humidity_2m")
        sp = current_reading.get("surface_pressure")

        # 1. Missing Data Check
        if temp is None or np.isnan(temp) or rh is None or np.isnan(rh) or sp is None or np.isnan(sp):
            flags["missing_data"] = True
            flags["details"].append("Missing or NaN sensor telemetry reading.")
            return flags

        # 2. Physical Range Bounds Check
        if not (self.temp_range[0] <= temp <= self.temp_range[1]):
            flags["range_violation"] = True
            flags["details"].append(f"Temperature {temp:.1f}°C out of valid range {self.temp_range}.")

        if not (self.rh_range[0] <= rh <= self.rh_range[1]):
            flags["range_violation"] = True
            flags["details"].append(f"Relative Humidity {rh:.1f}% out of valid range {self.rh_range}.")

        if not (self.press_range[0] <= sp <= self.press_range[1]):
            flags["range_violation"] = True
            flags["details"].append(f"Surface Pressure {sp:.1f} hPa out of valid range {self.press_range}.")

        # 3. Rate-of-Change (Delta) Check
        if recent_history and len(recent_history) >= 1:
            prev = recent_history[-1]
            p_temp = prev.get("temperature_2m")
            p_rh = prev.get("relative_humidity_2m")
            p_sp = prev.get("surface_pressure")

            if p_temp is not None and not np.isnan(p_temp):
                delta_temp = abs(temp - p_temp)
                if delta_temp > self.max_rate_temp:
                    flags["rate_of_change_violation"] = True
                    flags["details"].append(f"Temperature rate-of-change {delta_temp:.1f}°C/hr exceeds threshold {self.max_rate_temp}°C/hr.")

            if p_rh is not None and not np.isnan(p_rh):
                delta_rh = abs(rh - p_rh)
                if delta_rh > self.max_rate_rh:
                    flags["rate_of_change_violation"] = True
                    flags["details"].append(f"Humidity rate-of-change {delta_rh:.1f}%/hr exceeds threshold {self.max_rate_rh}%/hr.")

            if p_sp is not None and not np.isnan(p_sp):
                delta_sp = abs(sp - p_sp)
                if delta_sp > self.max_rate_press:
                    flags["rate_of_change_violation"] = True
                    flags["details"].append(f"Pressure rate-of-change {delta_sp:.1f}hPa/hr exceeds threshold {self.max_rate_press}hPa/hr.")

        # 4. Flatline / Stuck Value Check
        if recent_history and len(recent_history) >= self.flatline_window - 1:
            window = recent_history[-(self.flatline_window - 1):] + [current_reading]
            
            # Check temp flatline
            temps = [r.get("temperature_2m") for r in window if r.get("temperature_2m") is not None]
            if len(temps) == self.flatline_window and len(set(temps)) == 1:
                flags["flatline_violation"] = True
                flags["details"].append(f"Temperature flatlined static at {temps[0]}°C for {self.flatline_window} consecutive hours.")

            # Check pressure flatline
            sps = [r.get("surface_pressure") for r in window if r.get("surface_pressure") is not None]
            if len(sps) == self.flatline_window and len(set(sps)) == 1:
                flags["flatline_violation"] = True
                flags["details"].append(f"Surface Pressure flatlined static at {sps[0]} hPa for {self.flatline_window} consecutive hours.")

        return flags
