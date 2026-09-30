"""
Stage 3: Physics + Distance-Weighted Spatial Consistency Engine.
Verifies psychrometric relationship (T_dew <= T), hydrostatic elevation pressure delta,
and evaluates spatial consensus across the real 4-station network using distance-weighted IDW.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from skyguard.data.preprocessing import compute_dew_point, compute_theoretical_hydrostatic_delta


class PhysicsSpatialEngine:
    """
    Combines physical thermodynamic limits and network spatial consensus.
    """

    def __init__(self, target_elevation_m: float = 189.0):
        self.target_elevation_m = target_elevation_m

    def evaluate_physics(self, current_reading: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates physical consistency:
        - Dew point temperature bound: T_d <= T_c + 0.5°C
        - Hydrostatic MSL-surface pressure delta vs station elevation (189m)
        """
        temp = current_reading.get("temperature_2m")
        rh = current_reading.get("relative_humidity_2m")
        sp = current_reading.get("surface_pressure")
        msl = current_reading.get("pressure_msl")

        physics_res = {
            "physics_violation": False,
            "dew_point_c": None,
            "dew_point_violation": False,
            "hydrostatic_residual_hpa": None,
            "hydrostatic_violation": False,
            "details": []
        }

        if temp is None or rh is None or sp is None or msl is None:
            return physics_res

        # 1. Dew Point Calculation
        dew_pt = float(compute_dew_point(np.array([temp]), np.array([rh]))[0])
        physics_res["dew_point_c"] = dew_pt

        if dew_pt > (temp + 0.5):
            physics_res["dew_point_violation"] = True
            physics_res["physics_violation"] = True
            physics_res["details"].append(f"Psychrometric Violation: Dew point ({dew_pt:.1f}°C) exceeds ambient temperature ({temp:.1f}°C).")

        # 2. Hydrostatic Elevation Pressure Sanity
        actual_delta = msl - sp
        theoretical_delta = float(compute_theoretical_hydrostatic_delta(np.array([sp]), np.array([temp]), self.target_elevation_m)[0])
        hydro_residual = abs(actual_delta - theoretical_delta)
        physics_res["hydrostatic_residual_hpa"] = hydro_residual

        if hydro_residual > 4.5: # hPa residual threshold
            physics_res["hydrostatic_violation"] = True
            physics_res["physics_violation"] = True
            physics_res["details"].append(
                f"Hydrostatic Pressure Violation: MSL-Surface delta ({actual_delta:.1f} hPa) deviates {hydro_residual:.1f} hPa from theoretical elevation delta ({theoretical_delta:.1f} hPa)."
            )

        return physics_res

    def evaluate_spatial_consensus(
        self,
        target_reading: Dict[str, Any],
        neighbor_readings: Dict[str, Dict[str, Any]],
        neighbor_metadata: Dict[str, Dict[str, Any]],
        max_radius_km: float = 5000.0
    ) -> Dict[str, Any]:
        """
        Calculates Inverse Distance Weighting (IDW) consensus across the real neighbor stations.
        Returns expected spatial values, spatial residuals, and spatial Z-scores.
        """
        spatial_res = {
            "spatial_anomaly": False,
            "temp_spatial_z": 0.0,
            "rh_spatial_z": 0.0,
            "sp_spatial_z": 0.0,
            "expected_temp": None,
            "expected_rh": None,
            "expected_sp": None,
            "details": []
        }


        filtered_readings = {}
        for st_id, n_read in neighbor_readings.items():
            meta = neighbor_metadata.get(st_id, {})
            dist_km = meta.get("distance_km", 100.0)
            if dist_km <= max_radius_km:
                filtered_readings[st_id] = n_read
        neighbor_readings = filtered_readings

        if not neighbor_readings:
            spatial_res["insufficient_neighbors"] = True
            return spatial_res


        weights = []
        n_temps, n_rhs, n_sps = [], [], []

        for st_id, n_read in neighbor_readings.items():
            meta = neighbor_metadata.get(st_id, {})
            dist_km = meta.get("distance_km", 100.0)
            corr = meta.get("corr_temp", 0.95)
            
            # Inverse distance weighting combined with temperature correlation
            w = (corr / max(dist_km, 1.0))
            
            nt = n_read.get("temperature_2m")
            nrh = n_read.get("relative_humidity_2m")
            nsp = n_read.get("surface_pressure")

            if nt is not None and not np.isnan(nt):
                weights.append(w)
                n_temps.append(nt)
                n_rhs.append(nrh if nrh is not None else 50.0)
                n_sps.append(nsp if nsp is not None else 990.0)

        if not weights:
            return spatial_res

        weights = np.array(weights)
        weights /= np.sum(weights)

        # Expected Spatial Consensus Values
        exp_temp = float(np.sum(weights * np.array(n_temps)))
        exp_rh = float(np.sum(weights * np.array(n_rhs)))
        exp_sp = float(np.sum(weights * np.array(n_sps)))

        spatial_res["expected_temp"] = exp_temp
        spatial_res["expected_rh"] = exp_rh
        spatial_res["expected_sp"] = exp_sp

        # Compute Standard Deviations among neighbors
        std_temp = max(float(np.std(n_temps)), 3.5)
        std_rh = max(float(np.std(n_rhs)), 10.0)
        std_sp = max(float(np.std(n_sps)), 5.0)

        # Compute Spatial Z-Scores for target reading
        t_temp = target_reading.get("temperature_2m")
        t_rh = target_reading.get("relative_humidity_2m")
        t_sp = target_reading.get("surface_pressure")

        if t_temp is not None and not np.isnan(t_temp):
            spatial_res["temp_spatial_z"] = float(abs(t_temp - exp_temp) / std_temp)
        if t_rh is not None and not np.isnan(t_rh):
            spatial_res["rh_spatial_z"] = float(abs(t_rh - exp_rh) / std_rh)
        if t_sp is not None and not np.isnan(t_sp):
            spatial_res["sp_spatial_z"] = float(abs(t_sp - exp_sp) / std_sp)

        max_z = max(spatial_res["temp_spatial_z"], spatial_res["rh_spatial_z"], spatial_res["sp_spatial_z"])
        if max_z > 3.0:
            spatial_res["spatial_anomaly"] = True
            spatial_res["details"].append(f"Spatial Deviation: Target telemetry deviates {max_z:.1f} standard deviations from network neighbor consensus.")

        return spatial_res
