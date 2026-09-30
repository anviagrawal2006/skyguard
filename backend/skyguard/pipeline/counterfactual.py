"""
Stage 4: Multi-Evidence Counterfactual Diagnosis Engine.
Combines Edge QC, Temporal AI, Physics, and Spatial Evidence to disambiguate
sensor hardware faults from genuine meteorologically extreme weather events.
"""

from typing import Dict, Any, List
import numpy as np


class CounterfactualDiagnosisEngine:
    """
    Counterfactual Reasoning & Disambiguation Engine:
    Fuses evidence across Edge QC rules, Temporal AI MSE, Psychrometric/Hydrostatic physics,
    and Spatial neighbor consensus to pinpoint root causes.
    """

    def diagnose(
        self,
        edge_flags: Dict[str, Any],
        temporal_res: Dict[str, Any],
        physics_res: Dict[str, Any],
        spatial_res: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synthesizes multi-stage evidence into diagnosis verdict, severity score, and confidence %.
        """
        missing_data = edge_flags.get("missing_data", False)
        range_viol = edge_flags.get("range_violation", False)
        rate_viol = edge_flags.get("rate_of_change_violation", False)
        flatline_viol = edge_flags.get("flatline_violation", False)

        is_temporal_anomaly = temporal_res.get("is_temporal_anomaly", False)
        max_z = temporal_res.get("max_z_score", 0.0)
        lstm_mse = temporal_res.get("lstm_mse", 0.0)

        physics_viol = physics_res.get("physics_violation", False)
        dew_viol = physics_res.get("dew_point_violation", False)
        hydro_viol = physics_res.get("hydrostatic_violation", False)

        spatial_anomaly = spatial_res.get("spatial_anomaly", False)
        temp_spatial_z = spatial_res.get("temp_spatial_z", 0.0)
        sp_spatial_z = spatial_res.get("sp_spatial_z", 0.0)
        insufficient_neighbors = spatial_res.get("insufficient_neighbors", False)

        # Baseline Status
        diagnosis_type = "NORMAL"
        root_cause = "Station operating within normal physical, temporal, and spatial bounds."
        severity_score = 0.0
        confidence_score = 1.0
        evidence_chain = []

        has_any_anomaly = missing_data or range_viol or rate_viol or flatline_viol or is_temporal_anomaly or physics_viol or spatial_anomaly

        if not has_any_anomaly:
            return {
                "is_anomalous": False,
                "diagnosis_type": diagnosis_type,
                "root_cause": root_cause,
                "severity_score": 0.0,
                "confidence_score": 0.98,
                "evidence_chain": ["All edge, temporal, physics, and spatial checks passed clean."]
            }

        # Disambiguation logic: Genuine Extreme Weather vs Sensor Fault
        # Condition for Genuine Extreme Weather:
        # High temporal delta / MSE BUT low spatial deviation (neighbors confirm storm/front) AND physics respected (Td <= T)
        spatial_agreement = (temp_spatial_z < 2.0) and (sp_spatial_z < 2.0) and not spatial_anomaly and not insufficient_neighbors
        physics_respected = not dew_viol and not hydro_viol and not flatline_viol

        if (rate_viol or is_temporal_anomaly or range_viol) and spatial_agreement and physics_respected:
            diagnosis_type = "GENUINE_EXTREME_EVENT"
            root_cause = "Genuine Extreme Meteorological Event (e.g. sharp cold front, pressure dip, storm) confirmed by high spatial neighbor consensus."
            severity_score = min(7.0 + max_z * 0.5, 10.0)
            confidence_score = 0.92
            evidence_chain.append("Edge/Temporal rate jump detected.")
            evidence_chain.append(f"Neighbor consensus confirmed event (Spatial Z={max_z:.1f} <= 2.0).")
            evidence_chain.append("Psychrometric and hydrostatic physical bounds respected.")
        elif (rate_viol or is_temporal_anomaly or range_viol) and insufficient_neighbors and physics_respected:
            diagnosis_type = "UNCONFIRMED_ANOMALY"
            root_cause = "Temporal/Rate anomaly detected, but insufficient neighbors (<150km) to confirm if Genuine Event or Sensor Drift."
            severity_score = min(5.0 + max_z * 0.3, 7.5)
            confidence_score = 0.50
            evidence_chain.append("Edge/Temporal jump detected.")
            evidence_chain.append("Spatial validation SKIPPED (INSUFFICIENT_NEIGHBOURS).")
        else:
            diagnosis_type = "SENSOR_FAULT"
            confidence_score = 0.95
            
            if missing_data:
                root_cause = "Communication Failure / Sensor Dropout (Missing NaN Telemetry)."
                severity_score = 9.5
                evidence_chain.append("Signal drop detected on edge QC missing data rule.")
            elif flatline_viol:
                root_cause = "Sensor Hardware Freeze / Flatline Fault."
                severity_score = 8.5
                evidence_chain.append("Sensor output remained statically unchanged over consecutive hours.")
            elif dew_viol:
                root_cause = "Relative Humidity / Temperature Sensor Calibration Fault (Psychrometric Violation)."
                severity_score = 8.0
                evidence_chain.append("Dew point calculated exceeds physical dry-bulb temperature limit.")
            elif hydro_viol:
                root_cause = "Surface / MSL Pressure Sensor Calibration Drift."
                severity_score = 7.5
                evidence_chain.append("Hydrostatic MSL-surface pressure delta deviates from station 189m elevation baseline.")
            elif rate_viol and spatial_anomaly:
                root_cause = "Isolated Sensor Spike Fault."
                severity_score = 8.8
                evidence_chain.append(f"Abrupt single-station jump unconfirmed by neighboring AWS network (Spatial Z={max(temp_spatial_z, sp_spatial_z):.1f} > 3.0).")
            elif spatial_anomaly:
                root_cause = "Station Sensor Calibration Drift."
                severity_score = 6.5
                evidence_chain.append("Station readings show sustained deviation from distance-weighted neighbor network consensus.")
            else:
                root_cause = "Temporal Anomaly / Unspecified Sensor Degradation."
                severity_score = 6.0
                evidence_chain.append(f"Temporal AI reconstruction error (MSE={lstm_mse:.3f}) exceeded baseline threshold.")

        return {
            "is_anomalous": True,
            "diagnosis_type": diagnosis_type,
            "root_cause": root_cause,
            "severity_score": float(np.round(severity_score, 1)),
            "confidence_score": float(np.round(confidence_score, 2)),
            "evidence_chain": evidence_chain
        }
