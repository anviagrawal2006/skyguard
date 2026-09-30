"""
SkyGuard AI Master Pipeline Entry Point.
Integrates Stages 1 through 8 into a unified, high-performance process_reading() interface.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from skyguard.pipeline.edge_qc import EdgeQCEngine
from skyguard.pipeline.temporal_ai import TemporalAIEngine
from skyguard.pipeline.physics_spatial import PhysicsSpatialEngine
from skyguard.pipeline.counterfactual import CounterfactualDiagnosisEngine
from skyguard.pipeline.fault_classifier import FaultClassifierEngine
from skyguard.pipeline.explainability import ExplainabilityEngine
from skyguard.pipeline.self_healing import SelfHealingEngine
from skyguard.pipeline.health_intelligence import SensorHealthEngine


class SkyGuardPipeline:
    """
    Master Anomaly Detection, Diagnosis, Self-Healing, and Health Intelligence Pipeline.
    Exposes process_reading(current_reading, neighbor_readings) entry point.
    """

    def __init__(self, target_elevation_m: float = 189.0, neighbor_metadata: Dict[str, Dict] = None):
        self.target_elevation_m = target_elevation_m
        self.neighbor_metadata = neighbor_metadata or {}

        # Pipeline Stage Instances
        self.edge_qc = EdgeQCEngine()
        self.temporal_ai = TemporalAIEngine(seq_len=24)
        self.physics_spatial = PhysicsSpatialEngine(target_elevation_m=target_elevation_m)
        self.counterfactual = CounterfactualDiagnosisEngine()
        self.fault_classifier = FaultClassifierEngine()
        self.explainability = ExplainabilityEngine(self.fault_classifier)
        self.self_healing = SelfHealingEngine()
        self.health_engine = SensorHealthEngine(rolling_window_size=168)

        # Sliding History Buffer (24 hours)
        self.recent_history: List[Dict[str, Any]] = []

    def fit(self, train_df: pd.DataFrame, synthetic_corrupted_val_df: Optional[pd.DataFrame] = None):
        """
        Pre-trains Stage 2 Temporal AI (LSTM Autoencoder) and Stage 5 Fault Classifier (XGBoost).
        """
        print("Training Stage 2 Temporal AI Engine (Scaler, IsolationForest, PyTorch LSTM)...")
        self.temporal_ai.fit(train_df, epochs=3)

        if synthetic_corrupted_val_df is not None:
            print("Training Stage 5 Multi-Class Fault Classifier (XGBoost)...")
            X_train, y_train = self._prepare_classifier_training_data(synthetic_corrupted_val_df)
            if len(X_train) > 0:
                self.fault_classifier.fit(X_train, y_train)
                self.explainability.initialize_explainer(X_train)

    def _prepare_classifier_training_data(self, df: pd.DataFrame):
        """Generates synthetic training feature vectors for Fault Classifier."""
        X_list, y_list = [], []
        window_size = 24
        
        # Sample all anomalous rows + clean rows for balanced training set
        anomaly_indices = df[df["is_anomaly"] == 1].index.tolist()
        clean_indices = df[df["is_anomaly"] == 0].index.tolist()
        
        # Take all anomaly indices (above window_size) plus up to 1000 clean indices
        valid_anomaly = [i for i in anomaly_indices if i >= window_size]
        valid_clean = [i for i in clean_indices if i >= window_size][:1000]
        selected_indices = sorted(valid_anomaly + valid_clean)

        for i in selected_indices:
            sub_df = df.iloc[i - window_size : i + 1]
            curr_reading = sub_df.iloc[-1].to_dict()
            hist = sub_df.iloc[:-1].to_dict("records")
            
            edge = self.edge_qc.process_reading(curr_reading, hist)
            temp = self.temporal_ai.evaluate_window(sub_df.iloc[:-1])
            phys = self.physics_spatial.evaluate_physics(curr_reading)
            spat = {"temp_spatial_z": 0.5, "rh_spatial_z": 0.5, "sp_spatial_z": 0.5}

            vec = self.fault_classifier.extract_features(curr_reading, edge, temp, phys, spat, hist)
            label = curr_reading.get("anomaly_type", "CLEAN")
            
            X_list.append(vec)
            y_list.append(label if label in FaultClassifierEngine.CLASSES else "CLEAN")

        return np.array(X_list), y_list

    def process_reading(
        self,
        current_reading: Dict[str, Any],
        neighbor_readings: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Core Real-Time Telemetry Processing Entry Point.
        Input:
            current_reading: dict with time, temperature_2m, relative_humidity_2m, surface_pressure, pressure_msl
            neighbor_readings: optional dict of st_id -> telemetry dict for neighbors
        Output:
            Structured result containing anomaly status, diagnosis, SHAP explanations,
            self-healed corrected values, and sensor health index.
        """
        if neighbor_readings is None:
            neighbor_readings = {}

        # 1. Edge QC
        edge_flags = self.edge_qc.process_reading(current_reading, self.recent_history)

        # 2. Temporal AI Evaluation
        window_df = pd.DataFrame(self.recent_history + [current_reading])
        temporal_res = self.temporal_ai.evaluate_window(window_df)

        # 3. Physics & Spatial Consistency
        physics_res = self.physics_spatial.evaluate_physics(current_reading)
        spatial_res = self.physics_spatial.evaluate_spatial_consensus(
            current_reading, neighbor_readings, self.neighbor_metadata.get("neighbors", self.neighbor_metadata)
        )

        # 4. Counterfactual Diagnosis
        diag_res = self.counterfactual.diagnose(edge_flags, temporal_res, physics_res, spatial_res)

        # 5. Fault Classification & Disambiguation
        feat_vector = self.fault_classifier.extract_features(
            current_reading, edge_flags, temporal_res, physics_res, spatial_res, self.recent_history
        )

        # --- ROUTING FIX (Phase 1) ---
        # 1. Final Status comes strictly from Stage 4 (Counterfactual)
        diag_type = diag_res["diagnosis_type"]
        if not diag_res["is_anomalous"] or diag_type == "NORMAL":
            status = "Healthy"
            fault_type = "CLEAN"
            conf_prob = 0.98
        elif diag_type == "GENUINE_EXTREME_EVENT":
            status = "Warning"
            fault_type = "GENUINE_EXTREME"
            conf_prob = diag_res.get("confidence_score", 0.92)
        elif diag_type == "UNCONFIRMED_ANOMALY":
            status = "Warning"
            fault_type = "UNCONFIRMED_ANOMALY"
            conf_prob = diag_res.get("confidence_score", 0.50)
        else:
            status = "Faulty"
            
            # 2. Only if Faulty, we trust Stage 5 to classify the EXACT fault type.
            if edge_flags["missing_data"]:
                fault_type = "COMM_FAILURE"
                conf_prob = 0.98
            else:
                fault_type, conf_prob, class_probs = self.fault_classifier.classify(feat_vector)
                
                if fault_type in ["CLEAN", "NORMAL", "GENUINE_EXTREME"]:
                    fault_type = "UNKNOWN_FAULT" # Safety net
                else:
                    fault_type = fault_type

        # Synchronize diag_res root cause
        if status == "Faulty":
            if fault_type == "SPIKE":
                diag_res["root_cause"] = "Isolated Sensor Spike Fault (Abrupt rate-of-change jump)."
            elif fault_type == "FROZEN":
                diag_res["root_cause"] = "Sensor Hardware Freeze / Flatline Fault."
            elif fault_type == "DRIFT":
                diag_res["root_cause"] = "Sensor Calibration Drift."
            elif fault_type == "INCONSISTENT":
                diag_res["root_cause"] = "Psychrometric Physical Inconsistency (Dew Point Deficit)."
            elif fault_type == "COMM_FAILURE":
                diag_res["root_cause"] = "Communication Failure / Sensor Dropout (Missing NaN Telemetry)."
            elif fault_type == "UNKNOWN_FAULT":
                diag_res["root_cause"] = "Unspecified Sensor Fault (Failed XGBoost Classification)."

        # 6. SHAP Explainability
        explain_res = self.explainability.explain(feat_vector, fault_type, conf_prob, diag_res)

        # 7. Self-Healing Data Correction
        healing_res = self.self_healing.reconstruct(
            current_reading, self.recent_history, spatial_res, fault_type
        )

        # 8. Sensor Health Intelligence
        health_res = self.health_engine.update_and_evaluate(
            is_fault=(fault_type not in ["CLEAN", "NORMAL", "GENUINE_EXTREME"]),
            fault_type=fault_type,
            recon_error=temporal_res["lstm_mse"]
        )

        # Update sliding history buffer
        self.recent_history.append(current_reading)
        if len(self.recent_history) > 48:
            self.recent_history.pop(0)

        # Construct Master Output JSON
        master_output = {
            "timestamp": str(current_reading.get("time")),
            "station_id": self.neighbor_metadata.get("target", {}).get("station_id", "AWS_TARGET_30_25N_74_25E"),
            "status": status,
            "is_anomaly": (status != "Healthy"),
            "anomaly_type": str(fault_type),
            "fault_type": str(fault_type),
            "severity_score": float(diag_res["severity_score"]),
            "confidence_score": float(np.round(conf_prob, 4)),
            "diagnosis": {
                "diagnosis_type": diag_res["diagnosis_type"],
                "root_cause": diag_res["root_cause"],
                "evidence_chain": diag_res["evidence_chain"]
            },
            "explainability": {
                "reason_string": explain_res["reason_string"],
                "confidence_pct": explain_res["confidence_pct"],
                "shap_attributions": explain_res["shap_attributions"]
            },
            "original_telemetry": {
                "temperature_2m": current_reading.get("temperature_2m"),
                "relative_humidity_2m": current_reading.get("relative_humidity_2m"),
                "surface_pressure": current_reading.get("surface_pressure"),
                "pressure_msl": current_reading.get("pressure_msl")
            },
            "corrected_telemetry": {
                "needs_correction": healing_res["needs_correction"],
                "temperature_2m": healing_res["corrected_temperature_2m"],
                "relative_humidity_2m": healing_res["corrected_relative_humidity_2m"],
                "surface_pressure": healing_res["corrected_surface_pressure"],
                "pressure_msl": healing_res["corrected_pressure_msl"],
                "reconstruction_confidence_pct": healing_res["reconstruction_confidence_pct"],
                "imputation_method": healing_res["imputation_method"]
            },
            "sensor_health": health_res,
            "evidence_metrics": {
                "edge_flags": edge_flags,
                "temporal_lstm_mse": temporal_res["lstm_mse"],
                "physics_dew_point_c": physics_res["dew_point_c"],
                "hydrostatic_residual_hpa": physics_res["hydrostatic_residual_hpa"],
                "spatial_expected_temp": spatial_res["expected_temp"],
                "spatial_temp_z_score": spatial_res["temp_spatial_z"]
            }
        }

        return master_output
