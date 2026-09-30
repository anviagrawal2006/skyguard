"""
Stage 5: Multi-Class Fault Classifier.
Uses XGBoost / RandomForest trained over 14 engineered physical, temporal,
and spatial features to classify specific fault categories.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


class FaultClassifierEngine:
    """
    Multi-class classifier predicting exact fault type:
    [CLEAN, SPIKE, FROZEN, DRIFT, COMM_FAILURE, INCONSISTENT, GENUINE_EXTREME]
    """

    FEATURE_NAMES = [
        "rate_temp", "rate_rh", "rate_press", "flatline_length", "missing_flag",
        "lstm_mse", "iso_forest_score", "max_z_score", "hydrostatic_residual",
        "dew_point_deficit", "temp_spatial_z", "rh_spatial_z", "sp_spatial_z",
        "spatial_agreement_ratio"
    ]

    CLASSES = ["CLEAN", "SPIKE", "FROZEN", "DRIFT", "COMM_FAILURE", "INCONSISTENT", "GENUINE_EXTREME"]

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.CLASSES)
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.08,
            random_state=42,
            eval_metric="mlogloss"
        )
        self.is_fitted = False

    def extract_features(
        self,
        reading: Dict[str, Any],
        edge_flags: Dict[str, Any],
        temporal_res: Dict[str, Any],
        physics_res: Dict[str, Any],
        spatial_res: Dict[str, Any],
        recent_history: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Constructs 14-element feature vector from pipeline evaluation results."""
        # 1-3. Rates of change
        prev = recent_history[-1] if recent_history else reading
        rate_temp = abs(reading.get("temperature_2m", 0) - prev.get("temperature_2m", 0)) if reading.get("temperature_2m") is not None and prev.get("temperature_2m") is not None else 0.0
        rate_rh = abs(reading.get("relative_humidity_2m", 0) - prev.get("relative_humidity_2m", 0)) if reading.get("relative_humidity_2m") is not None and prev.get("relative_humidity_2m") is not None else 0.0
        rate_press = abs(reading.get("surface_pressure", 0) - prev.get("surface_pressure", 0)) if reading.get("surface_pressure") is not None and prev.get("surface_pressure") is not None else 0.0

        # 4. Flatline length
        flatline_len = 0
        if edge_flags.get("flatline_violation", False):
            flatline_len = 4
            if len(recent_history) >= 4:
                # Count back identical readings
                curr_t = reading.get("temperature_2m")
                for r in reversed(recent_history):
                    if r.get("temperature_2m") == curr_t:
                        flatline_len += 1
                    else:
                        break

        # 5. Missing flag
        missing_flag = 1.0 if edge_flags.get("missing_data", False) else 0.0

        # 6-8. Temporal AI scores
        lstm_mse = temporal_res.get("lstm_mse", 0.0)
        iso_score = temporal_res.get("iso_forest_score", 0.0)
        max_z = temporal_res.get("max_z_score", 0.0)

        # 9-10. Physics residuals
        hydro_res = physics_res.get("hydrostatic_residual_hpa", 0.0) or 0.0
        temp = reading.get("temperature_2m", 20.0) or 20.0
        dew = physics_res.get("dew_point_c", 15.0) or 15.0
        dew_deficit = temp - dew

        # 11-13. Spatial Z-scores
        sz_temp = spatial_res.get("temp_spatial_z", 0.0)
        sz_rh = spatial_res.get("rh_spatial_z", 0.0)
        sz_sp = spatial_res.get("sp_spatial_z", 0.0)

        # 14. Spatial agreement ratio
        spatial_agreement = 1.0 if (sz_temp < 2.0 and sz_sp < 2.0) else 0.0

        feature_vector = np.array([
            rate_temp, rate_rh, rate_press, flatline_len, missing_flag,
            lstm_mse, iso_score, max_z, hydro_res, dew_deficit,
            sz_temp, sz_rh, sz_sp, spatial_agreement
        ], dtype=np.float32)

        return feature_vector

    def fit(self, X: np.ndarray, y: List[str]):
        """Fits the multi-class fault classifier on engineered training vectors."""
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        self.model.fit(X, y_encoded)
        self.is_fitted = True

    def save(self, filepath: str):
        import joblib
        joblib.dump({"model": self.model, "encoder": self.label_encoder}, filepath)
        
    def load(self, filepath: str):
        import joblib
        import os
        if os.path.exists(filepath):
            data = joblib.load(filepath)
            self.model = data["model"]
            self.label_encoder = data["encoder"]
            self.is_fitted = True
            return True
        return False

    def classify(self, feature_vector: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Classifies feature vector into fault label, confidence probability, and class distribution.
        """
        if not self.is_fitted:
            # Rule fallback before training
            rate_temp = feature_vector[0]
            flatline = feature_vector[3]
            missing = feature_vector[4]
            temp_spat_z = feature_vector[10]
            
            if missing == 1.0: return "COMM_FAILURE", 0.9, {}
            if flatline > 2: return "FROZEN", 0.9, {}
            if abs(rate_temp) > 5.0 and temp_spat_z > 3.0: return "SPIKE", 0.9, {}
            if temp_spat_z > 3.0: return "DRIFT", 0.7, {}
            
            return "UNKNOWN_FAULT", 0.5, {}

        X_in = feature_vector.reshape(1, -1)
        probs = self.model.predict_proba(X_in)[0]
        pred_idx = np.argmax(probs)
        pred_label = self.label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(probs[pred_idx])
        prob_dict = {str(self.label_encoder.inverse_transform([i])[0]): float(p) for i, p in enumerate(probs)}

        return pred_label, confidence, prob_dict
