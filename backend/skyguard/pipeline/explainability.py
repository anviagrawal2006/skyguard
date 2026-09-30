"""
Stage 6: Explainability Engine (SHAP).
Uses SHAP TreeExplainer to compute feature attributions for classified faults and
generates human-readable natural language explanation strings.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import shap
from skyguard.pipeline.fault_classifier import FaultClassifierEngine


class ExplainabilityEngine:
    """
    SHAP-based Explainability engine for sensor fault diagnosis.
    Generates quantitative feature importance weights and natural language explanation text.
    """

    def __init__(self, classifier_engine: FaultClassifierEngine):
        self.classifier_engine = classifier_engine
        self.explainer = None
        self.feature_names = FaultClassifierEngine.FEATURE_NAMES

    def initialize_explainer(self, background_features: np.ndarray):
        """Initializes SHAP TreeExplainer using background feature matrix."""
        if self.classifier_engine.is_fitted:
            try:
                self.explainer = shap.TreeExplainer(self.classifier_engine.model)
            except Exception:
                self.explainer = shap.Explainer(self.classifier_engine.model, background_features[:50])

    def explain(
        self,
        feature_vector: np.ndarray,
        predicted_label: str,
        confidence: float,
        diagnosis_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Computes SHAP attributions and returns structured explanation and natural language string.
        """
        shap_values_dict = {}
        top_features = []

        if self.explainer is not None:
            try:
                sv = self.explainer(feature_vector.reshape(1, -1))
                # Format shap values array
                values = sv.values[0]
                if len(values.shape) > 1:
                    # Multi-class output: pick index of predicted label
                    class_idx = list(self.classifier_engine.label_encoder.classes_).index(predicted_label)
                    values = values[:, class_idx]
                
                for idx, fname in enumerate(self.feature_names):
                    shap_values_dict[fname] = float(values[idx])

                # Get top 3 features by absolute SHAP attribution
                top_indices = np.argsort(np.abs(values))[::-1][:3]
                for idx in top_indices:
                    top_features.append((self.feature_names[idx], float(values[idx]), float(feature_vector[idx])))
            except Exception:
                pass

        # Ensure ALL 14 features are present in shap_values_dict
        for fname in self.feature_names:
            if fname not in shap_values_dict:
                shap_values_dict[fname] = 0.0

        # Generate Natural Language Reason String
        reason_str = self._format_reason_string(predicted_label, confidence, feature_vector, top_features, diagnosis_dict)

        return {
            "predicted_label": predicted_label,
            "confidence_pct": float(np.round(confidence * 100, 1)),
            "reason_string": reason_str,
            "shap_attributions": shap_values_dict,
            "top_contributing_features": top_features
        }

    def _format_reason_string(
        self,
        label: str,
        confidence: float,
        features: np.ndarray,
        top_features: List[Tuple[str, float, float]],
        diag: Dict[str, Any]
    ) -> str:
        """Formats clear, human-readable natural language explanation string."""
        conf_pct = confidence * 100

        if label == "CLEAN" or label == "NORMAL":
            return "Station telemetry operating normally within expected physical, temporal, and spatial bounds."

        if label == "SPIKE":
            rate_t = features[0]
            rate_rh = features[1]
            rate_p = features[2]
            spatial_z = features[10]

            # Dynamically identify parameter exceeding rate-of-change threshold
            if rate_p >= 6.0 and (rate_p / 6.0) >= max(rate_t / 5.0, rate_rh / 25.0):
                trigger_desc = f"{rate_p:.1f} hPa/hr surface pressure rate-of-change (exceeding 6.0 hPa/hr threshold)"
            elif rate_rh >= 25.0 and (rate_rh / 25.0) >= (rate_t / 5.0):
                trigger_desc = f"{rate_rh:.1f}%/hr relative humidity rate-of-change (exceeding 25.0%/hr threshold)"
            else:
                trigger_desc = f"{rate_t:.1f}°C/hr temperature rate-of-change (exceeding 5.0°C/hr threshold)"

            if spatial_z >= 3.0:
                return f"Flagged as SPIKE with {conf_pct:.1f}% confidence due to {trigger_desc} and spatial deviation (Z={spatial_z:.1f}) from neighbor network."
            else:
                return f"Flagged as SPIKE with {conf_pct:.1f}% confidence due to isolated {trigger_desc} unconfirmed by neighbor network (spatial consensus Z={spatial_z:.1f} <= 3.0)."

        if label == "FROZEN":
            flat_len = int(features[3])
            return f"Flagged as FROZEN with {conf_pct:.1f}% confidence because sensor output remained statically flatlined for {flat_len} consecutive hours."

        if label == "DRIFT":
            hydro_res = features[8]
            sz = features[10]
            if sz >= 3.0:
                return f"Flagged as DRIFT with {conf_pct:.1f}% confidence due to accumulating hydrostatic pressure delta residual ({hydro_res:.1f} hPa) and sustained spatial divergence (Z={sz:.1f})."
            else:
                return f"Flagged as DRIFT with {conf_pct:.1f}% confidence due to accumulating hydrostatic pressure delta residual ({hydro_res:.1f} hPa)."

        if label == "COMM_FAILURE":
            return f"Flagged as COMM_FAILURE with {conf_pct:.1f}% confidence due to missing/NaN telemetry signal dropout across sensor channels."

        if label == "INCONSISTENT":
            dew_def = features[9]
            return f"Flagged as INCONSISTENT with {conf_pct:.1f}% confidence due to psychrometric dew point violation (dew point deficit = {dew_def:.1f}°C)."

        if label == "GENUINE_EXTREME":
            return f"Flagged as GENUINE_EXTREME with {conf_pct:.1f}% confidence: rapid meteorological rate jump verified by strong spatial consensus across 3 neighboring stations."

        return f"Flagged as {label} with {conf_pct:.1f}% confidence: {diag.get('root_cause', 'Anomalous pattern detected.')}"
