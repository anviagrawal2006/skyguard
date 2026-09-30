"""
Stage 2: Temporal AI Anomaly Model.
Includes PyTorch LSTM Autoencoder (sequence-to-sequence reconstruction) and
Scikit-Learn Isolation Forest / Rolling Z-Score baselines for hardware fallbacks.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Attempt PyTorch import with robust fallback
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class PyTorchLSTMAutoencoder(nn.Module):
        """
        Sequence-to-Sequence PyTorch LSTM Autoencoder.
        Encoder compresses 24-hour sequence into latent representation.
        Decoder reconstructs the 24-hour multivariate sequence.
        """
        def __init__(self, seq_len: int = 24, n_features: int = 4, hidden_dim: int = 32, latent_dim: int = 16):
            super().__init__()
            self.seq_len = seq_len
            self.n_features = n_features
            
            # Encoder
            self.encoder_lstm = nn.LSTM(n_features, hidden_dim, batch_first=True)
            self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
            
            # Decoder
            self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
            self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            self.output_layer = nn.Linear(hidden_dim, n_features)

        def forward(self, x):
            # x shape: (batch_size, seq_len, n_features)
            batch_size = x.size(0)
            _, (hn, _) = self.encoder_lstm(x)
            latent = self.encoder_fc(hn.squeeze(0)) # (batch_size, latent_dim)
            
            # Repeat latent vector for sequence length
            dec_in = self.decoder_fc(latent).unsqueeze(1).repeat(1, self.seq_len, 1)
            dec_out, _ = self.decoder_lstm(dec_in)
            reconstruction = self.output_layer(dec_out) # (batch_size, seq_len, n_features)
            return reconstruction


class TemporalAIEngine:
    """
    Temporal AI engine evaluating 24-hour sliding window patterns.
    Uses PyTorch LSTM Autoencoder if available, supplemented with
    Scikit-Learn IsolationForest and Rolling Z-Score models.
    """

    def __init__(self, seq_len: int = 24, features: List[str] = None):
        self.seq_len = seq_len
        self.features = features or ["temperature_2m", "relative_humidity_2m", "surface_pressure", "pressure_msl"]
        self.scaler = StandardScaler()
        self.iso_forest = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
        self.use_pytorch = TORCH_AVAILABLE
        self.lstm_model = None
        self.threshold_mse = 0.5
        self.is_fitted = False

    def _to_scale_invariant(self, df: pd.DataFrame) -> np.ndarray:
        # Compute first-order differences (rate of change) to remove absolute baseline
        data = df[self.features].ffill().bfill()
        diffs = data.diff().fillna(0).values
        return diffs

    def _prepare_sequences(self, df: pd.DataFrame) -> np.ndarray:
        """Converts DataFrame feature matrix into (num_samples, seq_len, num_features) array."""
        data_clean = self._to_scale_invariant(df)
        scaled = self.scaler.transform(data_clean)
        
        sequences = []
        for i in range(len(scaled) - self.seq_len + 1):
            sequences.append(scaled[i : i + self.seq_len])
        return np.array(sequences)

    def fit(self, train_df: pd.DataFrame, epochs: int = 5):
        """Fits Scaler, IsolationForest baseline, and PyTorch LSTM Autoencoder on clean training data."""
        data_clean = self._to_scale_invariant(train_df)
        
        # 1. Fit Scaler
        self.scaler.fit(data_clean)
        scaled_data = self.scaler.transform(data_clean)

        # 2. Fit Isolation Forest baseline
        self.iso_forest.fit(scaled_data)

        # 3. Fit PyTorch LSTM Autoencoder if PyTorch is functional
        if self.use_pytorch:
            try:
                torch.manual_seed(42)
                np.random.seed(42)
                seqs = self._prepare_sequences(train_df)
                if len(seqs) > 0:
                    tensor_seqs = torch.tensor(seqs, dtype=torch.float32)
                    # Train mini-batches
                    self.lstm_model = PyTorchLSTMAutoencoder(seq_len=self.seq_len, n_features=len(self.features))
                    optimizer = optim.Adam(self.lstm_model.parameters(), lr=1e-3)
                    criterion = nn.MSELoss()

                    self.lstm_model.train()
                    batch_size = 256
                    for epoch in range(epochs):
                        permutation = torch.randperm(tensor_seqs.size(0))
                        for i in range(0, tensor_seqs.size(0), batch_size):
                            indices = permutation[i:i+batch_size]
                            batch_x = tensor_seqs[indices]
                            optimizer.zero_grad()
                            recon = self.lstm_model(batch_x)
                            loss = criterion(recon, batch_x)
                            loss.backward()
                            optimizer.step()

                    # Compute 99th percentile validation MSE threshold
                    self.lstm_model.eval()
                    with torch.no_grad():
                        recon_all = self.lstm_model(tensor_seqs)
                        mse_losses = torch.mean((recon_all - tensor_seqs)**2, dim=(1, 2)).numpy()
                        self.threshold_mse = float(np.percentile(mse_losses, 99.0))
            except Exception as e:
                print(f"Notice: PyTorch LSTM training fallback to IsolationForest baseline ({e})")
                self.use_pytorch = False
                
        if not self.use_pytorch:
            # Fallback score threshold calculation
            iso_scores = -self.iso_forest.score_samples(scaled_data)
            # Approximate z-scores
            means = np.mean(data_clean, axis=0)
            stds = np.std(data_clean, axis=0) + 1e-6
            z_scores = (data_clean - means) / stds
            fallback_scores = np.mean(np.abs(z_scores), axis=1) * 0.1 + iso_scores * 0.2
            self.threshold_mse = float(np.percentile(fallback_scores, 99.0))

        self.is_fitted = True

    def save(self, filepath: str):
        import joblib, os
        base_dir = os.path.dirname(filepath)
        os.makedirs(base_dir, exist_ok=True)
        joblib.dump({"scaler": self.scaler, "iso": self.iso_forest, "thresh": self.threshold_mse}, filepath + "_sklearn.pkl")
        if self.use_pytorch and self.lstm_model is not None:
            import torch
            torch.save(self.lstm_model.state_dict(), filepath + "_lstm.pt")
            
    def load(self, filepath: str):
        import joblib, os
        if os.path.exists(filepath + "_sklearn.pkl"):
            data = joblib.load(filepath + "_sklearn.pkl")
            self.scaler = data["scaler"]
            self.iso_forest = data["iso"]
            self.threshold_mse = data.get("thresh", 0.5)
            self.is_fitted = True
            
            if self.use_pytorch and os.path.exists(filepath + "_lstm.pt"):
                import torch
                self.lstm_model = LSTMAutoencoder(self.seq_len, len(self.features))
                self.lstm_model.load_state_dict(torch.load(filepath + "_lstm.pt", weights_only=True))
                self.lstm_model.eval()
            return True
        return False

    def evaluate_window(self, window_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluates a 24-hour sequence window.
        Returns MSE reconstruction error, Isolation Forest score, and Z-scores.
        """
        # Cold Start Guard
        if len(window_df) < 3:
            return {
                "lstm_mse": 0.0,
                "threshold_mse": self.threshold_mse,
                "iso_forest_score": 0.0,
                "max_z_score": 0.0,
                "is_temporal_anomaly": False,
                "reason": "Cold start: insufficient history",
                "max_z_score": 0.0,
                "z_scores": {f: 0.0 for f in self.features},
                "is_temporal_anomaly": False
            }

        if len(window_df) < self.seq_len:
            # Padding if shorter than seq_len
            pad_rows = self.seq_len - len(window_df)
            padded_head = pd.DataFrame([window_df.iloc[0].to_dict()] * pad_rows)
            window_df = pd.concat([padded_head, window_df], ignore_index=True)

        window_df = window_df.tail(self.seq_len)
        data_clean = self._to_scale_invariant(window_df)

        # 1. Rolling Z-Score of the last timestep relative to the 24-hour window
        means = np.mean(data_clean[:-1], axis=0) if len(data_clean) > 1 else np.mean(data_clean, axis=0)
        stds = np.std(data_clean[:-1], axis=0)
        # Guard against zero variance in padded cold starts
        stds = np.maximum(stds, [0.5, 2.0, 1.0, 1.0])
        last_val = data_clean[-1]
        z_scores = (last_val - means) / stds

        # 2. Isolation Forest score
        scaled_last = self.scaler.transform([last_val])
        iso_score = float(-self.iso_forest.score_samples(scaled_last)[0])

        # 3. LSTM Reconstruction Error
        lstm_mse = 0.0
        if self.use_pytorch and self.lstm_model is not None:
            try:
                scaled_seq = self.scaler.transform(data_clean)
                tensor_seq = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(0)
                self.lstm_model.eval()
                with torch.no_grad():
                    recon = self.lstm_model(tensor_seq)
                    lstm_mse = float(torch.mean((recon - tensor_seq) ** 2).item())
            except Exception:
                lstm_mse = iso_score
        else:
            # Fallback score proportional to Isolation Forest & Z-scores
            lstm_mse = float(np.mean(np.abs(z_scores)) * 0.1 + iso_score * 0.2)

        is_temporal_anomaly = lstm_mse > self.threshold_mse or iso_score > 0.65 or np.max(np.abs(z_scores)) > 3.5

        return {
            "lstm_mse": lstm_mse,
            "threshold_mse": self.threshold_mse,
            "iso_forest_score": iso_score,
            "max_z_score": float(np.max(np.abs(z_scores))),
            "z_scores": {f: float(z_scores[i]) for i, f in enumerate(self.features)},
            "is_temporal_anomaly": bool(is_temporal_anomaly)
        }
