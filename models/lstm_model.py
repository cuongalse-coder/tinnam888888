"""
LSTM Model - Long Short-Term Memory neural network for lottery prediction.
Uses sequence-to-sequence learning to find patterns in historical draws.
"""
import numpy as np
import os
import json

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class LSTMModel:
    """LSTM neural network for lottery number prediction."""
    
    def __init__(self, max_number, pick_count, sequence_length=20):
        """
        Args:
            max_number: Maximum number in lottery (45 or 55)
            pick_count: Numbers to pick (6)
            sequence_length: How many past draws to use as input
        """
        self.max_number = max_number
        self.pick_count = pick_count
        self.seq_len = sequence_length
        self.model = None
        self.is_trained = False
        self.model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'data', f'lstm_{max_number}.weights.h5'
        )
    
    def _build_model(self):
        """Build the LSTM model architecture."""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
            from tensorflow.keras.optimizers import Adam
            
            model = Sequential([
                LSTM(128, input_shape=(self.seq_len, self.max_number), return_sequences=True),
                Dropout(0.3),
                BatchNormalization(),
                LSTM(64, return_sequences=False),
                Dropout(0.3),
                BatchNormalization(),
                Dense(128, activation='relu'),
                Dropout(0.2),
                Dense(self.max_number, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            self.model = model
            return True
        except ImportError:
            print("[LSTM] TensorFlow not available. Using fallback mode.")
            return False
    
    def _encode_draw(self, numbers):
        """Convert a draw to binary vector (one-hot style)."""
        vec = np.zeros(self.max_number)
        for n in numbers:
            if 1 <= n <= self.max_number:
                vec[n - 1] = 1
        return vec
    
    def _prepare_data(self, data):
        """Prepare training data from historical draws."""
        encoded = [self._encode_draw(draw[:self.pick_count]) for draw in data]
        
        X, y = [], []
        for i in range(len(encoded) - self.seq_len):
            X.append(encoded[i:i + self.seq_len])
            y.append(encoded[i + self.seq_len])
        
        return np.array(X), np.array(y)
    
    def fit(self, data, epochs=50, batch_size=32, verbose=0):
        """Train the LSTM model on historical data."""
        if len(data) < self.seq_len + 10:
            print(f"[LSTM] Not enough data (need {self.seq_len + 10}, got {len(data)})")
            self.is_trained = False
            return
        
        if not self._build_model():
            self.is_trained = False
            return
        
        X, y = self._prepare_data(data)
        
        if len(X) == 0:
            print("[LSTM] No training sequences could be generated")
            self.is_trained = False
            return
        
        print(f"[LSTM] Training on {len(X)} sequences (seq_len={self.seq_len})...")
        
        try:
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
            ]
            
            self.model.fit(
                X, y, 
                epochs=epochs, 
                batch_size=batch_size, 
                validation_split=0.2,
                callbacks=callbacks,
                verbose=verbose
            )
            
            # Save model weights
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.model.save_weights(self.model_path)
            self.is_trained = True
            print("[LSTM] Training complete!")
            
        except Exception as e:
            print(f"[LSTM] Training error: {e}")
            self.is_trained = False
    
    def predict(self, data, n_sets=5):
        """Generate predictions using the trained LSTM model."""
        if not self.is_trained or self.model is None:
            return self._fallback_predict(data, n_sets)
        
        # Use last seq_len draws as input
        recent = data[-self.seq_len:]
        encoded = [self._encode_draw(draw[:self.pick_count]) for draw in recent]
        X = np.array([encoded])
        
        # Get probability distribution
        probs = self.model.predict(X, verbose=0)[0]
        
        predictions = []
        for _ in range(n_sets):
            # Sample from probability distribution with some noise
            noisy_probs = probs + np.random.normal(0, 0.05, len(probs))
            noisy_probs = np.clip(noisy_probs, 0.01, 1.0)
            noisy_probs = noisy_probs / noisy_probs.sum()
            
            # Select top numbers
            indices = np.random.choice(
                range(self.max_number), 
                size=self.pick_count, 
                replace=False, 
                p=noisy_probs
            )
            numbers = sorted([i + 1 for i in indices])
            predictions.append(numbers)
        
        return predictions
    
    def _fallback_predict(self, data, n_sets=5):
        """Fallback prediction when TF is not available."""
        # Use simple pattern-based prediction
        from collections import Counter
        
        recent = data[-50:] if len(data) >= 50 else data
        freq = Counter()
        for draw in recent:
            for n in draw[:self.pick_count]:
                freq[n] += 1
        
        # Weight by recency
        total = sum(freq.values())
        probs = {}
        for n in range(1, self.max_number + 1):
            probs[n] = (freq.get(n, 0) + 1) / (total + self.max_number)
        
        numbers = list(probs.keys())
        weights = np.array(list(probs.values()))
        weights = weights / weights.sum()
        
        predictions = []
        for _ in range(n_sets):
            selected = np.random.choice(numbers, size=self.pick_count, 
                                        replace=False, p=weights)
            predictions.append(sorted(selected.tolist()))
        
        return predictions
    
    def get_probabilities(self, data):
        """Get probability for each number (1 to max_number)."""
        if not self.is_trained or self.model is None:
            # Fallback
            from collections import Counter
            recent = data[-50:] if len(data) >= 50 else data
            freq = Counter()
            for draw in recent:
                for n in draw[:self.pick_count]:
                    freq[n] += 1
            total = sum(freq.values())
            return {n: round(freq.get(n, 0) / total * 100, 2) for n in range(1, self.max_number + 1)}
        
        recent = data[-self.seq_len:]
        encoded = [self._encode_draw(draw[:self.pick_count]) for draw in recent]
        X = np.array([encoded])
        probs = self.model.predict(X, verbose=0)[0]
        
        return {i + 1: round(float(probs[i]) * 100, 2) for i in range(self.max_number)}
