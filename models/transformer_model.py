"""
Transformer Model - Self-Attention based prediction model.
Uses multi-head attention to find complex patterns in lottery sequences.
"""
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TransformerModel:
    """Transformer-based lottery number prediction."""
    
    def __init__(self, max_number, pick_count, sequence_length=30):
        self.max_number = max_number
        self.pick_count = pick_count
        self.seq_len = sequence_length
        self.model = None
        self.is_trained = False
        self.model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data', f'transformer_{max_number}.weights.h5'
        )
    
    def _build_model(self):
        """Build Transformer model."""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import (
                Input, Dense, Dropout, LayerNormalization, 
                MultiHeadAttention, GlobalAveragePooling1D, Add
            )
            
            inputs = Input(shape=(self.seq_len, self.max_number))
            
            # Positional encoding (simple learned)
            pos_encoding = Dense(self.max_number)(inputs)
            x = Add()([inputs, pos_encoding])
            
            # Transformer Block 1
            attn_output = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
            attn_output = Dropout(0.2)(attn_output)
            x1 = LayerNormalization()(Add()([x, attn_output]))
            
            ff = Dense(128, activation='relu')(x1)
            ff = Dense(self.max_number)(ff)
            ff = Dropout(0.2)(ff)
            x2 = LayerNormalization()(Add()([x1, ff]))
            
            # Transformer Block 2
            attn_output2 = MultiHeadAttention(num_heads=4, key_dim=32)(x2, x2)
            attn_output2 = Dropout(0.2)(attn_output2)
            x3 = LayerNormalization()(Add()([x2, attn_output2]))
            
            ff2 = Dense(128, activation='relu')(x3)
            ff2 = Dense(self.max_number)(ff2)
            ff2 = Dropout(0.2)(ff2)
            x4 = LayerNormalization()(Add()([x3, ff2]))
            
            # Output
            x5 = GlobalAveragePooling1D()(x4)
            x5 = Dense(128, activation='relu')(x5)
            x5 = Dropout(0.3)(x5)
            outputs = Dense(self.max_number, activation='sigmoid')(x5)
            
            model = Model(inputs, outputs)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            self.model = model
            return True
            
        except ImportError:
            print("[Transformer] TensorFlow not available. Using fallback mode.")
            return False
    
    def _encode_draw(self, numbers):
        """Convert draw to binary vector."""
        vec = np.zeros(self.max_number)
        for n in numbers:
            if 1 <= n <= self.max_number:
                vec[n - 1] = 1
        return vec
    
    def _prepare_data(self, data):
        """Prepare training sequences."""
        encoded = [self._encode_draw(draw[:self.pick_count]) for draw in data]
        X, y = [], []
        for i in range(len(encoded) - self.seq_len):
            X.append(encoded[i:i + self.seq_len])
            y.append(encoded[i + self.seq_len])
        return np.array(X), np.array(y)
    
    def fit(self, data, epochs=50, batch_size=32, verbose=0):
        """Train the Transformer model."""
        if len(data) < self.seq_len + 10:
            print(f"[Transformer] Not enough data")
            self.is_trained = False
            return
        
        if not self._build_model():
            self.is_trained = False
            return
        
        X, y = self._prepare_data(data)
        print(f"[Transformer] Training on {len(X)} sequences...")
        
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
            
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.model.save_weights(self.model_path)
            self.is_trained = True
            print("[Transformer] Training complete!")
            
        except Exception as e:
            print(f"[Transformer] Training error: {e}")
            self.is_trained = False
    
    def predict(self, data, n_sets=5):
        """Generate predictions."""
        if not self.is_trained or self.model is None:
            return self._fallback_predict(data, n_sets)
        
        recent = data[-self.seq_len:]
        encoded = [self._encode_draw(draw[:self.pick_count]) for draw in recent]
        X = np.array([encoded])
        
        probs = self.model.predict(X, verbose=0)[0]
        
        predictions = []
        for _ in range(n_sets):
            noisy_probs = probs + np.random.normal(0, 0.03, len(probs))
            noisy_probs = np.clip(noisy_probs, 0.01, 1.0)
            noisy_probs = noisy_probs / noisy_probs.sum()
            
            indices = np.random.choice(
                range(self.max_number), 
                size=self.pick_count,
                replace=False, 
                p=noisy_probs
            )
            predictions.append(sorted([i + 1 for i in indices]))
        
        return predictions
    
    def _fallback_predict(self, data, n_sets=5):
        """Fallback using weighted moving average."""
        from collections import Counter
        
        # Weighted by recency
        freq = Counter()
        total_weight = 0
        for idx, draw in enumerate(data):
            weight = 1.0 + idx * 0.01  # More recent = more weight
            for n in draw[:self.pick_count]:
                freq[n] += weight
            total_weight += weight
        
        probs = {}
        for n in range(1, self.max_number + 1):
            probs[n] = (freq.get(n, 0) + 0.5) / (total_weight + self.max_number * 0.5)
        
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
        """Get probability for each number."""
        if not self.is_trained or self.model is None:
            from collections import Counter
            recent = data[-50:] if len(data) >= 50 else data
            freq = Counter()
            for draw in recent:
                for n in draw[:self.pick_count]:
                    freq[n] += 1
            total = sum(freq.values())
            return {n: round(freq.get(n, 0) / max(total, 1) * 100, 2) 
                    for n in range(1, self.max_number + 1)}
        
        recent = data[-self.seq_len:]
        encoded = [self._encode_draw(draw[:self.pick_count]) for draw in recent]
        X = np.array([encoded])
        probs = self.model.predict(X, verbose=0)[0]
        return {i + 1: round(float(probs[i]) * 100, 2) for i in range(self.max_number)}
