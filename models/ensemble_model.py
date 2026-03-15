"""
Ensemble Model V2 - Combines ALL models including Advanced Engine.
Provides the final, most confident prediction using 10+ techniques.
"""
import numpy as np
from collections import Counter

from models.frequency_model import FrequencyModel
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from models.advanced_engine import AdvancedPredictionEngine


class EnsembleModel:
    """Combines ALL models for maximum prediction accuracy."""
    
    def __init__(self, max_number, pick_count):
        self.max_number = max_number
        self.pick_count = pick_count
        
        # Sub-models
        self.freq_model = FrequencyModel(max_number, pick_count)
        self.lstm_model = LSTMModel(max_number, pick_count, sequence_length=20)
        self.transformer_model = TransformerModel(max_number, pick_count, sequence_length=30)
        self.advanced_engine = AdvancedPredictionEngine(max_number, pick_count)
        
        self.is_trained = False
        self.training_info = {}
    
    def fit(self, data, train_deep=True, epochs=50, verbose=0):
        """Train all models."""
        print("=" * 60)
        print("  TRAINING ENSEMBLE MODEL (V2 - Advanced)")
        print("=" * 60)
        
        # 1. Frequency Model
        print("\n[1/4] Training Frequency Model...")
        self.freq_model.fit(data)
        self.training_info['frequency'] = {'status': 'trained', 'data_points': len(data)}
        
        # 2. Advanced Engine (8 methods)
        print("\n[2/4] Running Advanced Engine (8 methods)...")
        self.advanced_engine.fit(data)
        self.training_info['advanced'] = {'status': 'trained', 'data_points': len(data), 'methods': 8}
        
        if train_deep and len(data) >= 50:
            # 3. LSTM
            print("\n[3/4] Training LSTM Model...")
            self.lstm_model.fit(data, epochs=epochs, verbose=verbose)
            self.training_info['lstm'] = {
                'status': 'trained' if self.lstm_model.is_trained else 'fallback',
                'data_points': len(data)
            }
            
            # 4. Transformer
            print("\n[4/4] Training Transformer Model...")
            self.transformer_model.fit(data, epochs=epochs, verbose=verbose)
            self.training_info['transformer'] = {
                'status': 'trained' if self.transformer_model.is_trained else 'fallback',
                'data_points': len(data)
            }
        else:
            self.training_info['lstm'] = {'status': 'skipped'}
            self.training_info['transformer'] = {'status': 'skipped'}
            print("\n[3/4] LSTM: Skipped")
            print("[4/4] Transformer: Skipped")
        
        self.is_trained = True
        print("\n" + "=" * 60)
        print("  TRAINING COMPLETE! (10+ analysis methods active)")
        print("=" * 60)
    
    def predict(self, data, n_sets=5):
        """Generate final ensemble predictions."""
        # Get predictions from ALL sources
        freq_preds = self.freq_model.predict(n_sets=n_sets * 2)
        lstm_preds = self.lstm_model.predict(data, n_sets=n_sets * 2)
        trans_preds = self.transformer_model.predict(data, n_sets=n_sets * 2)
        adv_preds = self.advanced_engine.predict(n_sets=n_sets * 3)  # More weight to advanced
        
        # Weighted vote counting
        vote_counter = Counter()
        w = {'freq': 0.15, 'lstm': 0.15, 'trans': 0.15, 'adv': 0.55}  # Advanced gets most weight
        
        for pred in freq_preds:
            for num in pred: vote_counter[num] += w['freq']
        for pred in lstm_preds:
            for num in pred: vote_counter[num] += w['lstm']
        for pred in trans_preds:
            for num in pred: vote_counter[num] += w['trans']
        for pred in adv_preds:
            for num in pred: vote_counter[num] += w['adv']
        
        # Build distribution
        total = sum(vote_counter.values())
        probs = {}
        for n in range(1, self.max_number + 1):
            probs[n] = (vote_counter.get(n, 0) + 0.05) / (total + self.max_number * 0.05)
        
        numbers = list(probs.keys())
        weights = np.array(list(probs.values()))
        weights = weights / weights.sum()
        
        predictions = []
        for _ in range(n_sets):
            selected = np.random.choice(numbers, size=self.pick_count, 
                                        replace=False, p=weights)
            predictions.append(sorted(selected.tolist()))
        
        return predictions
    
    def predict_all_models(self, data, n_sets=3):
        """Get predictions from EACH model separately + ensemble."""
        results = {}
        
        # Advanced Engine (top priority)
        results['advanced'] = {
            'name': 'Advanced Engine (8 Phương Pháp)',
            'icon': '⚡',
            'predictions': self.advanced_engine.predict(n_sets=n_sets),
            'status': 'trained',
            'description': 'Markov + Bayesian + Chi-Square + Monte Carlo + Gap Analysis + Association Rules + Pattern Mining + Recency'
        }
        
        # Frequency
        results['frequency'] = {
            'name': 'Phân Tích Tần Suất',
            'icon': '📊',
            'predictions': self.freq_model.predict(n_sets=n_sets),
            'status': self.training_info.get('frequency', {}).get('status', 'unknown')
        }
        
        # LSTM
        results['lstm'] = {
            'name': 'LSTM Neural Network',
            'icon': '🧠',
            'predictions': self.lstm_model.predict(data, n_sets=n_sets),
            'status': self.training_info.get('lstm', {}).get('status', 'unknown')
        }
        
        # Transformer
        results['transformer'] = {
            'name': 'Transformer AI',
            'icon': '🤖',
            'predictions': self.transformer_model.predict(data, n_sets=n_sets),
            'status': self.training_info.get('transformer', {}).get('status', 'unknown')
        }
        
        # Final Ensemble
        results['ensemble'] = {
            'name': 'ULTIMATE ENSEMBLE (Tổng hợp 10+ kỹ thuật)',
            'icon': '🎯',
            'predictions': self.predict(data, n_sets=n_sets),
            'status': 'trained' if self.is_trained else 'unknown',
            'description': 'Kết hợp tất cả 10+ phương pháp AI & thống kê'
        }
        
        return results
    
    def get_analysis(self):
        """Get comprehensive analysis from frequency model."""
        return self.freq_model.get_analysis_summary()
    
    def get_advanced_analysis(self):
        """Get the advanced engine's full analysis."""
        return self.advanced_engine.get_full_analysis()
    
    def get_all_probabilities(self, data):
        """Get number probabilities from all models."""
        return {
            'frequency': self.freq_model.get_frequency_stats(),
            'lstm': self.lstm_model.get_probabilities(data),
            'transformer': self.transformer_model.get_probabilities(data)
        }
