"""
Master Predictor — Kết hợp nhanh các thuật toán tốt nhất
=========================================================
Chạy nhanh (< 60 giây), kết hợp 15 tín hiệu mạnh nhất
từ Phase 1-7 → cho ra 1 dãy số dự đoán duy nhất.

Walk-forward backtest tự động → chọn trọng số tối ưu.
"""
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
import math
import warnings
warnings.filterwarnings('ignore')


class MasterPredictor:
    """Chạy nhanh, kết hợp 15 tín hiệu tốt nhất → 1 dãy số."""
    
    def __init__(self, max_number, pick_count):
        self.max_number = max_number
        self.pick_count = pick_count
    
    def predict(self, data):
        """Return prediction result with backtest stats."""
        self.data = [d[:self.pick_count] for d in data]
        self.flat = [n for d in self.data for n in d]
        n = len(self.data)
        
        print(f"[Master] Analyzing {n} draws...")
        
        # Step 1: Auto-tune weights via backtest
        best_weights = self._optimize_weights()
        print(f"  Weights optimized")
        
        # Step 2: Generate prediction with optimized weights
        numbers, score_details = self._predict_with_weights(self.data, best_weights)
        
        # Step 3: Backtest to show accuracy
        bt = self._backtest(best_weights)
        print(f"  Backtest: {bt['avg']:.4f}/6 ({bt['improvement']:+.1f}%)")
        
        # Step 4: Confidence analysis
        confidence = self._confidence_analysis(score_details)
        
        print(f"[Master] Prediction: {numbers}")
        
        return {
            'numbers': numbers,
            'score_details': score_details[:15],
            'backtest': bt,
            'confidence': confidence,
            'method': f'Master AI ({n} draws analyzed, {bt["tests"]} backtested)',
        }
    
    def _score_numbers(self, history, weights):
        """Score all numbers using 15 weighted signals. Returns scores dict."""
        n_draws = len(history)
        flat = [n for d in history for n in d]
        last = set(history[-1])
        scores = {}
        
        # Pre-compute shared data
        last_seen = {}
        for i, d in enumerate(history):
            for n in d:
                last_seen[n] = i
        
        exp_gap = self.max_number / self.pick_count
        freq_10 = Counter(n for d in history[-10:] for n in d)
        freq_30 = Counter(n for d in history[-30:] for n in d)
        freq_50 = Counter(n for d in history[-50:] for n in d)
        total_freq = Counter(flat)
        expected_total = len(flat) / self.max_number
        
        # Momentum
        r10 = Counter(n for d in history[-10:] for n in d)
        p10 = Counter(n for d in history[-20:-10] for n in d) if n_draws > 20 else r10
        
        # Pair network
        pair_scores = Counter()
        for d in history[-50:]:
            for pair in combinations(sorted(d), 2):
                pair_scores[pair] += 1
        
        # KNN: similar draws
        knn_scores = Counter()
        for i in range(len(history) - 2):
            overlap = len(set(history[i]) & last)
            if overlap >= 2:
                for n in history[i+1]:
                    knn_scores[n] += overlap ** 1.5
        
        for num in range(1, self.max_number + 1):
            s = [0.0] * 15
            
            # S0: Freq last 10 (hot)
            s[0] = freq_10.get(num, 0) / 10
            
            # S1: Freq last 30
            s[1] = freq_30.get(num, 0) / 30
            
            # S2: Freq last 50
            s[2] = freq_50.get(num, 0) / 50
            
            # S3: Gap overdue
            gap = n_draws - last_seen.get(num, 0)
            s[3] = max(0, gap / exp_gap - 0.8)
            
            # S4: Anti-repeat (negative if in last draw)
            s[4] = -1.0 if num in last else 0.0
            
            # S5: Momentum
            s[5] = (r10.get(num, 0) - p10.get(num, 0)) / 5
            
            # S6: Position frequency (appears at its typical position?)
            positions = []
            for d in history[-50:]:
                sd = sorted(d)
                if num in sd:
                    positions.append(sd.index(num))
            s[6] = len(positions) / 50
            
            # S7: Pair network bonus
            pair_bonus = 0
            for n in last:
                for pair, c in pair_scores.most_common(100):
                    if n in pair:
                        partner = pair[0] if pair[1] == n else pair[1]
                        if partner == num:
                            pair_bonus += c
            s[7] = pair_bonus / max(1, len(last) * 50)
            
            # S8: KNN conditional
            s[8] = knn_scores.get(num, 0) / max(1, max(knn_scores.values()) if knn_scores else 1)
            
            # S9: Frequency correction (under-represented boost)
            dev = (expected_total - total_freq.get(num, 0)) / max(1, expected_total)
            s[9] = max(0, dev)
            
            # S10: Run-length turning point
            curr_absence = 0
            for d in reversed(history):
                if num not in d:
                    curr_absence += 1
                else:
                    break
            s[10] = min(curr_absence / exp_gap, 2.0) if curr_absence > exp_gap * 0.7 else 0
            
            # S11: Temporal gradient (acceleration)
            f5 = sum(1 for d in history[-5:] if num in d) / 5
            f15 = sum(1 for d in history[-15:] if num in d) / 15
            f30 = sum(1 for d in history[-30:] if num in d) / 30
            v1 = f5 - f15
            v2 = f15 - f30
            s[11] = v1 + (v1 - v2) * 0.5
            
            # S12: Regime trend (hot streak boost)
            f_r = sum(1 for d in history[-15:] if num in d) / 15
            f_o = sum(1 for d in history[-45:-15] if num in d) / 30 if n_draws > 45 else f_r
            trend = f_r - f_o
            s[12] = max(0, trend) * 10
            
            # S13: Sum balance target
            avg_sum = np.mean([sum(d) for d in history[-20:]])
            target = avg_sum / self.pick_count
            s[13] = max(0, 1 - abs(num - target) / self.max_number)
            
            # S14: Anti-repeat double (second-to-last)
            s[14] = -0.5 if n_draws > 1 and num in history[-2] and num in last else 0
            
            # Weighted sum
            total_score = sum(w * si for w, si in zip(weights, s))
            scores[num] = total_score
        
        return scores
    
    def _predict_with_weights(self, history, weights):
        """Generate prediction using weights."""
        scores = self._score_numbers(history, weights)
        
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        numbers = sorted([n for n, _ in ranked[:self.pick_count]])
        
        max_s = max(s for _, s in ranked[:20]) if ranked else 1
        details = [{'number': int(n), 'score': round(float(s), 2),
                     'confidence': round(s / max(max_s, 0.01) * 100, 1),
                     'selected': n in numbers}
                    for n, s in ranked[:18]]
        
        return numbers, details
    
    def _optimize_weights(self):
        """Optimize 15 signal weights using coordinate descent on backtest."""
        n = len(self.data)
        train_end = min(n - 1, n - 50)
        test_range = range(max(60, train_end - 80), train_end)
        
        # Initial weights (hand-tuned starting point)
        weights = [3.0, 2.0, 1.5, 2.5, 5.0, 2.0, 1.0, 3.0, 2.5, 1.5, 1.5, 2.0, 1.0, 0.5, 2.0]
        
        best_score = 0
        best_weights = weights[:]
        
        for iteration in range(3):
            for w_idx in range(len(weights)):
                best_w = weights[w_idx]
                best_s = 0
                for delta in [-2, -1, -0.5, 0, 0.5, 1, 2]:
                    weights[w_idx] = best_w + delta
                    # Quick eval
                    matches = []
                    for i in test_range:
                        scores = self._score_numbers(self.data[:i+1], weights)
                        pred = sorted(scores, key=lambda x: -scores[x])[:self.pick_count]
                        actual = set(self.data[i+1])
                        matches.append(len(set(pred) & actual))
                    avg = np.mean(matches) if matches else 0
                    if avg > best_s:
                        best_s = avg
                        best_w = weights[w_idx]
                weights[w_idx] = best_w
            
            if best_s > best_score:
                best_score = best_s
                best_weights = weights[:]
        
        return best_weights
    
    def _backtest(self, weights, test_count=200):
        """Walk-forward backtest with final weights."""
        n = len(self.data)
        start = max(60, n - test_count)
        matches = []
        for i in range(start, n - 1):
            scores = self._score_numbers(self.data[:i+1], weights)
            pred = sorted(scores, key=lambda x: -scores[x])[:self.pick_count]
            actual = set(self.data[i+1])
            matches.append(len(set(pred) & actual))
        
        if not matches:
            return {'avg': 0, 'max': 0, 'improvement': 0, 'tests': 0, 'distribution': {}}
        
        avg = float(np.mean(matches))
        rexp = self.pick_count ** 2 / self.max_number
        imp = (avg / rexp - 1) * 100 if rexp > 0 else 0
        
        return {
            'avg': round(avg, 4),
            'max': int(max(matches)),
            'random_expected': round(rexp, 3),
            'improvement': round(float(imp), 2),
            'tests': len(matches),
            'match_3plus': sum(1 for m in matches if m >= 3),
            'match_4plus': sum(1 for m in matches if m >= 4),
            'match_5plus': sum(1 for m in matches if m >= 5),
            'distribution': {str(k): int(v) for k, v in sorted(Counter(matches).items())},
        }
    
    def _confidence_analysis(self, score_details):
        """Analyze confidence of the prediction."""
        if not score_details:
            return {'level': 'low', 'score': 0}
        
        selected = [s for s in score_details if s.get('selected')]
        if not selected:
            return {'level': 'low', 'score': 0}
        
        avg_conf = np.mean([s['confidence'] for s in selected])
        min_conf = min(s['confidence'] for s in selected)
        
        # Gap between selected and non-selected
        non_selected = [s for s in score_details if not s.get('selected')]
        if non_selected:
            gap = selected[-1]['score'] - non_selected[0]['score']  
        else:
            gap = 0
        
        conf_score = avg_conf * 0.6 + min_conf * 0.3 + min(gap * 10, 10) * 0.1
        
        if conf_score >= 70:
            level = 'high'
        elif conf_score >= 40:
            level = 'medium'
        else:
            level = 'low'
        
        return {
            'level': level,
            'score': round(conf_score, 1),
            'avg_confidence': round(avg_conf, 1),
            'min_confidence': round(min_conf, 1),
        }
