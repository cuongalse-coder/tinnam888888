"""
Master Predictor V15.0 — Ultimate Ensemble with 25 Signals
=============================================================
V14: 20 signals + Column-Pool + Spectral Lag-10 + Regime-Adaptive
V15: 25 signals + Multi-Model Ensemble Voting + Enhanced Optimizer
     + Triplet Network + Zipf + Fibonacci Gap + Mirror Symmetry + Streak Surge
     + Population-based optimizer (3 restarts, 10 iterations)
     + Top-5 strategy ensemble voting for final selection
     + Adaptive anti-repeat strength

Walk-forward backtest tự động → chọn trọng số tối ưu.
"""
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
import math
import warnings
warnings.filterwarnings('ignore')


class MasterPredictor:
    """V15.0: 25 signals + Multi-Model Ensemble → maximum accuracy."""
    
    VERSION = "V15.0"
    NUM_SIGNALS = 25
    
    def __init__(self, max_number, pick_count):
        self.max_number = max_number
        self.pick_count = pick_count
    
    def predict(self, data):
        """Return prediction result with backtest stats."""
        self.data = [d[:self.pick_count] for d in data]
        self.flat = [n for d in self.data for n in d]
        n = len(self.data)
        
        print(f"[Master V15] Analyzing {n} draws with {self.NUM_SIGNALS} signals + Ensemble Voting...")
        
        # Step 1: Pre-compute column pool candidates
        self._column_pool = self._column_pool_candidates(self.data)
        print(f"  Column-Pool: {sum(len(v) for v in self._column_pool)} candidates across {self.pick_count} positions")
        
        # Step 2: Auto-tune weights via ENHANCED optimizer (population-based)
        best_weights = self._optimize_weights_v15()
        print(f"  Weights optimized ({len(best_weights)} signals, population-based)")
        
        # Step 3: Generate base prediction with optimized weights
        numbers_base, score_details = self._predict_with_weights(self.data, best_weights)
        
        # Step 4: Multi-Model Ensemble Voting
        numbers_ensemble = self._ensemble_voting(self.data)
        print(f"  Ensemble Vote result: {numbers_ensemble}")
        
        # Step 5: Merge base + ensemble (prefer ensemble if backtest is better)
        bt_base = self._backtest_quick(best_weights, test_count=100)
        bt_ens = self._backtest_ensemble(test_count=100)
        
        if bt_ens['avg'] > bt_base['avg']:
            numbers = numbers_ensemble
            method_note = "Ensemble Voting (5 models)"
            bt = bt_ens
            print(f"  → Using Ensemble (avg={bt_ens['avg']:.4f}) over Base (avg={bt_base['avg']:.4f})")
        else:
            numbers = numbers_base
            method_note = "Signal Optimizer (25 signals)"
            bt = bt_base
            print(f"  → Using Base (avg={bt_base['avg']:.4f}) over Ensemble (avg={bt_ens['avg']:.4f})")
        
        # Step 6: Full backtest with chosen method
        bt_full = self._backtest(best_weights) if method_note.startswith("Signal") else self._backtest_ensemble(test_count=200)
        print(f"  Full Backtest: {bt_full['avg']:.4f}/6 ({bt_full['improvement']:+.1f}%)")
        if bt_full.get('match_3plus', 0) > 0:
            print(f"  >=3 match: {bt_full['match_3plus']} times ({bt_full['hit_rate_3plus_pct']:.1f}%)")
        
        # Step 7: Confidence analysis
        confidence = self._confidence_analysis(score_details)
        
        print(f"[Master V15] Final Prediction: {numbers}")
        
        return {
            'numbers': numbers,
            'score_details': score_details[:15],
            'backtest': bt_full,
            'confidence': confidence,
            'version': self.VERSION,
            'method': f'Master AI V15 ({n} draws, {self.NUM_SIGNALS} signals, {method_note}, {bt_full["tests"]} backtested)',
            'ensemble_info': {
                'base_avg': round(bt_base['avg'], 4),
                'ensemble_avg': round(bt_ens['avg'], 4),
                'chosen': method_note,
            }
        }
    
    # ==========================================
    # COLUMN POOL (inherited from V14)
    # ==========================================
    def _column_pool_candidates(self, history):
        """Column-Pool: per-position block prediction + hot number filtering."""
        n = len(history)
        if n < 30:
            return [set(range(1, self.max_number + 1))] * self.pick_count
        
        pos_data = [[] for _ in range(self.pick_count)]
        for d in history:
            sd = sorted(d[:self.pick_count])
            for p in range(self.pick_count):
                if p < len(sd):
                    pos_data[p].append(sd[p])
        
        if self.max_number <= 45:
            blocks = {'A': (1, 9), 'B': (10, 19), 'C': (20, 29), 'D': (30, 39), 'E': (40, 45)}
        else:
            blocks = {'A': (1, 9), 'B': (10, 19), 'C': (20, 29), 'D': (30, 39), 'E': (40, 49), 'F': (50, 55)}
        
        def to_block(n_val):
            for bname, (lo, hi) in blocks.items():
                if lo <= n_val <= hi:
                    return bname
            return list(blocks.keys())[-1]
        
        candidates = []
        for pos in range(self.pick_count):
            h = pos_data[pos]
            if len(h) < 3:
                candidates.append(set(range(1, self.max_number + 1)))
                continue
            
            bseq = [to_block(v) for v in h]
            pred_blocks = None
            if len(bseq) >= 3:
                p3 = (bseq[-3], bseq[-2], bseq[-1])
                p3n = Counter()
                for i in range(len(bseq) - 3):
                    if (bseq[i], bseq[i+1], bseq[i+2]) == p3:
                        p3n[bseq[i+3]] += 1
                if sum(p3n.values()) >= 3:
                    pred_blocks = [b for b, _ in p3n.most_common(2)]
            if not pred_blocks:
                bc = Counter(bseq[-30:])
                pred_blocks = [b for b, _ in bc.most_common(3)]
            
            freq = Counter(h[-50:])
            valid = set()
            for b in pred_blocks:
                blo, bhi = blocks[b]
                for num in range(blo, bhi + 1):
                    if freq.get(num, 0) > 0:
                        valid.add(num)
            
            ranked = sorted(valid, key=lambda x: -freq.get(x, 0))
            total_pct = 0
            hot = set()
            nh = len(h[-50:])
            for num in ranked:
                hot.add(num)
                total_pct += freq[num] / nh * 100
                if total_pct >= 70 or len(hot) >= 10:
                    break
            
            candidates.append(hot if hot else set(range(1, self.max_number + 1)))
        
        return candidates
    
    # ==========================================
    # V14 SIGNALS (S0-S19) + V15 NEW (S20-S24)
    # ==========================================
    def _spectral_lag_signal(self, history, num):
        """Spectral Lag-10: detect periodicity via autocorrelation."""
        n = len(history)
        if n < 25:
            return 0.0
        
        window = min(100, n)
        seq = [1.0 if num in history[n - window + i] else 0.0 for i in range(window)]
        
        if len(seq) < 20:
            return 0.0
        
        mean = np.mean(seq)
        var = np.var(seq)
        if var < 1e-10:
            return 0.0
        
        lag = 10
        if len(seq) <= lag:
            return 0.0
        
        autocorr = np.mean([(seq[i] - mean) * (seq[i + lag] - mean) 
                           for i in range(len(seq) - lag)]) / var
        
        appeared_at_lag = 1.0 if num in history[-lag] else 0.0
        score = max(0, autocorr) * appeared_at_lag
        
        return float(min(score, 1.0))
    
    def _detect_regime(self, history, num, window=20):
        """Regime detection: hot/cold/neutral."""
        n_draws = len(history)
        if n_draws < window:
            return 0.0
        
        freq_recent = sum(1 for d in history[-window:] if num in d) / window
        freq_old = sum(1 for d in history[-window*2:-window] if num in d) / window if n_draws > window*2 else freq_recent
        
        expected = self.pick_count / self.max_number
        trend = freq_recent - freq_old
        level = freq_recent - expected
        
        if trend > 0.03 and level > 0:
            return trend * 5 + level * 3
        elif trend < -0.03 and level < 0:
            return -0.3
        else:
            return 0.0
    
    def _score_numbers(self, history, weights):
        """Score all numbers using 25 weighted signals. Returns scores dict."""
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
        
        # Column pool membership (S15)
        col_pool_flat = set()
        if hasattr(self, '_column_pool'):
            for cset in self._column_pool:
                col_pool_flat.update(cset)
        
        # Cross-scale data (S19)
        scale_appearances = {}
        scales = [5, 10, 20, 50, 100]
        for num in range(1, self.max_number + 1):
            count = 0
            for scale in scales:
                if n_draws >= scale:
                    if any(num in d for d in history[-scale:]):
                        count += 1
                elif any(num in d for d in history):
                    count += 1
            scale_appearances[num] = count
        
        # ===== V15 Pre-computations =====
        
        # Triplet network (S20)
        triplet_scores = Counter()
        for d in history[-50:]:
            sd = sorted(d)
            for trip in combinations(sd, 3):
                triplet_scores[trip] += 1
        
        # Pre-compute triplet bonus per number
        triplet_bonus = Counter()
        last_sorted = sorted(last)
        for trip, c in triplet_scores.most_common(200):
            overlap_with_last = len(set(trip) & last)
            if overlap_with_last >= 2:
                for n in trip:
                    if n not in last:
                        triplet_bonus[n] += c
        
        # Zipf rank (S21) - rank by total frequency, compare to Zipf expected
        freq_ranked = total_freq.most_common()
        zipf_rank = {}
        for rank_idx, (num, _) in enumerate(freq_ranked):
            zipf_rank[num] = rank_idx + 1
        
        # Fibonacci gaps (S22)
        fib_set = {1, 2, 3, 5, 8, 13, 21, 34, 55}
        
        # Mirror pairs (S23)
        # Mirror: n ↔ max_number - n + 1
        
        # Streak detection (S24)
        streak_nums = set()
        if n_draws >= 3:
            for num in range(1, self.max_number + 1):
                appeared_last_2 = (num in history[-1]) and (num in history[-2])
                appeared_last_3 = appeared_last_2 and n_draws >= 3 and (num in history[-3])
                if appeared_last_2 or appeared_last_3:
                    streak_nums.add(num)
        
        # Adaptive anti-repeat strength
        # Check how often last-draw numbers repeat in history
        repeat_rate = 0
        if n_draws >= 20:
            repeats = 0
            for i in range(max(0, n_draws - 20), n_draws - 1):
                repeats += len(set(history[i]) & set(history[i+1]))
            repeat_rate = repeats / (20 * self.pick_count)
        anti_repeat_strength = 1.0 - min(repeat_rate * 5, 0.5)  # Range 0.5 to 1.0
        
        for num in range(1, self.max_number + 1):
            s = [0.0] * self.NUM_SIGNALS
            
            # === Original 15 signals (S0-S14) ===
            
            # S0: Freq last 10 (hot)
            s[0] = freq_10.get(num, 0) / 10
            
            # S1: Freq last 30
            s[1] = freq_30.get(num, 0) / 30
            
            # S2: Freq last 50
            s[2] = freq_50.get(num, 0) / 50
            
            # S3: Gap overdue
            gap = n_draws - last_seen.get(num, 0)
            s[3] = max(0, gap / exp_gap - 0.8)
            
            # S4: Anti-repeat (ADAPTIVE strength)
            s[4] = -anti_repeat_strength if num in last else 0.0
            
            # S5: Momentum
            s[5] = (r10.get(num, 0) - p10.get(num, 0)) / 5
            
            # S6: Position frequency
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
            
            # S9: Frequency correction
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
            
            # S12: Regime trend
            f_r = sum(1 for d in history[-15:] if num in d) / 15
            f_o = sum(1 for d in history[-45:-15] if num in d) / 30 if n_draws > 45 else f_r
            trend = f_r - f_o
            s[12] = max(0, trend) * 10
            
            # S13: Sum balance target
            avg_sum = np.mean([sum(d) for d in history[-20:]])
            target = avg_sum / self.pick_count
            s[13] = max(0, 1 - abs(num - target) / self.max_number)
            
            # S14: Anti-repeat double
            s[14] = -0.5 if n_draws > 1 and num in history[-2] and num in last else 0
            
            # === V14 signals (S15-S19) ===
            
            # S15: Column-Pool membership
            if col_pool_flat:
                s[15] = 1.0 if num in col_pool_flat else -0.3
            else:
                s[15] = 0.0
            
            # S16: Spectral Lag-10
            s[16] = self._spectral_lag_signal(history, num)
            
            # S17: Regime-Adaptive
            s[17] = self._detect_regime(history, num)
            
            # S18: Enhanced Run-Length (sigmoid)
            if curr_absence > 0:
                seq = [1 if num in d else 0 for d in history]
                absence_runs = []
                run = 0
                for sv in seq:
                    if sv == 0:
                        run += 1
                    else:
                        if run > 0:
                            absence_runs.append(run)
                        run = 0
                avg_absence = np.mean(absence_runs) if absence_runs else exp_gap
                ratio = curr_absence / avg_absence if avg_absence > 0 else 0
                s[18] = 1 / (1 + math.exp(-3 * (ratio - 0.8)))
            else:
                s[18] = 0.0
            
            # S19: Cross-Scale Agreement
            appearances_count = scale_appearances.get(num, 0)
            s[19] = max(0, (appearances_count - 3)) * 0.5
            
            # === NEW V15 signals (S20-S24) ===
            
            # S20: Triplet Network — bonus for numbers in strong triplets with last draw
            s[20] = triplet_bonus.get(num, 0) / max(1, max(triplet_bonus.values()) if triplet_bonus else 1)
            
            # S21: Zipf Score — numbers at optimal Zipf rank get boost
            rank = zipf_rank.get(num, self.max_number)
            # Zipf expected frequency: 1/rank normalized
            expected_zipf_rank = self.max_number / 2  # Middle rank is neutral
            s[21] = max(0, 1 - rank / self.max_number) * 0.5  # Top-ranked numbers get more
            
            # S22: Fibonacci Gap — if current gap is a Fibonacci number, turning point
            if gap in fib_set:
                s[22] = 0.8
            elif gap + 1 in fib_set or gap - 1 in fib_set:
                s[22] = 0.3
            else:
                s[22] = 0.0
            
            # S23: Mirror Symmetry — balance with mirror number
            mirror = self.max_number - num + 1
            mirror_freq = freq_30.get(mirror, 0)
            num_freq = freq_30.get(num, 0)
            if mirror_freq > num_freq * 1.5:
                s[23] = 0.5  # Mirror appeared more, this num is "due"
            elif num_freq > mirror_freq * 1.5:
                s[23] = -0.3  # This num appeared more, mirror is "due"
            else:
                s[23] = 0.0
            
            # S24: Streak Surge — detect numbers on 2-3 draw streaks
            if num in streak_nums:
                # Currently on a streak — might continue or end
                # Use historical streak continuation rate
                continues = 0
                total_streaks = 0
                for i in range(2, n_draws):
                    if num in history[i-1] and num in history[i-2]:
                        total_streaks += 1
                        if num in history[i]:
                            continues += 1
                if total_streaks > 0:
                    s[24] = (continues / total_streaks - 0.5) * 2  # Positive if >50% continue
                else:
                    s[24] = -0.3  # Default: streaks tend to end
            else:
                s[24] = 0.0
            
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
    
    # ==========================================
    # V15 ENHANCED OPTIMIZER (Population-based)
    # ==========================================
    def _optimize_weights_v15(self):
        """Population-based optimizer: 3 random restarts × 10 iterations."""
        n = len(self.data)
        train_end = min(n - 1, n - 50)
        test_range = range(max(60, train_end - 120), train_end)  # Larger window (120)
        
        # Base weights
        base_weights = [
            3.0, 2.0, 1.5, 2.5, 5.0,   # S0-S4
            2.0, 1.0, 3.0, 2.5, 1.5,    # S5-S9
            1.5, 2.0, 1.0, 0.5, 2.0,    # S10-S14
            2.0, 3.0, 1.5, 2.0, 1.0,    # S15-S19
            2.5, 1.0, 1.5, 0.8, 1.2,    # S20-S24
        ]
        
        all_best = []
        
        # 3 random restarts
        for restart in range(3):
            if restart == 0:
                weights = base_weights[:]
            else:
                # Random perturbation of base weights
                weights = [w + np.random.normal(0, 0.5) for w in base_weights]
            
            best_score = 0
            best_weights = weights[:]
            
            # Finer deltas
            deltas = [-3, -2, -1, -0.5, -0.25, -0.15, -0.1, 0, 0.1, 0.15, 0.25, 0.5, 1, 2, 3]
            
            for iteration in range(10):
                for w_idx in range(len(weights)):
                    best_w = weights[w_idx]
                    best_s = 0
                    for delta in deltas:
                        weights[w_idx] = best_w + delta
                        matches = []
                        for i in test_range:
                            if i + 1 >= len(self.data):
                                continue
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
            
            all_best.append((best_score, best_weights[:]))
        
        # Return best across all restarts
        all_best.sort(key=lambda x: -x[0])
        return all_best[0][1]
    
    # ==========================================
    # MULTI-MODEL ENSEMBLE VOTING
    # ==========================================
    def _ensemble_voting(self, history):
        """Run top 5 diverse strategies, weighted vote → final 6 numbers."""
        n_draws = len(history)
        flat = [n for d in history for n in d]
        last = set(history[-1])
        
        votes = Counter()
        
        # --- Strategy 1: Weighted Frequency + Gap ---
        def strat_freq_gap():
            scores = Counter()
            for j, d in enumerate(history[-50:]):
                w = 1 + j / 50
                for n in d:
                    scores[n] += w * 0.3
            # Gap overdue
            last_seen = {}
            for i, d in enumerate(history):
                for n in d:
                    last_seen[n] = i
            for n in range(1, self.max_number + 1):
                gap = n_draws - last_seen.get(n, 0)
                exp = self.max_number / self.pick_count
                if gap > exp * 1.2:
                    scores[n] += (gap / exp) * 1.8
            for n in last:
                scores[n] -= 8
            return [n for n, _ in scores.most_common(self.pick_count)]
        
        # --- Strategy 2: KNN + Pair Network ---
        def strat_knn_pair():
            scores = Counter()
            for i in range(len(history) - 2):
                overlap = len(set(history[i]) & last)
                if overlap >= 2:
                    for n in history[i+1]:
                        scores[n] += overlap ** 1.5
            pair_sc = Counter()
            for d in history[-60:]:
                for pair in combinations(sorted(d), 2):
                    pair_sc[pair] += 1
            for n in last:
                for pair, c in pair_sc.most_common(120):
                    if n in pair:
                        partner = pair[0] if pair[1] == n else pair[1]
                        if partner not in last:
                            scores[partner] += c * 0.15
            for n in last:
                scores[n] -= 8
            return [n for n, _ in scores.most_common(self.pick_count)]
        
        # --- Strategy 3: Momentum + Regime ---
        def strat_momentum_regime():
            scores = {}
            for num in range(1, self.max_number + 1):
                f5 = sum(1 for d in history[-5:] if num in d) / 5
                f15 = sum(1 for d in history[-15:] if num in d) / 15
                f30 = sum(1 for d in history[-30:] if num in d) / 30
                v1 = f5 - f15
                v2 = f15 - f30
                accel = v1 - v2
                # Regime
                f_r = f15
                f_o = sum(1 for d in history[-45:-15] if num in d) / 30 if n_draws > 45 else f_r
                trend = f_r - f_o
                scores[num] = f5 * 4 + v1 * 8 + accel * 4 + max(0, trend) * 15
                if num in last:
                    scores[num] -= 3
            return sorted(scores, key=lambda x: -scores[x])[:self.pick_count]
        
        # --- Strategy 4: Run-Length Turning Point ---
        def strat_run_length():
            scores = {}
            for num in range(1, self.max_number + 1):
                seq = [1 if num in d else 0 for d in history]
                absence_runs = []
                run = 0
                for sv in seq:
                    if sv == 0:
                        run += 1
                    else:
                        if run > 0:
                            absence_runs.append(run)
                        run = 0
                curr_abs = 0
                for sv in reversed(seq):
                    if sv == 0:
                        curr_abs += 1
                    else:
                        break
                avg_abs = np.mean(absence_runs) if absence_runs else self.max_number / self.pick_count
                if avg_abs > 0 and curr_abs > 0:
                    ratio = curr_abs / avg_abs
                    turn_prob = 1 / (1 + math.exp(-3 * (ratio - 0.8)))
                    scores[num] = turn_prob * 5
                else:
                    scores[num] = 0
                if num in last:
                    scores[num] -= 3
            return sorted(scores, key=lambda x: -scores[x])[:self.pick_count]
        
        # --- Strategy 5: Multi-Scale Fusion ---
        def strat_multi_scale():
            scores = Counter()
            for scale, w in [(5, 3), (10, 2.5), (20, 2), (50, 1.5), (100, 1)]:
                window = history[-scale:] if len(history) >= scale else history
                freq = Counter(n for d in window for n in d)
                total = max(1, sum(freq.values()))
                for n, c in freq.items():
                    scores[n] += (c / total) * w * 8
            for num in range(1, self.max_number + 1):
                appear_in_scales = sum(1 for scale in [5, 10, 20, 50, 100]
                                        if any(num in d for d in history[-min(scale, len(history)):]))
                if appear_in_scales >= 4:
                    scores[num] += 2
            for n in last:
                scores[n] -= 6
            return [n for n, _ in scores.most_common(self.pick_count)]
        
        # Run all strategies and collect weighted votes
        strategies = [
            (strat_freq_gap, 3.0),         # Weight 3
            (strat_knn_pair, 2.5),         # Weight 2.5
            (strat_momentum_regime, 2.0),   # Weight 2
            (strat_run_length, 2.0),        # Weight 2
            (strat_multi_scale, 2.5),       # Weight 2.5
        ]
        
        for strat_fn, weight in strategies:
            try:
                pred = strat_fn()
                for n in pred:
                    votes[n] += weight
            except Exception:
                pass
        
        # Select top pick_count by vote score
        return sorted([n for n, _ in votes.most_common(self.pick_count)])
    
    # ==========================================
    # BACKTEST METHODS
    # ==========================================
    def _backtest_quick(self, weights, test_count=100):
        """Quick backtest with signal weights."""
        n = len(self.data)
        start = max(60, n - test_count)
        matches = []
        for i in range(start, n - 1):
            scores = self._score_numbers(self.data[:i+1], weights)
            pred = sorted(scores, key=lambda x: -scores[x])[:self.pick_count]
            actual = set(self.data[i+1])
            matches.append(len(set(pred) & actual))
        
        if not matches:
            return {'avg': 0, 'max': 0, 'improvement': 0, 'tests': 0}
        
        avg = float(np.mean(matches))
        rexp = self.pick_count ** 2 / self.max_number
        imp = (avg / rexp - 1) * 100 if rexp > 0 else 0
        return {'avg': round(avg, 4), 'max': int(max(matches)), 'improvement': round(float(imp), 2), 'tests': len(matches)}
    
    def _backtest_ensemble(self, test_count=200):
        """Walk-forward backtest for ensemble voting method."""
        n = len(self.data)
        start = max(60, n - test_count)
        matches = []
        for i in range(start, n - 1):
            pred = self._ensemble_voting(self.data[:i+1])
            actual = set(self.data[i+1])
            matches.append(len(set(pred) & actual))
        
        if not matches:
            return {'avg': 0, 'max': 0, 'improvement': 0, 'tests': 0, 'distribution': {},
                    'match_3plus': 0, 'match_4plus': 0, 'match_5plus': 0, 'match_6': 0,
                    'hit_rate_3plus_pct': 0, 'random_expected': 0, 'avg_last_50': 0}
        
        avg = float(np.mean(matches))
        rexp = self.pick_count ** 2 / self.max_number
        imp = (avg / rexp - 1) * 100 if rexp > 0 else 0
        m3plus = sum(1 for m in matches if m >= 3)
        total_tests = len(matches)
        
        return {
            'avg': round(avg, 4),
            'max': int(max(matches)),
            'random_expected': round(rexp, 3),
            'improvement': round(float(imp), 2),
            'tests': total_tests,
            'match_3plus': m3plus,
            'match_4plus': sum(1 for m in matches if m >= 4),
            'match_5plus': sum(1 for m in matches if m >= 5),
            'match_6': sum(1 for m in matches if m >= 6),
            'hit_rate_3plus_pct': round(m3plus / total_tests * 100, 2) if total_tests > 0 else 0,
            'avg_last_50': round(float(np.mean(matches[-50:])), 4) if len(matches) >= 50 else round(avg, 4),
            'distribution': {str(k): int(v) for k, v in sorted(Counter(matches).items())},
        }
    
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
        
        m3plus = sum(1 for m in matches if m >= 3)
        total_tests = len(matches)
        
        result = {
            'avg': round(avg, 4),
            'max': int(max(matches)),
            'random_expected': round(rexp, 3),
            'improvement': round(float(imp), 2),
            'tests': total_tests,
            'match_3plus': m3plus,
            'match_4plus': sum(1 for m in matches if m >= 4),
            'match_5plus': sum(1 for m in matches if m >= 5),
            'match_6': sum(1 for m in matches if m >= 6),
            'hit_rate_3plus_pct': round(m3plus / total_tests * 100, 2) if total_tests > 0 else 0,
            'avg_last_50': round(float(np.mean(matches[-50:])), 4) if len(matches) >= 50 else round(avg, 4),
            'distribution': {str(k): int(v) for k, v in sorted(Counter(matches).items())},
        }
        
        return result
    
    def _confidence_analysis(self, score_details):
        """Analyze confidence of the prediction."""
        if not score_details:
            return {'level': 'low', 'score': 0}
        
        selected = [s for s in score_details if s.get('selected')]
        if not selected:
            return {'level': 'low', 'score': 0}
        
        avg_conf = np.mean([s['confidence'] for s in selected])
        min_conf = min(s['confidence'] for s in selected)
        
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
