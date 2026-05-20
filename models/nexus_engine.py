"""
NEXUS ENGINE V700.0 — QUANTUM SUPREME
=======================================
Wraps MegaExploitV15 + StackingEngine V700 (5-model stacking ensemble).
V700 upgrades:
- 12 high-precision signals (was 8)
- Adaptive rolling calibration with exponential weighting
- 5-model stacking with 28 features, BayesianRidge meta-learner
- KNN Fractal V3, Pair-Triplet Hybrid, Regime Detector, Lag Correlation
"""
import math
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
try:
    from sklearn.ensemble import HistGradientBoostingRegressor
except ImportError:
    HistGradientBoostingRegressor = None

class NexusEngine:
    def __init__(self, max_number, pick_count):
        self.max_number = max_number
        self.pick_count = pick_count

    def predict(self, data, n_sets=5, use_elastic_filter=True):
        n = len(data)
        if n < 60:
            return {'predictions': [], 'top_pool': [], 'weights': {}, 'confidence': 0, 'absolute_final_6': [], 'constraints': {}, 'sum_mod7': []}

        # --- PHASE 1: Import base V15 signals ---
        try:
            from models.mega_exploit_v15 import MegaExploitV15
            base = MegaExploitV15(self.max_number, self.pick_count)
            base_result = base.predict(data, n_sets=n_sets)
        except Exception:
            base_result = {'predictions': [], 'top_pool': [], 'weights': {}, 'confidence': 0, 'absolute_final_6': [], 'constraints': {}, 'sum_mod7': []}

        # --- PHASE 2: 12 HIGH-PRECISION SIGNALS (V700) ---
        new_sigs = {}
        new_sigs['sliding_window'] = self._sig_sliding_window(data)
        new_sigs['conditional_prob'] = self._sig_conditional_probability(data)
        new_sigs['gap_acceleration'] = self._sig_gap_acceleration(data)
        new_sigs['hot_cold_cross'] = self._sig_hot_cold_intersection(data)
        new_sigs['delta_momentum'] = self._sig_delta_momentum(data)
        new_sigs['sector_rotation'] = self._sig_sector_rotation(data)
        new_sigs['pair_boost'] = self._sig_pair_boost(data)
        new_sigs['temporal_decay'] = self._sig_temporal_decay(data)
        # V700 NEW signals
        new_sigs['knn_fractal_v3'] = self._sig_knn_fractal_v3(data)
        new_sigs['pair_triplet_hybrid'] = self._sig_pair_triplet_hybrid(data)
        new_sigs['regime_detector'] = self._sig_regime_detector(data)
        new_sigs['lag_correlation'] = self._sig_lag_correlation(data)

        # --- PHASE 3: Calibrate new signals via rolling backtest ---
        new_weights = self._calibrate_rolling(data, new_sigs)

        # --- PHASE 4: Merge with base scores ---
        base_scores = base_result.get('scores', {})
        # Convert base scores dict to full range
        scores = {num: 0.0 for num in range(1, self.max_number + 1)}
        for num, s in base_scores.items():
            scores[int(num)] = float(s)

        # Add new signals with calibrated weights
        for sig_name, sig_scores in new_sigs.items():
            w = new_weights.get(sig_name, 1.0)
            if not sig_scores:
                continue
            vals = list(sig_scores.values())
            max_s = max(abs(v) for v in vals) if vals else 1
            if max_s < 0.001:
                continue
            for num, score in sig_scores.items():
                scores[num] += (score / max_s) * w * 2.0  # V700: Boosted factor for calibrated signals

        # --- PHASE 5: Re-rank and build final output ---
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        top_pool = [n for n, s in ranked if s > 0][:30]
        if len(top_pool) < self.pick_count:
            top_pool = [n for n, _ in ranked[:30]]

        # ALWAYS generate new constraints to respect use_elastic_filter setting
        constraints = self._learn_constraints(data, use_elastic_filter)
        sum_mod7 = base_result.get('sum_mod7', list(range(7)))
        if isinstance(sum_mod7, list):
            sum_mod7 = set(sum_mod7)

        predictions = []
        used = set()

        # Best combo via simulated annealing
        best = self._best_combo(top_pool[:20], scores, constraints, sum_mod7)
        if best:
            predictions.append({'numbers': best, 'strategy': '🧬 NEXUS V700 — Quantum Supreme'})
            used.add(tuple(best))

        # Diverse combos
        attempts = 0
        while len(predictions) < n_sets and attempts < n_sets * 500:
            attempts += 1
            combo = self._random_combo(top_pool[:25], scores, constraints, sum_mod7)
            if not combo:
                continue
            t = tuple(combo)
            if t in used:
                continue
            if all(len(set(combo) - set(p['numbers'])) >= 2 for p in predictions):
                used.add(t)
                predictions.append({'numbers': combo, 'strategy': '🧬 V700 Diversified'})

        predictions.sort(key=lambda x: -sum(scores[n] for n in x['numbers']))

        # V600: Use StackingEngine for Absolute Final 6 (no data leakage)
        try:
            from models.stacking_engine import StackingEngine
            stacker = StackingEngine(self.max_number, self.pick_count)
            stack_result = stacker.predict_top_pool(data, pool_size=20)
            absolute_final_6 = stack_result['top6']
            
            # Boost top_pool with stacking scores
            stack_scores = stack_result.get('scores', {})
            for num, ss in stack_scores.items():
                if ss > 0:
                    scores[num] = scores.get(num, 0) + ss * 12.0  # V700: Higher stacking boost
            
            # Re-rank after stacking boost
            ranked = sorted(scores.items(), key=lambda x: -x[1])
            top_pool = [n for n, s in ranked if s > 0][:30]
            if len(top_pool) < self.pick_count:
                top_pool = [n for n, _ in ranked[:30]]
        except Exception as e:
            print(f"V600 Stacking Fallback: {e}")
            absolute_final_6 = sorted([n for n, _ in ranked[:6]])

        # Merge weights
        all_weights = dict(base_result.get('weights', {}))
        for k, v in new_weights.items():
            all_weights[f"NEXUS_{k}"] = round(v, 3)

        confidence = min(base_result.get('confidence', 50) + len([w for w in new_weights.values() if w > 1.0]) * 3, 98)

        return {
            'predictions': predictions,
            'strategy': 'NexusEngine_V700_QUANTUM_SUPREME',
            'confidence': round(confidence, 1),
            'weights': {k: round(v, 3) if isinstance(v, float) else v for k, v in sorted(all_weights.items(), key=lambda x: -float(x[1]))},
            'scores': {n: round(s, 3) for n, s in ranked[:30]},
            'top_pool': top_pool[:25],
            'n_signals': 35 + len(new_sigs),  # V700: 47 signals total
            'constraints': constraints if isinstance(constraints, dict) else {},
            'sum_mod7': list(sum_mod7) if sum_mod7 else [],
            'absolute_final_6': absolute_final_6,
        }

    # ================================================================
    # V500 DEEP LEARNING (ABSOLUTE FINAL 6)
    # ================================================================

    # V500 _predict_absolute_final_6_v500 REMOVED — replaced by StackingEngine in predict()

    # ================================================================
    # 6 NEW HIGH-PRECISION SIGNALS
    # ================================================================

    def _sig_sliding_window(self, data):
        """Multi-scale sliding window frequency with exponential decay."""
        scores = {n: 0.0 for n in range(1, self.max_number + 1)}
        windows = [5, 10, 20, 40, 80]
        window_weights = [5.0, 3.0, 2.0, 1.0, 0.5]
        expected = self.pick_count / self.max_number

        for w_size, w_weight in zip(windows, window_weights):
            if len(data) < w_size:
                continue
            recent = data[-w_size:]
            freq = Counter(n for d in recent for n in d[:self.pick_count])
            for num in range(1, self.max_number + 1):
                observed = freq.get(num, 0) / w_size
                deviation = (observed - expected) / (expected + 0.001)
                scores[num] += deviation * w_weight
        return scores

    def _sig_conditional_probability(self, data):
        """P(num appears | specific numbers appeared in last draw)."""
        scores = {n: 0.0 for n in range(1, self.max_number + 1)}
        if len(data) < 30:
            return scores
        last = set(data[-1][:self.pick_count])

        # For each number, compute conditional probability given each number in last draw
        cond_counts = defaultdict(lambda: defaultdict(int))
        total_given = defaultdict(int)

        for i in range(len(data) - 1):
            for given_num in data[i][:self.pick_count]:
                total_given[given_num] += 1
                for next_num in data[i + 1][:self.pick_count]:
                    cond_counts[given_num][next_num] += 1

        for num in range(1, self.max_number + 1):
            prob_sum = 0.0
            for given in last:
                if total_given[given] > 0:
                    prob_sum += cond_counts[given].get(num, 0) / total_given[given]
            scores[num] = prob_sum * 3.0
        return scores

    def _sig_gap_acceleration(self, data):
        """Detect if gap is accelerating (number due sooner) or decelerating."""
        scores = {n: 0.0 for n in range(1, self.max_number + 1)}
        n = len(data)

        for num in range(1, self.max_number + 1):
            appearances = [i for i, d in enumerate(data) if num in d[:self.pick_count]]
            if len(appearances) < 4:
                continue
            gaps = [appearances[j + 1] - appearances[j] for j in range(len(appearances) - 1)]
            if len(gaps) < 3:
                continue
            # Gap acceleration = trend in gap changes
            recent_gaps = gaps[-5:]
            if len(recent_gaps) >= 2:
                diffs = [recent_gaps[i] - recent_gaps[i - 1] for i in range(1, len(recent_gaps))]
                avg_accel = sum(diffs) / len(diffs)
                current_gap = n - appearances[-1]
                mean_gap = sum(gaps) / len(gaps)
                overdue_ratio = current_gap / (mean_gap + 0.1)
                # Negative acceleration (gaps shrinking) + overdue = very likely to appear
                if avg_accel < 0 and overdue_ratio > 0.8:
                    scores[num] = abs(avg_accel) * overdue_ratio * 2.0
                elif overdue_ratio > 1.5:
                    scores[num] = overdue_ratio * 1.5
        return scores

    def _sig_hot_cold_intersection(self, data):
        """Find numbers that are hot in short-term AND cold in mid-term (breakout candidates)."""
        scores = {n: 0.0 for n in range(1, self.max_number + 1)}
        if len(data) < 50:
            return scores

        short_freq = Counter(n for d in data[-10:] for n in d[:self.pick_count])
        mid_freq = Counter(n for d in data[-30:-10] for n in d[:self.pick_count])
        long_freq = Counter(n for d in data[-80:] for n in d[:self.pick_count])

        for num in range(1, self.max_number + 1):
            s = short_freq.get(num, 0) / 10
            m = mid_freq.get(num, 0) / 20
            l = long_freq.get(num, 0) / 80
            expected = self.pick_count / self.max_number

            # Breakout: hot recently, cold before, historically average
            if s > expected * 1.3 and m < expected * 0.7:
                scores[num] = (s - m) * 8.0  # Strong breakout signal
            # Revival: cold recently, but historically hot
            elif s < expected * 0.5 and l > expected * 1.2:
                scores[num] = (l - s) * 3.0  # Due for revival
        return scores

    def _sig_delta_momentum(self, data):
        """Rate of change of momentum (2nd derivative of frequency)."""
        scores = {n: 0.0 for n in range(1, self.max_number + 1)}
        if len(data) < 30:
            return scores

        for num in range(1, self.max_number + 1):
            # Compute frequency in 3 consecutive windows
            f1 = sum(1 for d in data[-10:] if num in d[:self.pick_count]) / 10
            f2 = sum(1 for d in data[-20:-10] if num in d[:self.pick_count]) / 10
            f3 = sum(1 for d in data[-30:-20] if num in d[:self.pick_count]) / 10

            v1 = f1 - f2  # Recent velocity
            v2 = f2 - f3  # Previous velocity
            accel = v1 - v2  # Acceleration

            # Positive acceleration = momentum building up
            if accel > 0 and v1 > 0:
                scores[num] = accel * 15.0 + v1 * 5.0
            elif accel > 0:
                scores[num] = accel * 8.0
        return scores

    def _sig_sector_rotation(self, data):
        """Track which number sectors (decades) are rotating in/out of favor."""
        scores = {n: 0.0 for n in range(1, self.max_number + 1)}
        if len(data) < 40:
            return scores

        n_sectors = (self.max_number + 9) // 10
        
        # Sector frequency in recent vs previous period
        recent_sector = [0] * n_sectors
        prev_sector = [0] * n_sectors
        
        for d in data[-15:]:
            for num in d[:self.pick_count]:
                recent_sector[(num - 1) // 10] += 1
        for d in data[-30:-15]:
            for num in d[:self.pick_count]:
                prev_sector[(num - 1) // 10] += 1

        # Sectors gaining momentum
        for num in range(1, self.max_number + 1):
            sector = (num - 1) // 10
            r = recent_sector[sector]
            p = prev_sector[sector]
            if r > p * 1.2:  # Sector heating up
                scores[num] = (r - p) * 0.3
            elif r < p * 0.8:  # Sector cooling down
                scores[num] = -0.5
        return scores

    def _sig_pair_boost(self, data):
        """V400: Pair co-occurrence boost with last draw numbers."""
        scores = {n: 0.0 for n in range(1, self.max_number + 1)}
        last = set(data[-1][:self.pick_count])
        pf = Counter()
        for x in data[-150:]:
            for p in combinations(sorted(x[:self.pick_count]), 2):
                pf[p] += 1
        for n in range(1, self.max_number + 1):
            for p in last:
                key = tuple(sorted([p, n]))
                cnt = pf.get(key, 0)
                if cnt > 3:
                    scores[n] += cnt * 0.08
        return scores

    def _sig_temporal_decay(self, data):
        """V400: Exponential temporal decay weighting."""
        scores = {n: 0.0 for n in range(1, self.max_number + 1)}
        nd = len(data)
        lam = 0.05
        for i, draw in enumerate(data):
            age = nd - 1 - i
            w = math.exp(-lam * age)
            for n in draw[:self.pick_count]:
                scores[n] += w
        mx = max(scores.values()) if scores else 1
        if mx > 0:
            for n in scores:
                scores[n] = (scores[n] / mx) * 4
        return scores

    # ================================================================
    # V700 NEW SIGNALS
    # ================================================================

    def _sig_knn_fractal_v3(self, data):
        """V700: KNN Fractal V3 — 3-draw fingerprint matching with weighted decay."""
        scores = {n: 0.0 for n in range(1, self.max_number + 1)}
        nd = len(data)
        if nd < 20:
            return scores
        # Current fingerprint: union of last 3 draws
        fingerprint = set(data[-1][:self.pick_count]) | set(data[-2][:self.pick_count]) | set(data[-3][:self.pick_count])
        similarities = []
        for i in range(2, nd - 3):
            past_fp = set(data[i][:self.pick_count]) | set(data[i-1][:self.pick_count]) | set(data[i-2][:self.pick_count])
            overlap = len(fingerprint & past_fp)
            recency = 1.0 + 0.5 * (i / nd)
            if overlap >= 4:
                similarities.append((overlap * recency, i + 1))
        similarities.sort(key=lambda x: -x[0])
        for sim_score, next_idx in similarities[:25]:
            if next_idx < nd:
                for num in data[next_idx][:self.pick_count]:
                    scores[num] += sim_score
        return scores

    def _sig_pair_triplet_hybrid(self, data):
        """V700: Combined pair + triplet co-occurrence with recency weighting."""
        scores = {n: 0.0 for n in range(1, self.max_number + 1)}
        nd = len(data)
        if nd < 50:
            return scores
        last = set(data[-1][:self.pick_count])
        # Build pair and triplet counts with recency
        pair_c = Counter()
        trip_c = Counter()
        for idx in range(max(0, nd - 150), nd):
            w = 1.0 + (idx - max(0, nd - 150)) / 150
            draw = sorted(data[idx][:self.pick_count])
            for p in combinations(draw, 2):
                pair_c[p] += w
            for t in combinations(draw, 3):
                trip_c[t] += w * 0.5
        for num in range(1, self.max_number + 1):
            if num in last:
                continue
            for anchor in last:
                key = tuple(sorted([num, anchor]))
                scores[num] += pair_c.get(key, 0) * 0.3
            # Triplet bonus: 2 from last draw + this number
            for a1, a2 in combinations(sorted(last), 2):
                trip_key = tuple(sorted([num, a1, a2]))
                scores[num] += trip_c.get(trip_key, 0) * 0.8
        return scores

    def _sig_regime_detector(self, data):
        """V700: Detect current regime (hot/cold/mixed) and boost accordingly."""
        scores = {n: 0.0 for n in range(1, self.max_number + 1)}
        nd = len(data)
        if nd < 40:
            return scores
        expected = self.pick_count / self.max_number
        # Classify each number into regime based on short vs long term
        for num in range(1, self.max_number + 1):
            f5 = sum(1 for d in data[-5:] if num in d[:self.pick_count]) / 5
            f20 = sum(1 for d in data[-20:] if num in d[:self.pick_count]) / 20
            f50 = sum(1 for d in data[-min(50, nd):] if num in d[:self.pick_count]) / min(50, nd)
            # Hot regime: consistently above expected
            if f5 > expected * 1.3 and f20 > expected * 1.1:
                scores[num] = (f5 + f20) * 4.0  # Ride the hot streak
            # Reversal regime: cold recently but historically hot
            elif f5 < expected * 0.5 and f50 > expected * 1.2:
                scores[num] = (f50 - f5) * 3.0  # Due for bounce
            # Breakout: suddenly hot after being cold
            elif f5 > expected * 1.5 and f20 < expected * 0.8:
                scores[num] = f5 * 5.0  # Strong breakout
        return scores

    def _sig_lag_correlation(self, data):
        """V700: Multi-lag autocorrelation — numbers that follow themselves at lag 2,3,4."""
        scores = {n: 0.0 for n in range(1, self.max_number + 1)}
        nd = len(data)
        if nd < 30:
            return scores
        lags = [2, 3, 4, 5, 7]
        lag_weights = [3.0, 2.5, 2.0, 1.5, 1.0]
        for num in range(1, self.max_number + 1):
            for lag, lw in zip(lags, lag_weights):
                count = 0
                total = 0
                for i in range(lag, nd):
                    if num in data[i - lag][:self.pick_count]:
                        total += 1
                        if num in data[i][:self.pick_count]:
                            count += 1
                if total > 5:
                    ratio = count / total
                    expected = self.pick_count / self.max_number
                    if ratio > expected * 1.2:
                        # Check if this number appeared at the right lag recently
                        for lg in range(1, lag + 1):
                            if nd - lg >= 0 and num in data[nd - lg][:self.pick_count]:
                                scores[num] += (ratio - expected) * lw * 3
                                break
        return scores

    # ================================================================
    # IMPROVED ROLLING CALIBRATION (V700: expanded window)
    # ================================================================

    def _calibrate_rolling(self, data, signals):
        """Rolling window calibration with exponential weighting."""
        n = len(data)
        test_size = min(40, n - 70)
        if test_size < 8:
            return {name: 1.0 for name in signals}

        hits = {name: 0.0 for name in signals}
        total_weight = 0.0

        for idx in range(n - test_size, n):
            if idx < 1:
                continue
            actual = set(data[idx][:self.pick_count])
            recency = math.exp((idx - (n - test_size)) / 8.0)
            total_weight += recency

            for sig_name, sig_scores in signals.items():
                if not sig_scores:
                    continue
                top = set(num for num, _ in sorted(sig_scores.items(), key=lambda x: -x[1])[:self.pick_count])
                match_cnt = len(top & actual)
                hits[sig_name] += match_cnt * recency

        base = self.pick_count * (self.pick_count / self.max_number)
        expected = total_weight * base

        weights = {}
        for name in signals:
            if expected > 0 and hits[name] > 0:
                weights[name] = max(hits[name] / expected, 0.1)
            else:
                weights[name] = 0.1
        return weights

    # ================================================================
    # COMBO GENERATION
    # ================================================================

    def _learn_constraints(self, data, use_elastic_filter=True):
        recent = data[-50:]
        sums = [sum(d[:self.pick_count]) for d in recent]
        odds = [sum(1 for x in d[:self.pick_count] if x % 2 == 1) for d in recent]
        mid = self.max_number // 2
        highs = [sum(1 for x in d[:self.pick_count] if x > mid) for d in recent]
        ranges = [max(d[:self.pick_count]) - min(d[:self.pick_count]) for d in recent]
        
        # --- GLOBAL DELTA SYSTEM ---
        deltas = []
        for d in data[-200:]:
            md = d[0]
            for i in range(1, self.pick_count):
                if d[i] - d[i-1] > md: md = d[i] - d[i-1]
            deltas.append(md)
        delta_hi = int(np.percentile(deltas, 95))
        
        # --- COL BOUNDS FILTER ---
        col_bounds = []
        for i in range(self.pick_count):
            col_vals = [d[i] for d in data[-200:]] # Use 200 history for stable bounds
            col_bounds.append((int(np.percentile(col_vals, 3)), int(np.percentile(col_vals, 97))))
            
        sum_lo = int(np.percentile(sums, 8))
        sum_hi = int(np.percentile(sums, 92))
        
        range_lo = int(np.percentile(ranges, 8))
        range_hi = int(np.percentile(ranges, 92))
        
        banned_sum_block = None
        
        if len(data) >= 1:
            prev_sum = sum(data[-1][:self.pick_count])
            
            # Khối Tổng có tỷ lệ lặp lại tại chỗ chỉ 19% -> Block Khối +-10
            banned_sum_block = [prev_sum - 10, prev_sum + 10]
            
            # Hiệu ứng Bật Tường (Rebound) ở Cực hạn
            if prev_sum <= 100:
                sum_lo = max(sum_lo, 110)
            elif prev_sum >= 180:
                sum_hi = min(sum_hi, 170)
        
        # --- ELASTIC SPREAD FILTER (User Intuition) ---
        if use_elastic_filter and len(data) >= 2:
            s_t1 = max(data[-1][:self.pick_count]) - min(data[-1][:self.pick_count])
            s_t2 = max(data[-2][:self.pick_count]) - min(data[-2][:self.pick_count])
            
            if s_t1 >= 40: # Extreme Expansion -> Force Contraction
                range_hi = min(range_hi, 38)
            elif s_t1 <= 25: # Extreme Contraction -> Force Expansion
                range_lo = max(range_lo, 28)
            else:
                # Normal breathing: 65-68% chance of reversal
                if s_t1 > s_t2: # Was expanding -> Expect contraction
                    range_hi = min(range_hi, s_t1 - 1)
                elif s_t1 < s_t2: # Was contracting -> Expect expansion
                    range_lo = max(range_lo, s_t1 + 1)
                    
        # Sanity check
        if range_lo > range_hi:
            range_lo, range_hi = range_hi, range_lo
            
        if sum_lo > sum_hi:
            sum_lo, sum_hi = sum_hi, sum_lo
            
        go_board_liberties = set()
        prev_draw_set = set()
        if len(data) >= 1:
            prev_draw = data[-1][:self.pick_count]
            prev_draw_set = set(prev_draw)
            go_cols = 10  # Dynamic grid: 10 columns
            go_rows = (self.max_number + go_cols - 1) // go_cols
            for b in prev_draw:
                r, c = (b-1) // go_cols, (b-1) % go_cols
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < go_rows and 0 <= nc < go_cols:
                        adj = nr * go_cols + nc + 1
                        if 1 <= adj <= self.max_number:
                            go_board_liberties.add(adj)
            go_board_liberties -= prev_draw_set
                    
        missing_pool = set()
        hot_pool = set()
        if len(data) >= 10:
            window = data[-10:]
            counts = [0] * (self.max_number + 1)
            for d in window:
                for x in d[:self.pick_count]: counts[x] += 1
            for n in range(1, self.max_number + 1):
                if counts[n] == 0: missing_pool.add(n)
                elif counts[n] >= 2: hot_pool.add(n)
                
        top_frequent = set()
        if len(data) >= 100:
            window = data[-100:]
            freqs = [0] * (self.max_number + 1)
            for d in window:
                for x in d[:self.pick_count]: freqs[x] += 1
            arr = [(n, freqs[n]) for n in range(1, self.max_number + 1)]
            arr.sort(key=lambda x: x[1], reverse=True)
            for i in range(22):
                top_frequent.add(arr[i][0])
                
        markov_transitions = [{} for _ in range(self.pick_count)]
        alphabet_patterns = set()
        if len(data) >= 2:
            for i in range(1, len(data)):
                p_draw = data[i-1][:self.pick_count]
                c_draw = data[i][:self.pick_count]
                for c in range(self.pick_count):
                    p_val = p_draw[c]
                    c_val = c_draw[c]
                    if p_val not in markov_transitions[c]:
                        markov_transitions[c][p_val] = set()
                    markov_transitions[c][p_val].add(c_val)
                    
            for d in data:
                word = "".join("A" if x<=9 else "B" if x<=19 else "C" if x<=29 else "D" if x<=39 else "E" if x<=49 else "F" for x in d[:self.pick_count])
                alphabet_patterns.add(word)
            
        return {
            'sum_lo': sum_lo,
            'sum_hi': sum_hi,
            'odd_lo': max(0, int(np.percentile(odds, 8))),
            'odd_hi': min(self.pick_count, int(np.percentile(odds, 92))),
            'high_lo': max(0, int(np.percentile(highs, 8))),
            'high_hi': min(self.pick_count, int(np.percentile(highs, 92))),
            'range_lo': range_lo,
            'range_hi': range_hi,
            'banned_sum_block': banned_sum_block,
            'col_bounds': col_bounds,
            'delta_hi': delta_hi,
            'go_board_liberties': go_board_liberties,
            'prev_draw_set': prev_draw_set,
            'missing_pool': missing_pool,
            'hot_pool': hot_pool,
            'top_frequent': top_frequent,
            'markov_transitions': markov_transitions,
            'alphabet_patterns': alphabet_patterns,
            'prev_draw': data[-1][:self.pick_count] if len(data) >= 1 else None
        }

    def _validate(self, combo, c):
        s = sum(combo)
        if s < c.get('sum_lo', 0) or s > c.get('sum_hi', 999):
            return False
            
        banned_sum_block = c.get('banned_sum_block')
        if banned_sum_block and banned_sum_block[0] <= s <= banned_sum_block[1]:
            return False
            
        col_bounds = c.get('col_bounds')
        if col_bounds:
            for i, n in enumerate(combo):
                if n < col_bounds[i][0] or n > col_bounds[i][1]:
                    return False
                    
        max_delta = combo[0]
        for i in range(1, len(combo)):
            if combo[i] - combo[i-1] > max_delta:
                max_delta = combo[i] - combo[i-1]
        if max_delta > c.get('delta_hi', 45):
            return False
            
        digit_counts = [0] * 10
        for n in combo:
            s_val = str(n).zfill(2)
            digit_counts[int(s_val[0])] += 1
            digit_counts[int(s_val[1])] += 1
        if max(digit_counts) > 4:
            return False
            
        s_str = "".join([str(n).zfill(2) for n in combo])
        adj_pairs = 0
        for i in range(1, 12):
            if s_str[i] == s_str[i-1]:
                adj_pairs += 1
        if adj_pairs > 2:
            return False
            
        ones = [n % 10 for n in combo]
        breaks = 0
        currentDir = 0
        for i in range(1, 6):
            s_val_dir = 0
            if ones[i] > ones[i-1]: s_val_dir = 1
            elif ones[i] < ones[i-1]: s_val_dir = -1
            if s_val_dir != 0:
                if currentDir != 0 and currentDir != s_val_dir:
                    breaks += 1
                currentDir = s_val_dir
        if breaks == 0:
            return False
            
        matrix = [[0]*10 for _ in range(5)]
        for n in combo:
            r, c_idx = n // 10, n % 10
            if r < 5 and c_idx < 10:
                matrix[r][c_idx] = 1
        has_2x2 = False
        has_diag3 = False
        for r in range(4):
            for c_idx in range(9):
                if matrix[r][c_idx] and matrix[r][c_idx+1] and matrix[r+1][c_idx] and matrix[r+1][c_idx+1]:
                    has_2x2 = True
        for r in range(3):
            for c_idx in range(8):
                if matrix[r][c_idx] and matrix[r+1][c_idx+1] and matrix[r+2][c_idx+2]:
                    has_diag3 = True
            for c_idx in range(2, 10):
                if matrix[r][c_idx] and matrix[r+1][c_idx-1] and matrix[r+2][c_idx-2]:
                    has_diag3 = True
        if has_2x2 or has_diag3:
            return False
            
        colors = [0, 0, 0, 0, 0]
        for n in combo:
            ld = n % 10
            if ld == 1 or ld == 6: colors[0] += 1
            elif ld == 2 or ld == 7: colors[1] += 1
            elif ld == 3 or ld == 8: colors[2] += 1
            elif ld == 4 or ld == 9: colors[3] += 1
            elif ld == 5 or ld == 0: colors[4] += 1
        
        unique_colors = sum(1 for c in colors if c > 0)
        max_color = max(colors)
        if unique_colors <= 2 or max_color >= 4:
            return False
            
        # Go Board Filter
        prev_draw_set = c.get('prev_draw_set')
        go_board_liberties = c.get('go_board_liberties')
        if prev_draw_set is not None and go_board_liberties is not None:
            overlap = sum(1 for x in combo if x in prev_draw_set)
            contact = sum(1 for x in combo if x in go_board_liberties)
            if overlap > 2 or contact > 4:
                return False
                
        # Sliding Window Filter (Lô Gan / Hot)
        missing_pool = c.get('missing_pool')
        hot_pool = c.get('hot_pool')
        if missing_pool is not None and hot_pool is not None:
            missing_hit = sum(1 for x in combo if x in missing_pool)
            hot_hit = sum(1 for x in combo if x in hot_pool)
            if missing_hit > 3 or hot_hit > 3:
                return False
                
        # Markov Transition Filter
        markov_transitions = c.get('markov_transitions')
        prev_draw = c.get('prev_draw')
        if markov_transitions is not None and prev_draw is not None:
            markov_pass = 0
            for i in range(len(combo)):
                p_val = prev_draw[i]
                c_val = combo[i]
                if p_val in markov_transitions[i] and c_val in markov_transitions[i][p_val]:
                    markov_pass += 1
            if markov_pass < 4:
                return False
                
        # Hacker Cipher 12-bit Filter
        s_bin = ""
        for n in combo:
            n_str = str(n).zfill(2)
            s_bin += "0" if int(n_str[0]) % 2 == 0 else "1"
            s_bin += "0" if int(n_str[1]) % 2 == 0 else "1"
        max_0 = max(len(c) for c in s_bin.split("1"))
        max_1 = max(len(c) for c in s_bin.split("0"))
        is_pal = (s_bin == s_bin[::-1])
        is_alt = (s_bin == "010101010101" or s_bin == "101010101010")
        if max_0 >= 7 or max_1 >= 7 or is_pal or is_alt:
            return False
            
        # Frequency Polarity Filter
        top_frequent = c.get('top_frequent')
        if top_frequent:
            hit_top = sum(1 for x in combo if x in top_frequent)
            if hit_top < 2 or hit_top > 4:
                return False
                
        # Column Migration Filter
        if prev_draw is not None:
            for new_col in range(len(combo)):
                num = combo[new_col]
                if num in prev_draw_set:
                    old_col = prev_draw.index(num)
                    if abs(new_col - old_col) >= 3:
                        return False
                        
        # Alphabet Decade Cipher
        alphabet_patterns = c.get('alphabet_patterns')
        if alphabet_patterns is not None:
            word = "".join("A" if x<=9 else "B" if x<=19 else "C" if x<=29 else "D" if x<=39 else "E" if x<=49 else "F" for x in combo)
            if word not in alphabet_patterns:
                return False
            
        odd = sum(1 for x in combo if x % 2 == 1)
        if odd < c.get('odd_lo', 0) or odd > c.get('odd_hi', 6):
            return False
        mid = self.max_number // 2
        high = sum(1 for x in combo if x > mid)
        if high < c.get('high_lo', 0) or high > c.get('high_hi', 6):
            return False
        rng = max(combo) - min(combo)
        if rng < c.get('range_lo', 0) or rng > c.get('range_hi', 999):
            return False
        dec = [0] * 6
        for n in combo:
            dec[min((n - 1) // 10, 5)] += 1
        if max(dec) > 3:
            return False
        return True

    def _best_combo(self, pool, scores, constraints, sum_mod7=None):
        if len(pool) < self.pick_count:
            return None
        best, best_score = None, -float('inf')
        current = sorted(np.random.choice(pool, self.pick_count, replace=False).tolist())
        for _ in range(100):
            current = sorted(np.random.choice(pool, self.pick_count, replace=False).tolist())
            if self._validate(current, constraints):
                break

        best = current
        best_score = sum(scores.get(n, 0) for n in current)
        T, alpha = 10.0, 0.99

        for _ in range(5000):
            if T < 0.001:
                break
            neighbor = list(current)
            idx = np.random.randint(self.pick_count)
            new_num = np.random.choice(pool)
            while new_num in neighbor:
                new_num = np.random.choice(pool)
            neighbor[idx] = new_num
            neighbor = sorted(neighbor)

            if not self._validate(neighbor, constraints):
                T *= alpha
                continue
            if sum_mod7 and sum(neighbor) % 7 not in sum_mod7:
                T *= alpha
                continue

            ns = sum(scores.get(n, 0) for n in neighbor)
            if ns > best_score:
                current, best, best_score = neighbor, neighbor, ns
            elif np.random.rand() < math.exp((ns - best_score) / T):
                current = neighbor
            T *= alpha
        return best

    def _random_combo(self, pool, scores, constraints, sum_mod7=None):
        if len(pool) < self.pick_count:
            return None
        wts = np.array([max(scores.get(n, 0), 0.01) for n in pool])
        wts = wts / wts.sum()
        for _ in range(100):
            try:
                idx = np.random.choice(len(pool), self.pick_count, replace=False, p=wts)
                combo = sorted([pool[i] for i in idx])
                if self._validate(combo, constraints):
                    if sum_mod7 and sum(combo) % 7 not in sum_mod7:
                        continue
                    return combo
            except ValueError:
                continue
        return None
