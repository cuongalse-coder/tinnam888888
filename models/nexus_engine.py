"""
NEXUS ENGINE V400.0 — ADAPTIVE QUANTUM
=======================================
Wraps MegaExploitV15 and adds 8 high-impact signals + walk-forward calibration.
V400 improvements: pair co-occurrence, temporal decay, adaptive weights.
"""
import math
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations

class NexusEngine:
    def __init__(self, max_number, pick_count):
        self.max_number = max_number
        self.pick_count = pick_count

    def predict(self, data, n_sets=5):
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

        # --- PHASE 2: 6 NEW high-precision signals ---
        new_sigs = {}
        new_sigs['sliding_window'] = self._sig_sliding_window(data)
        new_sigs['conditional_prob'] = self._sig_conditional_probability(data)
        new_sigs['gap_acceleration'] = self._sig_gap_acceleration(data)
        new_sigs['hot_cold_cross'] = self._sig_hot_cold_intersection(data)
        new_sigs['delta_momentum'] = self._sig_delta_momentum(data)
        new_sigs['sector_rotation'] = self._sig_sector_rotation(data)
        new_sigs['pair_boost'] = self._sig_pair_boost(data)
        new_sigs['temporal_decay'] = self._sig_temporal_decay(data)

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
                scores[num] += (score / max_s) * w * 1.5  # Boost factor for new signals

        # --- PHASE 5: Re-rank and build final output ---
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        top_pool = [n for n, s in ranked if s > 0][:30]
        if len(top_pool) < self.pick_count:
            top_pool = [n for n, _ in ranked[:30]]

        constraints = base_result.get('constraints', self._learn_constraints(data))
        sum_mod7 = base_result.get('sum_mod7', list(range(7)))
        if isinstance(sum_mod7, list):
            sum_mod7 = set(sum_mod7)

        predictions = []
        used = set()

        # Best combo via simulated annealing
        best = self._best_combo(top_pool[:20], scores, constraints, sum_mod7)
        if best:
            predictions.append({'numbers': best, 'strategy': '🧬 NEXUS V200 — Quantum Optimal'})
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
                predictions.append({'numbers': combo, 'strategy': '🧬 NEXUS Diversified'})

        predictions.sort(key=lambda x: -sum(scores[n] for n in x['numbers']))

        # Absolute Final 6
        absolute_final_6 = sorted([n for n, _ in ranked[:6]])

        # Merge weights
        all_weights = dict(base_result.get('weights', {}))
        for k, v in new_weights.items():
            all_weights[f"NEXUS_{k}"] = round(v, 3)

        confidence = min(base_result.get('confidence', 50) + len([w for w in new_weights.values() if w > 1.0]) * 2, 97)

        return {
            'predictions': predictions,
            'strategy': 'NexusEngine_V200_QUANTUM',
            'confidence': round(confidence, 1),
            'weights': {k: round(v, 3) if isinstance(v, float) else v for k, v in sorted(all_weights.items(), key=lambda x: -float(x[1]))},
            'scores': {n: round(s, 3) for n, s in ranked[:30]},
            'top_pool': top_pool[:25],
            'n_signals': 35 + len(new_sigs),  # V400: 43 signals total
            'constraints': constraints if isinstance(constraints, dict) else {},
            'sum_mod7': list(sum_mod7) if sum_mod7 else [],
            'absolute_final_6': absolute_final_6,
        }

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
    # IMPROVED ROLLING CALIBRATION
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

    def _learn_constraints(self, data):
        recent = data[-50:]
        sums = [sum(d[:self.pick_count]) for d in recent]
        odds = [sum(1 for x in d[:self.pick_count] if x % 2 == 1) for d in recent]
        mid = self.max_number // 2
        highs = [sum(1 for x in d[:self.pick_count] if x > mid) for d in recent]
        ranges = [max(d[:self.pick_count]) - min(d[:self.pick_count]) for d in recent]
        return {
            'sum_lo': int(np.percentile(sums, 8)),
            'sum_hi': int(np.percentile(sums, 92)),
            'odd_lo': max(0, int(np.percentile(odds, 8))),
            'odd_hi': min(self.pick_count, int(np.percentile(odds, 92))),
            'high_lo': max(0, int(np.percentile(highs, 8))),
            'high_hi': min(self.pick_count, int(np.percentile(highs, 92))),
            'range_lo': int(np.percentile(ranges, 8)),
            'range_hi': int(np.percentile(ranges, 92)),
        }

    def _validate(self, combo, c):
        s = sum(combo)
        if s < c.get('sum_lo', 0) or s > c.get('sum_hi', 999):
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
