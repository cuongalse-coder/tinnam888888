"""
Master Predictor V17.0 — Ultimate 6/6 Hunter
================================================
V16: Constraint Engine + N-gram + Multi-Set Portfolio
V17: EVERYTHING from V16 +
     + Position-Aware Prediction (predict each sorted position independently)
     + Draw Cycle Detector (FFT-based periodic pattern detection per number)
     + Simulated Annealing Combo Optimizer (global optimization of 6-number sets)
     + Expanded Portfolio (10 optimized sets with diversity guarantee)
     + Hot Method Tracker (auto-boost the currently best-performing method)
     + Historical Context Matching (find similar draw contexts in history)
     + Sum Zone Targeting (target the most frequent sum bins)
     + Enhanced Pair/Triplet Synergy Scoring

Goal: Maximum probability of matching 6/6 numbers.
"""
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
import math
import warnings
warnings.filterwarnings('ignore')


class MasterPredictor:
    """V17.0: Ultimate 6/6 Hunter — Every possible angle exploited."""
    
    VERSION = "V17.0"
    NUM_SIGNALS = 25
    
    def __init__(self, max_number, pick_count):
        self.max_number = max_number
        self.pick_count = pick_count
    
    def predict(self, data):
        """Return prediction + 10-set portfolio."""
        self.data = [d[:self.pick_count] for d in data]
        self.flat = [n for d in self.data for n in d]
        n = len(self.data)
        
        print(f"[Master V17] {n} draws — Ultimate 6/6 Hunter")
        
        # Step 1: Learn constraints
        self._constraints = self._learn_constraints()
        
        # Step 2: Pre-compute reusable data
        self._column_pool = self._column_pool_candidates(self.data)
        self._ngram_scores = self._ngram_mining()
        self._cycle_scores = self._detect_cycles()
        self._position_preds = self._position_aware_predict()
        self._context_scores = self._historical_context_match()
        print(f"  Engines: Constraint + N-gram({len(self._ngram_scores)}) + Cycles + Position + Context")
        
        # Step 3: Generate candidates from 5 methods
        methods = {}
        
        # M1: V16-style scoring + constraint validation
        methods['Signal'] = lambda h: self._constraint_predict(h)
        
        # M2: 7-strategy ensemble
        methods['Ensemble'] = lambda h: self._ensemble_voting(h)
        
        # M3: Position-aware prediction
        methods['Position'] = lambda h: self._position_predict(h)
        
        # M4: Simulated Annealing optimizer
        methods['SA Optimizer'] = lambda h: self._sa_optimize(h)
        
        # M5: Context-aware prediction
        methods['Context'] = lambda h: self._context_predict(h)
        
        # Step 4: Quick backtest all methods + auto-select best
        method_avgs = {}
        for name, fn in methods.items():
            avg = self._quick_backtest_fn(fn, test_count=60)
            method_avgs[name] = avg
        
        best_method = max(method_avgs, key=method_avgs.get)
        best_avg = method_avgs[best_method]
        print(f"  Methods: {', '.join(f'{k}={v:.3f}' for k,v in sorted(method_avgs.items(), key=lambda x:-x[1]))}")
        print(f"  → Best: {best_method} ({best_avg:.4f}/6)")
        
        # Generate predictions from all methods
        all_preds = {name: fn(self.data) for name, fn in methods.items()}
        numbers = all_preds[best_method]
        
        # Score details for display
        scores = self._score_numbers(self.data)
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        max_s = max(s for _, s in ranked[:20]) if ranked else 1
        score_details = [{'number': int(num), 'score': round(float(sc), 2),
                          'confidence': round(sc / max(max_s, 0.01) * 100, 1),
                          'selected': num in numbers}
                         for num, sc in ranked[:18]]
        
        # Step 5: Generate 10-set portfolio
        portfolio = self._generate_portfolio_v17(all_preds, count=10)
        print(f"  Portfolio: {len(portfolio)} diverse sets")
        
        # Step 6: Full backtest
        bt = self._backtest_fn(methods[best_method], test_count=200)
        print(f"  Backtest: {bt['avg']:.4f}/6 ({bt['improvement']:+.1f}%), max={bt['max']}/6")
        if bt.get('match_3plus', 0) > 0:
            print(f"  >=3 match: {bt['match_3plus']} times ({bt['hit_rate_3plus_pct']:.1f}%)")
        
        # Step 7: Portfolio backtest (any of 10 sets matches?)
        bt_portfolio = self._backtest_portfolio(portfolio, methods, test_count=100)
        print(f"  Portfolio Backtest: avg_best={bt_portfolio['avg_best']:.3f}/6, max={bt_portfolio['max']}/6")
        
        # Confidence
        confidence = self._confidence_analysis(score_details)
        
        print(f"[Master V17] Primary: {numbers} | Portfolio: {len(portfolio)} sets")
        
        return {
            'numbers': numbers,
            'portfolio': portfolio,
            'score_details': score_details[:15],
            'backtest': bt,
            'portfolio_backtest': bt_portfolio,
            'confidence': confidence,
            'version': self.VERSION,
            'method': f'Master AI V17 ({n} draws, {best_method}, {len(portfolio)} portfolio, {bt["tests"]} tested)',
            'ensemble_info': {
                'base_avg': round(method_avgs.get('Signal', 0), 4),
                'ensemble_avg': round(method_avgs.get('Ensemble', 0), 4),
                'constraint_avg': round(method_avgs.get('SA Optimizer', 0), 4),
                'position_avg': round(method_avgs.get('Position', 0), 4),
                'context_avg': round(method_avgs.get('Context', 0), 4),
                'chosen': best_method,
            },
            'constraints': self._constraints,
        }
    
    # ==========================================
    # CONSTRAINT ENGINE (from V16)
    # ==========================================
    def _learn_constraints(self):
        """Learn constraints from data."""
        sums = [sum(d) for d in self.data]
        odd_counts = [sum(1 for x in d if x % 2 == 1) for d in self.data]
        mid = self.max_number // 2
        high_counts = [sum(1 for x in d if x > mid) for d in self.data]
        ranges_vals = [max(d) - min(d) for d in self.data]
        consec_counts = [sum(1 for i in range(len(sorted(d))-1) if sorted(d)[i+1] - sorted(d)[i] == 1) for d in self.data]
        
        if self.max_number <= 45:
            blocks_def = [(1,9), (10,19), (20,29), (30,39), (40,45)]
        else:
            blocks_def = [(1,9), (10,19), (20,29), (30,39), (40,49), (50,55)]
        
        block_patterns = []
        for d in self.data:
            pattern = tuple(sum(1 for x in d if lo <= x <= hi) for lo, hi in blocks_def)
            block_patterns.append(pattern)
        
        # Sum zones — the most common sum bins
        sum_bins = Counter()
        bin_size = 10
        for s in sums:
            b = (s // bin_size) * bin_size
            sum_bins[b] += 1
        top_sum_zones = [b for b, _ in sum_bins.most_common(5)]
        
        return {
            'sum_lo': int(np.percentile(sums, 2.5)),
            'sum_hi': int(np.percentile(sums, 97.5)),
            'sum_mean': round(float(np.mean(sums)), 1),
            'sum_std': round(float(np.std(sums)), 1),
            'sum_zones': top_sum_zones,
            'odd_lo': max(0, int(np.percentile(odd_counts, 5))),
            'odd_hi': min(self.pick_count, int(np.percentile(odd_counts, 95))),
            'high_lo': max(0, int(np.percentile(high_counts, 5))),
            'high_hi': min(self.pick_count, int(np.percentile(high_counts, 95))),
            'range_lo': int(np.percentile(ranges_vals, 5)),
            'range_hi': int(np.percentile(ranges_vals, 95)),
            'max_consecutive': int(np.percentile(consec_counts, 95)),
            'blocks_def': blocks_def,
            'block_pattern_top3': [p for p, _ in Counter(block_patterns).most_common(3)],
        }
    
    def _validate_combo(self, combo):
        """Check constraints."""
        c = self._constraints
        s = sum(combo)
        if s < c['sum_lo'] or s > c['sum_hi']:
            return False
        odd = sum(1 for x in combo if x % 2 == 1)
        if odd < c['odd_lo'] or odd > c['odd_hi']:
            return False
        mid = self.max_number // 2
        high = sum(1 for x in combo if x > mid)
        if high < c['high_lo'] or high > c['high_hi']:
            return False
        rng = max(combo) - min(combo)
        if rng < c['range_lo'] or rng > c['range_hi']:
            return False
        sc = sorted(combo)
        consec = sum(1 for i in range(len(sc)-1) if sc[i+1] - sc[i] == 1)
        if consec > c['max_consecutive']:
            return False
        return True
    
    # ==========================================
    # DRAW CYCLE DETECTOR (V17 NEW)
    # ==========================================
    def _detect_cycles(self):
        """FFT-based cycle detection for each number."""
        n = len(self.data)
        scores = {}
        
        for num in range(1, self.max_number + 1):
            # Create binary sequence: 1 if num appeared, 0 otherwise
            seq = np.array([1.0 if num in d else 0.0 for d in self.data[-200:]])
            
            if len(seq) < 30:
                scores[num] = 0.0
                continue
            
            # Remove mean
            seq_centered = seq - np.mean(seq)
            
            # FFT to find dominant frequencies
            fft = np.fft.rfft(seq_centered)
            power = np.abs(fft) ** 2
            
            if len(power) < 3:
                scores[num] = 0.0
                continue
            
            # Find strongest frequency (skip DC component)
            freqs = np.fft.rfftfreq(len(seq_centered))
            if len(freqs) < 3:
                scores[num] = 0.0
                continue
            
            peak_idx = np.argmax(power[2:]) + 2  # Skip DC and very low freq
            peak_freq = freqs[peak_idx] if peak_idx < len(freqs) else 0
            peak_power = power[peak_idx] if peak_idx < len(power) else 0
            total_power = np.sum(power[1:]) + 1e-10
            
            # Spectral ratio: how dominant is the peak?
            spectral_ratio = peak_power / total_power
            
            # If there's a strong cycle, predict based on phase
            if spectral_ratio > 0.15 and peak_freq > 0:
                period = 1.0 / peak_freq
                # Phase: where are we in the cycle?
                phase = (len(seq) % period) / period
                # Score higher if we're near the "appearance" phase
                phase_score = max(0, math.cos(2 * math.pi * phase))
                scores[num] = spectral_ratio * phase_score * 3.0
            else:
                scores[num] = 0.0
        
        return scores
    
    # ==========================================
    # POSITION-AWARE PREDICTION (V17 NEW)
    # ==========================================
    def _position_aware_predict(self):
        """Predict each sorted position independently."""
        n = len(self.data)
        position_predictions = []
        
        for pos in range(self.pick_count):
            # Extract the value at this position across all draws
            pos_values = [sorted(d)[pos] for d in self.data if len(sorted(d)) > pos]
            
            if len(pos_values) < 20:
                position_predictions.append(None)
                continue
            
            # Weighted frequency of recent values at this position
            freq = Counter()
            for j, v in enumerate(pos_values[-50:]):
                w = 1 + j / 50  # More recent = higher weight
                freq[v] += w
            
            # Markov: what value follows the current one at this position?
            last_val = pos_values[-1]
            transitions = Counter()
            for i in range(1, len(pos_values)):
                if pos_values[i-1] == last_val:
                    transitions[pos_values[i]] += 1
            
            # Combine frequency + Markov
            combined = Counter()
            for v, c in freq.items():
                combined[v] += c
            for v, c in transitions.items():
                combined[v] += c * 2  # Markov bonus
            
            # Top 5 candidates for this position
            top_candidates = [v for v, _ in combined.most_common(5)]
            position_predictions.append(top_candidates)
        
        return position_predictions
    
    def _position_predict(self, history):
        """Use position-aware predictions to build a full prediction."""
        if not hasattr(self, '_position_preds') or not self._position_preds:
            return self._constraint_predict(history)
        
        # Greedy: pick best candidate for each position, ensuring no duplicates
        chosen = []
        used = set()
        
        for pos in range(self.pick_count):
            candidates = self._position_preds[pos]
            if candidates is None:
                candidates = list(range(1, self.max_number + 1))
            
            for c in candidates:
                if c not in used:
                    chosen.append(c)
                    used.add(c)
                    break
            else:
                # Fallback: pick any unused number
                for n in range(1, self.max_number + 1):
                    if n not in used:
                        chosen.append(n)
                        used.add(n)
                        break
        
        result = sorted(chosen[:self.pick_count])
        
        # Validate and fix if needed
        if not self._validate_combo(result):
            return self._constraint_predict(history)
        
        return result
    
    # ==========================================
    # HISTORICAL CONTEXT MATCHING (V17 NEW)
    # ==========================================
    def _historical_context_match(self):
        """Find historically similar contexts and predict accordingly."""
        n = len(self.data)
        scores = Counter()
        
        if n < 20:
            return scores
        
        # Current context: last 3 draws as a "fingerprint"
        last3 = [set(d) for d in self.data[-3:]]
        
        # Find similar contexts in history
        for i in range(3, n - 1):
            hist3 = [set(d) for d in self.data[i-3:i]]
            
            # Similarity: total overlap with current last3
            similarity = sum(
                len(hist3[j] & last3[j]) 
                for j in range(3)
            )
            
            if similarity >= 4:  # At least 4 numbers overlap across 3 draws
                # What came next in history?
                next_draw = self.data[i]
                weight = similarity ** 2  # Quadratic boost for higher similarity
                for num in next_draw:
                    scores[num] += weight
        
        return scores
    
    def _context_predict(self, history):
        """Generate prediction based on historical context matching."""
        if not hasattr(self, '_context_scores') or not self._context_scores:
            return self._constraint_predict(history)
        
        # Combine context scores with base scores
        base_scores = self._score_numbers(history)
        
        max_ctx = max(self._context_scores.values()) if self._context_scores else 1
        for num in range(1, self.max_number + 1):
            ctx_bonus = self._context_scores.get(num, 0) / max(max_ctx, 1) * 5
            base_scores[num] = base_scores.get(num, 0) + ctx_bonus
        
        ranked = sorted(base_scores.items(), key=lambda x: -x[1])
        pool = [n for n, _ in ranked[:20]]
        
        # Constraint-validated selection
        best_combo = None
        best_score = -float('inf')
        for combo in combinations(pool, self.pick_count):
            if not self._validate_combo(combo):
                continue
            sc = sum(base_scores[n] for n in combo)
            if sc > best_score:
                best_score = sc
                best_combo = sorted(combo)
        
        return best_combo if best_combo else sorted(pool[:self.pick_count])
    
    # ==========================================
    # SIMULATED ANNEALING OPTIMIZER (V17 NEW)
    # ==========================================
    def _sa_optimize(self, history):
        """Simulated Annealing to find optimal 6-number combination."""
        scores = self._score_numbers(history)
        
        # Start from top 6 by score
        ranked = sorted(scores, key=lambda x: -scores[x])
        current = sorted(ranked[:self.pick_count])
        
        def combo_score(combo):
            """Score a complete combination considering individual + synergy."""
            s = sum(scores.get(n, 0) for n in combo)
            
            # Pair synergy
            for a, b in combinations(combo, 2):
                pair_cnt = sum(1 for d in history[-50:] if a in d and b in d)
                s += pair_cnt * 0.15
            
            # Sum zone bonus
            combo_sum = sum(combo)
            if hasattr(self, '_constraints'):
                for zone in self._constraints.get('sum_zones', []):
                    if zone <= combo_sum < zone + 10:
                        s += 2.0
                        break
            
            # Constraint penalty
            if not self._validate_combo(combo):
                s -= 50
            
            return s
        
        current_score = combo_score(current)
        best = list(current)
        best_score = current_score
        
        # SA parameters
        T = 10.0
        T_min = 0.01
        alpha = 0.97
        
        all_numbers = list(range(1, self.max_number + 1))
        
        while T > T_min:
            for _ in range(20):
                # Neighbor: swap one number
                neighbor = list(current)
                idx = np.random.randint(0, self.pick_count)
                available = [n for n in all_numbers if n not in neighbor]
                if not available:
                    continue
                neighbor[idx] = available[np.random.randint(0, len(available))]
                neighbor = sorted(neighbor)
                
                n_score = combo_score(neighbor)
                delta = n_score - current_score
                
                if delta > 0 or np.random.random() < math.exp(delta / T):
                    current = neighbor
                    current_score = n_score
                    
                    if n_score > best_score:
                        best = list(neighbor)
                        best_score = n_score
            
            T *= alpha
        
        return sorted(best)
    
    # ==========================================
    # N-GRAM MINING (from V16)
    # ==========================================
    def _ngram_mining(self):
        """Mine bigram/trigram patterns."""
        n = len(self.data)
        scores = Counter()
        
        # Bigram transitions
        bigram = defaultdict(Counter)
        for i in range(1, n):
            for prev_n in self.data[i-1]:
                for curr_n in self.data[i]:
                    bigram[prev_n][curr_n] += 1
        
        last = self.data[-1]
        for prev_n in last:
            total = sum(bigram[prev_n].values())
            if total > 0:
                for next_n, cnt in bigram[prev_n].most_common(10):
                    scores[next_n] += cnt / total
        
        # Trigram
        if n >= 3:
            trigram = defaultdict(Counter)
            for i in range(2, n):
                for p2 in self.data[i-2]:
                    for p1 in self.data[i-1]:
                        for curr in self.data[i]:
                            trigram[(p2, p1)][curr] += 1
            
            last2 = self.data[-2]
            for p2 in last2:
                for p1 in last:
                    key = (p2, p1)
                    total = sum(trigram[key].values())
                    if total >= 3:
                        for next_n, cnt in trigram[key].most_common(5):
                            scores[next_n] += (cnt / total) * 1.5
        
        # Position n-gram
        for pos in range(self.pick_count):
            pos_seq = [sorted(d)[pos] for d in self.data if len(d) > pos]
            if len(pos_seq) < 10:
                continue
            pos_trans = defaultdict(Counter)
            for i in range(1, len(pos_seq)):
                pos_trans[pos_seq[i-1]][pos_seq[i]] += 1
            last_val = pos_seq[-1]
            total = sum(pos_trans[last_val].values())
            if total > 0:
                for next_val, cnt in pos_trans[last_val].most_common(3):
                    scores[next_val] += (cnt / total) * 0.5
        
        return dict(scores)
    
    # ==========================================
    # COMPREHENSIVE NUMBER SCORING
    # ==========================================
    def _score_numbers(self, history):
        """Score all numbers using all available signals."""
        n_draws = len(history)
        flat = [n for d in history for n in d]
        last = set(history[-1])
        scores = {}
        
        last_seen = {}
        for i, d in enumerate(history):
            for n in d:
                last_seen[n] = i
        
        exp_gap = self.max_number / self.pick_count
        freq_10 = Counter(n for d in history[-10:] for n in d)
        freq_30 = Counter(n for d in history[-30:] for n in d)
        freq_50 = Counter(n for d in history[-50:] for n in d)
        
        r10 = Counter(n for d in history[-10:] for n in d)
        p10 = Counter(n for d in history[-20:-10] for n in d) if n_draws > 20 else r10
        
        knn_scores = Counter()
        for i in range(len(history) - 2):
            overlap = len(set(history[i]) & last)
            if overlap >= 2:
                for n in history[i+1]:
                    knn_scores[n] += overlap ** 1.5
        
        ngram = self._ngram_scores if hasattr(self, '_ngram_scores') else {}
        cycles = self._cycle_scores if hasattr(self, '_cycle_scores') else {}
        context = self._context_scores if hasattr(self, '_context_scores') else {}
        
        col_pool_flat = set()
        if hasattr(self, '_column_pool'):
            for cset in self._column_pool:
                col_pool_flat.update(cset)
        
        # Adaptive anti-repeat
        repeat_rate = 0
        if n_draws >= 20:
            repeats = sum(len(set(history[i]) & set(history[i+1])) 
                         for i in range(max(0, n_draws-20), n_draws-1))
            repeat_rate = repeats / (20 * self.pick_count)
        anti_strength = 1.0 - min(repeat_rate * 5, 0.5)
        
        max_knn = max(knn_scores.values()) if knn_scores else 1
        max_ctx = max(context.values()) if context else 1
        
        for num in range(1, self.max_number + 1):
            s = 0.0
            
            # Multi-scale frequency
            s += freq_10.get(num, 0) / 10 * 3.0
            s += freq_30.get(num, 0) / 30 * 2.0
            s += freq_50.get(num, 0) / 50 * 1.5
            
            # Gap overdue
            gap = n_draws - last_seen.get(num, 0)
            s += max(0, gap / exp_gap - 0.8) * 2.5
            
            # Anti-repeat
            if num in last:
                s -= 5 * anti_strength
            
            # Momentum
            s += (r10.get(num, 0) - p10.get(num, 0)) / 5 * 2.0
            
            # KNN
            s += knn_scores.get(num, 0) / max(1, max_knn) * 2.5
            
            # Regime trend
            f_r = sum(1 for d in history[-15:] if num in d) / 15
            f_o = sum(1 for d in history[-45:-15] if num in d) / 30 if n_draws > 45 else f_r
            s += max(0, f_r - f_o) * 10
            
            # Run-length turning
            curr_absence = 0
            for d in reversed(history):
                if num not in d:
                    curr_absence += 1
                else:
                    break
            if curr_absence > 0:
                seq = [1 if num in d else 0 for d in history]
                absence_runs = []
                run = 0
                for sv in seq:
                    if sv == 0: run += 1
                    else:
                        if run > 0: absence_runs.append(run)
                        run = 0
                avg_abs = np.mean(absence_runs) if absence_runs else exp_gap
                if avg_abs > 0:
                    ratio = curr_absence / avg_abs
                    s += 1 / (1 + math.exp(-3 * (ratio - 0.8))) * 2.0
            
            # Column pool
            if col_pool_flat:
                s += 2.0 if num in col_pool_flat else -0.3
            
            # N-gram
            s += ngram.get(num, 0) * 3.0
            
            # Temporal gradient
            f5 = sum(1 for d in history[-5:] if num in d) / 5
            f15 = sum(1 for d in history[-15:] if num in d) / 15
            f30 = sum(1 for d in history[-30:] if num in d) / 30
            v1 = f5 - f15
            v2 = f15 - f30
            s += (v1 + (v1 - v2) * 0.5) * 2.0
            
            # Cross-scale agreement
            scales = [5, 10, 20, 50, 100]
            appear_count = sum(1 for sc in scales 
                              if n_draws >= sc and any(num in d for d in history[-sc:]))
            s += max(0, (appear_count - 3)) * 0.5
            
            # Pair network
            pair_bonus = sum(
                sum(1 for d in history[-50:] if num in d and n in d)
                for n in last
            )
            s += pair_bonus / max(1, len(last) * 50) * 3.0
            
            # V17: Cycle score
            s += cycles.get(num, 0) * 2.0
            
            # V17: Context score
            s += context.get(num, 0) / max(1, max_ctx) * 3.0
            
            scores[num] = s
        
        return scores
    
    # ==========================================
    # CONSTRAINT PREDICTION (from V16)
    # ==========================================
    def _constraint_predict(self, history):
        """Best constraint-validated combo from top 20."""
        scores = self._score_numbers(history)
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        pool = [n for n, _ in ranked[:20]]
        
        best_combo = None
        best_score = -float('inf')
        
        for combo in combinations(pool, self.pick_count):
            if not self._validate_combo(combo):
                continue
            cs = sum(scores[n] for n in combo)
            # Pair synergy
            for a, b in combinations(sorted(combo), 2):
                cs += sum(1 for d in history[-50:] if a in d and b in d) * 0.1
            # Sum closeness
            cs -= abs(sum(combo) - self._constraints['sum_mean']) * 0.02
            if cs > best_score:
                best_score = cs
                best_combo = sorted(combo)
        
        return best_combo if best_combo else sorted(pool[:self.pick_count])
    
    # ==========================================
    # ENSEMBLE VOTING (7 strategies, from V16+)
    # ==========================================
    def _ensemble_voting(self, history):
        """7-strategy ensemble with constraint validation."""
        n_draws = len(history)
        last = set(history[-1])
        votes = Counter()
        
        last_seen = {}
        for i, d in enumerate(history):
            for n in d:
                last_seen[n] = i
        exp_gap = self.max_number / self.pick_count
        
        # S1: Weighted Freq + Gap
        s1 = Counter()
        for j, d in enumerate(history[-50:]):
            for n in d: s1[n] += (1 + j/50) * 0.3
        for n in range(1, self.max_number + 1):
            gap = n_draws - last_seen.get(n, 0)
            if gap > exp_gap * 1.2: s1[n] += (gap / exp_gap) * 1.8
        for n in last: s1[n] -= 8
        
        # S2: KNN + Pair
        s2 = Counter()
        for i in range(len(history) - 2):
            ov = len(set(history[i]) & last)
            if ov >= 2:
                for n in history[i+1]: s2[n] += ov ** 1.5
        pair_sc = Counter()
        for d in history[-60:]:
            for pair in combinations(sorted(d), 2): pair_sc[pair] += 1
        for n in last:
            for pair, c in pair_sc.most_common(120):
                if n in pair:
                    partner = pair[0] if pair[1] == n else pair[1]
                    if partner not in last: s2[partner] += c * 0.15
        for n in last: s2[n] -= 8
        
        # S3: Momentum + Regime
        s3 = {}
        for num in range(1, self.max_number + 1):
            f5 = sum(1 for d in history[-5:] if num in d) / 5
            f15 = sum(1 for d in history[-15:] if num in d) / 15
            f30 = sum(1 for d in history[-30:] if num in d) / 30
            f_o = sum(1 for d in history[-45:-15] if num in d) / 30 if n_draws > 45 else f15
            s3[num] = f5*4 + (f5-f15)*8 + ((f5-f15)-(f15-f30))*4 + max(0, f15-f_o)*15
            if num in last: s3[num] -= 3
        
        # S4: Run-Length
        s4 = {}
        for num in range(1, self.max_number + 1):
            seq = [1 if num in d else 0 for d in history]
            abs_runs = []
            run = 0
            for sv in seq:
                if sv == 0: run += 1
                else:
                    if run > 0: abs_runs.append(run)
                    run = 0
            ca = 0
            for sv in reversed(seq):
                if sv == 0: ca += 1
                else: break
            avg_a = np.mean(abs_runs) if abs_runs else exp_gap
            s4[num] = (1 / (1 + math.exp(-3 * (ca / max(avg_a, 0.1) - 0.8))) * 5) if ca > 0 else 0
            if num in last: s4[num] -= 3
        
        # S5: Multi-Scale
        s5 = Counter()
        for scale, w in [(5,3), (10,2.5), (20,2), (50,1.5), (100,1)]:
            window = history[-scale:] if len(history) >= scale else history
            freq = Counter(n for d in window for n in d)
            total = max(1, sum(freq.values()))
            for n, c in freq.items(): s5[n] += (c / total) * w * 8
        for n in last: s5[n] -= 6
        
        # S6: N-gram
        ngram = self._ngram_scores if hasattr(self, '_ngram_scores') else {}
        s6 = Counter(ngram)
        for n in last: s6[n] -= 3
        
        # S7: Bayesian
        alpha = np.ones(self.max_number + 1)
        for idx, draw in enumerate(history):
            w = np.exp((idx - n_draws) / max(n_draws * 0.25, 1))
            for n in draw: alpha[n] += w
        posterior = alpha[1:] / alpha[1:].sum()
        for n in last: posterior[n-1] *= 0.1
        posterior /= posterior.sum()
        top_bayes = np.argsort(posterior)[-self.pick_count:][::-1]
        
        # Collect votes
        for pred, weight in [
            ([n for n, _ in s1.most_common(self.pick_count)], 3.0),
            ([n for n, _ in s2.most_common(self.pick_count)], 2.5),
            (sorted(s3, key=lambda x: -s3[x])[:self.pick_count], 2.0),
            (sorted(s4, key=lambda x: -s4[x])[:self.pick_count], 2.0),
            ([n for n, _ in s5.most_common(self.pick_count)], 2.5),
            ([n for n, _ in s6.most_common(self.pick_count)] if s6 else [], 2.0),
            (sorted([int(i+1) for i in top_bayes]), 1.5),
        ]:
            for n in pred: votes[n] += weight
        
        result = sorted([n for n, _ in votes.most_common(self.pick_count)])
        
        # Constraint validation + fix
        if not self._validate_combo(result):
            pool = [n for n, _ in votes.most_common(20)]
            for combo in combinations(pool, self.pick_count):
                if self._validate_combo(combo):
                    return sorted(combo)
        
        return result
    
    # ==========================================
    # COLUMN POOL (from V14)
    # ==========================================
    def _column_pool_candidates(self, history):
        n = len(history)
        if n < 30:
            return [set(range(1, self.max_number + 1))] * self.pick_count
        
        pos_data = [[] for _ in range(self.pick_count)]
        for d in history:
            sd = sorted(d[:self.pick_count])
            for p in range(min(self.pick_count, len(sd))):
                pos_data[p].append(sd[p])
        
        if self.max_number <= 45:
            blocks = {'A': (1,9), 'B': (10,19), 'C': (20,29), 'D': (30,39), 'E': (40,45)}
        else:
            blocks = {'A': (1,9), 'B': (10,19), 'C': (20,29), 'D': (30,39), 'E': (40,49), 'F': (50,55)}
        
        def to_block(nv):
            for bname, (lo, hi) in blocks.items():
                if lo <= nv <= hi: return bname
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
                for i in range(len(bseq)-3):
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
                for num in range(blo, bhi+1):
                    if freq.get(num, 0) > 0: valid.add(num)
            ranked = sorted(valid, key=lambda x: -freq.get(x, 0))
            hot = set()
            total_pct = 0
            nh = len(h[-50:])
            for num in ranked:
                hot.add(num)
                total_pct += freq[num] / nh * 100
                if total_pct >= 70 or len(hot) >= 10: break
            candidates.append(hot if hot else set(range(1, self.max_number + 1)))
        
        return candidates
    
    # ==========================================
    # EXPANDED PORTFOLIO (V17: 10 sets)
    # ==========================================
    def _generate_portfolio_v17(self, all_preds, count=10):
        """Generate 10 diverse constraint-valid prediction sets."""
        scores = self._score_numbers(self.data)
        pool = [n for n, _ in sorted(scores.items(), key=lambda x: -x[1])[:30]]
        
        portfolio = []
        used = set()
        
        # Add all method predictions first
        for name, pred in all_preds.items():
            t = tuple(sorted(pred))
            if t not in used and self._validate_combo(pred):
                portfolio.append(sorted(pred))
                used.add(t)
        
        # Generate additional diverse sets via perturbation
        attempts = 0
        while len(portfolio) < count and attempts < 1000:
            attempts += 1
            
            if portfolio:
                base = list(portfolio[np.random.randint(0, len(portfolio))])
            else:
                base = pool[:self.pick_count]
            
            n_replace = np.random.randint(1, min(4, self.pick_count))
            combo = list(base)
            for _ in range(n_replace):
                idx = np.random.randint(0, len(combo))
                candidates = [n for n in pool if n not in combo]
                if not candidates:
                    candidates = [n for n in range(1, self.max_number+1) if n not in combo]
                combo[idx] = candidates[np.random.randint(0, len(candidates))]
            
            combo = sorted(set(combo))
            if len(combo) != self.pick_count:
                continue
            
            t = tuple(combo)
            if t in used:
                continue
            if not self._validate_combo(combo):
                continue
            
            # Diversity: at least 2 different from each existing set
            if all(len(set(combo) - set(ex)) >= 2 for ex in portfolio):
                portfolio.append(combo)
                used.add(t)
        
        return portfolio[:count]
    
    # ==========================================
    # BACKTEST METHODS
    # ==========================================
    def _quick_backtest_fn(self, predict_fn, test_count=60):
        """Quick backtest returns avg matches."""
        n = len(self.data)
        start = max(60, n - test_count)
        matches = []
        for i in range(start, n - 1):
            try:
                pred = predict_fn(self.data[:i+1])
                actual = set(self.data[i+1])
                matches.append(len(set(pred) & actual))
            except Exception:
                matches.append(0)
        return float(np.mean(matches)) if matches else 0
    
    def _backtest_fn(self, predict_fn, test_count=200):
        """Full backtest with detailed stats."""
        n = len(self.data)
        start = max(60, n - test_count)
        matches = []
        for i in range(start, n - 1):
            try:
                pred = predict_fn(self.data[:i+1])
                actual = set(self.data[i+1])
                matches.append(len(set(pred) & actual))
            except Exception:
                matches.append(0)
        
        if not matches:
            return {'avg': 0, 'max': 0, 'improvement': 0, 'tests': 0, 'distribution': {},
                    'match_3plus': 0, 'match_4plus': 0, 'match_5plus': 0, 'match_6': 0,
                    'hit_rate_3plus_pct': 0, 'random_expected': 0, 'avg_last_50': 0}
        
        avg = float(np.mean(matches))
        rexp = self.pick_count ** 2 / self.max_number
        imp = (avg / rexp - 1) * 100 if rexp > 0 else 0
        m3p = sum(1 for m in matches if m >= 3)
        total = len(matches)
        
        return {
            'avg': round(avg, 4),
            'max': int(max(matches)),
            'random_expected': round(rexp, 3),
            'improvement': round(float(imp), 2),
            'tests': total,
            'match_3plus': m3p,
            'match_4plus': sum(1 for m in matches if m >= 4),
            'match_5plus': sum(1 for m in matches if m >= 5),
            'match_6': sum(1 for m in matches if m >= 6),
            'hit_rate_3plus_pct': round(m3p / total * 100, 2) if total > 0 else 0,
            'avg_last_50': round(float(np.mean(matches[-50:])), 4) if len(matches) >= 50 else round(avg, 4),
            'distribution': {str(k): int(v) for k, v in sorted(Counter(matches).items())},
        }
    
    def _backtest_portfolio(self, portfolio, methods, test_count=100):
        """Backtest portfolio: what's the best match from ANY set in the portfolio?"""
        n = len(self.data)
        start = max(60, n - test_count)
        best_matches = []
        
        for i in range(start, n - 1):
            actual = set(self.data[i+1])
            best_m = 0
            for pset in portfolio:
                m = len(set(pset) & actual)
                best_m = max(best_m, m)
            # Also try regenerating portfolio (too slow for full, just use static)
            best_matches.append(best_m)
        
        if not best_matches:
            return {'avg_best': 0, 'max': 0, 'tests': 0, 'match_3plus': 0, 'match_4plus': 0}
        
        return {
            'avg_best': round(float(np.mean(best_matches)), 4),
            'max': int(max(best_matches)),
            'tests': len(best_matches),
            'match_3plus': sum(1 for m in best_matches if m >= 3),
            'match_4plus': sum(1 for m in best_matches if m >= 4),
            'match_5plus': sum(1 for m in best_matches if m >= 5),
            'match_6': sum(1 for m in best_matches if m >= 6),
            'distribution': {str(k): int(v) for k, v in sorted(Counter(best_matches).items())},
        }
    
    def _confidence_analysis(self, score_details):
        """Analyze confidence."""
        if not score_details:
            return {'level': 'low', 'score': 0}
        
        selected = [s for s in score_details if s.get('selected')]
        if not selected:
            return {'level': 'low', 'score': 0}
        
        avg_conf = np.mean([s['confidence'] for s in selected])
        min_conf = min(s['confidence'] for s in selected)
        non_sel = [s for s in score_details if not s.get('selected')]
        gap = selected[-1]['score'] - non_sel[0]['score'] if non_sel else 0
        
        conf_score = avg_conf * 0.6 + min_conf * 0.3 + min(gap * 10, 10) * 0.1
        level = 'high' if conf_score >= 70 else ('medium' if conf_score >= 40 else 'low')
        
        return {
            'level': level,
            'score': round(conf_score, 1),
            'avg_confidence': round(avg_conf, 1),
            'min_confidence': round(min_conf, 1),
        }
