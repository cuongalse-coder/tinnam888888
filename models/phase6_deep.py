"""
Phase 6: Deep Intelligence Engine
===================================
10 completely novel strategies not used in Phases 1-5.
Each strategy walk-forward backtested 200 draws.

Strategies:
1. HMM Regime Detection - Detect hot/cold/neutral regimes
2. Bayesian Number Network - Prior→Posterior probability update
3. Simulated Annealing - SA optimization of number sets
4. Information Theory Selector - Conditional entropy + MI
5. Copula Dependency Model - Rank-based dependency structure
6. Stacking Meta-Learner - Stack Phase 4+5 predictions
7. Run-Length Predictor - Predict appearance/absence turning points
8. Frequency Decomposition - Trend/seasonal/residual per number
9. Benford Evolution Tracker - Digit distribution anomaly detection
10. Combinatorial Coverage Optimizer - Maximum uncovered number coverage
"""
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
import math
import warnings
warnings.filterwarnings('ignore')


class DeepIntelligenceEngine:
    """Phase 6: Novel prediction strategies combining ML, information theory, and optimization."""
    
    def __init__(self, max_number, pick_count):
        self.max_number = max_number
        self.pick_count = pick_count
    
    def analyze(self, data):
        self.data = [d[:self.pick_count] for d in data]
        self.flat = [n for d in self.data for n in d]
        n = len(self.data)
        
        results = {}
        print(f"[Phase 6 Deep] Running on {n} draws...")
        
        results['hmm_regime'] = self._hmm_regime()
        print("  [1/10] HMM Regime Detection done")
        
        results['bayesian_net'] = self._bayesian_network()
        print("  [2/10] Bayesian Network done")
        
        results['simulated_annealing'] = self._simulated_annealing()
        print("  [3/10] Simulated Annealing done")
        
        results['info_theory'] = self._information_theory()
        print("  [4/10] Information Theory done")
        
        results['copula'] = self._copula_model()
        print("  [5/10] Copula Model done")
        
        results['stacking'] = self._stacking_meta()
        print("  [6/10] Stacking Meta-Learner done")
        
        results['run_length'] = self._run_length_predictor()
        print("  [7/10] Run-Length Predictor done")
        
        results['freq_decomp'] = self._frequency_decomposition()
        print("  [8/10] Frequency Decomposition done")
        
        results['benford_evo'] = self._benford_evolution()
        print("  [9/10] Benford Evolution done")
        
        results['coverage_opt'] = self._coverage_optimizer()
        print("  [10/10] Coverage Optimizer done")
        
        results['next_prediction'] = self._deep_prediction()
        results['verdict'] = self._verdict(results)
        
        print("[Phase 6 Deep] Complete!")
        return results
    
    def _bt(self, predict_fn, test_count=200):
        """Walk-forward backtest."""
        n = len(self.data)
        start = max(60, n - test_count)
        matches = []
        for i in range(start, n - 1):
            pred = predict_fn(self.data[:i+1])
            actual = set(self.data[i+1])
            matches.append(len(set(pred) & actual))
        if not matches:
            return {'avg_matches': 0, 'max_matches': 0, 'improvement': 0, 'distribution': {}, 'tests': 0}
        avg = float(np.mean(matches))
        rexp = self.pick_count ** 2 / self.max_number
        imp = (avg / rexp - 1) * 100 if rexp > 0 else 0
        return {
            'avg_matches': round(avg, 4),
            'max_matches': int(max(matches)),
            'random_expected': round(rexp, 3),
            'improvement': round(float(imp), 2),
            'distribution': {str(k): int(v) for k, v in sorted(Counter(matches).items())},
            'tests': len(matches),
            'match_3plus': sum(1 for m in matches if m >= 3),
            'match_4plus': sum(1 for m in matches if m >= 4),
        }
    
    # ===== 1. HMM REGIME DETECTION =====
    def _hmm_regime(self):
        """
        Simple Hidden Markov Model to detect hot/cold/neutral regimes.
        Each number is classified into a regime based on recent frequency patterns.
        Predictions are weighted by the current regime.
        """
        def detect_regime(history, window=20):
            """Returns regime scores for each number: hot(+1), neutral(0), cold(-1)"""
            n_draws = len(history)
            regimes = {}
            
            for num in range(1, self.max_number + 1):
                # Frequency in recent windows
                freq_recent = sum(1 for d in history[-window:] if num in d) / window
                freq_mid = sum(1 for d in history[-window*2:-window] if num in d) / window if n_draws > window*2 else freq_recent
                freq_old = sum(1 for d in history[-window*3:-window*2] if num in d) / window if n_draws > window*3 else freq_mid
                
                expected = self.pick_count / self.max_number
                
                # Regime detection: trending up, trending down, or stable
                trend = freq_recent - freq_old
                level = freq_recent - expected
                
                if trend > 0.03 and level > 0:
                    regime = 'hot'
                    score = 2.0 + trend * 10 + level * 5
                elif trend < -0.03 and level < 0:
                    regime = 'cold'
                    score = -1.0
                else:
                    regime = 'neutral'
                    score = freq_recent * 5
                
                # Overdue boost for cold numbers
                gap = n_draws
                for i in range(n_draws-1, max(0, n_draws-100), -1):
                    if num in history[i]:
                        gap = n_draws - 1 - i
                        break
                exp_gap = self.max_number / self.pick_count
                if gap > exp_gap * 1.3:
                    score += (gap / exp_gap - 1) * 3
                
                regimes[num] = {'regime': regime, 'score': score}
            
            return regimes
        
        def predict_fn(history):
            regimes = detect_regime(history)
            last = set(history[-1])
            
            scores = {n: r['score'] for n, r in regimes.items()}
            for n in last:
                scores[n] -= 5  # anti-repeat
            
            return sorted(scores, key=lambda x: -scores[x])[:self.pick_count]
        
        bt = self._bt(predict_fn)
        
        # Get current regime breakdown
        regimes = detect_regime(self.data)
        regime_counts = Counter(r['regime'] for r in regimes.values())
        
        return {
            'name': 'HMM Regime Detection',
            'is_pattern': bt['improvement'] > 15,
            'regime_breakdown': {k: int(v) for k, v in regime_counts.items()},
            **bt,
            'conclusion': f"HMM Regime: {bt['avg_matches']:.4f}/6 ({bt['improvement']:+.2f}%)"
        }
    
    # ===== 2. BAYESIAN NUMBER NETWORK =====
    def _bayesian_network(self):
        """
        Bayesian approach: start with uniform prior, update with evidence from 
        recent draws using likelihood ratios.
        """
        def predict_fn(history):
            n_draws = len(history)
            
            # Prior: uniform
            prior = np.ones(self.max_number + 1) / self.max_number
            
            # Likelihood updates from multiple evidence sources
            
            # Evidence 1: Frequency in last 30 draws
            freq_30 = Counter(n for d in history[-30:] for n in d)
            lik_freq = np.zeros(self.max_number + 1)
            for n in range(1, self.max_number + 1):
                lik_freq[n] = (freq_30.get(n, 0) + 1) / (30 * self.pick_count + self.max_number)
            lik_freq /= lik_freq.sum() + 1e-10
            
            # Evidence 2: Gap-based (overdue numbers more likely)
            lik_gap = np.zeros(self.max_number + 1)
            last_seen = {}
            for i, d in enumerate(history):
                for n in d:
                    last_seen[n] = i
            exp_gap = self.max_number / self.pick_count
            for n in range(1, self.max_number + 1):
                gap = n_draws - last_seen.get(n, 0)
                # Geometric distribution likelihood
                p = self.pick_count / self.max_number
                lik_gap[n] = p * ((1-p) ** (gap - 1)) if gap > 0 else p
                if gap > exp_gap * 1.2:
                    lik_gap[n] *= (gap / exp_gap)  # Overdue boost
            lik_gap /= lik_gap.sum() + 1e-10
            
            # Evidence 3: Conditional on last draw (numbers that follow similar draws)
            lik_cond = np.zeros(self.max_number + 1)
            last = set(history[-1])
            for i in range(len(history) - 2):
                overlap = len(set(history[i]) & last)
                if overlap >= 2:
                    for n in history[i+1]:
                        lik_cond[n] += overlap
            lik_cond += 0.1
            lik_cond /= lik_cond.sum() + 1e-10
            
            # Posterior = Prior * Lik1 * Lik2 * Lik3
            posterior = prior[1:] * lik_freq[1:] * lik_gap[1:] * lik_cond[1:]
            
            # Anti-repeat
            for n in history[-1]:
                posterior[n-1] *= 0.1
            
            posterior /= posterior.sum() + 1e-10
            
            top_idx = np.argsort(posterior)[-self.pick_count:][::-1]
            return [int(i+1) for i in top_idx]
        
        bt = self._bt(predict_fn)
        return {
            'name': 'Bayesian Number Network',
            'is_pattern': bt['improvement'] > 15,
            **bt,
            'conclusion': f"Bayesian: {bt['avg_matches']:.4f}/6 ({bt['improvement']:+.2f}%)"
        }
    
    # ===== 3. SIMULATED ANNEALING =====
    def _simulated_annealing(self):
        """
        Use SA to optimize a set of 6 numbers based on composite scoring function.
        """
        def score_set(nums, history):
            """Score a candidate set based on multiple signals."""
            total = 0
            flat = [n for d in history[-50:] for n in d]
            freq = Counter(flat)
            last = set(history[-1])
            n_draws = len(history)
            
            for n in nums:
                # Frequency score
                total += freq.get(n, 0) * 0.3
                # Gap score
                gap = n_draws
                for i in range(n_draws-1, max(0, n_draws-100), -1):
                    if n in history[i]:
                        gap = n_draws - 1 - i
                        break
                exp_gap = self.max_number / self.pick_count
                if gap > exp_gap:
                    total += (gap / exp_gap) * 2
                # Anti-repeat
                if n in last:
                    total -= 10
            
            # Pair bonus
            for a, b in combinations(sorted(nums), 2):
                pair_count = sum(1 for d in history[-50:] if a in d and b in d)
                total += pair_count * 0.2
            
            # Sum constraint
            avg_sum = np.mean([sum(d) for d in history[-30:]])
            sum_diff = abs(sum(nums) - avg_sum)
            total -= sum_diff * 0.05
            
            return total
        
        def predict_fn(history):
            # Initial solution: top frequency numbers
            freq = Counter(n for d in history[-30:] for n in d)
            for n in history[-1]:
                freq[n] -= 3
            current = sorted([n for n, _ in freq.most_common(self.pick_count)])
            current_score = score_set(current, history)
            
            T = 10.0
            T_min = 0.1
            alpha = 0.92
            
            best = current[:]
            best_score = current_score
            
            for _ in range(200):
                # Neighbor: replace one number
                neighbor = current[:]
                idx = np.random.randint(0, self.pick_count)
                new_num = np.random.randint(1, self.max_number + 1)
                while new_num in neighbor:
                    new_num = np.random.randint(1, self.max_number + 1)
                neighbor[idx] = new_num
                neighbor.sort()
                
                neighbor_score = score_set(neighbor, history)
                delta = neighbor_score - current_score
                
                if delta > 0 or np.random.random() < math.exp(delta / max(T, 0.01)):
                    current = neighbor
                    current_score = neighbor_score
                    
                    if current_score > best_score:
                        best = current[:]
                        best_score = current_score
                
                T *= alpha
                if T < T_min:
                    break
            
            return best
        
        bt = self._bt(predict_fn)
        return {
            'name': 'Simulated Annealing Optimizer',
            'is_pattern': bt['improvement'] > 15,
            **bt,
            'conclusion': f"SA: {bt['avg_matches']:.4f}/6 ({bt['improvement']:+.2f}%)"
        }
    
    # ===== 4. INFORMATION THEORY =====
    def _information_theory(self):
        """
        Use conditional entropy and mutual information to select numbers.
        Numbers with lower conditional entropy are more predictable.
        """
        def predict_fn(history):
            n_draws = len(history)
            scores = {}
            
            for num in range(1, self.max_number + 1):
                # Build binary sequence: 1 if num appeared, 0 otherwise
                seq = [1 if num in d else 0 for d in history[-100:]]
                
                if len(seq) < 10:
                    scores[num] = 0
                    continue
                
                # Conditional entropy H(X_t | X_{t-1}, X_{t-2})
                # Lower = more predictable
                transitions = defaultdict(Counter)
                for i in range(2, len(seq)):
                    context = (seq[i-2], seq[i-1])
                    transitions[context][seq[i]] += 1
                
                cond_entropy = 0
                total = sum(sum(v.values()) for v in transitions.values())
                for context, next_counts in transitions.items():
                    context_total = sum(next_counts.values())
                    p_context = context_total / max(1, total)
                    for outcome, count in next_counts.items():
                        p = count / context_total
                        if p > 0:
                            cond_entropy -= p_context * p * math.log2(p)
                
                # Current context
                if len(seq) >= 2:
                    current_context = (seq[-2], seq[-1])
                    if current_context in transitions:
                        # Probability of appearing next
                        ctx_counts = transitions[current_context]
                        prob_next = ctx_counts.get(1, 0) / max(1, sum(ctx_counts.values()))
                    else:
                        prob_next = self.pick_count / self.max_number
                else:
                    prob_next = self.pick_count / self.max_number
                
                # Score: high probability + low entropy = good
                predictability = 1.0 - cond_entropy
                scores[num] = prob_next * 5 + predictability * 2
                
                # Anti-repeat
                if num in history[-1]:
                    scores[num] -= 3
            
            return sorted(scores, key=lambda x: -scores[x])[:self.pick_count]
        
        bt = self._bt(predict_fn)
        return {
            'name': 'Information Theory Selector',
            'is_pattern': bt['improvement'] > 15,
            **bt,
            'conclusion': f"InfoTheory: {bt['avg_matches']:.4f}/6 ({bt['improvement']:+.2f}%)"
        }
    
    # ===== 5. COPULA DEPENDENCY MODEL =====
    def _copula_model(self):
        """
        Model dependencies between the 6 number positions using rank-based approach.
        If position 1 is high, what tends to happen to positions 2-6?
        """
        def predict_fn(history):
            if len(history) < 50:
                freq = Counter(n for d in history for n in d)
                return [n for n, _ in freq.most_common(self.pick_count)]
            
            # Build rank data for each position
            sorted_draws = [sorted(d) for d in history[-100:]]
            
            # Current draw's ranks
            last_sorted = sorted(history[-1])
            last_ranks = []
            for pos in range(self.pick_count):
                # Get percentile of current value in historical values for this position
                historical_vals = [d[pos] for d in sorted_draws if pos < len(d)]
                rank = np.searchsorted(sorted(historical_vals), last_sorted[pos]) / max(1, len(historical_vals))
                last_ranks.append(rank)
            
            # Find similar rank patterns in history
            scores = Counter()
            for i in range(len(sorted_draws) - 1):
                ranks_i = []
                for pos in range(self.pick_count):
                    historical_vals = [d[pos] for d in sorted_draws[:max(1, i)] if pos < len(d)]
                    if not historical_vals:
                        ranks_i.append(0.5)
                        continue
                    rank = np.searchsorted(sorted(historical_vals), sorted_draws[i][pos]) / max(1, len(historical_vals))
                    ranks_i.append(rank)
                
                # Similarity via rank correlation
                if len(ranks_i) == len(last_ranks):
                    diff = sum(abs(a-b) for a, b in zip(ranks_i, last_ranks))
                    similarity = max(0, self.pick_count - diff * 3)
                    
                    if similarity > 0:
                        # What came after this similar draw?
                        next_draw = sorted_draws[i+1] if i+1 < len(sorted_draws) else sorted_draws[-1]
                        for n in next_draw:
                            scores[n] += similarity
            
            # Anti-repeat
            for n in history[-1]:
                scores[n] -= 5
            
            if not scores:
                freq = Counter(n for d in history[-30:] for n in d)
                return [n for n, _ in freq.most_common(self.pick_count)]
            
            return [n for n, _ in scores.most_common(self.pick_count)]
        
        bt = self._bt(predict_fn)
        return {
            'name': 'Copula Dependency Model',
            'is_pattern': bt['improvement'] > 15,
            **bt,
            'conclusion': f"Copula: {bt['avg_matches']:.4f}/6 ({bt['improvement']:+.2f}%)"
        }
    
    # ===== 6. STACKING META-LEARNER =====
    def _stacking_meta(self):
        """
        Stack multiple simple predictors and let a meta-model combine them.
        """
        def base_predictors(history):
            """Returns dict of {predictor_name: set_of_predicted_numbers}"""
            preds = {}
            flat = [n for d in history for n in d]
            last = set(history[-1])
            n_draws = len(history)
            
            # P1: Hot frequency (last 20)
            freq20 = Counter(n for d in history[-20:] for n in d)
            preds['hot20'] = set(n for n, _ in freq20.most_common(self.pick_count * 2))
            
            # P2: Hot frequency (last 50)
            freq50 = Counter(n for d in history[-50:] for n in d)
            preds['hot50'] = set(n for n, _ in freq50.most_common(self.pick_count * 2))
            
            # P3: Overdue numbers
            last_seen = {}
            for i, d in enumerate(history):
                for n in d:
                    last_seen[n] = i
            overdue = sorted(range(1, self.max_number + 1), 
                           key=lambda n: n_draws - last_seen.get(n, 0), reverse=True)
            preds['overdue'] = set(overdue[:self.pick_count * 2])
            
            # P4: Position mode
            pos_nums = set()
            for pos in range(self.pick_count):
                vals = [sorted(d)[pos] for d in history[-50:] if len(d) >= pos+1]
                if vals:
                    mode = Counter(vals).most_common(1)[0][0]
                    pos_nums.add(mode)
            preds['position'] = pos_nums
            
            # P5: Markov transition
            markov_nums = set()
            if len(flat) >= 3:
                trans = defaultdict(Counter)
                for i in range(2, len(flat)):
                    trans[(flat[i-2], flat[i-1])][flat[i]] += 1
                key = (flat[-2], flat[-1])
                if key in trans:
                    markov_nums = set(n for n, _ in trans[key].most_common(self.pick_count * 2))
            preds['markov'] = markov_nums
            
            # P6: Momentum
            mom_nums = set()
            if n_draws > 20:
                r10 = Counter(n for d in history[-10:] for n in d)
                p10 = Counter(n for d in history[-20:-10] for n in d)
                momentum = {n: r10.get(n, 0) - p10.get(n, 0) for n in range(1, self.max_number + 1)}
                top_mom = sorted(momentum, key=lambda x: -momentum[x])[:self.pick_count * 2]
                mom_nums = set(top_mom)
            preds['momentum'] = mom_nums
            
            return preds
        
        def predict_fn(history):
            preds = base_predictors(history)
            last = set(history[-1])
            
            # Meta-learner: vote counting with learned weights
            scores = Counter()
            weights = {'hot20': 3, 'hot50': 2, 'overdue': 2.5, 'position': 2, 'markov': 3, 'momentum': 2}
            
            for name, nums in preds.items():
                w = weights.get(name, 1)
                for n in nums:
                    scores[n] += w
            
            # Anti-repeat
            for n in last:
                scores[n] -= 8
            
            return [n for n, _ in scores.most_common(self.pick_count)]
        
        bt = self._bt(predict_fn)
        
        # Analyze which predictors are best
        preds = base_predictors(self.data)
        pred_quality = {}
        actual_last = set(self.data[-1])
        for name, nums in preds.items():
            overlap = len(nums & actual_last)
            pred_quality[name] = overlap
        
        return {
            'name': 'Stacking Meta-Learner (6 base)',
            'is_pattern': bt['improvement'] > 15,
            'base_predictors': list(preds.keys()),
            'predictor_quality': pred_quality,
            **bt,
            'conclusion': f"Stacking: {bt['avg_matches']:.4f}/6 ({bt['improvement']:+.2f}%)"
        }
    
    # ===== 7. RUN-LENGTH PREDICTOR =====
    def _run_length_predictor(self):
        """
        For each number, track run-lengths of appearance and absence.
        Predict turning points (absent→present) based on average run length.
        """
        def predict_fn(history):
            n_draws = len(history)
            scores = {}
            
            for num in range(1, self.max_number + 1):
                # Build appearance sequence
                seq = [1 if num in d else 0 for d in history]
                
                # Compute run lengths of 0s (absence runs)
                absence_runs = []
                current_run = 0
                for s in seq:
                    if s == 0:
                        current_run += 1
                    else:
                        if current_run > 0:
                            absence_runs.append(current_run)
                        current_run = 0
                
                # Current absence run
                current_absence = 0
                for s in reversed(seq):
                    if s == 0:
                        current_absence += 1
                    else:
                        break
                
                # Average absence run length
                avg_absence = np.mean(absence_runs) if absence_runs else self.max_number / self.pick_count
                
                # Turning point probability: higher when current_absence approaches avg_absence
                if avg_absence > 0:
                    ratio = current_absence / avg_absence
                    # Sigmoid-like: probability increases as ratio approaches 1
                    turn_prob = 1 / (1 + math.exp(-3 * (ratio - 0.8)))
                else:
                    turn_prob = 0.5
                
                # Also compute presence run lengths
                presence_runs = []
                current_run = 0
                for s in seq:
                    if s == 1:
                        current_run += 1
                    else:
                        if current_run > 0:
                            presence_runs.append(current_run)
                        current_run = 0
                
                current_presence = 0
                for s in reversed(seq):
                    if s == 1:
                        current_presence += 1
                    else:
                        break
                
                avg_presence = np.mean(presence_runs) if presence_runs else 1
                
                if current_absence > 0:
                    # Currently absent - predict turning point
                    scores[num] = turn_prob * 5
                else:
                    # Currently appearing - predict if streak will continue
                    if avg_presence > 0:
                        continue_prob = max(0, 1 - current_presence / avg_presence)
                        scores[num] = continue_prob * 2
                    else:
                        scores[num] = 0
                
                # Anti-repeat
                if num in history[-1]:
                    scores[num] -= 3
            
            return sorted(scores, key=lambda x: -scores[x])[:self.pick_count]
        
        bt = self._bt(predict_fn)
        return {
            'name': 'Run-Length Turning Point Predictor',
            'is_pattern': bt['improvement'] > 15,
            **bt,
            'conclusion': f"RunLength: {bt['avg_matches']:.4f}/6 ({bt['improvement']:+.2f}%)"
        }
    
    # ===== 8. FREQUENCY DECOMPOSITION =====
    def _frequency_decomposition(self):
        """
        Decompose per-number frequency time series into trend + seasonal + residual.
        Use trend+seasonal to predict next.
        """
        def predict_fn(history):
            n_draws = len(history)
            window = min(30, n_draws // 3)
            if window < 5:
                window = 5
            
            scores = {}
            
            for num in range(1, self.max_number + 1):
                # Build frequency time series (rolling window)
                freq_series = []
                for i in range(window, n_draws + 1):
                    cnt = sum(1 for d in history[i-window:i] if num in d)
                    freq_series.append(cnt / window)
                
                if len(freq_series) < 5:
                    scores[num] = 0
                    continue
                
                ts = np.array(freq_series)
                
                # Trend: moving average
                trend_window = min(10, len(ts) // 2)
                if trend_window < 2:
                    trend = ts
                else:
                    trend = np.convolve(ts, np.ones(trend_window)/trend_window, mode='valid')
                    # Pad to same length
                    pad = len(ts) - len(trend)
                    trend = np.concatenate([np.full(pad, trend[0]), trend])
                
                # Seasonal: residual after trend removal, find dominant cycle
                detrended = ts - trend
                
                # Simple extrapolation: use last trend value and direction
                if len(trend) >= 2:
                    trend_slope = trend[-1] - trend[-2]
                    next_trend = trend[-1] + trend_slope
                else:
                    next_trend = trend[-1] if len(trend) > 0 else 0
                
                # Seasonal estimate: average of detrended values
                next_seasonal = np.mean(detrended[-5:]) if len(detrended) >= 5 else 0
                
                predicted_freq = next_trend + next_seasonal
                scores[num] = predicted_freq * 10
                
                # Anti-repeat
                if num in history[-1]:
                    scores[num] -= 2
            
            return sorted(scores, key=lambda x: -scores[x])[:self.pick_count]
        
        bt = self._bt(predict_fn)
        return {
            'name': 'Frequency Decomposition (Trend+Seasonal)',
            'is_pattern': bt['improvement'] > 15,
            **bt,
            'conclusion': f"FreqDecomp: {bt['avg_matches']:.4f}/6 ({bt['improvement']:+.2f}%)"
        }
    
    # ===== 9. BENFORD EVOLUTION =====
    def _benford_evolution(self):
        """
        Track how the first-digit distribution evolves over time.
        When distribution deviates from Benford's law, certain number ranges become more likely.
        """
        def predict_fn(history):
            n_draws = len(history)
            
            # Benford expected distribution for first digits
            benford = {d: math.log10(1 + 1/d) for d in range(1, 10)}
            
            # Actual first-digit distribution in recent draws
            recent_digits = Counter()
            for d in history[-50:]:
                for n in d:
                    fd = int(str(n)[0])
                    recent_digits[fd] += 1
            
            total = sum(recent_digits.values())
            
            # Find digits that are over/under represented vs Benford
            digit_scores = {}
            for d in range(1, min(10, self.max_number)):
                actual_pct = recent_digits.get(d, 0) / max(1, total)
                expected_pct = benford.get(d, 0.1)
                # If under-represented, boost numbers starting with this digit
                deviation = expected_pct - actual_pct
                digit_scores[d] = deviation
            
            scores = {}
            for num in range(1, self.max_number + 1):
                fd = int(str(num)[0])
                # Base score from Benford deviation
                scores[num] = digit_scores.get(fd, 0) * 20
                
                # Also add recent frequency
                freq = sum(1 for d in history[-30:] if num in d)
                scores[num] += freq * 0.5
                
                # Momentum
                r10 = sum(1 for d in history[-10:] if num in d)
                p10 = sum(1 for d in history[-20:-10] if num in d) if n_draws > 20 else r10
                scores[num] += (r10 - p10) * 1.5
                
                # Anti-repeat
                if num in history[-1]:
                    scores[num] -= 4
            
            return sorted(scores, key=lambda x: -scores[x])[:self.pick_count]
        
        bt = self._bt(predict_fn)
        
        # Current Benford analysis
        benford = {d: math.log10(1 + 1/d) for d in range(1, 10)}
        recent_digits = Counter()
        for d in self.data[-50:]:
            for n in d:
                fd = int(str(n)[0])
                recent_digits[fd] += 1
        total = sum(recent_digits.values())
        digit_analysis = {}
        for d in range(1, min(10, self.max_number)):
            actual = round(recent_digits.get(d, 0) / max(1, total) * 100, 1)
            expected = round(benford.get(d, 0.1) * 100, 1)
            digit_analysis[str(d)] = {'actual_pct': actual, 'benford_pct': expected, 
                                       'deviation': round(actual - expected, 1)}
        
        return {
            'name': 'Benford Evolution Tracker',
            'is_pattern': bt['improvement'] > 15,
            'digit_analysis': digit_analysis,
            **bt,
            'conclusion': f"Benford: {bt['avg_matches']:.4f}/6 ({bt['improvement']:+.2f}%)"
        }
    
    # ===== 10. COMBINATORIAL COVERAGE OPTIMIZER =====
    def _coverage_optimizer(self):
        """
        Optimize selection to maximize coverage of number groups
        that haven't appeared recently.
        """
        def predict_fn(history):
            n_draws = len(history)
            
            # Group numbers into zones (low, mid, high)
            zone_size = self.max_number // 3
            zones = {
                'low': range(1, zone_size + 1),
                'mid': range(zone_size + 1, zone_size * 2 + 1),
                'high': range(zone_size * 2 + 1, self.max_number + 1),
            }
            
            # Recent coverage per zone
            zone_recent = {}
            for zname, zrange in zones.items():
                zset = set(zrange)
                covered = set()
                for d in history[-10:]:
                    for n in d:
                        if n in zset:
                            covered.add(n)
                zone_recent[zname] = len(covered) / len(zset)
            
            # Find uncovered numbers
            all_recent = set(n for d in history[-5:] for n in d)
            
            scores = {}
            for num in range(1, self.max_number + 1):
                # Base: frequency
                freq = sum(1 for d in history[-30:] if num in d)
                scores[num] = freq * 0.5
                
                # Bonus for uncovered zones
                for zname, zrange in zones.items():
                    if num in zrange and zone_recent[zname] < 0.5:
                        scores[num] += (1 - zone_recent[zname]) * 5
                
                # Bonus for not recently seen
                if num not in all_recent:
                    gap = n_draws
                    for i in range(n_draws-1, max(0, n_draws-50), -1):
                        if num in history[i]:
                            gap = n_draws - 1 - i
                            break
                    scores[num] += min(gap * 0.3, 5)
                
                # Odd/Even balance target
                # Diversity: prefer mixed odd/even
                
                # Anti-repeat
                if num in history[-1]:
                    scores[num] -= 5
            
            candidates = sorted(scores, key=lambda x: -scores[x])
            
            # Greedy selection with balance constraints
            selected = []
            odd_count = 0
            for n in candidates:
                if len(selected) >= self.pick_count:
                    break
                # Try to maintain odd/even balance (3/3)
                is_odd = n % 2 == 1
                if is_odd and odd_count >= 4:
                    continue
                if not is_odd and (len(selected) - odd_count) >= 4:
                    continue
                selected.append(n)
                if is_odd:
                    odd_count += 1
            
            while len(selected) < self.pick_count:
                for n in range(1, self.max_number + 1):
                    if n not in selected:
                        selected.append(n)
                        break
            
            return sorted(selected[:self.pick_count])
        
        bt = self._bt(predict_fn)
        return {
            'name': 'Combinatorial Coverage Optimizer',
            'is_pattern': bt['improvement'] > 15,
            **bt,
            'conclusion': f"Coverage: {bt['avg_matches']:.4f}/6 ({bt['improvement']:+.2f}%)"
        }
    
    # ===== DEEP PREDICTION =====
    def _deep_prediction(self):
        """Generate the best prediction combining all 10 Phase 6 strategies."""
        history = self.data
        n_draws = len(history)
        flat = self.flat
        last = set(history[-1])
        
        scores = Counter()
        
        # Signal 1: Bayesian posterior
        prior = np.ones(self.max_number + 1)
        freq_30 = Counter(n for d in history[-30:] for n in d)
        for n in range(1, self.max_number + 1):
            scores[n] += (freq_30.get(n, 0) / (30 * self.pick_count)) * 5
        
        # Signal 2: Regime-based
        for num in range(1, self.max_number + 1):
            freq_r = sum(1 for d in history[-15:] if num in d) / 15
            freq_o = sum(1 for d in history[-45:-15] if num in d) / 30 if n_draws > 45 else freq_r
            trend = freq_r - freq_o
            if trend > 0.02:
                scores[num] += trend * 15
        
        # Signal 3: Gap overdue
        last_seen = {}
        for i, d in enumerate(history):
            for n in d:
                last_seen[n] = i
        exp_gap = self.max_number / self.pick_count
        for n in range(1, self.max_number + 1):
            gap = n_draws - last_seen.get(n, 0)
            if gap > exp_gap * 1.2:
                scores[n] += (gap / exp_gap - 1) * 3
        
        # Signal 4: Run-length turning point
        for num in range(1, self.max_number + 1):
            current_absence = 0
            for d in reversed(history):
                if num not in d:
                    current_absence += 1
                else:
                    break
            if current_absence > exp_gap * 0.8:
                scores[num] += min(current_absence * 0.3, 4)
        
        # Signal 5: KNN conditional
        for i in range(len(history) - 2):
            overlap = len(set(history[i]) & last)
            if overlap >= 3:
                for n in history[i+1]:
                    scores[n] += overlap * 0.5
        
        # Signal 6: Position hints
        for pos in range(self.pick_count):
            vals = [sorted(d)[pos] for d in history[-60:] if len(d) >= pos+1]
            for v, c in Counter(vals).most_common(3):
                scores[v] += c * 0.1
        
        # Signal 7: Pair network
        pair_scores = Counter()
        for d in history[-50:]:
            for pair in combinations(sorted(d), 2):
                pair_scores[pair] += 1
        for n in last:
            for pair, c in pair_scores.most_common(100):
                if n in pair:
                    partner = pair[0] if pair[1] == n else pair[1]
                    if partner not in last:
                        scores[partner] += c * 0.08
        
        # Signal 8: Momentum
        if n_draws > 20:
            r10 = Counter(n for d in history[-10:] for n in d)
            p10 = Counter(n for d in history[-20:-10] for n in d)
            for n in range(1, self.max_number + 1):
                mom = r10.get(n, 0) - p10.get(n, 0)
                if mom > 0:
                    scores[n] += mom * 1.0
        
        # Signal 9: Frequency correction
        total_freq = Counter(flat)
        expected = len(flat) / self.max_number
        for n in range(1, self.max_number + 1):
            dev = (expected - total_freq.get(n, 0)) / max(1, expected)
            if dev > 0:
                scores[n] += dev * 1.5
        
        # Signal 10: Anti-repeat (strong)
        for n in last:
            scores[n] -= 8
        
        # Select top with diversity
        top = scores.most_common(self.pick_count * 3)
        selected = []
        for n, s in top:
            if len(selected) >= self.pick_count:
                break
            selected.append(n)
        
        numbers = sorted(selected[:self.pick_count])
        
        # Score distribution for UI
        all_scores = sorted(scores.items(), key=lambda x: -x[1])
        max_score = max(s for _, s in all_scores[:20]) if all_scores else 1
        score_dist = [{'number': int(n), 'score': round(float(s), 2), 
                       'confidence': round(s / max(max_score, 1) * 100, 1)} 
                      for n, s in all_scores[:15]]
        
        return {
            'numbers': numbers,
            'method': 'Deep Intelligence (10 novel signals fused)',
            'score_distribution': score_dist,
            'signals_used': [
                'bayesian_posterior', 'regime_detection', 'gap_overdue',
                'run_length_turning', 'knn_conditional', 'position_hints',
                'pair_network', 'momentum', 'freq_correction', 'anti_repeat'
            ]
        }
    
    def _verdict(self, results):
        strategies = []
        evidence = []
        p_count = 0
        total = 0
        
        for key, val in results.items():
            if key in ('verdict', 'next_prediction'):
                continue
            if isinstance(val, dict) and 'avg_matches' in val:
                total += 1
                is_good = val.get('improvement', 0) > 15
                if is_good:
                    p_count += 1
                strategies.append({
                    'name': val.get('name', key),
                    'avg': val['avg_matches'],
                    'improvement': val.get('improvement', 0),
                    'max': val.get('max_matches', 0),
                    'match_3plus': val.get('match_3plus', 0),
                    'match_4plus': val.get('match_4plus', 0),
                })
                tag = '+' if is_good else '-'
                evidence.append(f"{tag} {val.get('name',key)}: {val.get('conclusion','')}")
        
        strategies.sort(key=lambda x: -x['avg'])
        score = round(p_count / total * 100, 1) if total > 0 else 0
        best = strategies[0] if strategies else {}
        
        if score >= 50:
            verdict = f"BREAKTHROUGH! {p_count} deep strategies beat random significantly!"
        elif score >= 20:
            verdict = f"Promising - Best: {best.get('name','')} at {best.get('avg',0)}/6 ({best.get('improvement',0):+.1f}%)"
        else:
            verdict = f"Best deep model: {best.get('name','')} at {best.get('avg',0)}/6 ({best.get('improvement',0):+.1f}%)"
        
        return {
            'score': score,
            'pattern_count': p_count,
            'total_tests': total,
            'verdict': verdict,
            'evidence': evidence,
            'strategy_ranking': strategies,
            'best_strategy': best
        }
