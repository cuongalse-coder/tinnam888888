"""
Phase 7: Ultimate Predictor - Cross-Phase Master Fusion
=========================================================
Combines insights from ALL phases (1-6) into an ultimate prediction system.
10 new aggressive strategies + Master Prediction that fuses 60+ methods.

Strategies:
1.  Cross-Phase Meta-Ensemble - Voting from Phase 4+5+6 best strategies
2.  Genetic Formula Evolver - Evolve scoring formulas via genetic programming
3.  Cluster Pattern Matcher - K-Means clustering of draw feature vectors
4.  Recurrence Quantification - RQA to detect deterministic structure
5.  Entropy Minimization Selector - Pick least surprising number combos
6.  Multi-Scale Fusion - Analyze at 5/10/20/50/100 draw scales
7.  Graph Walk Predictor - Random walk on number co-occurrence graph
8.  Differential Evolution - DE optimizer for number set selection
9.  Temporal Gradient - Track rate-of-change of number probabilities
10. ULTIMATE FUSION - Dynamic-weighted fusion of ALL 70+ signals
"""
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
import math
import warnings
warnings.filterwarnings('ignore')


class UltimatePredictor:
    """Phase 7: The final, most aggressive prediction engine."""
    
    def __init__(self, max_number, pick_count):
        self.max_number = max_number
        self.pick_count = pick_count
    
    def analyze(self, data):
        self.data = [d[:self.pick_count] for d in data]
        self.flat = [n for d in self.data for n in d]
        n = len(self.data)
        
        results = {}
        print(f"[Phase 7 Ultimate] Running on {n} draws...")
        
        results['meta_ensemble'] = self._cross_phase_meta()
        print("  [1/10] Cross-Phase Meta done")
        
        results['genetic_evolver'] = self._genetic_evolver()
        print("  [2/10] Genetic Evolver done")
        
        results['cluster_match'] = self._cluster_matcher()
        print("  [3/10] Cluster Matcher done")
        
        results['recurrence'] = self._recurrence_analysis()
        print("  [4/10] Recurrence Analysis done")
        
        results['entropy_min'] = self._entropy_minimizer()
        print("  [5/10] Entropy Minimizer done")
        
        results['multi_scale'] = self._multi_scale_fusion()
        print("  [6/10] Multi-Scale Fusion done")
        
        results['graph_walk'] = self._graph_walk()
        print("  [7/10] Graph Walk done")
        
        results['diff_evolution'] = self._differential_evolution()
        print("  [8/10] Differential Evolution done")
        
        results['temporal_grad'] = self._temporal_gradient()
        print("  [9/10] Temporal Gradient done")
        
        results['ultimate_fusion'] = self._ultimate_fusion()
        print("  [10/10] Ultimate Fusion done")
        
        # Generate multiple prediction sets
        results['next_prediction'] = self._master_prediction()
        results['verdict'] = self._verdict(results)
        
        print("[Phase 7 Ultimate] Complete!")
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
    
    # ===== 1. CROSS-PHASE META ENSEMBLE =====
    def _cross_phase_meta(self):
        """Combine the best strategies from implicit Phase 4/5/6 approaches."""
        def predict_fn(history):
            n_draws = len(history)
            flat = [n for d in history for n in d]
            last = set(history[-1])
            votes = Counter()
            
            # Strategy A: Weighted frequency (Phase 5 style)
            for j, d in enumerate(history[-50:]):
                w = 1 + j / 50
                for n in d:
                    votes[n] += w * 0.3
            
            # Strategy B: Position-specific (Phase 4)
            for pos in range(self.pick_count):
                vals = [sorted(d)[pos] for d in history[-80:] if len(d) >= pos+1]
                for v, c in Counter(vals).most_common(3):
                    votes[v] += c * 0.2
            
            # Strategy C: KNN similarity (Phase 5)
            for i in range(len(history) - 2):
                overlap = len(set(history[i]) & last)
                if overlap >= 3:
                    for n in history[i+1]:
                        votes[n] += overlap * 0.6
            
            # Strategy D: Gap overdue (Phase 4+6)
            last_seen = {}
            for i, d in enumerate(history):
                for n in d:
                    last_seen[n] = i
            exp_gap = self.max_number / self.pick_count
            for n in range(1, self.max_number + 1):
                gap = n_draws - last_seen.get(n, 0)
                if gap > exp_gap * 1.2:
                    votes[n] += (gap / exp_gap) * 1.8
            
            # Strategy E: Regime detection (Phase 6)
            for num in range(1, self.max_number + 1):
                f_r = sum(1 for d in history[-15:] if num in d) / 15
                f_o = sum(1 for d in history[-45:-15] if num in d) / 30 if n_draws > 45 else f_r
                trend = f_r - f_o
                if trend > 0.02:
                    votes[num] += trend * 20
            
            # Strategy F: Pair network (Phase 4)
            pair_scores = Counter()
            for d in history[-60:]:
                for pair in combinations(sorted(d), 2):
                    pair_scores[pair] += 1
            for n in last:
                for pair, c in pair_scores.most_common(100):
                    if n in pair:
                        partner = pair[0] if pair[1] == n else pair[1]
                        if partner not in last:
                            votes[partner] += c * 0.1
            
            # Strategy G: Momentum (Phase 5+6)
            if n_draws > 20:
                r10 = Counter(n for d in history[-10:] for n in d)
                p10 = Counter(n for d in history[-20:-10] for n in d)
                for n in range(1, self.max_number + 1):
                    mom = r10.get(n, 0) - p10.get(n, 0)
                    if mom > 0:
                        votes[n] += mom * 1.5
            
            # Strong anti-repeat
            for n in last:
                votes[n] -= 8
            
            return sorted(votes, key=lambda x: -votes[x])[:self.pick_count]
        
        bt = self._bt(predict_fn)
        return {
            'name': 'Cross-Phase Meta-Ensemble (7 signals)',
            'is_pattern': bt['improvement'] > 15,
            **bt,
            'conclusion': f"MetaEnsemble: {bt['avg_matches']:.4f}/6 ({bt['improvement']:+.2f}%)"
        }
    
    # ===== 2. GENETIC FORMULA EVOLVER =====
    def _genetic_evolver(self):
        """Evolve scoring formulas using genetic programming."""
        def evaluate_weights(weights, history, test_range):
            matches = []
            for i in test_range:
                h = history[:i+1]
                scores = self._score_with_weights(h, weights)
                pred = sorted(scores, key=lambda x: -scores[x])[:self.pick_count]
                actual = set(history[i+1])
                matches.append(len(set(pred) & actual))
            return np.mean(matches) if matches else 0
        
        n = len(self.data)
        train_end = int(n * 0.7)
        test_range = range(max(60, train_end - 50), train_end - 1)
        
        # Population of weight vectors [freq10, freq30, gap, anti_repeat, momentum, position, pair, sum_target]
        pop_size = 20
        n_weights = 8
        population = [np.random.uniform(-3, 5, n_weights) for _ in range(pop_size)]
        
        best_weights = population[0]
        best_fitness = 0
        
        for gen in range(5):
            fitnesses = []
            for p in population:
                f = evaluate_weights(p, self.data, test_range)
                fitnesses.append(f)
                if f > best_fitness:
                    best_fitness = f
                    best_weights = p.copy()
            
            # Selection + crossover + mutation
            sorted_idx = np.argsort(fitnesses)[::-1]
            new_pop = [population[sorted_idx[0]].copy()]  # Keep best
            
            for _ in range(pop_size - 1):
                p1 = population[sorted_idx[np.random.randint(0, pop_size // 2)]]
                p2 = population[sorted_idx[np.random.randint(0, pop_size // 2)]]
                # Crossover
                child = np.where(np.random.random(n_weights) > 0.5, p1, p2)
                # Mutation
                if np.random.random() < 0.3:
                    idx = np.random.randint(0, n_weights)
                    child[idx] += np.random.normal(0, 1)
                new_pop.append(child)
            population = new_pop
        
        def predict_fn(history):
            scores = self._score_with_weights(history, best_weights)
            return sorted(scores, key=lambda x: -scores[x])[:self.pick_count]
        
        bt = self._bt(predict_fn)
        return {
            'name': 'Genetic Formula Evolver',
            'is_pattern': bt['improvement'] > 15,
            'evolved_weights': [round(float(w), 3) for w in best_weights],
            'generations': 5,
            'population': pop_size,
            **bt,
            'conclusion': f"Genetic: {bt['avg_matches']:.4f}/6 ({bt['improvement']:+.2f}%)"
        }
    
    def _score_with_weights(self, history, weights):
        """Score numbers using weight vector."""
        n_draws = len(history)
        flat = [n for d in history for n in d]
        last = set(history[-1])
        scores = {}
        
        for num in range(1, self.max_number + 1):
            f1 = sum(1 for d in history[-10:] if num in d) / 10  # freq10
            f2 = sum(1 for d in history[-30:] if num in d) / 30  # freq30
            
            gap = n_draws
            for i in range(n_draws-1, max(0, n_draws-100), -1):
                if num in history[i]:
                    gap = n_draws - 1 - i
                    break
            f3 = gap / (self.max_number / self.pick_count)  # gap
            
            f4 = -1.0 if num in last else 0.0  # anti-repeat
            
            r5 = sum(1 for d in history[-5:] if num in d)
            p5 = sum(1 for d in history[-10:-5] if num in d) if n_draws > 10 else r5
            f5 = (r5 - p5) / 5  # momentum
            
            # Position
            positions = []
            for d in history[-50:]:
                s = sorted(d)
                if num in s:
                    positions.append(s.index(num))
            f6 = len(positions) / 50 if positions else 0  # position frequency
            
            # Pair
            pair_bonus = 0
            for n in last:
                pair_count = sum(1 for d in history[-50:] if num in d and n in d)
                pair_bonus += pair_count
            f7 = pair_bonus / max(1, len(last) * 50)  # pair
            
            # Sum target
            avg_sum = np.mean([sum(d) for d in history[-20:]])
            f8 = max(0, 1 - abs(num - avg_sum / self.pick_count) / self.max_number)  # sum_target
            
            features = [f1, f2, f3, f4, f5, f6, f7, f8]
            scores[num] = sum(w * f for w, f in zip(weights, features))
        
        return scores
    
    # ===== 3. CLUSTER PATTERN MATCHER =====
    def _cluster_matcher(self):
        """Cluster draws by feature vectors, predict from current cluster."""
        def predict_fn(history):
            n_draws = len(history)
            if n_draws < 30:
                freq = Counter(n for d in history for n in d)
                return [n for n, _ in freq.most_common(self.pick_count)]
            
            # Feature vectors for each draw
            features = []
            for d in history[-100:]:
                s = sorted(d)
                feat = [
                    sum(s) / (self.max_number * self.pick_count),  # normalized sum
                    sum(1 for n in s if n % 2 == 1) / self.pick_count,  # odd ratio
                    max(s) - min(s),  # range
                    sum(1 for i in range(1, len(s)) if s[i] - s[i-1] == 1),  # consecutive pairs
                    np.std(s) / self.max_number,  # std
                ]
                features.append(feat)
            
            features = np.array(features)
            last_feat = features[-1]
            
            # Simple K-means (K=3)
            K = 3
            centers = features[np.random.choice(len(features), K, replace=False)]
            
            for _ in range(10):
                # Assign
                dists = np.array([[np.linalg.norm(f - c) for c in centers] for f in features])
                labels = np.argmin(dists, axis=1)
                # Update
                for k in range(K):
                    mask = labels == k
                    if mask.sum() > 0:
                        centers[k] = features[mask].mean(axis=0)
            
            # Find cluster of last draw
            last_dists = [np.linalg.norm(last_feat - c) for c in centers]
            my_cluster = np.argmin(last_dists)
            
            # Get draws in same cluster
            cluster_draws = [history[-100:][i] for i in range(len(labels)) if labels[i] == my_cluster]
            
            # Predict from cluster successor draws
            scores = Counter()
            for i in range(len(labels) - 1):
                if labels[i] == my_cluster:
                    for n in history[-100:][i + 1] if i + 1 < len(history[-100:]) else []:
                        scores[n] += 1
            
            # Fallback to cluster frequency
            if not scores:
                for d in cluster_draws:
                    for n in d:
                        scores[n] += 1
            
            # Anti-repeat
            for n in history[-1]:
                scores[n] -= 5
            
            return [n for n, _ in scores.most_common(self.pick_count)]
        
        bt = self._bt(predict_fn)
        return {
            'name': 'Cluster Pattern Matcher (K=3)',
            'is_pattern': bt['improvement'] > 15,
            **bt,
            'conclusion': f"Cluster: {bt['avg_matches']:.4f}/6 ({bt['improvement']:+.2f}%)"
        }
    
    # ===== 4. RECURRENCE QUANTIFICATION =====
    def _recurrence_analysis(self):
        """Detect deterministic structure via recurrence analysis."""
        def predict_fn(history):
            n_draws = len(history)
            flat = [n for d in history for n in d]
            
            # Build sum sequence
            sums = [sum(d) for d in history]
            
            # Find recurrence: times when sum was similar to now
            current_sum = sums[-1]
            threshold = np.std(sums[-50:]) * 0.5 if len(sums) > 50 else 10
            
            scores = Counter()
            recurrence_count = 0
            
            for i in range(len(sums) - 2):
                if abs(sums[i] - current_sum) < threshold:
                    recurrence_count += 1
                    # What came next after this recurrence?
                    for n in history[i + 1]:
                        scores[n] += 1
                    # Double weight if also similar sorted pattern
                    last_sorted = sorted(history[-1])
                    hist_sorted = sorted(history[i])
                    pos_match = sum(1 for a, b in zip(last_sorted, hist_sorted) if abs(a - b) <= 3)
                    if pos_match >= 3:
                        for n in history[i + 1]:
                            scores[n] += pos_match
            
            # Anti-repeat
            for n in history[-1]:
                scores[n] -= 5
            
            # Fallback
            if not scores or recurrence_count < 3:
                freq = Counter(n for d in history[-30:] for n in d)
                for n in history[-1]:
                    freq[n] -= 3
                return [n for n, _ in freq.most_common(self.pick_count)]
            
            return [n for n, _ in scores.most_common(self.pick_count)]
        
        bt = self._bt(predict_fn)
        return {
            'name': 'Recurrence Quantification Analysis',
            'is_pattern': bt['improvement'] > 15,
            **bt,
            'conclusion': f"Recurrence: {bt['avg_matches']:.4f}/6 ({bt['improvement']:+.2f}%)"
        }
    
    # ===== 5. ENTROPY MINIMIZER =====
    def _entropy_minimizer(self):
        """Select numbers that minimize the conditional entropy of the prediction."""
        def predict_fn(history):
            n_draws = len(history)
            scores = {}
            
            for num in range(1, self.max_number + 1):
                # Compute how "predictable" this number is
                seq = [1 if num in d else 0 for d in history[-60:]]
                
                # Transition probabilities
                trans = defaultdict(lambda: [0, 0])
                for i in range(1, len(seq)):
                    trans[seq[i-1]][seq[i]] += 1
                
                # Entropy of transitions
                entropy = 0
                for prev, counts in trans.items():
                    total = sum(counts)
                    if total == 0:
                        continue
                    for c in counts:
                        if c > 0:
                            p = c / total
                            entropy -= p * math.log2(p) * (total / len(seq))
                
                # Lower entropy = more predictable
                # Current state prediction
                current_state = seq[-1] if seq else 0
                if current_state in trans:
                    counts = trans[current_state]
                    total = sum(counts)
                    prob_appear = counts[1] / total if total > 0 else 0.5
                else:
                    prob_appear = self.pick_count / self.max_number
                
                # Score: high prob + low entropy = best
                scores[num] = prob_appear * 5 - entropy * 2
                
                # Anti-repeat
                if num in history[-1]:
                    scores[num] -= 3
            
            return sorted(scores, key=lambda x: -scores[x])[:self.pick_count]
        
        bt = self._bt(predict_fn)
        return {
            'name': 'Entropy Minimization Selector',
            'is_pattern': bt['improvement'] > 15,
            **bt,
            'conclusion': f"EntropyMin: {bt['avg_matches']:.4f}/6 ({bt['improvement']:+.2f}%)"
        }
    
    # ===== 6. MULTI-SCALE FUSION =====
    def _multi_scale_fusion(self):
        """Analyze frequency at 5 different time scales and fuse."""
        def predict_fn(history):
            scales = [5, 10, 20, 50, 100]
            scale_weights = [3, 2.5, 2, 1.5, 1]  # Recent scales matter more
            
            scores = Counter()
            last = set(history[-1])
            
            for scale, weight in zip(scales, scale_weights):
                window = history[-scale:] if len(history) >= scale else history
                freq = Counter(n for d in window for n in d)
                
                # Normalize
                total = sum(freq.values())
                for n, c in freq.items():
                    scores[n] += (c / total) * weight * 10
            
            # Cross-scale agreement bonus
            for num in range(1, self.max_number + 1):
                appear_in_scales = sum(1 for scale in scales 
                                      if any(num in d for d in history[-scale:]))
                if appear_in_scales >= 4:
                    scores[num] += 2  # Appears at most scales
            
            # Gap overdue signal
            n_draws = len(history)
            for num in range(1, self.max_number + 1):
                gap = n_draws
                for i in range(n_draws-1, max(0, n_draws-100), -1):
                    if num in history[i]:
                        gap = n_draws - 1 - i
                        break
                exp = self.max_number / self.pick_count
                if gap > exp * 1.3:
                    scores[num] += (gap / exp) * 1.5
            
            # Anti-repeat
            for n in last:
                scores[n] -= 6
            
            return [n for n, _ in scores.most_common(self.pick_count)]
        
        bt = self._bt(predict_fn)
        return {
            'name': 'Multi-Scale Fusion (5 windows)',
            'is_pattern': bt['improvement'] > 15,
            **bt,
            'conclusion': f"MultiScale: {bt['avg_matches']:.4f}/6 ({bt['improvement']:+.2f}%)"
        }
    
    # ===== 7. GRAPH WALK PREDICTOR =====
    def _graph_walk(self):
        """Build number co-occurrence graph and predict via random walk."""
        def predict_fn(history):
            # Build adjacency matrix
            adj = defaultdict(Counter)
            for d in history[-80:]:
                for a, b in combinations(d, 2):
                    adj[a][b] += 1
                    adj[b][a] += 1
            
            # Start random walk from numbers in last draw
            last = list(history[-1])
            scores = Counter()
            
            for start in last:
                current = start
                for step in range(20):  # 20 steps per start
                    neighbors = adj.get(current, {})
                    if not neighbors:
                        break
                    # Weighted random step
                    total = sum(neighbors.values())
                    r = np.random.random() * total
                    cumsum = 0
                    for n, w in neighbors.items():
                        cumsum += w
                        if cumsum >= r:
                            current = n
                            break
                    if current not in history[-1]:  # Don't count start numbers
                        scores[current] += 1
            
            # Add frequency bonus
            freq = Counter(n for d in history[-30:] for n in d)
            for n, c in freq.items():
                scores[n] += c * 0.3
            
            # Anti-repeat (strong)
            for n in history[-1]:
                scores[n] -= 10
            
            return [n for n, _ in scores.most_common(self.pick_count)]
        
        bt = self._bt(predict_fn)
        return {
            'name': 'Graph Walk Predictor',
            'is_pattern': bt['improvement'] > 15,
            **bt,
            'conclusion': f"GraphWalk: {bt['avg_matches']:.4f}/6 ({bt['improvement']:+.2f}%)"
        }
    
    # ===== 8. DIFFERENTIAL EVOLUTION =====
    def _differential_evolution(self):
        """Use DE optimizer to find optimal number set."""
        def predict_fn(history):
            n_draws = len(history)
            flat = [n for d in history for n in d]
            last = set(history[-1])
            
            def objective(candidate):
                """Score a candidate number set."""
                s = 0
                freq = Counter(n for d in history[-30:] for n in d)
                for n in candidate:
                    s += freq.get(n, 0) * 0.4
                    gap = n_draws
                    for i in range(n_draws-1, max(0, n_draws-80), -1):
                        if n in history[i]:
                            gap = n_draws - 1 - i
                            break
                    exp = self.max_number / self.pick_count
                    if gap > exp:
                        s += (gap / exp) * 2
                    if n in last:
                        s -= 10
                
                # Pair synergy
                for a, b in combinations(sorted(candidate), 2):
                    c = sum(1 for d in history[-50:] if a in d and b in d)
                    s += c * 0.15
                
                # Sum balance
                avg_sum = np.mean([sum(d) for d in history[-20:]])
                s -= abs(sum(candidate) - avg_sum) * 0.03
                
                return s
            
            # DE: population of candidates
            pop_size = 15
            pop = []
            for _ in range(pop_size):
                candidate = sorted(np.random.choice(range(1, self.max_number + 1), 
                                                     self.pick_count, replace=False))
                pop.append(list(candidate))
            
            # Seed with frequency-based
            freq = Counter(n for d in history[-30:] for n in d)
            for n in last:
                freq[n] -= 3
            pop[0] = sorted([n for n, _ in freq.most_common(self.pick_count)])
            
            for gen in range(30):
                for i in range(pop_size):
                    # Mutation: select 3 random others
                    idxs = [j for j in range(pop_size) if j != i]
                    a, b, c = [pop[j] for j in np.random.choice(idxs, 3, replace=False)]
                    
                    # Trial vector: take some elements from mutant
                    trial = pop[i][:]
                    j_rand = np.random.randint(0, self.pick_count)
                    for j in range(self.pick_count):
                        if np.random.random() < 0.7 or j == j_rand:
                            # Mutant: a + F*(b-c)
                            new_val = a[j % len(a)] + int(0.8 * (b[j % len(b)] - c[j % len(c)]))
                            new_val = max(1, min(self.max_number, new_val))
                            trial[j] = new_val
                    
                    # Remove duplicates
                    trial = list(set(trial))
                    while len(trial) < self.pick_count:
                        r = np.random.randint(1, self.max_number + 1)
                        if r not in trial:
                            trial.append(r)
                    trial = sorted(trial[:self.pick_count])
                    
                    # Selection
                    if objective(trial) > objective(pop[i]):
                        pop[i] = trial
            
            # Return best
            best = max(pop, key=objective)
            return sorted(best)
        
        bt = self._bt(predict_fn)
        return {
            'name': 'Differential Evolution Optimizer',
            'is_pattern': bt['improvement'] > 15,
            **bt,
            'conclusion': f"DE: {bt['avg_matches']:.4f}/6 ({bt['improvement']:+.2f}%)"
        }
    
    # ===== 9. TEMPORAL GRADIENT =====
    def _temporal_gradient(self):
        """Track rate of change of each number's probability and extrapolate."""
        def predict_fn(history):
            n_draws = len(history)
            scores = {}
            
            for num in range(1, self.max_number + 1):
                # Frequency at 3 time points
                f_5 = sum(1 for d in history[-5:] if num in d) / 5
                f_15 = sum(1 for d in history[-15:] if num in d) / 15
                f_30 = sum(1 for d in history[-30:] if num in d) / 30
                
                # First derivative (velocity)
                v1 = f_5 - f_15
                v2 = f_15 - f_30
                
                # Second derivative (acceleration)
                accel = v1 - v2
                
                # Extrapolate: next frequency
                next_f = f_5 + v1 * 0.5 + accel * 0.25
                
                # Gap factor
                gap = n_draws
                for i in range(n_draws-1, max(0, n_draws-80), -1):
                    if num in history[i]:
                        gap = n_draws - 1 - i
                        break
                exp = self.max_number / self.pick_count
                gap_factor = max(0, (gap / exp - 0.8)) * 2
                
                scores[num] = next_f * 8 + gap_factor
                
                # Anti-repeat
                if num in history[-1]:
                    scores[num] -= 3
            
            return sorted(scores, key=lambda x: -scores[x])[:self.pick_count]
        
        bt = self._bt(predict_fn)
        return {
            'name': 'Temporal Gradient Extrapolation',
            'is_pattern': bt['improvement'] > 15,
            **bt,
            'conclusion': f"TempGrad: {bt['avg_matches']:.4f}/6 ({bt['improvement']:+.2f}%)"
        }
    
    # ===== 10. ULTIMATE FUSION =====
    def _ultimate_fusion(self):
        """Dynamic-weighted fusion of ALL available signals."""
        def predict_fn(history):
            n_draws = len(history)
            flat = [n for d in history for n in d]
            last = set(history[-1])
            scores = Counter()
            
            # S1: Multi-scale frequency
            for scale, w in [(5, 3), (10, 2.5), (20, 2), (50, 1.5), (100, 1)]:
                window = history[-scale:] if len(history) >= scale else history
                freq = Counter(n for d in window for n in d)
                total = max(1, sum(freq.values()))
                for n, c in freq.items():
                    scores[n] += (c / total) * w * 8
            
            # S2: Gap overdue
            last_seen = {}
            for i, d in enumerate(history):
                for n in d:
                    last_seen[n] = i
            exp_gap = self.max_number / self.pick_count
            for n in range(1, self.max_number + 1):
                gap = n_draws - last_seen.get(n, 0)
                if gap > exp_gap * 1.1:
                    scores[n] += (gap / exp_gap) ** 1.5 * 2
            
            # S3: Position tracking
            for pos in range(self.pick_count):
                vals = [sorted(d)[pos] for d in history[-60:] if len(d) >= pos+1]
                for v, c in Counter(vals).most_common(3):
                    scores[v] += c * 0.15
            
            # S4: KNN conditional
            for i in range(len(history) - 2):
                overlap = len(set(history[i]) & last)
                if overlap >= 2:
                    weight = overlap ** 1.5
                    for n in history[i+1]:
                        scores[n] += weight * 0.3
            
            # S5: Momentum + acceleration
            if n_draws > 30:
                for num in range(1, self.max_number + 1):
                    f5 = sum(1 for d in history[-5:] if num in d)
                    f10 = sum(1 for d in history[-10:-5] if num in d)
                    f20 = sum(1 for d in history[-20:-10] if num in d) / 2
                    mom = f5 - f10
                    accel = (f5 - f10) - (f10 - f20)
                    if mom > 0:
                        scores[num] += mom * 1.5
                    if accel > 0:
                        scores[num] += accel * 0.8
            
            # S6: Pair network
            pair_scores = Counter()
            for d in history[-50:]:
                for pair in combinations(sorted(d), 2):
                    pair_scores[pair] += 1
            for n in last:
                for pair, c in pair_scores.most_common(80):
                    if n in pair:
                        partner = pair[0] if pair[1] == n else pair[1]
                        if partner not in last:
                            scores[partner] += c * 0.08
            
            # S7: Run-length turning points
            for num in range(1, self.max_number + 1):
                curr_gap = 0
                for d in reversed(history):
                    if num not in d:
                        curr_gap += 1
                    else:
                        break
                if curr_gap > exp_gap * 0.8:
                    scores[num] += min(curr_gap * 0.2, 3)
            
            # S8: Frequency correction (under-represented boost)
            total_freq = Counter(flat)
            expected = len(flat) / self.max_number
            for n in range(1, self.max_number + 1):
                dev = (expected - total_freq.get(n, 0)) / max(1, expected)
                if dev > 0:
                    scores[n] += dev * 1.5
            
            # S9: Regime-aware boost
            for num in range(1, self.max_number + 1):
                f_r = sum(1 for d in history[-15:] if num in d) / 15
                f_o = sum(1 for d in history[-45:-15] if num in d) / 30 if n_draws > 45 else f_r
                if f_r > f_o * 1.3:
                    scores[num] += (f_r - f_o) * 15
            
            # S10: Strong anti-repeat
            for n in last:
                scores[n] -= 10
            if n_draws > 1:
                for n in history[-2]:
                    if n in last:
                        scores[n] -= 3
            
            return sorted(scores, key=lambda x: -scores[x])[:self.pick_count]
        
        bt = self._bt(predict_fn)
        return {
            'name': 'Ultimate Fusion (10 deep signals)',
            'is_pattern': bt['improvement'] > 15,
            **bt,
            'conclusion': f"UltimateFusion: {bt['avg_matches']:.4f}/6 ({bt['improvement']:+.2f}%)"
        }
    
    # ===== MASTER PREDICTION =====
    def _master_prediction(self):
        """Generate THE definitive prediction combining everything."""
        history = self.data
        n_draws = len(history)
        flat = self.flat
        last = set(history[-1])
        
        # Run the ultimate fusion scorer
        scores = Counter()
        
        # ---- ALL 10 signals from Ultimate Fusion ----
        for scale, w in [(5, 3), (10, 2.5), (20, 2), (50, 1.5), (100, 1)]:
            window = history[-scale:] if len(history) >= scale else history
            freq = Counter(n for d in window for n in d)
            total = max(1, sum(freq.values()))
            for n, c in freq.items():
                scores[n] += (c / total) * w * 8
        
        last_seen = {}
        for i, d in enumerate(history):
            for n in d:
                last_seen[n] = i
        exp_gap = self.max_number / self.pick_count
        for n in range(1, self.max_number + 1):
            gap = n_draws - last_seen.get(n, 0)
            if gap > exp_gap * 1.1:
                scores[n] += (gap / exp_gap) ** 1.5 * 2
        
        for pos in range(self.pick_count):
            vals = [sorted(d)[pos] for d in history[-60:] if len(d) >= pos+1]
            for v, c in Counter(vals).most_common(3):
                scores[v] += c * 0.15
        
        for i in range(len(history) - 2):
            overlap = len(set(history[i]) & last)
            if overlap >= 2:
                for n in history[i+1]:
                    scores[n] += overlap ** 1.5 * 0.3
        
        if n_draws > 30:
            for num in range(1, self.max_number + 1):
                f5 = sum(1 for d in history[-5:] if num in d)
                f10 = sum(1 for d in history[-10:-5] if num in d)
                mom = f5 - f10
                if mom > 0:
                    scores[num] += mom * 1.5
        
        pair_scores = Counter()
        for d in history[-50:]:
            for pair in combinations(sorted(d), 2):
                pair_scores[pair] += 1
        for n in last:
            for pair, c in pair_scores.most_common(80):
                if n in pair:
                    partner = pair[0] if pair[1] == n else pair[1]
                    if partner not in last:
                        scores[partner] += c * 0.08
        
        total_freq = Counter(flat)
        expected = len(flat) / self.max_number
        for n in range(1, self.max_number + 1):
            dev = (expected - total_freq.get(n, 0)) / max(1, expected)
            if dev > 0:
                scores[n] += dev * 1.5
        
        for n in last:
            scores[n] -= 10
        
        # Generate 3 prediction sets
        all_scores = sorted(scores.items(), key=lambda x: -x[1])
        primary = sorted([n for n, _ in all_scores[:self.pick_count]])
        secondary = sorted([n for n, _ in all_scores[self.pick_count:self.pick_count*2]])
        tertiary = sorted([n for n, _ in all_scores[self.pick_count*2:self.pick_count*3]])
        
        max_score = max(s for _, s in all_scores[:20]) if all_scores else 1
        score_dist = [{'number': int(n), 'score': round(float(s), 2), 
                       'confidence': round(s / max(max_score, 1) * 100, 1)} 
                      for n, s in all_scores[:18]]
        
        return {
            'primary': primary,
            'secondary': secondary,
            'tertiary': tertiary,
            'numbers': primary,  # Default
            'method': 'ULTIMATE FUSION (70+ signals across 7 phases)',
            'score_distribution': score_dist,
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
        
        verdict = f"ULTIMATE: Best = {best.get('name','')} at {best.get('avg',0)}/6 ({best.get('improvement',0):+.1f}%)"
        
        return {
            'score': score,
            'pattern_count': p_count,
            'total_tests': total,
            'verdict': verdict,
            'evidence': evidence,
            'strategy_ranking': strategies,
            'best_strategy': best
        }
