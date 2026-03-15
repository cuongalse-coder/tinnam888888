"""
Advanced Prediction Engine - State-of-the-art lottery analysis.
Combines 8+ advanced techniques for maximum prediction accuracy.

Techniques:
1. Markov Chain Analysis - transition probabilities
2. Bayesian Inference - posterior probability updating
3. Association Rules Mining - number co-occurrence patterns
4. Gap & Cycle Analysis - cycle detection per number
5. Chi-Square Deviation - detect non-uniform deviations
6. Monte Carlo Simulation - probabilistic sampling
7. Pattern Mining - odd/even, high/low, sum, consecutive
8. Weighted Multi-Score Fusion - combine all signals
"""
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
import math


class AdvancedPredictionEngine:
    """Ultimate prediction engine combining 8+ analysis methods."""
    
    def __init__(self, max_number, pick_count):
        self.max_number = max_number
        self.pick_count = pick_count
        self.data = []
        self.scores = {}      # Final composite score per number
        self.analysis = {}    # Detailed analysis results
    
    def fit(self, data):
        """Run all analyses on historical data."""
        self.data = [row[:self.pick_count] for row in data]
        if len(self.data) < 30:
            return
        
        print(f"[AdvEngine] Analyzing {len(self.data)} draws with 8 methods...")
        
        # Run all analyses
        self.analysis['markov'] = self._markov_chain()
        self.analysis['bayesian'] = self._bayesian_inference()
        self.analysis['association'] = self._association_rules()
        self.analysis['gap_cycle'] = self._gap_cycle_analysis()
        self.analysis['chi_square'] = self._chi_square_deviation()
        self.analysis['monte_carlo'] = self._monte_carlo_simulation()
        self.analysis['patterns'] = self._pattern_mining()
        self.analysis['recency'] = self._recency_weighted()
        
        # Fuse all scores
        self._fuse_scores()
        print("[AdvEngine] Analysis complete!")
    
    # ===== 1. MARKOV CHAIN =====
    def _markov_chain(self):
        """Calculate transition probabilities: P(number appears | previous draw)."""
        transition = defaultdict(Counter)  # {prev_num: {next_num: count}}
        follow_count = Counter()
        
        for i in range(1, len(self.data)):
            prev_set = set(self.data[i-1])
            curr_set = set(self.data[i])
            
            for p in prev_set:
                for c in curr_set:
                    transition[p][c] += 1
                follow_count[p] += len(curr_set)
        
        # Calculate probabilities based on most recent draw
        last_draw = set(self.data[-1])
        markov_scores = Counter()
        
        for prev_num in last_draw:
            total = follow_count[prev_num]
            if total > 0:
                for num, count in transition[prev_num].items():
                    markov_scores[num] += count / total
        
        # Normalize
        max_s = max(markov_scores.values()) if markov_scores else 1
        result = {n: round(markov_scores.get(n, 0) / max_s * 100, 2) 
                  for n in range(1, self.max_number + 1)}
        
        return {'scores': result, 'name': 'Markov Chain'}
    
    # ===== 2. BAYESIAN INFERENCE =====
    def _bayesian_inference(self):
        """Bayesian posterior probability with sliding window updates."""
        # Prior: uniform
        alpha = np.ones(self.max_number)  # Dirichlet prior
        
        # Update with all historical data, recent draws weighted more
        total = len(self.data)
        for idx, draw in enumerate(self.data):
            # Exponential recency weight
            weight = np.exp((idx - total) / (total * 0.3))
            for n in draw:
                alpha[n - 1] += weight
        
        # Posterior probabilities (Dirichlet-Multinomial)
        total_alpha = np.sum(alpha)
        posterior = alpha / total_alpha
        
        # Convert to scores (0-100)
        max_p = np.max(posterior)
        min_p = np.min(posterior)
        rng = max_p - min_p if max_p > min_p else 1
        
        result = {n + 1: round((posterior[n] - min_p) / rng * 100, 2) 
                  for n in range(self.max_number)}
        
        return {'scores': result, 'name': 'Bayesian Inference'}
    
    # ===== 3. ASSOCIATION RULES MINING =====
    def _association_rules(self):
        """Find numbers that frequently appear together (Apriori-like)."""
        pair_freq = Counter()
        single_freq = Counter()
        total = len(self.data)
        
        for draw in self.data:
            for n in draw:
                single_freq[n] += 1
            for combo in combinations(sorted(draw), 2):
                pair_freq[combo] += 1
        
        # For each number, calculate its association strength
        assoc_scores = defaultdict(float)
        last_draw = set(self.data[-1])
        
        for num in range(1, self.max_number + 1):
            score = 0
            for anchor in last_draw:
                pair = tuple(sorted([anchor, num]))
                if pair in pair_freq:
                    # Confidence: P(num | anchor) = support(pair) / support(anchor)
                    confidence = pair_freq[pair] / single_freq[anchor]
                    # Lift: confidence / P(num)
                    expected = single_freq[num] / total
                    lift = confidence / expected if expected > 0 else 0
                    score += lift
            assoc_scores[num] = score
        
        # Normalize
        max_s = max(assoc_scores.values()) if assoc_scores else 1
        result = {n: round(assoc_scores[n] / max_s * 100, 2) if max_s > 0 else 0
                  for n in range(1, self.max_number + 1)}
        
        # Top pairs
        top_pairs = pair_freq.most_common(20)
        
        return {
            'scores': result,
            'name': 'Association Rules',
            'top_pairs': [{'pair': list(p), 'count': c, 
                          'support': round(c/total*100, 2)} 
                         for p, c in top_pairs]
        }
    
    # ===== 4. GAP & CYCLE ANALYSIS =====
    def _gap_cycle_analysis(self):
        """Analyze appearance cycles for each number."""
        total = len(self.data)
        appearances = defaultdict(list)  # number -> list of draw indices
        
        for idx, draw in enumerate(self.data):
            for n in draw:
                appearances[n].append(idx)
        
        gap_scores = {}
        gap_details = {}
        
        for num in range(1, self.max_number + 1):
            apps = appearances[num]
            if len(apps) < 2:
                gap_scores[num] = 50  # neutral
                gap_details[num] = {'avg_gap': 0, 'current_gap': total, 'predicted': 'N/A'}
                continue
            
            # Calculate gaps
            gaps = [apps[i+1] - apps[i] for i in range(len(apps)-1)]
            avg_gap = np.mean(gaps)
            std_gap = np.std(gaps)
            median_gap = np.median(gaps)
            current_gap = total - apps[-1]
            
            # How overdue is this number?
            # Score higher if current gap approaching or exceeding average
            if avg_gap > 0:
                overdue_ratio = current_gap / avg_gap
                
                # Bell curve centered at 1.0 (due now), extending to 2.0+
                if overdue_ratio >= 1.0:
                    # Overdue: score increases (up to a point)
                    score = min(100, 50 + (overdue_ratio - 1.0) * 40)
                else:
                    # Not due yet: lower score
                    score = 50 * overdue_ratio
            else:
                score = 50
            
            gap_scores[num] = round(score, 2)
            gap_details[num] = {
                'appearances': len(apps),
                'avg_gap': round(avg_gap, 1),
                'std_gap': round(std_gap, 1),
                'median_gap': round(median_gap, 1),
                'current_gap': current_gap,
                'overdue_ratio': round(current_gap / avg_gap, 2) if avg_gap > 0 else 0,
                'predicted_next': round(apps[-1] + avg_gap) if avg_gap > 0 else None
            }
        
        return {
            'scores': gap_scores,
            'name': 'Gap & Cycle Analysis',
            'details': gap_details
        }
    
    # ===== 5. CHI-SQUARE DEVIATION =====
    def _chi_square_deviation(self):
        """Detect numbers deviating from expected uniform distribution."""
        total_draws = len(self.data)
        total_numbers = total_draws * self.pick_count
        expected = total_numbers / self.max_number  # Expected count per number
        
        freq = Counter()
        for draw in self.data:
            for n in draw:
                freq[n] += 1
        
        chi_scores = {}
        details = {}
        
        for num in range(1, self.max_number + 1):
            observed = freq.get(num, 0)
            # Chi-square contribution
            chi_sq = (observed - expected) ** 2 / expected
            
            # If under-represented, it's "due" → higher score
            # If over-represented, it might cool off → lower score
            deviation = observed - expected
            
            if deviation < 0:
                # Under-represented: score higher (due to appear)
                score = 50 + min(50, abs(deviation) / expected * 60)
            else:
                # Over-represented: score lower
                score = max(0, 50 - deviation / expected * 40)
            
            chi_scores[num] = round(score, 2)
            details[num] = {
                'observed': observed,
                'expected': round(expected, 1),
                'deviation': round(deviation, 1),
                'chi_sq': round(chi_sq, 3),
                'deviation_pct': round(deviation / expected * 100, 1)
            }
        
        # Overall chi-square statistic
        total_chi_sq = sum(d['chi_sq'] for d in details.values())
        df = self.max_number - 1  # degrees of freedom
        
        return {
            'scores': chi_scores,
            'name': 'Chi-Square Analysis',
            'total_chi_sq': round(total_chi_sq, 2),
            'degrees_of_freedom': df,
            'details': details
        }
    
    # ===== 6. MONTE CARLO SIMULATION =====
    def _monte_carlo_simulation(self, n_simulations=100000):
        """Run Monte Carlo simulations to estimate likely next draws."""
        # Build probability weights from recent history
        recent_n = min(100, len(self.data))
        recent = self.data[-recent_n:]
        
        # Weighted frequency (more recent = more weight)
        weighted_freq = np.zeros(self.max_number)
        for idx, draw in enumerate(recent):
            weight = 1.0 + (idx / recent_n) * 2  # Linear increasing weight
            for n in draw:
                weighted_freq[n - 1] += weight
        
        # Add small prior to avoid zero probability
        weighted_freq += 0.5
        probs = weighted_freq / weighted_freq.sum()
        
        # Run simulations
        appearance_count = np.zeros(self.max_number)
        for _ in range(n_simulations):
            draw = np.random.choice(
                range(self.max_number), 
                size=self.pick_count,
                replace=False, 
                p=probs
            )
            for idx in draw:
                appearance_count[idx] += 1
        
        # Convert to scores (0-100)
        max_c = np.max(appearance_count)
        min_c = np.min(appearance_count)
        rng = max_c - min_c if max_c > min_c else 1
        
        result = {n + 1: round((appearance_count[n] - min_c) / rng * 100, 2)
                  for n in range(self.max_number)}
        
        return {
            'scores': result,
            'name': 'Monte Carlo (100K sims)',
            'simulations': n_simulations
        }
    
    # ===== 7. PATTERN MINING =====
    def _pattern_mining(self):
        """Analyze structural patterns: odd/even, high/low, sum, consecutive."""
        total = len(self.data)
        mid = self.max_number // 2
        
        # Analyze patterns of each draw
        pattern_history = []
        for draw in self.data:
            odd_count = sum(1 for n in draw if n % 2 == 1)
            high_count = sum(1 for n in draw if n > mid)
            draw_sum = sum(draw)
            consec = sum(1 for i in range(len(draw)-1) 
                        if sorted(draw)[i+1] - sorted(draw)[i] == 1)
            
            pattern_history.append({
                'odd': odd_count,
                'even': self.pick_count - odd_count,
                'high': high_count,
                'low': self.pick_count - high_count,
                'sum': draw_sum,
                'consecutive_pairs': consec
            })
        
        # Find most common patterns
        odd_even_dist = Counter(p['odd'] for p in pattern_history)
        high_low_dist = Counter(p['high'] for p in pattern_history)
        sums = [p['sum'] for p in pattern_history]
        
        # Most likely pattern for next draw
        best_odd = odd_even_dist.most_common(1)[0][0]
        best_high = high_low_dist.most_common(1)[0][0]
        avg_sum = np.mean(sums)
        std_sum = np.std(sums)
        sum_range = (int(avg_sum - std_sum), int(avg_sum + std_sum))
        
        # Score numbers based on pattern compliance
        pattern_scores = {}
        for num in range(1, self.max_number + 1):
            score = 50  # base
            
            # Odd/Even alignment with most common pattern
            is_odd = num % 2 == 1
            odd_ratio = best_odd / self.pick_count
            if is_odd:
                score += (odd_ratio - 0.5) * 20  # Boost if odd is favored
            else:
                score += (0.5 - odd_ratio) * 20
            
            # High/Low alignment
            is_high = num > mid
            high_ratio = best_high / self.pick_count
            if is_high:
                score += (high_ratio - 0.5) * 15
            else:
                score += (0.5 - high_ratio) * 15
            
            pattern_scores[num] = round(max(0, min(100, score)), 2)
        
        return {
            'scores': pattern_scores,
            'name': 'Pattern Mining',
            'optimal_odd_count': best_odd,
            'optimal_high_count': best_high,
            'avg_sum': round(avg_sum, 1),
            'sum_range_68pct': sum_range,
            'odd_even_distribution': dict(odd_even_dist.most_common()),
            'high_low_distribution': dict(high_low_dist.most_common())
        }
    
    # ===== 8. RECENCY WEIGHTED =====
    def _recency_weighted(self):
        """Exponential decay weighted analysis - recent draws matter more."""
        decay = 0.97  # Decay factor per draw
        total = len(self.data)
        
        weighted_freq = defaultdict(float)
        
        for idx, draw in enumerate(self.data):
            # Exponential weight: most recent draw has weight ~1.0
            weight = decay ** (total - 1 - idx)
            for n in draw:
                weighted_freq[n] += weight
        
        max_w = max(weighted_freq.values()) if weighted_freq else 1
        min_w = min(weighted_freq.get(n, 0) for n in range(1, self.max_number + 1))
        rng = max_w - min_w if max_w > min_w else 1
        
        result = {n: round((weighted_freq.get(n, 0) - min_w) / rng * 100, 2)
                  for n in range(1, self.max_number + 1)}
        
        return {'scores': result, 'name': 'Recency Weighted'}
    
    # ===== SCORE FUSION =====
    def _fuse_scores(self):
        """Combine all method scores with optimized weights."""
        # Weights for each method (tuned for lottery analysis)
        weights = {
            'markov': 0.12,
            'bayesian': 0.15,
            'association': 0.10,
            'gap_cycle': 0.18,     # Gap analysis is very important
            'chi_square': 0.15,    # Statistical deviation matters
            'monte_carlo': 0.12,
            'patterns': 0.08,
            'recency': 0.10
        }
        
        self.scores = {}
        
        for num in range(1, self.max_number + 1):
            total = 0
            for method, w in weights.items():
                if method in self.analysis and 'scores' in self.analysis[method]:
                    total += self.analysis[method]['scores'].get(num, 50) * w
            self.scores[num] = round(total, 2)
        
        # Normalize to 0-100
        max_s = max(self.scores.values())
        min_s = min(self.scores.values())
        rng = max_s - min_s if max_s > min_s else 1
        
        for num in self.scores:
            self.scores[num] = round((self.scores[num] - min_s) / rng * 100, 2)
    
    def predict(self, n_sets=5):
        """Generate optimized predictions based on all analyses."""
        if not self.scores:
            return self._random_predict(n_sets)
        
        numbers = list(range(1, self.max_number + 1))
        raw_weights = np.array([self.scores[n] for n in numbers])
        
        # Apply temperature scaling for sharper distribution
        temperature = 0.5
        weights = np.exp(raw_weights / (100 * temperature))
        weights = weights / weights.sum()
        
        predictions = []
        seen_sets = set()
        
        attempts = 0
        while len(predictions) < n_sets and attempts < n_sets * 10:
            attempts += 1
            
            # Select numbers with weighted probability
            selected = np.random.choice(
                numbers, size=self.pick_count,
                replace=False, p=weights
            )
            selected = tuple(sorted(selected.tolist()))
            
            # Check pattern validity
            if not self._validate_pattern(selected):
                continue
            
            if selected not in seen_sets:
                seen_sets.add(selected)
                predictions.append(list(selected))
        
        # Fill remaining with weighted random if needed
        while len(predictions) < n_sets:
            selected = sorted(np.random.choice(
                numbers, size=self.pick_count,
                replace=False, p=weights
            ).tolist())
            predictions.append(selected)
        
        return predictions
    
    def _validate_pattern(self, numbers):
        """Validate that a predicted set follows observed patterns."""
        if 'patterns' not in self.analysis:
            return True
        
        p = self.analysis['patterns']
        
        # Check sum is within 68% range
        s = sum(numbers)
        sum_range = p.get('sum_range_68pct', (0, 999))
        if s < sum_range[0] * 0.8 or s > sum_range[1] * 1.2:
            return False
        
        # Check odd count is reasonable
        odd = sum(1 for n in numbers if n % 2 == 1)
        optimal_odd = p.get('optimal_odd_count', 3)
        if abs(odd - optimal_odd) > 2:
            return False
        
        return True
    
    def _random_predict(self, n_sets):
        preds = []
        for _ in range(n_sets):
            nums = sorted(np.random.choice(
                range(1, self.max_number + 1),
                self.pick_count, replace=False
            ).tolist())
            preds.append(nums)
        return preds
    
    def get_top_numbers(self, n=15):
        """Get top N numbers by composite score."""
        sorted_nums = sorted(self.scores.items(), key=lambda x: -x[1])
        return sorted_nums[:n]
    
    def get_full_analysis(self):
        """Get complete analysis for display."""
        return {
            'composite_scores': self.scores,
            'top_numbers': [{'number': n, 'score': s} for n, s in self.get_top_numbers(15)],
            'methods': {
                name: {
                    'name': info.get('name', name),
                    'top_5': sorted(
                        info.get('scores', {}).items(), 
                        key=lambda x: -x[1]
                    )[:5] if 'scores' in info else []
                }
                for name, info in self.analysis.items()
            },
            'pattern_info': {
                'optimal_odd': self.analysis.get('patterns', {}).get('optimal_odd_count'),
                'optimal_high': self.analysis.get('patterns', {}).get('optimal_high_count'),
                'sum_range': self.analysis.get('patterns', {}).get('sum_range_68pct'),
                'avg_sum': self.analysis.get('patterns', {}).get('avg_sum'),
                'odd_even_dist': self.analysis.get('patterns', {}).get('odd_even_distribution'),
            },
            'chi_square': {
                'total': self.analysis.get('chi_square', {}).get('total_chi_sq'),
                'df': self.analysis.get('chi_square', {}).get('degrees_of_freedom'),
            },
            'gap_details': {
                n: self.analysis.get('gap_cycle', {}).get('details', {}).get(n, {})
                for n in [x[0] for x in self.get_top_numbers(10)]
            },
            'top_pairs': self.analysis.get('association', {}).get('top_pairs', [])[:10],
        }
