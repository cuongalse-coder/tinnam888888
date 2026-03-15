"""
Phase 3: Forensic Deep Dive - "Thủ thuật" phát hiện bất thường
================================================================
10 kỹ thuật pháp y số học chuyên dùng để phát hiện gian lận,
thiên lệch máy quay, lỗi phần mềm RNG.

1.  Benford's Law - Luật chữ số đầu tiên (phát hiện data giả)
2.  Birthday Spacing Test - Khoảng cách trùng lặp  
3.  Streak Analysis - Chuỗi nóng/lạnh bất thường
4.  Digit Decomposition - Tách hàng chục/đơn vị phân tích riêng
5.  Position Bias - Thiên lệch theo vị trí (n1 luôn nhỏ, n6 luôn lớn)
6.  Consecutive Number Anomaly - Số liên tiếp xuất hiện quá nhiều?
7.  Pair/Triple Hotspot - Cặp/bộ ba xuất hiện bất thường
8.  Sum Distribution vs Theory - So sánh phân bố tổng với lý thuyết
9.  Coupon Collector - Tốc độ phủ hết các số
10. RNG Seed Hunter - Thử dò seed bằng brute force
"""
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
import math
import warnings
warnings.filterwarnings('ignore')


class ForensicAnalyzer:
    """Forensic tricks to find non-random evidence."""
    
    def __init__(self, max_number, pick_count):
        self.max_number = max_number
        self.pick_count = pick_count
    
    def analyze(self, data):
        self.data = data
        self.flat = [n for d in data for n in d[:self.pick_count]]
        
        results = {}
        print("[Phase 3 Forensic] Running 10 forensic analyses...")
        
        results['benford'] = self._benford_law()
        print("  [1/10] Benford's Law complete")
        
        results['birthday'] = self._birthday_spacing()
        print("  [2/10] Birthday Spacing complete")
        
        results['streaks'] = self._streak_analysis()
        print("  [3/10] Streak Analysis complete")
        
        results['digits'] = self._digit_decomposition()
        print("  [4/10] Digit Decomposition complete")
        
        results['position_bias'] = self._position_bias()
        print("  [5/10] Position Bias complete")
        
        results['consecutive'] = self._consecutive_anomaly()
        print("  [6/10] Consecutive Anomaly complete")
        
        results['pairs'] = self._pair_hotspot()
        print("  [7/10] Pair Hotspot complete")
        
        results['sum_theory'] = self._sum_vs_theory()
        print("  [8/10] Sum vs Theory complete")
        
        results['coupon'] = self._coupon_collector()
        print("  [9/10] Coupon Collector complete")
        
        results['seed_hunt'] = self._seed_hunter()
        print("  [10/10] Seed Hunter complete")
        
        results['verdict'] = self._verdict(results)
        print("[Phase 3 Forensic] Complete!")
        return results
    
    # ===== 1. BENFORD'S LAW =====
    def _benford_law(self):
        """
        Benford's Law: In natural data, leading digit '1' appears ~30.1%, '2' ~17.6%, etc.
        Deviation from Benford = data may be fabricated/algorithmically generated.
        """
        # Leading digits of all numbers
        leading = [int(str(n)[0]) for n in self.flat if n > 0]
        freq = Counter(leading)
        total = len(leading)
        
        # Benford expected distribution
        benford_expected = {d: math.log10(1 + 1/d) for d in range(1, 10)}
        
        # Chi-square test
        chi_sq = 0
        comparison = {}
        for d in range(1, 10):
            observed = freq.get(d, 0) / total if total > 0 else 0
            expected = benford_expected[d]
            if expected > 0:
                chi_sq += total * (observed - expected) ** 2 / expected
            comparison[str(d)] = {
                'observed_pct': round(observed * 100, 1),
                'benford_pct': round(expected * 100, 1),
                'deviation': round((observed - expected) * 100, 1)
            }
        
        # For lottery numbers 1-45/55, Benford doesn't perfectly apply
        # but deviations can still be informative
        # Critical chi-square for df=8 at 95% = 15.51
        is_significant = chi_sq > 15.51
        
        # Also check: for uniform [1, max_number], what would we expect?
        uniform_leading = Counter()
        for n in range(1, self.max_number + 1):
            uniform_leading[int(str(n)[0])] += 1
        uniform_total = sum(uniform_leading.values())
        
        uniform_chi = 0
        for d in range(1, 10):
            obs = freq.get(d, 0) / total if total > 0 else 0
            exp = uniform_leading.get(d, 0) / uniform_total if uniform_total > 0 else 0
            if exp > 0:
                uniform_chi += total * (obs - exp) ** 2 / exp
        
        return {
            'name': "Benford's Law (Leading Digit)",
            'is_pattern': uniform_chi > 20,
            'benford_chi_square': round(chi_sq, 2),
            'uniform_chi_square': round(uniform_chi, 2),
            'comparison': comparison,
            'conclusion': f"Benford χ²={chi_sq:.1f}, Uniform χ²={uniform_chi:.1f}. " +
                         ('SIGNIFICANT deviation from expected distribution!' if uniform_chi > 20
                          else 'Distribution consistent with uniform random.')
        }
    
    # ===== 2. BIRTHDAY SPACING =====
    def _birthday_spacing(self):
        """
        Birthday Spacing Test: Analyze gaps between sorted numbers within each draw.
        PRNG often has characteristic spacing patterns.
        """
        spacings = []
        for draw in self.data:
            s = sorted(draw[:self.pick_count])
            for i in range(1, len(s)):
                spacings.append(s[i] - s[i-1])
        
        spacing_freq = Counter(spacings)
        total = len(spacings)
        
        # Expected spacing for uniform random selection
        # For k numbers from {1..n}, expected avg spacing ≈ n/(k+1)
        expected_avg = self.max_number / (self.pick_count + 1)
        actual_avg = float(np.mean(spacings))
        
        # Distribution of spacings
        spacing_dist = {}
        for gap in sorted(spacing_freq.keys())[:20]:
            spacing_dist[str(gap)] = {
                'count': int(spacing_freq[gap]),
                'pct': round(spacing_freq[gap] / total * 100, 1)
            }
        
        # Chi-square for spacing distribution
        # Under uniform, spacing follows geometric-like distribution
        chi_sq = 0
        for gap in range(1, min(self.max_number, 25)):
            observed = spacing_freq.get(gap, 0)
            # Approximate expected (geometric-like)
            p = self.pick_count / self.max_number
            expected = total * p * (1 - p) ** (gap - 1)
            if expected > 1:
                chi_sq += (observed - expected) ** 2 / expected
        
        deviation = abs(actual_avg - expected_avg) / expected_avg * 100
        
        # Check for unusually common spacings
        max_concentration = max(spacing_freq.values()) / total * 100 if spacing_freq else 0
        
        return {
            'name': 'Birthday Spacing Test',
            'is_pattern': deviation > 5 or max_concentration > 25,
            'expected_avg_spacing': round(expected_avg, 2),
            'actual_avg_spacing': round(actual_avg, 2),
            'deviation_pct': round(deviation, 1),
            'spacing_distribution': spacing_dist,
            'max_concentration': round(max_concentration, 1),
            'chi_square': round(chi_sq, 2),
            'conclusion': f'Avg spacing: {actual_avg:.2f} (expected: {expected_avg:.2f}, dev: {deviation:.1f}%). ' +
                         ('SPACING ANOMALY detected!' if deviation > 5 or max_concentration > 25
                          else 'Spacing consistent with random.')
        }
    
    # ===== 3. STREAK ANALYSIS =====
    def _streak_analysis(self):
        """
        Analyze hot/cold streaks for each number.
        In random data, max streak length follows a predictable distribution.
        Unusually long streaks = non-random.
        """
        # For each number, find: appearances in consecutive draws
        number_presence = defaultdict(list)
        for i, draw in enumerate(self.data):
            draw_set = set(draw[:self.pick_count])
            for n in range(1, self.max_number + 1):
                number_presence[n].append(1 if n in draw_set else 0)
        
        # Find streak statistics
        streaks = {}
        max_hot_streak = 0
        max_cold_streak = 0
        max_hot_num = 0
        max_cold_num = 0
        
        anomalous = []
        
        for n in range(1, self.max_number + 1):
            seq = number_presence[n]
            
            # Hot streaks (consecutive appearances)
            hot_max = 0
            current = 0
            for v in seq:
                if v == 1:
                    current += 1
                    hot_max = max(hot_max, current)
                else:
                    current = 0
            
            # Cold streaks (consecutive absences)
            cold_max = 0
            current = 0
            for v in seq:
                if v == 0:
                    current += 1
                    cold_max = max(cold_max, current)
                else:
                    current = 0
            
            if hot_max > max_hot_streak:
                max_hot_streak = hot_max
                max_hot_num = n
            if cold_max > max_cold_streak:
                max_cold_streak = cold_max
                max_cold_num = n
            
            # Expected max hot streak for p=6/45 over N draws
            p = self.pick_count / self.max_number
            expected_max_hot = math.log(len(seq)) / abs(math.log(p)) if p > 0 and p < 1 else 0
            expected_max_cold = math.log(len(seq)) / abs(math.log(1 - p)) if p < 1 else 0
            
            if hot_max > expected_max_hot * 2 or cold_max > expected_max_cold * 1.5:
                anomalous.append({
                    'number': int(n),
                    'hot_streak': int(hot_max),
                    'cold_streak': int(cold_max),
                    'expected_hot': round(expected_max_hot, 1),
                    'expected_cold': round(expected_max_cold, 1)
                })
        
        p = self.pick_count / self.max_number
        exp_hot = math.log(len(self.data)) / abs(math.log(p)) if p > 0 and p < 1 else 0
        exp_cold = math.log(len(self.data)) / abs(math.log(1 - p)) if p < 1 else 0
        
        return {
            'name': 'Streak Analysis (Hot/Cold)',
            'is_pattern': len(anomalous) > 3,
            'max_hot_streak': {'number': int(max_hot_num), 'length': int(max_hot_streak), 'expected': round(exp_hot, 1)},
            'max_cold_streak': {'number': int(max_cold_num), 'length': int(max_cold_streak), 'expected': round(exp_cold, 1)},
            'anomalous_numbers': anomalous[:10],
            'num_anomalous': len(anomalous),
            'conclusion': f'Max hot streak: {max_hot_streak} (expected ~{exp_hot:.0f}). '
                         f'Max cold streak: {max_cold_streak} (expected ~{exp_cold:.0f}). '
                         f'{len(anomalous)} anomalous numbers. ' +
                         ('STREAK ANOMALY detected!' if len(anomalous) > 3
                          else 'Streaks within expected range.')
        }
    
    # ===== 4. DIGIT DECOMPOSITION =====
    def _digit_decomposition(self):
        """
        Separate tens digit and units digit, analyze independently.
        If RNG has modular bias, it shows here.
        """
        tens = [n // 10 for n in self.flat]
        units = [n % 10 for n in self.flat]
        
        tens_freq = Counter(tens)
        units_freq = Counter(units)
        total = len(self.flat)
        
        # Expected for uniform [1, max_number]
        expected_tens = Counter()
        expected_units = Counter()
        for n in range(1, self.max_number + 1):
            expected_tens[n // 10] += 1
            expected_units[n % 10] += 1
        
        exp_total = sum(expected_tens.values())
        
        # Chi-square for tens
        chi_tens = 0
        tens_detail = {}
        for t in sorted(set(list(tens_freq.keys()) + list(expected_tens.keys()))):
            obs = tens_freq.get(t, 0) / total
            exp = expected_tens.get(t, 0) / exp_total
            if exp > 0:
                chi_tens += total * (obs - exp) ** 2 / exp
            tens_detail[str(t)] = {
                'observed_pct': round(obs * 100, 1),
                'expected_pct': round(exp * 100, 1),
                'deviation': round((obs - exp) * 100, 1)
            }
        
        # Chi-square for units
        chi_units = 0
        units_detail = {}
        for u in range(10):
            obs = units_freq.get(u, 0) / total
            exp = expected_units.get(u, 0) / exp_total
            if exp > 0:
                chi_units += total * (obs - exp) ** 2 / exp
            units_detail[str(u)] = {
                'observed_pct': round(obs * 100, 1),
                'expected_pct': round(exp * 100, 1),
                'deviation': round((obs - exp) * 100, 1)
            }
        
        return {
            'name': 'Digit Decomposition (Tens/Units)',
            'is_pattern': chi_tens > 15 or chi_units > 18,
            'tens_chi_square': round(chi_tens, 2),
            'units_chi_square': round(chi_units, 2),
            'tens_breakdown': tens_detail,
            'units_breakdown': units_detail,
            'conclusion': f'Tens χ²={chi_tens:.1f}, Units χ²={chi_units:.1f}. ' +
                         ('DIGIT BIAS detected!' if chi_tens > 15 or chi_units > 18
                          else 'Digit distribution consistent with uniform.')
        }
    
    # ===== 5. POSITION BIAS =====
    def _position_bias(self):
        """
        In sorted draws, n1 is always smallest, n6 always largest.
        Check if the DISTRIBUTION at each position matches theoretical.
        If machine has physical bias, specific positions will be skewed.
        """
        position_data = defaultdict(list)
        for draw in self.data:
            s = sorted(draw[:self.pick_count])
            for i, v in enumerate(s):
                position_data[i].append(v)
        
        # Theoretical expected value for position k (order statistics of uniform)
        # E[X_(k:n)] = k * (N+1) / (n+1) where N=max_number, n=pick_count
        stats = {}
        max_dev = 0
        
        for pos in range(self.pick_count):
            vals = position_data[pos]
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            
            # Expected for order statistic
            expected_mean = (pos + 1) * (self.max_number + 1) / (self.pick_count + 1)
            expected_std = math.sqrt(
                (pos + 1) * (self.pick_count - pos) * (self.max_number + 1) ** 2 
                / ((self.pick_count + 1) ** 2 * (self.pick_count + 2))
            )
            
            dev = abs(mean - expected_mean)
            dev_sigma = dev / (expected_std / math.sqrt(len(vals))) if expected_std > 0 else 0
            max_dev = max(max_dev, dev_sigma)
            
            stats[f'n{pos+1}'] = {
                'mean': round(mean, 2),
                'expected_mean': round(expected_mean, 2),
                'deviation': round(dev, 2),
                'deviation_sigma': round(dev_sigma, 2),
                'std': round(std, 2),
                'expected_std': round(expected_std, 2),
                'mode': int(Counter(vals).most_common(1)[0][0])
            }
        
        return {
            'name': 'Position Bias Analysis',
            'is_pattern': max_dev > 3.0,
            'position_stats': stats,
            'max_deviation_sigma': round(max_dev, 2),
            'conclusion': f'Max position deviation: {max_dev:.2f}σ. ' +
                         ('POSITION BIAS detected!' if max_dev > 3.0
                          else 'Positions consistent with order statistics theory.')
        }
    
    # ===== 6. CONSECUTIVE NUMBER ANOMALY =====
    def _consecutive_anomaly(self):
        """
        Check if consecutive numbers (e.g., 14-15, 22-23) appear more/less
        than expected in random draws.
        """
        consec_counts = []
        for draw in self.data:
            s = sorted(draw[:self.pick_count])
            consec = 0
            for i in range(1, len(s)):
                if s[i] == s[i-1] + 1:
                    consec += 1
            consec_counts.append(consec)
        
        avg_consec = float(np.mean(consec_counts))
        
        # Theoretical expected consecutive pairs in random draw
        # P(two specific adjacent numbers both drawn) ≈ C(n-2, k-2) / C(n, k)
        n, k = self.max_number, self.pick_count
        # Number of possible consecutive pairs: n-1
        # P(specific pair drawn) = C(n-2, k-2) / C(n, k) = k*(k-1) / (n*(n-1))
        p_pair = k * (k - 1) / (n * (n - 1))
        expected_consec = (n - 1) * p_pair
        
        deviation = abs(avg_consec - expected_consec) / expected_consec * 100 if expected_consec > 0 else 0
        
        consec_dist = Counter(consec_counts)
        
        # Draws with many consecutives (3+)
        many_consec = [i for i, c in enumerate(consec_counts) if c >= 3]
        
        return {
            'name': 'Consecutive Number Anomaly',
            'is_pattern': deviation > 15,
            'avg_consecutive_pairs': round(avg_consec, 3),
            'expected_consecutive': round(expected_consec, 3),
            'deviation_pct': round(deviation, 1),
            'distribution': {str(k): int(v) for k, v in sorted(consec_dist.items())},
            'draws_with_3plus': len(many_consec),
            'conclusion': f'Avg consecutive: {avg_consec:.3f} (expected: {expected_consec:.3f}, dev: {deviation:.1f}%). ' +
                         ('CONSECUTIVE ANOMALY!' if deviation > 15
                          else 'Consecutive frequency within expected range.')
        }
    
    # ===== 7. PAIR/TRIPLE HOTSPOT =====
    def _pair_hotspot(self):
        """
        Find number pairs that appear together more often than expected.
        If RNG is biased, certain pairs will be "linked".
        """
        pair_counts = Counter()
        for draw in self.data:
            s = sorted(draw[:self.pick_count])
            for pair in combinations(s, 2):
                pair_counts[pair] += 1
        
        total_draws = len(self.data)
        # Expected frequency for any pair: C(n-2, k-2) / C(n, k) * total
        expected = total_draws * self.pick_count * (self.pick_count - 1) / (self.max_number * (self.max_number - 1))
        
        # Find outliers
        pair_values = list(pair_counts.values())
        if pair_values:
            mean_freq = float(np.mean(pair_values))
            std_freq = float(np.std(pair_values))
        else:
            mean_freq = std_freq = 0
        
        hottest = []
        coldest = []
        
        for pair, count in pair_counts.most_common(10):
            sigma = (count - mean_freq) / std_freq if std_freq > 0 else 0
            hottest.append({
                'pair': [int(pair[0]), int(pair[1])],
                'count': int(count),
                'expected': round(expected, 1),
                'sigma': round(sigma, 2)
            })
        
        # Find pairs that never or rarely appear
        all_possible = set(combinations(range(1, self.max_number + 1), 2))
        appeared = set(pair_counts.keys())
        never_appeared = len(all_possible) - len(appeared)
        
        # Find most over-expected
        max_sigma = max((c - mean_freq) / std_freq for c in pair_values) if pair_values and std_freq > 0 else 0
        
        return {
            'name': 'Pair Hotspot Analysis',
            'is_pattern': max_sigma > 4.0,
            'hottest_pairs': hottest,
            'expected_pair_freq': round(expected, 2),
            'actual_mean_freq': round(mean_freq, 2),
            'max_sigma': round(max_sigma, 2),
            'pairs_never_appeared': int(never_appeared),
            'total_possible_pairs': len(all_possible),
            'conclusion': f'Hottest pair: {max_sigma:.1f}σ above mean. '
                         f'{never_appeared} pairs never appeared. ' +
                         ('PAIR HOTSPOT detected!' if max_sigma > 4.0
                          else 'Pair frequencies within expected variation.')
        }
    
    # ===== 8. SUM VS THEORY =====
    def _sum_vs_theory(self):
        """
        Compare actual sum distribution to theoretical.
        For 6 numbers from {1..45}, the sum follows a near-normal distribution.
        """
        sums = [sum(d[:self.pick_count]) for d in self.data]
        
        # Theoretical: mean = k*(N+1)/2, var = k*(N-1)(N+1)/(12)
        # But for selection without replacement, adjusted
        n, k = self.max_number, self.pick_count
        expected_mean = k * (n + 1) / 2
        expected_var = k * (n - k) * (n + 1) / (12)
        expected_std = math.sqrt(expected_var)
        
        actual_mean = float(np.mean(sums))
        actual_std = float(np.std(sums))
        
        # Chi-square for binned distribution
        bins = 15
        hist, edges = np.histogram(sums, bins=bins)
        
        # Compare with normal
        from_normal = 0
        total = len(sums)
        for i in range(bins):
            lo, hi = edges[i], edges[i + 1]
            # Expected from normal
            z_lo = (lo - expected_mean) / expected_std
            z_hi = (hi - expected_mean) / expected_std
            # Approximate with error function
            exp_prob = (self._phi(z_hi) - self._phi(z_lo))
            expected_count = exp_prob * total
            if expected_count > 1:
                from_normal += (hist[i] - expected_count) ** 2 / expected_count
        
        mean_dev = abs(actual_mean - expected_mean)
        mean_dev_sigma = mean_dev / (expected_std / math.sqrt(len(sums)))
        
        # Skewness and kurtosis
        z_scores = [(s - actual_mean) / actual_std for s in sums]
        skewness = float(np.mean([z ** 3 for z in z_scores]))
        kurtosis = float(np.mean([z ** 4 for z in z_scores])) - 3  # Excess kurtosis
        
        return {
            'name': 'Sum Distribution vs Theory',
            'is_pattern': from_normal > 25 or mean_dev_sigma > 3,
            'expected_mean': round(expected_mean, 1),
            'actual_mean': round(actual_mean, 1),
            'mean_deviation_sigma': round(mean_dev_sigma, 2),
            'expected_std': round(expected_std, 1),
            'actual_std': round(actual_std, 1),
            'chi_square_vs_normal': round(from_normal, 2),
            'skewness': round(skewness, 4),
            'kurtosis': round(kurtosis, 4),
            'conclusion': f'Mean: {actual_mean:.1f} (exp: {expected_mean:.1f}, {mean_dev_sigma:.1f}σ). '
                         f'Skew: {skewness:.4f}, Kurt: {kurtosis:.4f}. ' +
                         ('SUM DISTRIBUTION ANOMALY!' if from_normal > 25 or mean_dev_sigma > 3
                          else 'Sum distribution matches theory well.')
        }
    
    def _phi(self, x):
        """Approximate standard normal CDF."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    # ===== 9. COUPON COLLECTOR =====
    def _coupon_collector(self):
        """
        How many draws until all N numbers have appeared at least once?
        Theoretical: N * H(N) where H(N) = harmonic number.
        Faster/slower than expected = non-random.
        """
        seen = set()
        draws_to_complete = None
        coverage_history = []
        
        for i, draw in enumerate(self.data):
            for n in draw[:self.pick_count]:
                seen.add(n)
            pct = len(seen) / self.max_number * 100
            coverage_history.append(round(pct, 1))
            
            if len(seen) >= self.max_number and draws_to_complete is None:
                draws_to_complete = i + 1
        
        # Theoretical expected draws
        # E = N/k * H(N) approximately, where k=pick_count
        harmonic = sum(1.0 / i for i in range(1, self.max_number + 1))
        expected_draws = self.max_number * harmonic / self.pick_count
        
        # Half-coverage
        half_coverage = None
        for i, pct in enumerate(coverage_history):
            if pct >= 50 and half_coverage is None:
                half_coverage = i + 1
        
        # 90% coverage
        ninety_coverage = None
        for i, pct in enumerate(coverage_history):
            if pct >= 90 and ninety_coverage is None:
                ninety_coverage = i + 1
        
        deviation = 0
        if draws_to_complete and expected_draws > 0:
            deviation = abs(draws_to_complete - expected_draws) / expected_draws * 100
        
        return {
            'name': 'Coupon Collector Analysis',
            'is_pattern': deviation > 30,
            'draws_to_complete': draws_to_complete,
            'expected_draws': round(expected_draws, 0),
            'deviation_pct': round(deviation, 1),
            'half_coverage_draw': half_coverage,
            'ninety_coverage_draw': ninety_coverage,
            'current_coverage': coverage_history[-1] if coverage_history else 0,
            'milestones': {
                '25pct': next((i+1 for i, p in enumerate(coverage_history) if p >= 25), None),
                '50pct': half_coverage,
                '75pct': next((i+1 for i, p in enumerate(coverage_history) if p >= 75), None),
                '90pct': ninety_coverage,
                '100pct': draws_to_complete
            },
            'conclusion': f'Full coverage at draw {draws_to_complete} (expected: {expected_draws:.0f}, dev: {deviation:.1f}%). ' +
                         ('COVERAGE ANOMALY!' if deviation > 30
                          else 'Coverage rate consistent with random.')
        }
    
    # ===== 10. SEED HUNTER =====
    def _seed_hunter(self):
        """
        Brute-force try common PRNG algorithms with various seeds
        to see if any can reproduce the observed sequence.
        Tests: Python random, LCG variants, xorshift.
        """
        target = self.flat[:30]  # First 30 numbers to match
        
        best_match = 0
        best_seed = None
        best_algo = None
        
        # Test 1: Python's Mersenne Twister
        import random
        for seed in range(0, 10000):
            random.seed(seed)
            generated = [random.randint(1, self.max_number) for _ in range(30)]
            matches = sum(1 for a, b in zip(target, generated) if a == b)
            if matches > best_match:
                best_match = matches
                best_seed = seed
                best_algo = f'Python random.seed({seed})'
        
        # Test 2: Simple LCG variants
        for a in [1103515245, 6364136223846793005, 1664525]:
            for c in [12345, 1, 1013904223]:
                m = 2 ** 32
                for seed in range(0, 1000):
                    state = seed
                    generated = []
                    for _ in range(30):
                        state = (a * state + c) % m
                        generated.append(state % self.max_number + 1)
                    matches = sum(1 for x, y in zip(target, generated) if x == y)
                    if matches > best_match:
                        best_match = matches
                        best_seed = seed
                        best_algo = f'LCG(a={a}, c={c}, m=2^32, seed={seed})'
        
        # Test 3: XORShift
        for seed in range(1, 5000):
            state = seed
            generated = []
            for _ in range(30):
                state ^= (state << 13) & 0xFFFFFFFF
                state ^= (state >> 17)
                state ^= (state << 5) & 0xFFFFFFFF
                generated.append(state % self.max_number + 1)
            matches = sum(1 for x, y in zip(target, generated) if x == y)
            if matches > best_match:
                best_match = matches
                best_seed = seed
                best_algo = f'XORShift(seed={seed})'
        
        # Expected random matches: 30 * (1/max_number) ≈ 0.67 for max=45
        expected_random = 30 / self.max_number
        improvement = best_match / expected_random if expected_random > 0 else 0
        
        return {
            'name': 'RNG Seed Hunter (Brute Force)',
            'is_pattern': best_match >= 5,
            'best_match': int(best_match),
            'out_of': 30,
            'best_algorithm': best_algo,
            'best_seed': best_seed,
            'expected_random_matches': round(expected_random, 2),
            'improvement_over_random': round(float(improvement), 1),
            'seeds_tested': '~16,000 seeds across 3 algorithms',
            'conclusion': f'Best: {best_match}/30 matches ({best_algo}). '
                         f'Random expected: {expected_random:.1f}. ' +
                         ('POSSIBLE SEED FOUND!' if best_match >= 5
                          else 'No seed matches found - not a common PRNG.')
        }
    
    def _verdict(self, results):
        p_count = 0
        total = 0
        evidence = []
        
        for key, val in results.items():
            if key == 'verdict':
                continue
            if isinstance(val, dict) and 'is_pattern' in val:
                total += 1
                if val['is_pattern']:
                    p_count += 1
                    evidence.append(f"+ {val.get('name', key)}: {val.get('conclusion', '')}")
                else:
                    evidence.append(f"- {val.get('name', key)}: {val.get('conclusion', '')}")
        
        score = round(p_count / total * 100, 1) if total > 0 else 0
        
        if score >= 60:
            verdict = "STRONG forensic evidence of non-random generation!"
        elif score >= 30:
            verdict = "SOME forensic anomalies detected - needs deeper investigation"
        else:
            verdict = "CLEAN - No forensic anomalies found"
        
        return {
            'score': score,
            'pattern_count': p_count,
            'total_tests': total,
            'verdict': verdict,
            'evidence': evidence
        }
