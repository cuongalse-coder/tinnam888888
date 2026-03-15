"""
Deep Temporal & Financial Pattern Analyzer
===========================================
Analyzes patterns based on:
1. Day of week - do specific days produce specific numbers?
2. Month / Season - monthly patterns
3. Draw sequence (ID) - sequential patterns
4. Jackpot amounts - does money flow affect numbers?
5. Date arithmetic - mod patterns in dates
6. Lunar calendar - Vietnamese lunar calendar analysis
7. Number-Date correlation - specific numbers on specific dates
8. Fibonacci / Prime day patterns
9. Year-over-year comparison
10. Cross-draw temporal autocorrelation
"""
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import math
import warnings
warnings.filterwarnings('ignore')


class DeepTemporalAnalyzer:
    """Analyze temporal, financial, and sequential patterns in lottery data."""
    
    def __init__(self, max_number, pick_count):
        self.max_number = max_number
        self.pick_count = pick_count
    
    def analyze(self, full_data):
        """
        Run all temporal/financial analyses.
        full_data: list of dicts with keys: draw_date, n1..n6, jackpot, id
        """
        self.data = full_data
        self.dates = []
        self.numbers = []
        self.jackpots = []
        self.draw_ids = []
        
        for row in full_data:
            try:
                dt = datetime.strptime(row['draw_date'], '%Y-%m-%d')
            except:
                try:
                    dt = datetime.strptime(row['draw_date'], '%d/%m/%Y')
                except:
                    continue
            
            self.dates.append(dt)
            nums = [row['n1'], row['n2'], row['n3'], row['n4'], row['n5'], row['n6']]
            self.numbers.append(nums)
            self.draw_ids.append(row.get('id', 0))
            
            # Parse jackpot
            jp = row.get('jackpot', '0')
            if jp and isinstance(jp, str):
                jp = jp.replace('.', '').replace(',', '').replace('đ', '').replace(' ', '')
                try:
                    jp = int(jp)
                except:
                    jp = 0
            self.jackpots.append(int(jp) if jp else 0)
        
        results = {}
        print("[Deep Temporal] Running 10 temporal/financial analyses...")
        
        results['day_of_week'] = self._day_of_week_analysis()
        print("  [1/10] Day of Week analysis complete")
        
        results['monthly'] = self._monthly_analysis()
        print("  [2/10] Monthly analysis complete")
        
        results['jackpot_correlation'] = self._jackpot_correlation()
        print("  [3/10] Jackpot Correlation complete")
        
        results['sequence_pattern'] = self._sequence_pattern()
        print("  [4/10] Sequence Pattern complete")
        
        results['date_number_corr'] = self._date_number_correlation()
        print("  [5/10] Date-Number Correlation complete")
        
        results['lunar'] = self._lunar_analysis()
        print("  [6/10] Lunar Calendar Analysis complete")
        
        results['fibonacci_prime'] = self._fibonacci_prime_days()
        print("  [7/10] Fibonacci/Prime Days complete")
        
        results['year_comparison'] = self._year_over_year()
        print("  [8/10] Year-over-Year Comparison complete")
        
        results['sum_by_date'] = self._sum_temporal_analysis()
        print("  [9/10] Sum Temporal Analysis complete")
        
        results['temporal_prediction'] = self._temporal_predictor()
        print("  [10/10] Temporal Predictor complete")
        
        # Generate verdict
        results['verdict'] = self._generate_verdict(results)
        
        print("[Deep Temporal] All analyses complete!")
        return results
    
    # ===== 1. DAY OF WEEK =====
    def _day_of_week_analysis(self):
        """Do specific days of the week favor specific numbers?"""
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        day_numbers = defaultdict(list)  # day -> list of all numbers drawn
        day_sums = defaultdict(list)
        
        for dt, nums in zip(self.dates, self.numbers):
            dow = dt.weekday()
            day_numbers[dow].extend(nums)
            day_sums[dow].append(sum(nums))
        
        # Find which day has highest average numbers
        day_stats = {}
        day_hot_numbers = {}
        
        for dow in sorted(day_numbers.keys()):
            all_nums = day_numbers[dow]
            freq = Counter(all_nums)
            top5 = freq.most_common(5)
            avg = float(np.mean(all_nums))
            avg_sum = float(np.mean(day_sums[dow])) if day_sums[dow] else 0
            
            day_stats[day_names[dow]] = {
                'count': len(day_sums[dow]),
                'avg_number': round(avg, 2),
                'avg_sum': round(avg_sum, 1),
                'std_sum': round(float(np.std(day_sums[dow])), 1) if day_sums[dow] else 0,
                'top_5_numbers': [[int(n), int(c)] for n, c in top5]
            }
            day_hot_numbers[day_names[dow]] = [int(n) for n, _ in top5]
        
        # Chi-square test: are number distributions different across days?
        all_freq = Counter()
        for nums in day_numbers.values():
            all_freq.update(nums)
        
        chi_sq = 0
        for dow in day_numbers:
            observed = Counter(day_numbers[dow])
            total_dow = len(day_numbers[dow])
            total_all = sum(all_freq.values())
            for n in range(1, self.max_number + 1):
                expected = all_freq.get(n, 0) * total_dow / total_all
                obs = observed.get(n, 0)
                if expected > 0:
                    chi_sq += (obs - expected) ** 2 / expected
        
        df = (len(day_numbers) - 1) * (self.max_number - 1)
        is_significant = chi_sq > df * 1.2  # Rough check
        
        return {
            'name': 'Day of Week Analysis',
            'is_pattern': is_significant,
            'day_stats': day_stats,
            'day_hot_numbers': day_hot_numbers,
            'chi_square': round(chi_sq, 2),
            'degrees_of_freedom': df,
            'conclusion': f'Chi-square = {chi_sq:.1f} (df={df}). ' +
                         ('SIGNIFICANT day-of-week pattern!' if is_significant
                          else 'No significant day-of-week bias.')
        }
    
    # ===== 2. MONTHLY ANALYSIS =====
    def _monthly_analysis(self):
        """Monthly number distribution patterns."""
        month_numbers = defaultdict(list)
        month_sums = defaultdict(list)
        
        for dt, nums in zip(self.dates, self.numbers):
            m = dt.month
            month_numbers[m].extend(nums)
            month_sums[m].append(sum(nums))
        
        month_stats = {}
        for m in range(1, 13):
            if m not in month_numbers:
                continue
            all_nums = month_numbers[m]
            freq = Counter(all_nums)
            top5 = freq.most_common(5)
            
            month_stats[str(m)] = {
                'draws': len(month_sums[m]),
                'avg_sum': round(float(np.mean(month_sums[m])), 1),
                'std_sum': round(float(np.std(month_sums[m])), 1),
                'top_5': [[int(n), int(c)] for n, c in top5]
            }
        
        # Check if avg sums differ significantly across months
        month_avgs = [np.mean(month_sums[m]) for m in sorted(month_sums.keys())]
        overall_avg = np.mean([s for sums in month_sums.values() for s in sums])
        max_dev = max(abs(a - overall_avg) for a in month_avgs) if month_avgs else 0
        
        return {
            'name': 'Monthly Pattern Analysis',
            'is_pattern': max_dev > overall_avg * 0.05,
            'month_stats': month_stats,
            'overall_avg_sum': round(float(overall_avg), 1),
            'max_monthly_deviation': round(float(max_dev), 1),
            'conclusion': f'Max monthly deviation from average: {max_dev:.1f}. ' +
                         ('MONTHLY PATTERN detected!' if max_dev > overall_avg * 0.05
                          else 'No significant monthly patterns.')
        }
    
    # ===== 3. JACKPOT CORRELATION =====
    def _jackpot_correlation(self):
        """Does jackpot size correlate with which numbers are drawn?"""
        valid = [(jp, nums) for jp, nums in zip(self.jackpots, self.numbers) if jp > 0]
        
        if len(valid) < 50:
            return {
                'name': 'Jackpot Correlation',
                'is_pattern': False,
                'conclusion': 'Not enough jackpot data for analysis.'
            }
        
        jps = [v[0] for v in valid]
        nums_list = [v[1] for v in valid]
        
        # Split into low/medium/high jackpot groups
        jp_sorted = sorted(jps)
        low_thresh = jp_sorted[len(jp_sorted) // 3]
        high_thresh = jp_sorted[2 * len(jp_sorted) // 3]
        
        groups = {'low': [], 'medium': [], 'high': []}
        for jp, nums in valid:
            if jp <= low_thresh:
                groups['low'].extend(nums)
            elif jp >= high_thresh:
                groups['high'].extend(nums)
            else:
                groups['medium'].extend(nums)
        
        group_stats = {}
        for gname, gnums in groups.items():
            if not gnums:
                continue
            freq = Counter(gnums)
            top5 = freq.most_common(5)
            group_stats[gname] = {
                'total_numbers': len(gnums),
                'avg': round(float(np.mean(gnums)), 2),
                'top_5': [[int(n), int(c)] for n, c in top5]
            }
        
        # Correlate jackpot with sum of numbers
        sums = [sum(nums) for _, nums in valid]
        jps_norm = np.array(jps, dtype=float)
        sums_norm = np.array(sums, dtype=float)
        
        if np.std(jps_norm) > 0 and np.std(sums_norm) > 0:
            corr = float(np.corrcoef(jps_norm, sums_norm)[0, 1])
        else:
            corr = 0
        
        # Correlate jackpot with each number position
        pos_corrs = {}
        for pos in range(min(6, len(nums_list[0]))):
            pos_vals = [nums[pos] for _, nums in valid]
            if np.std(pos_vals) > 0:
                c = float(np.corrcoef(jps_norm, pos_vals)[0, 1])
                pos_corrs[f'n{pos+1}'] = round(c, 4)
        
        max_corr = max(abs(v) for v in pos_corrs.values()) if pos_corrs else abs(corr)
        
        return {
            'name': 'Jackpot-Number Correlation',
            'is_pattern': max_corr > 0.1,
            'sum_jackpot_correlation': round(corr, 4),
            'position_correlations': pos_corrs,
            'group_comparison': group_stats,
            'jackpot_range': {
                'min': int(min(jps)), 
                'max': int(max(jps)),
                'low_threshold': int(low_thresh),
                'high_threshold': int(high_thresh)
            },
            'conclusion': f'Sum-Jackpot correlation: {corr:.4f}. Max position corr: {max_corr:.4f}. ' +
                         ('SIGNIFICANT jackpot-number correlation!' if max_corr > 0.1
                          else 'No significant jackpot influence on numbers.')
        }
    
    # ===== 4. SEQUENCE PATTERN =====
    def _sequence_pattern(self):
        """Look for patterns in draw sequence IDs."""
        if len(self.draw_ids) < 50:
            return {'name': 'Sequence Pattern', 'is_pattern': False, 'conclusion': 'Not enough data.'}
        
        # Check if draw ID mod various values correlates with numbers
        mod_correlations = {}
        
        for mod in [2, 3, 4, 5, 7, 10]:
            groups = defaultdict(list)
            for did, nums in zip(self.draw_ids, self.numbers):
                r = did % mod
                groups[r].append(sum(nums))
            
            group_avgs = {str(k): round(float(np.mean(v)), 1) for k, v in groups.items() if v}
            
            # F-test: are group means different?
            grand_mean = np.mean([s for v in groups.values() for s in v])
            ssb = sum(len(v) * (np.mean(v) - grand_mean) ** 2 for v in groups.values() if v)
            ssw = sum(sum((x - np.mean(v)) ** 2 for x in v) for v in groups.values() if v)
            
            k = len(groups)
            n = sum(len(v) for v in groups.values())
            
            if ssw > 0 and k > 1 and n > k:
                f_stat = (ssb / (k - 1)) / (ssw / (n - k))
            else:
                f_stat = 0
            
            mod_correlations[str(mod)] = {
                'group_averages': group_avgs,
                'f_statistic': round(float(f_stat), 3),
                'significant': f_stat > 2.5
            }
        
        sig_count = sum(1 for v in mod_correlations.values() if v['significant'])
        
        return {
            'name': 'Draw Sequence Pattern',
            'is_pattern': sig_count >= 2,
            'mod_analysis': mod_correlations,
            'significant_mods': sig_count,
            'conclusion': f'{sig_count} significant mod patterns found. ' +
                         ('SEQUENCE PATTERN detected!' if sig_count >= 2
                          else 'No significant sequence patterns.')
        }
    
    # ===== 5. DATE-NUMBER CORRELATION =====
    def _date_number_correlation(self):
        """Do specific date components predict specific numbers?"""
        correlations = {}
        
        # Day of month vs numbers
        day_nums = defaultdict(list)
        for dt, nums in zip(self.dates, self.numbers):
            day_nums[dt.day].extend(nums)
        
        # Find hot numbers for each day-of-month
        day_hot = {}
        for day in sorted(day_nums.keys())[:15]:  # First 15 days
            freq = Counter(day_nums[day])
            top3 = freq.most_common(3)
            day_hot[str(day)] = [[int(n), int(c)] for n, c in top3]
        
        # Does day of month appear in the drawn numbers more often?
        day_match_count = 0
        total = 0
        for dt, nums in zip(self.dates, self.numbers):
            day = dt.day
            if day <= self.max_number and day in nums:
                day_match_count += 1
            total += 1
        
        expected_match = total * self.pick_count / self.max_number
        day_match_ratio = day_match_count / expected_match if expected_match > 0 else 0
        
        # Does month appear in numbers?
        month_match_count = 0
        for dt, nums in zip(self.dates, self.numbers):
            m = dt.month
            if m in nums:
                month_match_count += 1
        
        month_expected = total * self.pick_count / self.max_number
        month_match_ratio = month_match_count / month_expected if month_expected > 0 else 0
        
        # Year digit analysis
        year_digit_matches = 0
        for dt, nums in zip(self.dates, self.numbers):
            y_digits = [int(d) for d in str(dt.year) if d != '0']
            for yd in y_digits:
                if yd in nums:
                    year_digit_matches += 1
                    break
        
        is_pattern = day_match_ratio > 1.2 or month_match_ratio > 1.2
        
        return {
            'name': 'Date-Number Correlation',
            'is_pattern': is_pattern,
            'day_in_numbers': {
                'matches': day_match_count,
                'expected': round(float(expected_match), 1),
                'ratio': round(day_match_ratio, 3),
                'interpretation': 'Day appears MORE than expected!' if day_match_ratio > 1.1
                                 else 'Day appears at random rate.'
            },
            'month_in_numbers': {
                'matches': month_match_count,
                'expected': round(float(month_expected), 1),
                'ratio': round(month_match_ratio, 3)
            },
            'day_hot_numbers': day_hot,
            'conclusion': f'Day-in-numbers ratio: {day_match_ratio:.3f}x expected. ' +
                         ('DATE-NUMBER PATTERN detected!' if is_pattern
                          else 'No significant date-number correlation.')
        }
    
    # ===== 6. LUNAR CALENDAR =====
    def _lunar_analysis(self):
        """Approximate lunar phase analysis."""
        # Use simple lunar cycle approximation (29.53 days)
        lunar_cycle = 29.53
        ref_new_moon = datetime(2000, 1, 6)  # Known new moon
        
        phase_numbers = defaultdict(list)
        phase_sums = defaultdict(list)
        
        for dt, nums in zip(self.dates, self.numbers):
            days_since = (dt - ref_new_moon).days
            phase = (days_since % lunar_cycle) / lunar_cycle
            
            # 4 phases
            if phase < 0.25:
                phase_name = 'New Moon'
            elif phase < 0.5:
                phase_name = 'First Quarter'
            elif phase < 0.75:
                phase_name = 'Full Moon'
            else:
                phase_name = 'Last Quarter'
            
            phase_numbers[phase_name].extend(nums)
            phase_sums[phase_name].append(sum(nums))
        
        phase_stats = {}
        for pname in ['New Moon', 'First Quarter', 'Full Moon', 'Last Quarter']:
            if pname not in phase_numbers:
                continue
            freq = Counter(phase_numbers[pname])
            top5 = freq.most_common(5)
            phase_stats[pname] = {
                'draws': len(phase_sums[pname]),
                'avg_sum': round(float(np.mean(phase_sums[pname])), 1),
                'std_sum': round(float(np.std(phase_sums[pname])), 1),
                'top_5': [[int(n), int(c)] for n, c in top5]
            }
        
        # Check variance between phases
        avgs = [s['avg_sum'] for s in phase_stats.values()]
        max_diff = max(avgs) - min(avgs) if avgs else 0
        overall_avg = np.mean(avgs) if avgs else 0
        
        return {
            'name': 'Lunar Phase Analysis',
            'is_pattern': max_diff > overall_avg * 0.03,
            'phase_stats': phase_stats,
            'max_phase_difference': round(float(max_diff), 1),
            'conclusion': f'Max difference between lunar phases: {max_diff:.1f}. ' +
                         ('LUNAR PATTERN detected!' if max_diff > overall_avg * 0.03
                          else 'No significant lunar patterns.')
        }
    
    # ===== 7. FIBONACCI/PRIME DAYS =====
    def _fibonacci_prime_days(self):
        """Are draws on Fibonacci/Prime days different?"""
        def is_prime(n):
            if n < 2: return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0: return False
            return True
        
        fibs = set()
        a, b = 1, 1
        while a <= 31:
            fibs.add(a)
            a, b = b, a + b
        
        fib_sums = []
        non_fib_sums = []
        prime_sums = []
        non_prime_sums = []
        
        for dt, nums in zip(self.dates, self.numbers):
            s = sum(nums)
            if dt.day in fibs:
                fib_sums.append(s)
            else:
                non_fib_sums.append(s)
            
            if is_prime(dt.day):
                prime_sums.append(s)
            else:
                non_prime_sums.append(s)
        
        fib_avg = float(np.mean(fib_sums)) if fib_sums else 0
        non_fib_avg = float(np.mean(non_fib_sums)) if non_fib_sums else 0
        prime_avg = float(np.mean(prime_sums)) if prime_sums else 0
        non_prime_avg = float(np.mean(non_prime_sums)) if non_prime_sums else 0
        
        fib_diff = abs(fib_avg - non_fib_avg)
        prime_diff = abs(prime_avg - non_prime_avg)
        
        return {
            'name': 'Fibonacci/Prime Day Analysis',
            'is_pattern': fib_diff > 5 or prime_diff > 5,
            'fibonacci_days': {
                'avg_sum': round(fib_avg, 1),
                'count': len(fib_sums),
                'vs_normal': round(fib_avg - non_fib_avg, 1)
            },
            'prime_days': {
                'avg_sum': round(prime_avg, 1),
                'count': len(prime_sums),
                'vs_normal': round(prime_avg - non_prime_avg, 1)
            },
            'conclusion': f'Fibonacci diff: {fib_diff:.1f}, Prime diff: {prime_diff:.1f}. ' +
                         ('SPECIAL DAY PATTERN!' if fib_diff > 5 or prime_diff > 5
                          else 'No Fibonacci/Prime day patterns.')
        }
    
    # ===== 8. YEAR-OVER-YEAR =====
    def _year_over_year(self):
        """Compare patterns across years."""
        year_data = defaultdict(lambda: {'sums': [], 'numbers': []})
        
        for dt, nums in zip(self.dates, self.numbers):
            y = dt.year
            year_data[y]['sums'].append(sum(nums))
            year_data[y]['numbers'].extend(nums)
        
        year_stats = {}
        for y in sorted(year_data.keys()):
            d = year_data[y]
            freq = Counter(d['numbers'])
            top5 = freq.most_common(5)
            year_stats[str(y)] = {
                'draws': len(d['sums']),
                'avg_sum': round(float(np.mean(d['sums'])), 1),
                'std_sum': round(float(np.std(d['sums'])), 1),
                'top_5': [[int(n), int(c)] for n, c in top5]
            }
        
        # Check trend over years
        years = sorted(year_data.keys())
        avgs = [np.mean(year_data[y]['sums']) for y in years]
        
        if len(avgs) >= 3:
            trend = float(np.polyfit(range(len(avgs)), avgs, 1)[0])
        else:
            trend = 0
        
        return {
            'name': 'Year-over-Year Comparison',
            'is_pattern': abs(trend) > 1.0,
            'year_stats': year_stats,
            'trend_per_year': round(trend, 2),
            'conclusion': f'Sum trend per year: {trend:+.2f}. ' +
                         ('YEARLY TREND detected!' if abs(trend) > 1.0
                          else 'No significant year trend.')
        }
    
    # ===== 9. SUM TEMPORAL ANALYSIS =====
    def _sum_temporal_analysis(self):
        """Analyze the sum of drawn numbers over time."""
        sums = [sum(nums) for nums in self.numbers]
        
        # Moving average
        window = 20
        if len(sums) >= window:
            ma = [float(np.mean(sums[max(0,i-window):i+1])) for i in range(len(sums))]
        else:
            ma = sums
        
        # Autocorrelation of sums
        s_arr = np.array(sums, dtype=float)
        s_norm = s_arr - np.mean(s_arr)
        var_s = np.sum(s_norm ** 2)
        
        sum_autocorr = []
        for lag in range(1, min(50, len(sums) // 3)):
            if var_s > 0:
                corr = np.sum(s_norm[:len(sums)-lag] * s_norm[lag:]) / var_s
                sum_autocorr.append({'lag': lag, 'corr': round(float(corr), 4)})
        
        # Is there a significant period?
        max_corr = max(abs(c['corr']) for c in sum_autocorr) if sum_autocorr else 0
        threshold = 2.0 / np.sqrt(len(sums))
        
        # Recent trend
        recent_sums = sums[-30:] if len(sums) >= 30 else sums
        recent_avg = float(np.mean(recent_sums))
        overall_avg = float(np.mean(sums))
        
        return {
            'name': 'Sum Temporal Analysis',
            'is_pattern': max_corr > threshold * 2,
            'overall_avg_sum': round(overall_avg, 1),
            'recent_30_avg': round(recent_avg, 1),
            'recent_vs_overall': round(recent_avg - overall_avg, 1),
            'max_sum_autocorr': round(max_corr, 4),
            'autocorr_threshold': round(float(threshold), 4),
            'top_autocorrelations': sorted(sum_autocorr, key=lambda x: -abs(x['corr']))[:5],
            'conclusion': f'Max sum autocorrelation: {max_corr:.4f} (threshold: {threshold:.4f}). ' +
                         ('SUM TEMPORAL PATTERN!' if max_corr > threshold * 2
                          else 'Sum appears random over time.')
        }
    
    # ===== 10. TEMPORAL PREDICTOR =====
    def _temporal_predictor(self):
        """Use ALL temporal features to predict next draw and backtest."""
        if len(self.dates) < 100:
            return {'name': 'Temporal Predictor', 'is_pattern': False, 'conclusion': 'Not enough data.'}
        
        # For each draw, compute day-based "hot numbers" from same-day-of-week history
        correct_counts = []
        
        # Walk-forward test on last 100 draws
        test_start = max(50, len(self.data) - 100)
        
        for i in range(test_start, len(self.dates) - 1):
            target_dow = self.dates[i + 1].weekday() if i + 1 < len(self.dates) else self.dates[i].weekday()
            target_month = self.dates[i + 1].month if i + 1 < len(self.dates) else self.dates[i].month
            
            # Collect same-day-of-week numbers from history
            dow_freq = Counter()
            month_freq = Counter()
            recent_freq = Counter()
            
            for j in range(i):
                if self.dates[j].weekday() == target_dow:
                    for n in self.numbers[j]:
                        dow_freq[n] += 1
                if self.dates[j].month == target_month:
                    for n in self.numbers[j]:
                        month_freq[n] += 1
            
            # Recent 10 draws
            for j in range(max(0, i-10), i+1):
                for n in self.numbers[j]:
                    recent_freq[n] += 1
            
            # Combined score
            scores = {}
            for n in range(1, self.max_number + 1):
                scores[n] = (dow_freq.get(n, 0) * 0.3 + 
                            month_freq.get(n, 0) * 0.3 + 
                            recent_freq.get(n, 0) * 0.4)
            
            # Pick top 6
            predicted = set(sorted(scores, key=lambda x: -scores[x])[:self.pick_count])
            
            if i + 1 < len(self.numbers):
                actual = set(self.numbers[i + 1])
                matches = len(predicted & actual)
                correct_counts.append(matches)
        
        if not correct_counts:
            return {'name': 'Temporal Predictor', 'is_pattern': False, 'conclusion': 'No test results.'}
        
        avg_matches = float(np.mean(correct_counts))
        max_matches = int(max(correct_counts))
        random_expected = self.pick_count * self.pick_count / self.max_number
        improvement = (avg_matches / random_expected - 1) * 100 if random_expected > 0 else 0
        
        dist = Counter(correct_counts)
        
        return {
            'name': 'Temporal Predictor (Day+Month+Recent)',
            'is_pattern': avg_matches > random_expected * 1.3,
            'avg_matches': round(avg_matches, 3),
            'max_matches': max_matches,
            'random_expected': round(random_expected, 3),
            'improvement': round(float(improvement), 1),
            'distribution': {str(k): int(v) for k, v in sorted(dist.items())},
            'tests_run': len(correct_counts),
            'conclusion': f'Temporal predictor avg: {avg_matches:.3f}/6 '
                          f'(random: {random_expected:.3f}, improvement: {improvement:+.1f}%). ' +
                         ('TEMPORAL PREDICTION WORKS!' if avg_matches > random_expected * 1.3
                          else 'Temporal features do not improve prediction.')
        }
    
    def _generate_verdict(self, results):
        """Generate final temporal verdict."""
        pattern_count = 0
        total = 0
        evidence = []
        
        for key, val in results.items():
            if key == 'verdict':
                continue
            if isinstance(val, dict) and 'is_pattern' in val:
                total += 1
                if val['is_pattern']:
                    pattern_count += 1
                    evidence.append(f"+ {val.get('name', key)}: {val.get('conclusion', 'Pattern found')}")
                else:
                    evidence.append(f"- {val.get('name', key)}: {val.get('conclusion', 'No pattern')}")
        
        score = round(pattern_count / total * 100, 1) if total > 0 else 0
        
        if score >= 60:
            verdict = "STRONG temporal/financial patterns detected!"
        elif score >= 30:
            verdict = "SOME temporal patterns found - worth investigating further"
        else:
            verdict = "No significant temporal/financial patterns"
        
        return {
            'score': score,
            'pattern_count': pattern_count,
            'total_tests': total,
            'verdict': verdict,
            'evidence': evidence
        }
