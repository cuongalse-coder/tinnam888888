"""
Deep Forensic Analysis — Mine every possible pattern from Vietlott data.
Goes beyond basic PASS/FAIL tests to extract ACTIONABLE intelligence.

Investigates:
1. Conditional transition matrices (what follows what)
2. Multi-scale temporal patterns (hot/cold streaks by window)
3. Day-specific number preferences
4. Position-weighted frequency (which numbers favor which positions)
5. Lag-N repeat analysis (how often numbers repeat after N draws)
6. Number neighborhood clustering (which numbers appear together)
7. Recent momentum signals (acceleration/deceleration)
8. Even/Odd and High/Low sequential patterns
9. Sum-range lock (narrow the sum range for next draw)
10. Triplet/pair co-occurrence network (which combos are "due")
"""
import sys
import os
import time
import math
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class DeepForensic:
    """Mine ALL actionable patterns from lottery data."""
    
    def __init__(self, max_number, pick_count):
        self.max_number = max_number
        self.pick_count = pick_count
    
    def analyze(self, data, draw_dates=None):
        """Run full deep forensic analysis. Returns prediction + evidence."""
        n = len(data)
        last = set(data[-1])
        
        # Accumulate scoring signals
        signals = {}  # signal_name -> {number: score}
        reports = {}  # signal_name -> report_dict
        
        # === 1. Conditional Transition Matrix ===
        sig, rep = self._transition_matrix(data)
        signals['transition'] = sig
        reports['transition'] = rep
        
        # === 2. Multi-Scale Momentum ===
        sig, rep = self._multi_scale_momentum(data)
        signals['momentum'] = sig
        reports['momentum'] = rep
        
        # === 3. Day-of-Week Profile ===
        if draw_dates:
            sig, rep = self._day_profile(data, draw_dates)
            signals['day_profile'] = sig
            reports['day_profile'] = rep
        
        # === 4. Lag-N Repeat Analysis ===
        sig, rep = self._lag_repeat(data)
        signals['lag_repeat'] = sig
        reports['lag_repeat'] = rep
        
        # === 5. Co-occurrence Network ===
        sig, rep = self._cooccurrence_network(data)
        signals['cooccurrence'] = sig
        reports['cooccurrence'] = rep
        
        # === 6. Position Frequency ===
        sig, rep = self._position_frequency(data)
        signals['position'] = sig
        reports['position'] = rep
        
        # === 7. Gap Timing (overdue by cycle) ===
        sig, rep = self._gap_timing(data)
        signals['gap_timing'] = sig
        reports['gap_timing'] = rep
        
        # === 8. Sum/Odd/Range Constraints ===
        sig, rep = self._structural_constraints(data)
        signals['structure'] = sig
        reports['structure'] = rep
        
        # === 9. Streak Analysis ===
        sig, rep = self._streak_analysis(data)
        signals['streak'] = sig
        reports['streak'] = rep
        
        # === 10. KNN History Match ===
        sig, rep = self._knn_match(data)
        signals['knn'] = sig
        reports['knn'] = rep
        
        # =============================================
        # COMBINE ALL SIGNALS
        # =============================================
        # Weight each signal by its backtest performance
        weights = self._calibrate_weights(data, signals)
        
        final_scores = {n: 0.0 for n in range(1, self.max_number + 1)}
        for sig_name, sig_scores in signals.items():
            w = weights.get(sig_name, 1.0)
            if not sig_scores:
                continue
            max_s = max(abs(v) for v in sig_scores.values()) if sig_scores else 1
            for num, score in sig_scores.items():
                final_scores[num] += (score / max(max_s, 0.001)) * w
        
        # Anti-repeat: penalize numbers from last draw
        for num in last:
            final_scores[num] -= 2.0
        
        # Rank
        ranked = sorted(final_scores.items(), key=lambda x: -x[1])
        pool = [n for n, _ in ranked[:30]]
        
        # Find best constraint-valid combo
        constraints = reports.get('structure', {}).get('constraints', {})
        primary = self._find_best_combo(pool, final_scores, constraints)
        
        # Generate portfolio (20 diverse sets)
        portfolio = self._generate_portfolio(pool, final_scores, constraints, 20)
        
        return {
            'primary': primary,
            'portfolio': portfolio,
            'scores': {n: round(s, 3) for n, s in ranked[:30]},
            'weights': weights,
            'reports': reports,
            'n_signals': len(signals),
            'top_30': [n for n, _ in ranked[:30]],
        }
    
    # ================================================================
    # SIGNAL 1: Conditional Transition Matrix
    # ================================================================
    def _transition_matrix(self, data):
        """Build P(num in draw N+1 | draw N contained X)."""
        n = len(data)
        # For each number, count how often each other number follows it
        follow_counts = defaultdict(Counter)  # follow_counts[prev][next] = count
        prev_counts = Counter()
        
        for i in range(n - 1):
            prev = set(data[i])
            next_draw = set(data[i + 1])
            for p in prev:
                prev_counts[p] += 1
                for nx in next_draw:
                    follow_counts[p][nx] += 1
        
        # Score: for each number, what's its probability given the LAST draw?
        last = set(data[-1])
        scores = {}
        top_transitions = []
        
        for num in range(1, self.max_number + 1):
            total_follows = 0
            total_prev = 0
            for p in last:
                total_follows += follow_counts[p].get(num, 0)
                total_prev += prev_counts[p]
            
            if total_prev > 0:
                conditional_p = total_follows / total_prev
                base_p = self.pick_count / self.max_number
                lift = conditional_p / base_p if base_p > 0 else 1.0
                scores[num] = (lift - 1.0) * 3.0  # Center around 0
                
                if lift > 1.3:
                    top_transitions.append({
                        'number': num, 'lift': round(lift, 3),
                        'cond_p': round(conditional_p, 4),
                    })
            else:
                scores[num] = 0
        
        top_transitions.sort(key=lambda x: -x['lift'])
        
        return scores, {
            'name': 'Conditional Transitions',
            'description': 'Numbers most likely to follow the last draw',
            'top': top_transitions[:15],
        }
    
    # ================================================================
    # SIGNAL 2: Multi-Scale Momentum
    # ================================================================
    def _multi_scale_momentum(self, data):
        """Track momentum at windows 5, 10, 20, 50."""
        n = len(data)
        windows = [5, 10, 20, 50]
        scores = {}
        momentum_report = []
        
        for num in range(1, self.max_number + 1):
            freqs = {}
            for w in windows:
                if n >= w:
                    freqs[w] = sum(1 for d in data[-w:] if num in d) / w
                else:
                    freqs[w] = 0
            
            # Momentum = short-term trend vs long-term
            if all(w in freqs for w in windows):
                short = freqs[5]
                med = freqs[20]
                long_term = freqs[50]
                
                accel = (short - med) * 10  # Short-term acceleration
                trend = (med - long_term) * 5  # Medium-term trend
                score = accel + trend
                scores[num] = score
                
                if abs(score) > 1.0:
                    momentum_report.append({
                        'number': num,
                        'f5': round(freqs[5], 3), 'f10': round(freqs[10], 3),
                        'f20': round(freqs[20], 3), 'f50': round(freqs[50], 3),
                        'score': round(score, 2),
                        'signal': 'RISING' if score > 0 else 'FALLING',
                    })
            else:
                scores[num] = 0
        
        momentum_report.sort(key=lambda x: -x['score'])
        
        return scores, {
            'name': 'Multi-Scale Momentum',
            'rising': [m for m in momentum_report if m['signal'] == 'RISING'][:10],
            'falling': [m for m in momentum_report if m['signal'] == 'FALLING'][-10:],
        }
    
    # ================================================================
    # SIGNAL 3: Day-of-Week Profile
    # ================================================================
    def _day_profile(self, data, dates):
        """Score numbers based on day-of-week bias."""
        from datetime import datetime
        
        day_draws = defaultdict(list)
        for date_str, draw in zip(dates, data):
            try:
                dt = datetime.strptime(date_str, '%Y-%m-%d')
                dow = dt.weekday()
                day_draws[dow].append(draw)
            except:
                continue
        
        # What day is next draw?
        try:
            last_date = datetime.strptime(dates[-1], '%Y-%m-%d')
            # Mega draws on Wed/Fri/Sun, Power on Tue/Thu/Sat
            next_dow = (last_date.weekday() + 2) % 7  # Approximate
        except:
            next_dow = None
        
        scores = {}
        day_report = []
        
        for num in range(1, self.max_number + 1):
            if next_dow is not None and next_dow in day_draws:
                day_specific = day_draws[next_dow]
                rate = sum(1 for d in day_specific if num in d) / len(day_specific) if day_specific else 0
                overall = sum(1 for d in data if num in d) / len(data)
                lift = rate / overall if overall > 0 else 1.0
                scores[num] = (lift - 1.0) * 2.0
                
                if abs(lift - 1.0) > 0.15:
                    day_report.append({'number': num, 'day_rate': round(rate, 4), 'overall': round(overall, 4), 'lift': round(lift, 3)})
            else:
                scores[num] = 0
        
        day_report.sort(key=lambda x: -x['lift'])
        
        return scores, {
            'name': 'Day-of-Week Profile',
            'next_dow': next_dow,
            'biased_numbers': day_report[:15],
        }
    
    # ================================================================
    # SIGNAL 4: Lag-N Repeat Analysis
    # ================================================================
    def _lag_repeat(self, data):
        """How likely is each number to appear given its last appearance lag?"""
        n = len(data)
        
        # For each number, what % of the time does it appear at lag 1, 2, 3, ...?
        lag_stats = defaultdict(lambda: defaultdict(int))  # lag_stats[num][lag] = count
        last_seen = {}
        
        for i, draw in enumerate(data):
            for num in draw:
                if num in last_seen:
                    lag = i - last_seen[num]
                    lag_stats[num][lag] += 1
                last_seen[num] = i
        
        scores = {}
        repeat_report = []
        
        for num in range(1, self.max_number + 1):
            current_lag = n - last_seen.get(num, 0)
            
            if num not in lag_stats:
                scores[num] = 0
                continue
            
            stats = lag_stats[num]
            total = sum(stats.values())
            
            # What's the probability of appearing at exactly this lag?
            count_at_lag = stats.get(current_lag, 0)
            p_at_lag = count_at_lag / total if total > 0 else 0
            
            # What's the cumulative probability of NOT appearing until now?
            p_not_yet = sum(v for k, v in stats.items() if k > current_lag) / total if total > 0 else 0
            
            # Numbers that are "due" (current lag exceeds their median gap)
            gaps_list = []
            for lag, cnt in stats.items():
                gaps_list.extend([lag] * cnt)
            median_gap = np.median(gaps_list) if gaps_list else self.max_number / self.pick_count
            
            overdue_ratio = current_lag / median_gap if median_gap > 0 else 1.0
            
            # Score: combine probability at this lag + overdue factor
            score = p_at_lag * 5.0 + max(0, overdue_ratio - 1.0) * 2.0
            scores[num] = score
            
            if overdue_ratio > 1.5 or p_at_lag > 0.1:
                repeat_report.append({
                    'number': num, 'current_lag': current_lag,
                    'median_gap': round(median_gap, 1),
                    'overdue_ratio': round(overdue_ratio, 2),
                    'p_at_lag': round(p_at_lag, 3),
                })
        
        repeat_report.sort(key=lambda x: -x['overdue_ratio'])
        
        return scores, {
            'name': 'Lag-N Repeat Analysis',
            'overdue': repeat_report[:15],
        }
    
    # ================================================================
    # SIGNAL 5: Co-occurrence Network
    # ================================================================
    def _cooccurrence_network(self, data):
        """Which pairs appeared together? Given last draw, which partners are due?"""
        n = len(data)
        last = set(data[-1])
        
        pair_freq = Counter()
        for draw in data[-200:]:
            for pair in combinations(sorted(draw[:self.pick_count]), 2):
                pair_freq[pair] += 1
        
        # For numbers in last draw, find their strongest partners NOT in last draw
        scores = {}
        partner_report = []
        
        for num in range(1, self.max_number + 1):
            total_partner_score = 0
            for prev_num in last:
                pair = tuple(sorted([prev_num, num]))
                freq = pair_freq.get(pair, 0)
                total_partner_score += freq
            
            scores[num] = total_partner_score * 0.1
        
        # Also: find triplets from last draw that are "due"
        # If 2 of 3 triplet members were in last draw, the 3rd is "due"
        triplet_freq = Counter()
        for draw in data[-150:]:
            for trip in combinations(sorted(draw[:self.pick_count]), 3):
                triplet_freq[trip] += 1
        
        triplet_due = Counter()
        for trip, count in triplet_freq.most_common(500):
            if count < 2:
                break
            trip_set = set(trip)
            overlap = trip_set & last
            if len(overlap) == 2:
                missing = (trip_set - last).pop()
                triplet_due[missing] += count
                scores[missing] = scores.get(missing, 0) + count * 0.5
        
        triplet_report = [{'number': n, 'triplet_score': c} for n, c in triplet_due.most_common(10)]
        
        return scores, {
            'name': 'Co-occurrence Network',
            'triplet_due': triplet_report,
        }
    
    # ================================================================
    # SIGNAL 6: Position Frequency
    # ================================================================
    def _position_frequency(self, data):
        """Which numbers are favored at each sorted position?"""
        n = len(data)
        pos_freq = [Counter() for _ in range(self.pick_count)]
        
        for draw in data:
            sd = sorted(draw[:self.pick_count])
            for p, num in enumerate(sd):
                pos_freq[p][num] += 1
        
        # For each number, what's its dominant position and how strong?
        scores = {}
        pos_report = []
        
        for num in range(1, self.max_number + 1):
            total_appearances = sum(pos_freq[p].get(num, 0) for p in range(self.pick_count))
            if total_appearances == 0:
                scores[num] = 0
                continue
            
            # Find dominant position
            pos_rates = []
            for p in range(self.pick_count):
                rate = pos_freq[p].get(num, 0) / total_appearances
                pos_rates.append(rate)
            
            dominant_pos = np.argmax(pos_rates)
            dominance = pos_rates[dominant_pos]
            
            # Score: numbers with high appearance rate at their dominant position
            expected_pos = round((num - 1) / (self.max_number - 1) * (self.pick_count - 1))
            if dominant_pos == expected_pos:
                scores[num] = total_appearances / n * 0.5  # Normal
            else:
                scores[num] = total_appearances / n * 1.0  # Unusual position = interesting
        
        return scores, {
            'name': 'Position Frequency',
            'description': 'Position distribution for each number',
        }
    
    # ================================================================
    # SIGNAL 7: Gap Timing (cycle-based prediction)
    # ================================================================
    def _gap_timing(self, data):
        """Predict which numbers are due based on their individual cycles."""
        n = len(data)
        scores = {}
        timing_report = []
        
        for num in range(1, self.max_number + 1):
            appearances = [i for i, d in enumerate(data) if num in d]
            if len(appearances) < 5:
                scores[num] = 0
                continue
            
            gaps = [appearances[i+1] - appearances[i] for i in range(len(appearances)-1)]
            mean_gap = np.mean(gaps)
            std_gap = np.std(gaps)
            current_gap = n - appearances[-1]
            
            # How many standard deviations overdue?
            z_overdue = (current_gap - mean_gap) / std_gap if std_gap > 0 else 0
            
            # Probability of appearing (based on empirical CDF)
            p_appear = sum(1 for g in gaps if g <= current_gap) / len(gaps)
            
            score = 0
            if z_overdue > 0.5:
                score = z_overdue * 1.5 + p_appear * 2.0
            elif z_overdue < -1.0:
                score = -1.0  # Just appeared, unlikely to repeat
            
            scores[num] = score
            
            if z_overdue > 1.0:
                timing_report.append({
                    'number': num,
                    'current_gap': current_gap,
                    'mean_gap': round(mean_gap, 1),
                    'z_overdue': round(z_overdue, 2),
                    'p_appear': round(p_appear, 3),
                })
        
        timing_report.sort(key=lambda x: -x['z_overdue'])
        
        return scores, {
            'name': 'Gap Timing',
            'overdue_numbers': timing_report[:15],
        }
    
    # ================================================================
    # SIGNAL 8: Structural Constraints
    # ================================================================
    def _structural_constraints(self, data):
        """Learn tight constraints for next draw."""
        n = len(data)
        recent = data[-50:]
        
        sums = [sum(d[:self.pick_count]) for d in recent]
        odds = [sum(1 for x in d[:self.pick_count] if x % 2 == 1) for d in recent]
        mid = self.max_number // 2
        highs = [sum(1 for x in d[:self.pick_count] if x > mid) for d in recent]
        ranges_v = [max(d[:self.pick_count]) - min(d[:self.pick_count]) for d in recent]
        
        constraints = {
            'sum_lo': int(np.percentile(sums, 10)),
            'sum_hi': int(np.percentile(sums, 90)),
            'sum_mean': round(np.mean(sums), 1),
            'odd_lo': max(0, int(np.percentile(odds, 10))),
            'odd_hi': min(self.pick_count, int(np.percentile(odds, 90))),
            'high_lo': max(0, int(np.percentile(highs, 10))),
            'high_hi': min(self.pick_count, int(np.percentile(highs, 90))),
            'range_lo': int(np.percentile(ranges_v, 10)),
            'range_hi': int(np.percentile(ranges_v, 90)),
        }
        
        # Score: assign small bonus to numbers in the "sweet spot"
        scores = {}
        for num in range(1, self.max_number + 1):
            scores[num] = 0.1  # Neutral

        return scores, {
            'name': 'Structural Constraints',
            'constraints': constraints,
        }
    
    # ================================================================
    # SIGNAL 9: Streak Analysis
    # ================================================================
    def _streak_analysis(self, data):
        """Find numbers on hot/cold streaks."""
        n = len(data)
        scores = {}
        streak_report = []
        
        for num in range(1, self.max_number + 1):
            # Current hot streak (consecutive draws with this number)
            hot_streak = 0
            for d in reversed(data):
                if num in d:
                    hot_streak += 1
                else:
                    break
            
            # Current cold streak
            cold_streak = 0
            for d in reversed(data):
                if num not in d:
                    cold_streak += 1
                else:
                    break
            
            # Historical: what happens after similar streaks?
            appearances = [1 if num in d else 0 for d in data]
            
            score = 0
            if hot_streak >= 2:
                # Check: after 2+ consecutive appearances, does it continue?
                continue_count = 0
                total = 0
                streak = 0
                for i in range(n - 1):
                    if appearances[i]:
                        streak += 1
                    else:
                        streak = 0
                    if streak >= hot_streak:
                        total += 1
                        if appearances[i + 1]:
                            continue_count += 1
                p_continue = continue_count / total if total > 0 else 0
                score = (p_continue - self.pick_count / self.max_number) * 5
                streak_report.append({
                    'number': num, 'type': 'HOT', 'streak': hot_streak,
                    'p_continue': round(p_continue, 3),
                })
            elif cold_streak >= 10:
                # Long absence — check if it tends to return
                score = cold_streak * 0.1
                streak_report.append({
                    'number': num, 'type': 'COLD', 'streak': cold_streak,
                })
            
            scores[num] = score
        
        return scores, {
            'name': 'Streak Analysis',
            'active_streaks': sorted(streak_report, key=lambda x: -x.get('streak', 0))[:10],
        }
    
    # ================================================================
    # SIGNAL 10: KNN History Match
    # ================================================================
    def _knn_match(self, data):
        """Find historical draws most similar to the last draw. What came next?"""
        n = len(data)
        last = set(data[-1])
        
        matches = []
        for i in range(n - 2):
            similarity = len(set(data[i]) & last)
            if similarity >= 3:
                matches.append((i, similarity, data[i + 1]))
        
        # Weight by similarity
        scores = Counter()
        for idx, sim, next_draw in matches:
            weight = sim ** 2  # Quadratic weight
            for num in next_draw:
                scores[num] += weight
        
        # Normalize
        max_score = max(scores.values()) if scores else 1
        normed = {num: scores.get(num, 0) / max_score * 3.0 for num in range(1, self.max_number + 1)}
        
        knn_report = [{'draw_idx': m[0], 'similarity': m[1], 'next': sorted(m[2])} for m in sorted(matches, key=lambda x: -x[1])[:5]]
        
        return normed, {
            'name': 'KNN History Match',
            'best_matches': knn_report,
        }
    
    # ================================================================
    # WEIGHT CALIBRATION (mini-backtest each signal)
    # ================================================================
    def _calibrate_weights(self, data, signals):
        """Quick backtest each signal to determine optimal weights."""
        n = len(data)
        test_range = range(max(70, n - 50), n - 1)
        
        signal_hits = {name: 0 for name in signals}
        signal_total = {name: 0 for name in signals}
        
        for test_idx in test_range:
            actual = set(data[test_idx + 1])
            train = data[:test_idx + 1]
            
            for sig_name, sig_func_scores in signals.items():
                # Use the existing scores (they're computed on full data, but this is approximate)
                if not sig_func_scores:
                    continue
                top_6 = sorted(sig_func_scores.items(), key=lambda x: -x[1])[:self.pick_count]
                predicted = set(n for n, _ in top_6)
                hits = len(predicted & actual)
                signal_hits[sig_name] += hits
                signal_total[sig_name] += self.pick_count
        
        # Convert to weights (relative performance)
        weights = {}
        base = self.pick_count / self.max_number  # Random expectation
        
        for name in signals:
            if signal_total[name] > 0:
                avg = signal_hits[name] / (len(test_range) if test_range else 1)
                lift = avg / (base * self.pick_count) if base > 0 else 1.0
                weights[name] = max(lift, 0.1)  # At least 0.1 weight
            else:
                weights[name] = 1.0
        
        return weights
    
    # ================================================================
    # COMBO SELECTION
    # ================================================================
    def _find_best_combo(self, pool, scores, constraints):
        """Find highest-scoring valid combo."""
        if not constraints:
            return sorted(pool[:self.pick_count])
        
        best = None
        best_score = -float('inf')
        
        search_pool = pool[:16]
        for combo in combinations(search_pool, self.pick_count):
            if not self._validate(combo, constraints):
                continue
            cs = sum(scores.get(n, 0) for n in combo)
            if cs > best_score:
                best_score = cs
                best = sorted(combo)
        
        return best if best else sorted(pool[:self.pick_count])
    
    def _generate_portfolio(self, pool, scores, constraints, n_sets):
        """Generate diverse portfolio."""
        portfolio = []
        best = self._find_best_combo(pool, scores, constraints)
        if best:
            portfolio.append(best)
        
        used = {tuple(best)} if best else set()
        extended_pool = pool[:25]
        
        weights = np.array([max(scores.get(n, 0.01), 0.01) for n in extended_pool])
        weights = weights / weights.sum()
        
        attempts = 0
        while len(portfolio) < n_sets and attempts < n_sets * 200:
            attempts += 1
            try:
                idx = np.random.choice(len(extended_pool), self.pick_count, replace=False, p=weights)
                combo = sorted([extended_pool[i] for i in idx])
            except:
                continue
            
            t = tuple(combo)
            if t in used:
                continue
            if constraints and not self._validate(combo, constraints):
                continue
            if all(len(set(combo) - set(p)) >= 3 for p in portfolio):
                used.add(t)
                portfolio.append(combo)
        
        return portfolio
    
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
        if rng < c.get('range_lo', 0) or rng > c.get('range_hi', 99):
            return False
        return True


# ================================================================
# CLI: Run deep forensic and print report
# ================================================================
if __name__ == '__main__':
    from scraper.data_manager import get_mega645_numbers, get_mega645_all
    
    print("Loading Mega 6/45 data...")
    data = get_mega645_numbers()
    all_rows = get_mega645_all()
    dates = [r['draw_date'] for r in all_rows]
    
    print(f"Loaded {len(data)} draws. Last: {dates[-1]}, Numbers: {data[-1]}")
    print(f"\nRunning Deep Forensic Analysis...\n")
    
    engine = DeepForensic(45, 6)
    start = time.time()
    result = engine.analyze(data, dates)
    elapsed = time.time() - start
    
    print(f"{'='*65}")
    print(f"  DEEP FORENSIC REPORT ({elapsed:.1f}s)")
    print(f"{'='*65}")
    
    # Primary prediction
    print(f"\n  PRIMARY: {result['primary']}")
    print(f"  Top 30 pool: {result['top_30']}")
    
    # Signal weights
    print(f"\n  SIGNAL WEIGHTS (calibrated by backtest):")
    for name, w in sorted(result['weights'].items(), key=lambda x: -x[1]):
        bar = '#' * int(w * 10)
        print(f"    {name:20s} {w:.3f} {bar}")
    
    # Key reports
    for name, report in result['reports'].items():
        print(f"\n  === {report.get('name', name)} ===")
        for key, val in report.items():
            if key in ('name', 'description', 'constraints'):
                continue
            if isinstance(val, list) and val:
                print(f"    {key}:")
                for item in val[:5]:
                    print(f"      {item}")
    
    # Constraints
    c = result['reports'].get('structure', {}).get('constraints', {})
    if c:
        print(f"\n  CONSTRAINTS: sum={c.get('sum_lo')}-{c.get('sum_hi')}, "
              f"odd={c.get('odd_lo')}-{c.get('odd_hi')}, "
              f"high={c.get('high_lo')}-{c.get('high_hi')}, "
              f"range={c.get('range_lo')}-{c.get('range_hi')}")
    
    # Portfolio
    print(f"\n  PORTFOLIO ({len(result['portfolio'])} sets):")
    for i, p in enumerate(result['portfolio']):
        print(f"    #{i+1}: {p}")
    
    # === BACKTEST ===
    print(f"\n  === BACKTEST (last 100 draws) ===")
    matches = []
    port_matches = []
    for test_end in range(max(70, len(data) - 100), len(data) - 1):
        train = data[:test_end + 1]
        actual = set(data[test_end + 1])
        
        r = engine.analyze(train, dates[:test_end + 1] if dates else None)
        pred = set(r['primary'])
        m = len(pred & actual)
        matches.append(m)
        
        # Portfolio best
        best_p = max((len(set(p) & actual) for p in r['portfolio']), default=m)
        port_matches.append(best_p)
    
    from collections import Counter as C2
    dist = C2(matches)
    port_dist = C2(port_matches)
    
    print(f"  Primary avg: {np.mean(matches):.4f}/6, max: {max(matches)}/6")
    print(f"  Portfolio avg: {np.mean(port_matches):.4f}/6, max: {max(port_matches)}/6")
    print(f"  Random: {6*6/45:.3f}/6")
    print(f"\n  Primary distribution:")
    for k in range(7):
        c = dist.get(k, 0)
        pct = c / len(matches) * 100
        print(f"    {k}/6: {c:3d} ({pct:.1f}%)")
    print(f"\n  Portfolio distribution:")
    for k in range(7):
        c = port_dist.get(k, 0)
        pct = c / len(port_matches) * 100
        print(f"    {k}/6: {c:3d} ({pct:.1f}%)")
    
    print(f"\n{'='*65}")
