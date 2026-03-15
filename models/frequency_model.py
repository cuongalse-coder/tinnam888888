"""
Frequency Analysis Model - Statistical analysis of lottery numbers.
Analyzes hot/cold numbers, frequency distribution, gap analysis, and patterns.
"""
import numpy as np
from collections import Counter, defaultdict


class FrequencyModel:
    """Statistical frequency analysis for lottery number prediction."""
    
    def __init__(self, max_number, pick_count):
        """
        Args:
            max_number: Maximum number in lottery (45 for Mega, 55 for Power)
            pick_count: Numbers to pick (6)
        """
        self.max_number = max_number
        self.pick_count = pick_count
        self.history = []
        self.freq = Counter()
        self.pair_freq = Counter()
        self.gap_data = defaultdict(list)  # number -> list of gaps
        
    def fit(self, data):
        """Train on historical data. data = list of [n1, n2, ..., n6]."""
        self.history = [row[:self.pick_count] for row in data]
        self.freq = Counter()
        self.pair_freq = Counter()
        self.gap_data = defaultdict(list)
        
        last_seen = {}
        
        for draw_idx, numbers in enumerate(self.history):
            for n in numbers:
                self.freq[n] += 1
                
                # Gap analysis
                if n in last_seen:
                    gap = draw_idx - last_seen[n]
                    self.gap_data[n].append(gap)
                last_seen[n] = draw_idx
            
            # Pair frequency
            sorted_nums = sorted(numbers)
            for i in range(len(sorted_nums)):
                for j in range(i + 1, len(sorted_nums)):
                    self.pair_freq[(sorted_nums[i], sorted_nums[j])] += 1
    
    def get_hot_numbers(self, n=15):
        """Get the N most frequently drawn numbers."""
        return self.freq.most_common(n)
    
    def get_cold_numbers(self, n=15):
        """Get the N least frequently drawn numbers."""
        all_nums = {i: self.freq.get(i, 0) for i in range(1, self.max_number + 1)}
        sorted_nums = sorted(all_nums.items(), key=lambda x: x[1])
        return sorted_nums[:n]
    
    def get_overdue_numbers(self, n=10):
        """Get numbers that haven't appeared for the longest time (overdue)."""
        total_draws = len(self.history)
        last_seen = {}
        for draw_idx, numbers in enumerate(self.history):
            for num in numbers:
                last_seen[num] = draw_idx
        
        overdue = {}
        for num in range(1, self.max_number + 1):
            if num in last_seen:
                overdue[num] = total_draws - last_seen[num]
            else:
                overdue[num] = total_draws
        
        sorted_overdue = sorted(overdue.items(), key=lambda x: -x[1])
        return sorted_overdue[:n]
    
    def get_frequency_stats(self):
        """Get complete frequency statistics for all numbers."""
        total_draws = len(self.history)
        stats = {}
        
        last_seen = {}
        for draw_idx, numbers in enumerate(self.history):
            for num in numbers:
                last_seen[num] = draw_idx
        
        for num in range(1, self.max_number + 1):
            count = self.freq.get(num, 0)
            expected = total_draws * self.pick_count / self.max_number
            avg_gap = np.mean(self.gap_data[num]) if self.gap_data[num] else total_draws
            current_gap = total_draws - last_seen.get(num, 0)
            
            stats[num] = {
                'number': num,
                'count': count,
                'frequency': round(count / total_draws * 100, 2) if total_draws else 0,
                'expected': round(expected, 1),
                'deviation': round(count - expected, 1),
                'avg_gap': round(avg_gap, 1),
                'current_gap': current_gap,
                'overdue': bool(current_gap > avg_gap * 1.5)
            }
        
        return stats
    
    def predict(self, n_sets=5):
        """Generate predictions based on frequency analysis.
        
        Strategy: Combine hot numbers, overdue numbers, and balanced selection.
        Returns list of n_sets, each containing pick_count numbers.
        """
        total_draws = len(self.history)
        if total_draws < 10:
            # Not enough data - return random
            predictions = []
            for _ in range(n_sets):
                nums = sorted(np.random.choice(range(1, self.max_number + 1), 
                                               self.pick_count, replace=False).tolist())
                predictions.append(nums)
            return predictions
        
        stats = self.get_frequency_stats()
        
        # Score each number
        scores = {}
        for num, s in stats.items():
            # Hot number bonus
            hot_score = s['frequency']
            # Overdue bonus (if current gap > average gap)
            overdue_score = max(0, (s['current_gap'] - s['avg_gap']) / s['avg_gap'] * 20) if s['avg_gap'] > 0 else 0
            # Deviation from expected
            dev_score = max(0, -s['deviation']) * 2  # Boost under-represented numbers
            
            scores[num] = hot_score + overdue_score + dev_score
        
        # Normalize scores to probabilities
        total_score = sum(scores.values())
        probs = {num: score / total_score for num, score in scores.items()}
        
        numbers = list(probs.keys())
        weights = list(probs.values())
        
        predictions = []
        for _ in range(n_sets):
            # Weighted random selection
            selected = np.random.choice(numbers, size=self.pick_count, 
                                        replace=False, p=weights).tolist()
            predictions.append(sorted(selected))
        
        return predictions
    
    def get_analysis_summary(self):
        """Get a comprehensive analysis summary."""
        if not self.history:
            return {"error": "No data available"}
        
        total_draws = len(self.history)
        hot = self.get_hot_numbers(10)
        cold = self.get_cold_numbers(10)
        overdue = self.get_overdue_numbers(10)
        
        # Recent trend (last 30 draws)
        recent = self.history[-30:] if len(self.history) >= 30 else self.history
        recent_freq = Counter()
        for draw in recent:
            for n in draw:
                recent_freq[n] += 1
        recent_hot = recent_freq.most_common(10)
        
        # Sum distribution
        sums = [sum(draw) for draw in self.history]
        
        # Odd/Even distribution
        odd_counts = [sum(1 for n in draw if n % 2 == 1) for draw in self.history]
        
        return {
            'total_draws': total_draws,
            'hot_numbers': [{'number': n, 'count': c} for n, c in hot],
            'cold_numbers': [{'number': n, 'count': c} for n, c in cold],
            'overdue_numbers': [{'number': n, 'gap': g} for n, g in overdue],
            'recent_hot': [{'number': n, 'count': c} for n, c in recent_hot],
            'avg_sum': round(np.mean(sums), 1),
            'sum_range': [int(np.min(sums)), int(np.max(sums))],
            'avg_odd_count': round(np.mean(odd_counts), 1),
            'all_frequency': self.get_frequency_stats()
        }
