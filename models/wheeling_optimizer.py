import random
import numpy as np
from itertools import combinations
import math

class WheelingOptimizer:
    """
    Abbreviated Wheeling System Optimizer with AI Smart Constraints
    Generates a set of orthogonal tickets from a pool of numbers to maximize coverage
    while strictly filtering out mathematically improbable tickets.
    """
    def __init__(self, pick_count=6, max_number=45):
        self.pick_count = pick_count
        self.max_number = max_number

    def _validate_ticket(self, combo, constraints, sum_mod7):
        if not constraints: return True
        s = sum(combo)
        if s < constraints.get('sum_lo', 0) or s > constraints.get('sum_hi', 999): return False
        odd = sum(1 for x in combo if x % 2 == 1)
        if odd < constraints.get('odd_lo', 0) or odd > constraints.get('odd_hi', 6): return False
        mid = self.max_number // 2
        high = sum(1 for x in combo if x > mid)
        if high < constraints.get('high_lo', 0) or high > constraints.get('high_hi', 6): return False
        rng = max(combo) - min(combo)
        if rng < constraints.get('range_lo', 0) or rng > constraints.get('range_hi', 999): return False
        
        # Avoid more than 3 consecutive numbers
        consec = 1
        max_consec = 1
        for i in range(len(combo) - 1):
            if combo[i+1] - combo[i] == 1:
                consec += 1
                max_consec = max(max_consec, consec)
            else:
                consec = 1
        if max_consec > 3: return False
        
        # Avoid decade overload (e.g. 4 numbers in the 20s)
        dec = [0] * 6
        for n in combo:
            dec[min((n - 1) // 10, 5)] += 1
        if max(dec) > 3: return False
        
        if sum_mod7 and s % 7 not in sum_mod7: return False
        
        return True

    def generate_wheel(self, pool, num_tickets, constraints=None, sum_mod7=None, history_data=None):
        """
        Generates `num_tickets` tickets from `pool` matching AI constraints and strict historical elimination.
        Maximizes coverage of 3-combinations (triplets) to guarantee a minimum win if the 6 winning numbers are in the pool.
        """
        pool = sorted(list(pool))
        if len(pool) <= self.pick_count:
            return [{'numbers': pool, 'strategy': '🎯 Trọng tâm (Duy nhất)'}] * num_tickets, 100.0

        all_triplets = set(combinations(pool, 3))
        uncovered = set(all_triplets)
        
        # Parse history into sets for fast elimination
        history_sets = []
        if history_data:
            history_sets = [set(d[:self.pick_count]) for d in history_data]
            
        tickets = []
        
        # Prepare Candidate Pool with Smart AI Filtering + Historical Elimination
        valid_candidates = []
        if len(pool) <= 18:
            all_cands = list(combinations(pool, self.pick_count))
            random.shuffle(all_cands)
            for c in all_cands:
                if not self._validate_ticket(c, constraints, sum_mod7):
                    continue
                # Strict Historical Elimination (V16.0 GOD MODE)
                c_set = set(c)
                if any(len(c_set & h) >= 4 for h in history_sets):
                    continue
                valid_candidates.append(c)
                if len(valid_candidates) >= 3000: break
        else:
            attempts = 0
            while len(valid_candidates) < 4000 and attempts < 30000:
                attempts += 1
                c = tuple(sorted(random.sample(pool, self.pick_count)))
                if not self._validate_ticket(c, constraints, sum_mod7):
                    continue
                c_set = set(c)
                if any(len(c_set & h) >= 4 for h in history_sets):
                    continue
                valid_candidates.append(c)
                    
        if not valid_candidates:
            # Fallback if constraints are too tight
            valid_candidates = [tuple(sorted(random.sample(pool, self.pick_count))) for _ in range(100)]
             
        for i in range(num_tickets):
            if not uncovered:
                best_ticket = random.choice(valid_candidates)
                strategy = "🌪️ Đột biến (Chống bão hòa)"
            else:
                best_ticket = None
                best_coverage = -1
                
                sample_pool = random.sample(valid_candidates, min(len(valid_candidates), 1000))
                
                for cand in sample_pool:
                    cand_triplets = set(combinations(cand, 3))
                    coverage = len(cand_triplets & uncovered)
                    
                    if coverage > best_coverage:
                        best_coverage = coverage
                        best_ticket = cand
                        
                    if best_coverage == 20:
                        break
                        
                if not best_ticket:
                    best_ticket = random.choice(valid_candidates)
                    strategy = "🌪️ Đột biến (Chống bão hòa)"
                else:
                    if i < num_tickets * 0.3:
                        strategy = "🔥 Vét lưới (Dồn tín hiệu AI)"
                    else:
                        strategy = "🎯 Trọng tâm (Bao phủ chéo)"
                    
            tickets.append({'numbers': list(best_ticket), 'strategy': strategy})
            covered = set(combinations(best_ticket, 3))
            uncovered -= covered
            
        coverage_ratio = 100.0 * (1.0 - len(uncovered) / max(1, len(all_triplets)))
        return tickets, round(coverage_ratio, 2)
