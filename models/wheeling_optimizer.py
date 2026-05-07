import random
import numpy as np
from itertools import combinations
import math

class WheelingOptimizer:
    """
    Abbreviated Wheeling System Optimizer
    Generates a set of orthogonal tickets from a pool of numbers to maximize coverage.
    """
    def __init__(self, pick_count=6):
        self.pick_count = pick_count

    def generate_wheel(self, pool, num_tickets):
        """
        Generates `num_tickets` tickets from `pool`.
        Maximizes coverage of 3-combinations (triplets) to guarantee a minimum win if the 6 winning numbers are in the pool.
        """
        pool = sorted(list(pool))
        if len(pool) <= self.pick_count:
            return [pool] * num_tickets, 100.0

        all_triplets = set(combinations(pool, 3))
        uncovered = set(all_triplets)
        
        tickets = []
        
        # Prepare candidate pool
        if len(pool) <= 18:
             sampled_candidates = list(combinations(pool, self.pick_count))
             random.shuffle(sampled_candidates)
             sampled_candidates = sampled_candidates[:3000] # Limit size for performance
        else:
             sampled_candidates = [tuple(sorted(random.sample(pool, self.pick_count))) for _ in range(4000)]
             
        for _ in range(num_tickets):
            if not uncovered:
                # If fully covered but we still need tickets, just add random orthogonal ones
                best_ticket = tuple(sorted(random.sample(pool, self.pick_count)))
            else:
                best_ticket = None
                best_coverage = -1
                
                for cand in sampled_candidates:
                    cand_triplets = set(combinations(cand, 3))
                    coverage = len(cand_triplets & uncovered)
                    
                    if coverage > best_coverage:
                        best_coverage = coverage
                        best_ticket = cand
                        
                    if best_coverage == 20: # 6 choose 3 = 20, max possible
                        break
                        
                if not best_ticket:
                    best_ticket = tuple(sorted(random.sample(pool, self.pick_count)))
                    
            tickets.append(list(best_ticket))
            covered = set(combinations(best_ticket, 3))
            uncovered -= covered
            
        coverage_ratio = 100.0 * (1.0 - len(uncovered) / max(1, len(all_triplets)))
        return tickets, round(coverage_ratio, 2)
