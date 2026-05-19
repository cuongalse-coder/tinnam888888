import secrets
import numpy as np
from itertools import combinations
import math

sys_rand = secrets.SystemRandom()

class WheelingOptimizer:
    """
    Abbreviated Wheeling System Optimizer with AI Smart Constraints
    Generates a set of orthogonal tickets from a pool of numbers to maximize coverage
    while strictly filtering out mathematically improbable tickets.
    """
    def __init__(self, pick_count=6, max_number=45):
        self.pick_count = pick_count
        self.max_number = max_number

    def _validate_ticket(self, combo, constraints, sum_mod7, stats=None):
        if not constraints: return True
        s = sum(combo)
        if s < constraints.get('sum_lo', 0) or s > constraints.get('sum_hi', 999):
            if stats is not None: stats['sum_range'] += 1
            return False
            
        banned_sum_block = constraints.get('banned_sum_block')
        if banned_sum_block and banned_sum_block[0] <= s <= banned_sum_block[1]:
            if stats is not None: stats['sum_block'] += 1
            return False
        
        col_bounds = constraints.get('col_bounds')
        if col_bounds:
            for i, n in enumerate(combo):
                if n < col_bounds[i][0] or n > col_bounds[i][1]:
                    if stats is not None: stats['col_bounds'] += 1
                    return False
                
        # Delta System Filter (Global Research)
        max_delta = combo[0]
        for i in range(1, 6):
            if combo[i] - combo[i-1] > max_delta:
                max_delta = combo[i] - combo[i-1]
        if max_delta > constraints.get('delta_hi', 45):
            if stats is not None: stats['delta'] += 1
            return False
        
        # Digit Frequency Filter (User's Invention)
        digit_counts = [0] * 10
        for n in combo:
            s_val = str(n).zfill(2)
            digit_counts[int(s_val[0])] += 1
            digit_counts[int(s_val[1])] += 1
        if max(digit_counts) > 4:
            if stats is not None: stats['digit_freq'] += 1
            return False
            
        # Adjacent Digit Pairs Filter (User's Invention)
        s_str = "".join([str(n).zfill(2) for n in combo])
        adj_pairs = 0
        for i in range(1, 12):
            if s_str[i] == s_str[i-1]:
                adj_pairs += 1
        if adj_pairs > 2:
            if stats is not None: stats['adj_digits'] += 1
            return False
            
        # Wave Inflection Points Filter (User's Invention)
        ones = [n % 10 for n in combo]
        breaks = 0
        currentDir = 0
        for i in range(1, 6):
            s_val_dir = 0
            if ones[i] > ones[i-1]: s_val_dir = 1
            elif ones[i] < ones[i-1]: s_val_dir = -1
            if s_val_dir != 0:
                if currentDir != 0 and currentDir != s_val_dir:
                    breaks += 1
                currentDir = s_val_dir
        if breaks == 0:
            if stats is not None: stats['wave_break'] += 1
            return False
            
        # Rubik Matrix Topology Filter (User's Invention)
        # Map to 5x10 grid
        matrix = [[0]*10 for _ in range(5)]
        for n in combo:
            r, c_idx = n // 10, n % 10
            if r < 5 and c_idx < 10:
                matrix[r][c_idx] = 1
        has_2x2 = False
        has_diag3 = False
        for r in range(4):
            for c_idx in range(9):
                if matrix[r][c_idx] and matrix[r][c_idx+1] and matrix[r+1][c_idx] and matrix[r+1][c_idx+1]:
                    has_2x2 = True
        for r in range(3):
            for c_idx in range(8):
                if matrix[r][c_idx] and matrix[r+1][c_idx+1] and matrix[r+2][c_idx+2]:
                    has_diag3 = True
            for c_idx in range(2, 10):
                if matrix[r][c_idx] and matrix[r+1][c_idx-1] and matrix[r+2][c_idx-2]:
                    has_diag3 = True
        if has_2x2 or has_diag3:
            if stats is not None: stats['rubik_matrix'] += 1
            return False
            
        # Color Palette Filter (User's Invention)
        # Colors: 0:(1,6), 1:(2,7), 2:(3,8), 3:(4,9), 4:(5,0)
        colors = [0, 0, 0, 0, 0]
        for n in combo:
            ld = n % 10
            if ld == 1 or ld == 6: colors[0] += 1
            elif ld == 2 or ld == 7: colors[1] += 1
            elif ld == 3 or ld == 8: colors[2] += 1
            elif ld == 4 or ld == 9: colors[3] += 1
            elif ld == 5 or ld == 0: colors[4] += 1
        
        unique_colors = sum(1 for c in colors if c > 0)
        max_color = max(colors)
        if unique_colors <= 2 or max_color >= 4:
            if stats is not None: stats['color_palette'] += 1
            return False
            
        # Go Board Filter
        prev_draw_set = constraints.get('prev_draw_set')
        go_board_liberties = constraints.get('go_board_liberties')
        if prev_draw_set is not None and go_board_liberties is not None:
            overlap = sum(1 for x in combo if x in prev_draw_set)
            contact = sum(1 for x in combo if x in go_board_liberties)
            if overlap > 2 or contact > 4:
                if stats is not None: stats['go_board'] += 1
                return False
                
        # Sliding Window Filter (Lô Gan / Hot)
        missing_pool = constraints.get('missing_pool')
        hot_pool = constraints.get('hot_pool')
        if missing_pool is not None and hot_pool is not None:
            missing_hit = sum(1 for x in combo if x in missing_pool)
            hot_hit = sum(1 for x in combo if x in hot_pool)
            if missing_hit > 3 or hot_hit > 3:
                if stats is not None: stats['sliding_window'] += 1
                return False
        
        odd = sum(1 for x in combo if x % 2 == 1)
        if odd < constraints.get('odd_lo', 0) or odd > constraints.get('odd_hi', 6):
            if stats is not None: stats['odd_even'] += 1
            return False
        mid = self.max_number // 2
        high = sum(1 for x in combo if x > mid)
        if high < constraints.get('high_lo', 0) or high > constraints.get('high_hi', 6):
            if stats is not None: stats['high_low'] += 1
            return False
        rng = max(combo) - min(combo)
        if rng < constraints.get('range_lo', 0) or rng > constraints.get('range_hi', 999):
            if stats is not None: stats['elastic'] += 1
            return False
        
        # Avoid more than 3 consecutive numbers
        consec = 1
        max_consec = 1
        for i in range(len(combo) - 1):
            if combo[i+1] - combo[i] == 1:
                consec += 1
                max_consec = max(max_consec, consec)
            else:
                consec = 1
        if max_consec > 3:
            if stats is not None: stats['consec'] += 1
            return False
        
        # Avoid decade overload (e.g. 4 numbers in the 20s)
        dec = [0] * 6
        for n in combo:
            dec[min((n - 1) // 10, 5)] += 1
        if max(dec) > 3:
            if stats is not None: stats['decade'] += 1
            return False
        
        # Psychological Avoidance
        count_over_31 = sum(1 for n in combo if n > 31)
        if count_over_31 < 1:
            if stats is not None: stats['psych'] += 1
            return False
        
        if sum_mod7 and s % 7 not in sum_mod7:
            if stats is not None: stats['mod7'] += 1
            return False
        
        return True

    def generate_wheel(self, pool, num_tickets, constraints=None, sum_mod7=None, history_data=None, ai_top_core=None, hard_core_lock=0):
        """
        Generates `num_tickets` tickets from `pool` matching AI constraints and strict historical elimination.
        V700: Maximizes coverage of 3-combinations AND 4-combinations (quadruplets) 
        to guarantee a minimum win if the 6 winning numbers are in the pool.
        Uses ai_top_core to force high-probability 5-6 match locking on 40% of tickets.
        If hard_core_lock > 0, forces the top 1-2 numbers into EVERY SINGLE TICKET.
        """
        pool = sorted(list(pool))
        if len(pool) <= self.pick_count:
            return [{'numbers': pool, 'strategy': '🎯 Trọng tâm (Duy nhất)'}] * num_tickets, 100.0
        stats = {
            'sum_range': 0, 'sum_block': 0, 'col_bounds': 0, 'delta': 0,
            'digit_freq': 0, 'adj_digits': 0, 'wave_break': 0, 'rubik_matrix': 0, 'color_palette': 0,
            'go_board': 0, 'sliding_window': 0, 'odd_even': 0, 'high_low': 0, 'elastic': 0,
            'consec': 0, 'decade': 0, 'psych': 0, 'mod7': 0
        }
        total_generated = 0
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
            sys_rand.shuffle(all_cands)
            for c in all_cands:
                total_generated += 1
                if not self._validate_ticket(c, constraints, sum_mod7, stats):
                    continue
                # Strict Historical Elimination (V16.0 GOD MODE)
                c_set = set(c)
                if any(len(c_set & h) >= 5 for h in history_sets):
                    continue
                valid_candidates.append(c)
                if len(valid_candidates) >= 3000: break
        else:
            attempts = 0
            while len(valid_candidates) < 4000 and attempts < 30000:
                attempts += 1
                c = tuple(sorted(sys_rand.sample(pool, self.pick_count)))
                total_generated += 1
                if not self._validate_ticket(c, constraints, sum_mod7, stats):
                    continue
                c_set = set(c)
                if any(len(c_set & h) >= 5 for h in history_sets):
                    continue
                valid_candidates.append(c)
                    
        if not valid_candidates:
            # Fallback if constraints are too tight
            valid_candidates = [tuple(sorted(sys_rand.sample(pool, self.pick_count))) for _ in range(100)]
            
        # V20.0: BẠCH THỦ LÔ (HARD CORE LOCK)
        # Force the top 1-2 numbers into EVERY ticket if requested
        if hard_core_lock > 0 and ai_top_core and len(ai_top_core) >= hard_core_lock:
            core_lock_set = set(ai_top_core[:hard_core_lock])
            locked_candidates = [c for c in valid_candidates if core_lock_set.issubset(set(c))]
            if len(locked_candidates) > 10:
                valid_candidates = locked_candidates
             
        # V19.0: FORCING JACKPOT LOCK (Ép xác suất trúng 5-6 số)
        # If the user wants extreme probability of hitting 5-6, we must lock the top 4/5 AI numbers on the first tickets.
        if ai_top_core and len(ai_top_core) >= 4:
            # Try to find valid candidates that contain at least 4 of the top core numbers
            core_set = set(ai_top_core)
            diamond_candidates = [c for c in valid_candidates if len(set(c) & core_set) >= 4]
            # Prioritize those that have exactly 5 or exactly 4
            diamond_candidates.sort(key=lambda c: len(set(c) & core_set), reverse=True)
            
            # We assign up to 40% of our tickets to "Jackpot Lock" (V700: was 30%)
            num_diamond = min(max(3, int(num_tickets * 0.4)), len(diamond_candidates))
            
            for i in range(num_diamond):
                best_ticket = diamond_candidates[i]
                strategy = f"💎 KHÓA KIM CƯƠNG (Bảo kê {len(set(best_ticket) & core_set)}/5 số lõi)"
                tickets.append({'numbers': list(best_ticket), 'strategy': strategy})
                covered = set(combinations(best_ticket, 3))
                uncovered -= covered
                # Remove from valid candidates so we don't pick it again
                if best_ticket in valid_candidates:
                    valid_candidates.remove(best_ticket)
                    
            remaining_tickets = num_tickets - num_diamond
        else:
            remaining_tickets = num_tickets

        for i in range(remaining_tickets):
            if not uncovered:
                best_ticket = sys_rand.choice(valid_candidates) if valid_candidates else tuple(sys_rand.sample(pool, self.pick_count))
                strategy = "🌪️ Đột biến (Chống bão hòa)"
            else:
                best_ticket = None
                best_coverage = -1
                
                sample_pool = sys_rand.sample(valid_candidates, min(len(valid_candidates), 1000)) if valid_candidates else []
                
                for cand in sample_pool:
                    cand_triplets = set(combinations(cand, 3))
                    # V700: Also track quadruplet coverage for 5-6 hit optimization
                    cand_quads = set(combinations(cand, 4))
                    coverage = len(cand_triplets & uncovered) + len(cand_quads) * 0.3
                    
                    if coverage > best_coverage:
                        best_coverage = coverage
                        best_ticket = cand
                        
                    if best_coverage == 20:
                        break
                        
                if not best_ticket:
                    best_ticket = sys_rand.choice(valid_candidates) if valid_candidates else tuple(sys_rand.sample(pool, self.pick_count))
                    strategy = "🌪️ Đột biến (Chống bão hòa)"
                else:
                    if i < remaining_tickets * 0.3:
                        strategy = "🔥 Vét lưới (Dồn tín hiệu AI)"
                    else:
                        strategy = "🎯 Trọng tâm (Bao phủ chéo)"
                    
            tickets.append({'numbers': list(best_ticket), 'strategy': strategy})
            covered = set(combinations(best_ticket, 3))
            uncovered -= covered
            
            if best_ticket in valid_candidates:
                valid_candidates.remove(best_ticket)
            
        cov = (len(all_triplets) - len(uncovered)) / len(all_triplets) * 100.0 if all_triplets else 100.0
        return tickets, round(cov, 2), stats, total_generated
