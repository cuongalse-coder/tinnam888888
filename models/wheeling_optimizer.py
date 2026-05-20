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
                
        # Markov Transition Filter
        markov_transitions = constraints.get('markov_transitions')
        prev_draw = constraints.get('prev_draw')
        if markov_transitions is not None and prev_draw is not None:
            markov_pass = 0
            for i in range(len(combo)):
                p_val = prev_draw[i]
                c_val = combo[i]
                if p_val in markov_transitions[i] and c_val in markov_transitions[i][p_val]:
                    markov_pass += 1
            if markov_pass < 4:
                if stats is not None: stats['markov_chain'] += 1
                return False
        
        # Hacker Cipher 12-bit Filter (User's Invention)
        s_bin = ""
        for n in combo:
            n_str = str(n).zfill(2)
            s_bin += "0" if int(n_str[0]) % 2 == 0 else "1"
            s_bin += "0" if int(n_str[1]) % 2 == 0 else "1"
        max_0 = max(len(c) for c in s_bin.split("1"))
        max_1 = max(len(c) for c in s_bin.split("0"))
        is_pal = (s_bin == s_bin[::-1])
        is_alt = (s_bin == "010101010101" or s_bin == "101010101010")
        if max_0 >= 7 or max_1 >= 7 or is_pal or is_alt:
            if stats is not None: stats['hacker_cipher'] += 1
            return False
            
        # Frequency Polarity Filter
        top_frequent = constraints.get('top_frequent')
        if top_frequent:
            hit_top = sum(1 for x in combo if x in top_frequent)
            if hit_top < 2 or hit_top > 4:
                if stats is not None: stats['freq_polarity'] += 1
                return False
                
        # Column Migration Filter
        if prev_draw is not None:
            prev_draw_set_loc = constraints.get('prev_draw_set')
            if prev_draw_set_loc:
                for new_col in range(len(combo)):
                    num = combo[new_col]
                    if num in prev_draw_set_loc:
                        old_col = prev_draw.index(num)
                        if abs(new_col - old_col) >= 3:
                            if stats is not None: stats['col_migration'] += 1
                            return False
                            
        # Alphabet Decade Cipher
        alphabet_patterns = constraints.get('alphabet_patterns')
        if alphabet_patterns is not None:
            word = "".join("A" if x<=9 else "B" if x<=19 else "C" if x<=29 else "D" if x<=39 else "E" if x<=49 else "F" for x in combo)
            if word not in alphabet_patterns:
                if stats is not None: stats['alphabet_cipher'] += 1
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
            
        # Micro-Sector Targeting Filter
        micro_sector = constraints.get('micro_sector')
        if micro_sector:
            if 'odd' in micro_sector and odd != micro_sector['odd']:
                if stats is not None: stats['micro_sector'] += 1
                return False
            if 'high' in micro_sector and high != micro_sector['high']:
                if stats is not None: stats['micro_sector'] += 1
                return False
            if 'overlap' in micro_sector:
                prev_draw_set_loc = constraints.get('prev_draw_set')
                if prev_draw_set_loc:
                    overlap_count = sum(1 for x in combo if x in prev_draw_set_loc)
                    ov_target = micro_sector['overlap']
                    if ov_target == 3:
                        if overlap_count < 3:
                            if stats is not None: stats['micro_sector'] += 1
                            return False
                    elif overlap_count != ov_target:
                        if stats is not None: stats['micro_sector'] += 1
                        return False
            if 'alphabet' in micro_sector and alphabet_patterns is not None:
                word = "".join("A" if x<=9 else "B" if x<=19 else "C" if x<=29 else "D" if x<=39 else "E" if x<=49 else "F" for x in combo)
                if word != micro_sector['alphabet']:
                    if stats is not None: stats['micro_sector'] += 1
                    return False
            if 'mod_x' in micro_sector:
                if sum(combo[:3]) % 8 != micro_sector['mod_x']:
                    if stats is not None: stats['micro_sector'] += 1
                    return False
            if 'mod_y' in micro_sector:
                if sum(combo[3:]) % 8 != micro_sector['mod_y']:
                    if stats is not None: stats['micro_sector'] += 1
                    return False
            if 'sub_delta' in micro_sector:
                if (combo[5] - combo[0]) != micro_sector['sub_delta']:
                    if stats is not None: stats['micro_sector'] += 1
                    return False
            if 'sub_midsum' in micro_sector:
                if (combo[2] + combo[3]) != micro_sector['sub_midsum']:
                    if stats is not None: stats['micro_sector'] += 1
                    return False
        
        return True


    def _radar_scan_chessboard(self, history_data):
        radar_map = {}
        for num in range(1, self.max_number + 1):
            radar_map[num] = {'hits': 0, 'last_seen': 999}
            
        if not history_data:
            return radar_map
            
        n = len(history_data)
        for i, draw in enumerate(history_data):
            for num in draw[:self.pick_count]:
                if num in radar_map:
                    radar_map[num]['hits'] += 1
                    radar_map[num]['last_seen'] = n - 1 - i
        return radar_map
        

    def _build_entanglement_matrix(self, history_data):
        parasitic = set()
        symbiotic = {}
        if not history_data or len(history_data) < 50:
            return parasitic, symbiotic
            
        pair_counts = {}
        single_counts = {}
        
        # Build co-occurrence matrix from last 500 draws max
        scan_data = history_data[-500:]
        
        for d in scan_data:
            draw = d[:self.pick_count]
            for num in draw:
                single_counts[num] = single_counts.get(num, 0) + 1
            for i in range(len(draw)):
                for j in range(i+1, len(draw)):
                    p = tuple(sorted((draw[i], draw[j])))
                    pair_counts[p] = pair_counts.get(p, 0) + 1
                    
        # Parasitic (Tương Khắc): Cả 2 số nổ khá nhiều (>10 lần), nhưng CHƯA BAO GIỜ nổ chung
        import itertools
        for a, b in itertools.combinations(range(1, self.max_number + 1), 2):
            if single_counts.get(a, 0) > 10 and single_counts.get(b, 0) > 10:
                p = tuple(sorted((a, b)))
                if pair_counts.get(p, 0) == 0:
                    parasitic.add(p)
                    
        # Symbiotic (Tương Sinh): Tỉ lệ nổ chung / tổng số lần nổ của A rất cao (>60%)
        for p, count in pair_counts.items():
            if count >= 4:
                a, b = p
                ca = single_counts.get(a, 1)
                cb = single_counts.get(b, 1)
                if count / ca > 0.6: symbiotic[a] = b
                if count / cb > 0.6: symbiotic[b] = a
                
        return parasitic, symbiotic
        
    def _calculate_spatial_score(self, ticket, radar_map, ai_top_core, symbiotic):
        score = 0.0
        dead_zone_count = 0
        epicenter_count = 0
        
        for num in ticket:
            if ai_top_core and num in ai_top_core:
                idx = ai_top_core.index(num)
                score += (30 - idx) * 2.0
                
            hits = radar_map[num]['hits']
            last_seen = radar_map[num]['last_seen']
            
            if last_seen > 15:
                dead_zone_count += 1
                score -= 10.0
            elif last_seen <= 2:
                epicenter_count += 1
                score += 5.0
                
        # V2500 Symbiotic Check
        for num in ticket:
            if num in symbiotic:
                partner = symbiotic[num]
                if partner in ticket:
                    score += 50.0 # Thưởng khủng cho Cặp Bài Trùng
                else:
                    score -= 20.0 # Trừ điểm nặng vì chia cắt Cặp Bài Trùng
                    
        return score, epicenter_count, dead_zone_count

    def generate_wheel(self, pool, num_tickets, constraints=None, sum_mod7=None, history_data=None, ai_top_core=None, hard_core_lock=0, micro_sector=None):
        pool = sorted(list(pool))
        if len(pool) <= self.pick_count:
            return [{'numbers': pool, 'strategy': '🎯 Trọng tâm (Duy nhất)'}] * num_tickets, 100.0
            
        stats = {
            'sum_range': 0, 'sum_block': 0, 'col_bounds': 0, 'delta': 0,
            'digit_freq': 0, 'adj_digits': 0, 'wave_break': 0, 'rubik_matrix': 0, 'color_palette': 0,
            'go_board': 0, 'sliding_window': 0, 'markov_chain': 0, 'hacker_cipher': 0, 'freq_polarity': 0,
            'col_migration': 0, 'alphabet_cipher': 0, 'odd_even': 0, 'high_low': 0, 'elastic': 0,
            'consec': 0, 'decade': 0, 'psych': 0, 'mod7': 0, 'micro_sector': 0,
            'entangled_rejected': 0
        }
        
        parasitic, symbiotic = self._build_entanglement_matrix(history_data)
        self.current_parasitic = parasitic
        
        history_sets = [set(d[:self.pick_count]) for d in history_data] if history_data else []
        radar_map = self._radar_scan_chessboard(history_data[-30:] if history_data else [])
        
        # V2000 DYNAMIC FILTERS (REINFORCEMENT LEARNING BOUNDS)
        if history_data and len(history_data) >= 5:
            recent_5 = history_data[-5:]
            recent_sums = [sum(d[:self.pick_count]) for d in recent_5]
            avg_recent_sum = sum(recent_sums) / 5
            
            recent_evens = sum(1 for d in recent_5 for n in d[:self.pick_count] if n % 2 == 0)
            
            target_sum_mean = 122 if self.max_number == 45 else 150
            
            if constraints is None:
                constraints = {}
                
            # Regression to the mean for Sums
            if avg_recent_sum < target_sum_mean - 20:
                constraints['sum_min'] = target_sum_mean
                constraints['sum_max'] = target_sum_mean + 40
            elif avg_recent_sum > target_sum_mean + 20:
                constraints['sum_min'] = target_sum_mean - 40
                constraints['sum_max'] = target_sum_mean
                
            # Regression for Odd/Even
            if recent_evens > 20: # Quá nhiều chẵn (trung bình là 15)
                constraints['odd_even'] = [4, 5] # Ép ra lẻ
            elif recent_evens < 10:
                constraints['odd_even'] = [1, 2] # Ép ra chẵn

        import itertools
        if len(pool) > 22:
            pool = pool[:22]
            
        all_cands = list(itertools.combinations(pool, self.pick_count))
        
        valid_candidates = []
        for c in all_cands:
            if not self._validate_ticket(c, constraints, sum_mod7, stats):
                continue
            c_set = set(c)
            if any(len(c_set & h) >= 5 for h in history_sets):
                continue
            valid_candidates.append(c)
            
        if not valid_candidates:
            valid_candidates = list(all_cands[:100])
            
        # V2700 Darwinian Survival Clustering
        if stats is not None:
            stats['survival_pool_size'] = len(valid_candidates)
            
        survival_densities = {}
        # Áp dụng thuật toán sinh tồn nếu đàn còn dưới 5000 cá thể (Tránh O(N^2) quá lớn)
        if 0 < len(valid_candidates) <= 5000:
            for i, cand_a in enumerate(valid_candidates):
                density = 0
                set_a = set(cand_a)
                # Tính lượng cá thể "cùng đàn" (chia sẻ ít nhất 4 gen giống nhau)
                for j, cand_b in enumerate(valid_candidates):
                    if i == j: continue
                    if len(set_a & set(cand_b)) >= 4:
                        density += 1
                survival_densities[tuple(cand_a)] = density
            
        scored_candidates = []
        for cand in valid_candidates:
            score, epi_count, dead_count = self._calculate_spatial_score(cand, radar_map, ai_top_core, symbiotic)
            if hard_core_lock > 0 and ai_top_core and len(ai_top_core) >= hard_core_lock:
                core_set = set(ai_top_core[:hard_core_lock])
                if not core_set.issubset(set(cand)):
                    score -= 9999.0
                    
            # V2700: Cộng điểm sinh tồn (Survival Points)
            cand_tuple = tuple(cand)
            if cand_tuple in survival_densities:
                d_score = survival_densities[cand_tuple]
                if d_score == 0:
                    score -= 50 # Cá thể mồ côi đột biến, trảm!
                else:
                    score += min(300, d_score * 3) # Nằm sâu trong lõi bầy đàn, cộng điểm cực lớn
                    
            scored_candidates.append((score, epi_count, dead_count, cand))
            
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        tickets = []
        epicenter_pool = [x for x in scored_candidates if x[1] >= 2 and x[2] == 0]
        recovery_pool = [x for x in scored_candidates if x[2] >= 1]
        perimeter_pool = [x for x in scored_candidates if x not in epicenter_pool and x not in recovery_pool]
        
        def add_ticket(pool_list, fallback_list, count, strategy_name):
            added = 0
            for item in pool_list:
                if added >= count: break
                tickets.append({'numbers': list(item[3]), 'strategy': strategy_name})
                added += 1
            if added < count:
                for item in fallback_list:
                    if added >= count: break
                    if not any(set(item[3]) == set(t['numbers']) for t in tickets):
                        tickets.append({'numbers': list(item[3]), 'strategy': strategy_name})
                        added += 1

        add_ticket(epicenter_pool, scored_candidates, 5, '🔥 Tâm chấn Radar (Khóa mục tiêu)')
        add_ticket(perimeter_pool, scored_candidates, 10, '🛰️ Vành đai không gian (Bao vây)')
        add_ticket(recovery_pool, scored_candidates, 5, '♻️ Vùng phục hồi (Đón lõng)')
        
        if num_tickets != 20:
            if num_tickets < 20:
                tickets = tickets[:num_tickets]
            else:
                for item in scored_candidates:
                    if len(tickets) >= num_tickets: break
                    if not any(set(item[3]) == set(t['numbers']) for t in tickets):
                        tickets.append({'numbers': list(item[3]), 'strategy': '🛸 Radar mở rộng'})
                        
        return tickets, 100.0, stats, len(all_cands)
