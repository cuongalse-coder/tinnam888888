const fs = require('fs');
const filepath = 'C:\\\\Users\\\\HQSP\\\\.gemini\\\\antigravity\\\\scratch\\\\tinnam888888_test\\\\models\\\\wheeling_optimizer.py';
let content = fs.readFileSync(filepath, 'utf8');

const radar_func = `
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
        
    def _calculate_spatial_score(self, ticket, radar_map, ai_top_core):
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
                
        return score, epicenter_count, dead_zone_count

    def generate_wheel`;

content = content.replace('    def generate_wheel', radar_func);

const start_idx = content.indexOf('    def generate_wheel');
const end_idx = content.length;

const new_wheel = `    def generate_wheel(self, pool, num_tickets, constraints=None, sum_mod7=None, history_data=None, ai_top_core=None, hard_core_lock=0, micro_sector=None):
        pool = sorted(list(pool))
        if len(pool) <= self.pick_count:
            return [{'numbers': pool, 'strategy': '🎯 Trọng tâm (Duy nhất)'}] * num_tickets, 100.0
            
        stats = {
            'sum_range': 0, 'sum_block': 0, 'col_bounds': 0, 'delta': 0,
            'digit_freq': 0, 'adj_digits': 0, 'wave_break': 0, 'rubik_matrix': 0, 'color_palette': 0,
            'go_board': 0, 'sliding_window': 0, 'markov_chain': 0, 'hacker_cipher': 0, 'freq_polarity': 0,
            'col_migration': 0, 'alphabet_cipher': 0, 'odd_even': 0, 'high_low': 0, 'elastic': 0,
            'consec': 0, 'decade': 0, 'psych': 0, 'mod7': 0, 'micro_sector': 0
        }
        
        history_sets = [set(d[:self.pick_count]) for d in history_data] if history_data else []
        radar_map = self._radar_scan_chessboard(history_data[-30:] if history_data else [])
        
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
            
        scored_candidates = []
        for cand in valid_candidates:
            score, epi_count, dead_count = self._calculate_spatial_score(cand, radar_map, ai_top_core)
            if hard_core_lock > 0 and ai_top_core and len(ai_top_core) >= hard_core_lock:
                core_set = set(ai_top_core[:hard_core_lock])
                if not core_set.issubset(set(cand)):
                    score -= 9999.0
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
`;

content = content.substring(0, start_idx) + new_wheel;
fs.writeFileSync(filepath, content, 'utf8');
console.log('Update success');
