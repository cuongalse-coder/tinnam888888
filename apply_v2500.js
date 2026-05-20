const fs = require('fs');
const filepath = 'C:\\\\Users\\\\HQSP\\\\.gemini\\\\antigravity\\\\scratch\\\\tinnam888888_test\\\\models\\\\wheeling_optimizer.py';
let content = fs.readFileSync(filepath, 'utf8');

const matrix_func = `
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
        
    def _calculate_spatial_score`;

content = content.replace("    def _calculate_spatial_score", matrix_func);

const spatial_old = `            if last_seen > 15:
                dead_zone_count += 1
                score -= 10.0
            elif last_seen <= 2:
                epicenter_count += 1
                score += 5.0
                
        return score, epicenter_count, dead_zone_count`;
        
const spatial_new = `            if last_seen > 15:
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
                    
        return score, epicenter_count, dead_zone_count`;
        
content = content.replace(spatial_old, spatial_new);
content = content.replace("def _calculate_spatial_score(self, ticket, radar_map, ai_top_core):", "def _calculate_spatial_score(self, ticket, radar_map, ai_top_core, symbiotic):");

const val_old = `        if 'mod7' in stats: stats['mod7'] += 1
        
        # Micro sector targeting`;
        
const val_new = `        if 'mod7' in stats: stats['mod7'] += 1
        
        # V2500 Parasitic Check (Tương Khắc)
        if hasattr(self, 'current_parasitic'):
            import itertools
            for a, b in itertools.combinations(combo, 2):
                if (a, b) in self.current_parasitic:
                    if stats is not None: 
                        stats['entangled_rejected'] = stats.get('entangled_rejected', 0) + 1
                    return False
        
        # Micro sector targeting`;

content = content.replace(val_old, val_new);

const gen_old = `        stats = {
            'sum_range': 0, 'sum_block': 0, 'col_bounds': 0, 'delta': 0,
            'digit_freq': 0, 'adj_digits': 0, 'wave_break': 0, 'rubik_matrix': 0, 'color_palette': 0,
            'go_board': 0, 'sliding_window': 0, 'markov_chain': 0, 'hacker_cipher': 0, 'freq_polarity': 0,
            'col_migration': 0, 'alphabet_cipher': 0, 'odd_even': 0, 'high_low': 0, 'elastic': 0,
            'consec': 0, 'decade': 0, 'psych': 0, 'mod7': 0, 'micro_sector': 0
        }`;

const gen_new = `        stats = {
            'sum_range': 0, 'sum_block': 0, 'col_bounds': 0, 'delta': 0,
            'digit_freq': 0, 'adj_digits': 0, 'wave_break': 0, 'rubik_matrix': 0, 'color_palette': 0,
            'go_board': 0, 'sliding_window': 0, 'markov_chain': 0, 'hacker_cipher': 0, 'freq_polarity': 0,
            'col_migration': 0, 'alphabet_cipher': 0, 'odd_even': 0, 'high_low': 0, 'elastic': 0,
            'consec': 0, 'decade': 0, 'psych': 0, 'mod7': 0, 'micro_sector': 0,
            'entangled_rejected': 0
        }
        
        parasitic, symbiotic = self._build_entanglement_matrix(history_data)
        self.current_parasitic = parasitic`;

content = content.replace(gen_old, gen_new);

const score_old = `score, epi_count, dead_count = self._calculate_spatial_score(cand, radar_map, ai_top_core)`;
const score_new = `score, epi_count, dead_count = self._calculate_spatial_score(cand, radar_map, ai_top_core, symbiotic)`;
content = content.replace(score_old, score_new);

fs.writeFileSync(filepath, content, 'utf8');
console.log('Update success');
