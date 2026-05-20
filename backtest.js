const fs = require('fs');

async function main() {
    console.log("📡 Đang tải dữ liệu từ GitHub...");
    const url = "https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl";
    const resp = await fetch(url);
    const text = await resp.text();
    
    let data = [];
    for (let line of text.trim().split('\n')) {
        if (line) {
            let obj = JSON.parse(line);
            if (obj.result && obj.result.length >= 6) {
                let draw = obj.result.slice(0, 6).map(Number).sort((a,b)=>a-b);
                data.push(draw);
            }
        }
    }
    
    console.log(`✅ Đã tải ${data.length} kỳ quay lịch sử (Mega 6/45)`);

    const MAX_NUMBER = 45;
    const PICK = 6;
    const ALL_NUMBERS = Array.from({length: MAX_NUMBER}, (_, i) => i + 1);

    // Helpers
    function getCombinations(arr, k) {
        let results = [];
        function helper(start, combo) {
            if (combo.length === k) {
                results.push([...combo]);
                return;
            }
            for (let i = start; i < arr.length; i++) {
                combo.push(arr[i]);
                helper(i + 1, combo);
                combo.pop();
            }
        }
        helper(0, []);
        return results;
    }

    class Counter {
        constructor() { this.map = new Map(); }
        add(key, val = 1) { this.map.set(key, (this.map.get(key) || 0) + val); }
        get(key) { return this.map.get(key) || 0; }
        mostCommon(n) {
            return [...this.map.entries()].sort((a, b) => b[1] - a[1]).slice(0, n).map(x => x[0]);
        }
    }

    function model_markov_chain(hist) {
        let transitions = new Map();
        for (let i = 0; i < hist.length - 1; i++) {
            let curr = hist[i].join(',');
            if (!transitions.has(curr)) transitions.set(curr, new Counter());
            for (let num of hist[i + 1]) transitions.get(curr).add(num);
        }
        if (hist.length > 0) {
            let last = hist[hist.length - 1].join(',');
            if (transitions.has(last)) {
                return transitions.get(last).mostCommon(6).map(Number);
            }
        }
        let freq = new Counter();
        let recent = hist.slice(-20);
        for (let d of recent) for (let n of d) freq.add(n);
        return freq.mostCommon(6).map(Number);
    }

    function model_gap_overdue(hist, top_n = 15) {
        let last_seen = {};
        for (let n of ALL_NUMBERS) last_seen[n] = -1;
        for (let i = 0; i < hist.length; i++) {
            for (let num of hist[i]) last_seen[num] = i;
        }
        let current_idx = hist.length;
        let gaps = {};
        for (let n of ALL_NUMBERS) gaps[n] = current_idx - last_seen[n];

        let avg_gaps = {};
        let last_idx = {};
        for (let n of ALL_NUMBERS) avg_gaps[n] = [];
        
        for (let i = 0; i < hist.length; i++) {
            for (let num of hist[i]) {
                if (last_idx[num] !== undefined) {
                    avg_gaps[num].push(i - last_idx[num]);
                }
                last_idx[num] = i;
            }
        }

        let due_scores = [];
        for (let num of ALL_NUMBERS) {
            if (avg_gaps[num].length > 0) {
                let mean_gap = avg_gaps[num].reduce((a,b)=>a+b, 0) / avg_gaps[num].length;
                due_scores.push({num: num, score: gaps[num] / (mean_gap + 0.1)});
            } else {
                due_scores.push({num: num, score: 0});
            }
        }
        due_scores.sort((a,b) => b.score - a.score);
        return due_scores.slice(0, top_n).map(x => x.num);
    }

    function model_momentum_neural(hist) {
        let weights = {};
        for (let n of ALL_NUMBERS) weights[n] = 0;
        let total = hist.length;
        for (let i = 0; i < total; i++) {
            let decay = 1 / (1 + Math.exp(-(i - total + 20) / 5));
            for (let num of hist[i]) weights[num] += decay;
        }
        let arr = ALL_NUMBERS.map(n => ({num: n, score: weights[n]}));
        arr.sort((a,b) => b.score - a.score);
        return arr.slice(0, 6).map(x => x.num);
    }

    function model_advanced_ml(hist) {
        return model_momentum_neural(hist);
    }

    function setUnion(s1, s2) { return new Set([...s1, ...s2]); }
    function setIntersection(s1, s2) { return new Set([...s1].filter(x => s2.has(x))); }
    function setDiff(s1, s2) { return new Set([...s1].filter(x => !s2.has(x))); }

    function model_knn_mirror(hist) {
        if (hist.length < 20) return model_momentum_neural(hist);
        let pattern = setUnion(setUnion(new Set(hist[hist.length-1]), new Set(hist[hist.length-2])), new Set(hist[hist.length-3]));
        if (hist.length > 3) pattern = setUnion(pattern, new Set(hist[hist.length-4]));
        let n = hist.length;
        let sims = [];
        for (let i = 3; i < n - 3; i++) {
            let past = setUnion(setUnion(setUnion(new Set(hist[i]), new Set(hist[i-1])), new Set(hist[i-2])), new Set(hist[i-3]));
            let inter = setIntersection(pattern, past).size;
            let recency = 1.0 + 0.5 * (i / n);
            if (inter >= 5) sims.push({score: inter * recency, next: i + 1});
        }
        sims.sort((a,b) => b.score - a.score);
        let votes = new Counter();
        for (let item of sims.slice(0, 30)) {
            if (item.next < n) {
                for (let num of hist[item.next]) votes.add(num, item.score);
            }
        }
        if (votes.map.size === 0) return model_momentum_neural(hist);
        return votes.mostCommon(20).map(Number);
    }

    function model_pair_matrix(hist) {
        if (hist.length < 30) return model_gap_overdue(hist);
        let pair_scores = new Counter();
        let n = hist.length;
        for (let i = 0; i < n; i++) {
            let decay = 0.3 + 0.7 * (i / n);
            let combos = getCombinations(hist[i].slice(0,6), 2);
            for (let c of combos) pair_scores.add(c.join(','), decay);
        }
        let last_draw = new Set(hist[hist.length-1].slice(0,6));
        let cand = new Counter();
        for (let num of ALL_NUMBERS) {
            if (last_draw.has(num)) continue;
            for (let anchor of last_draw) {
                let key = [num, anchor].sort((a,b)=>a-b).join(',');
                cand.add(num, pair_scores.get(key));
            }
        }
        let triplet_bonus = new Counter();
        for (let i = Math.max(0, n - 100); i < n; i++) {
            let trips = getCombinations(hist[i].slice(0,6), 3);
            for (let trip of trips) {
                let ts = new Set(trip);
                let overlap = setIntersection(ts, last_draw).size;
                if (overlap >= 2) {
                    for (let num of setDiff(ts, last_draw)) triplet_bonus.add(num, 1.5);
                }
            }
        }
        for (let [num, score] of triplet_bonus.map.entries()) cand.add(num, score);
        return cand.mostCommon(15).map(Number);
    }

    function model_delta_momentum(hist) {
        if (hist.length < 30) return model_momentum_neural(hist);
        let scores = [];
        for (let num of ALL_NUMBERS) {
            let last5 = hist.slice(-5);
            let last10_5 = hist.slice(-10, -5);
            let last15 = hist.slice(-15);
            let last30_15 = hist.slice(-30, -15);
            
            let f5 = last5.filter(d => d.includes(num)).length / 5;
            let f5p = last10_5.filter(d => d.includes(num)).length / 5;
            let f15 = last15.filter(d => d.includes(num)).length / 15;
            let f15p = last30_15.filter(d => d.includes(num)).length / 15;
            
            let ds = f5 - f5p;
            let dm = f15 - f15p;
            let m = ds * 3 + dm * 2;
            if (hist[hist.length-1].includes(num)) m += 0.5;
            if (hist.length >= 2 && hist[hist.length-2].includes(num)) m += 0.3;
            scores.push({num: num, score: m});
        }
        scores.sort((a,b) => b.score - a.score);
        return scores.slice(0, 15).map(x => x.num);
    }

    function model_cond_prob(hist) {
        if (hist.length < 30) return [];
        let last = new Set(hist[hist.length-1]);
        let cc = new Map();
        let tg = new Map();
        for (let i = 0; i < hist.length - 1; i++) {
            for (let g of hist[i]) {
                tg.set(g, (tg.get(g)||0) + 1);
                if (!cc.has(g)) cc.set(g, new Map());
                for (let nn of hist[i+1]) {
                    cc.get(g).set(nn, (cc.get(g).get(nn)||0) + 1);
                }
            }
        }
        let scores = [];
        for (let num of ALL_NUMBERS) {
            let sum = 0;
            for (let g of last) {
                if (tg.get(g) > 0 && cc.has(g)) {
                    sum += (cc.get(g).get(num) || 0) / tg.get(g);
                }
            }
            scores.push({num: num, score: sum});
        }
        scores.sort((a,b) => b.score - a.score);
        return scores.slice(0, 15).map(x => x.num);
    }

    function model_freq_gap_hybrid(hist) {
        if (hist.length < 30) return model_gap_overdue(hist);
        let expected = 6 / MAX_NUMBER;
        let scores = [];
        for (let num of ALL_NUMBERS) {
            let f5 = hist.slice(-5).filter(d => d.includes(num)).length / 5;
            let f15 = hist.slice(-15).filter(d => d.includes(num)).length / 15;
            let fs = (f5 / (expected + 0.01)) * 0.6 + (f15 / (expected + 0.01)) * 0.4;
            
            let last_seen = -1;
            for (let i = hist.length - 1; i >= 0; i--) {
                if (hist[i].includes(num)) { last_seen = i; break; }
            }
            let gap = last_seen >= 0 ? hist.length - last_seen : hist.length;
            
            let appearances = [];
            for (let i = 0; i < hist.length; i++) if (hist[i].includes(num)) appearances.push(i);
            
            let mg = MAX_NUMBER / 6;
            if (appearances.length >= 2) {
                let gs = [];
                for (let i = 0; i < appearances.length - 1; i++) gs.push(appearances[i+1] - appearances[i]);
                mg = gs.reduce((a,b)=>a+b, 0) / gs.length;
            }
            
            let od = gap / (mg + 0.1);
            let s = 0;
            if (fs > 0.8 && od > 0.7) s = fs * od * 3;
            else if (od > 1.5) s = od * 1.5;
            else if (fs > 1.3) s = fs * 2;
            else s = fs * 0.5 + od * 0.5;
            scores.push({num: num, score: s});
        }
        scores.sort((a,b) => b.score - a.score);
        return scores.slice(0, 15).map(x => x.num);
    }

    function run_ensemble(hist, pool_size = 20) {
        let m1 = model_markov_chain(hist);
        let m2 = model_gap_overdue(hist, 15);
        let m3 = model_momentum_neural(hist);
        let m4 = model_advanced_ml(hist);
        let m5 = model_knn_mirror(hist);
        let m6 = model_pair_matrix(hist);
        let m7 = model_delta_momentum(hist);
        let m8 = model_cond_prob(hist);
        let m9 = model_freq_gap_hybrid(hist);

        let votes = new Counter();
        for (let n of m5.slice(0,15)) votes.add(n, 12);
        for (let n of m6.slice(0,15)) votes.add(n, 8);
        for (let n of m8.slice(0,15)) votes.add(n, 6);
        for (let n of m9.slice(0,15)) votes.add(n, 5);
        for (let n of m4.slice(0,15)) votes.add(n, 4);
        for (let n of m7.slice(0,15)) votes.add(n, 4);
        for (let n of m2.slice(0,15)) votes.add(n, 3);
        for (let n of m3.slice(0,6)) votes.add(n, 2);
        for (let n of m1.slice(0,6)) votes.add(n, 1);

        let strong = [new Set(m5.slice(0,12)), new Set(m6.slice(0,12)), new Set(m8.slice(0,12)), new Set(m7.slice(0,12))];
        for (let num of ALL_NUMBERS) {
            let c = strong.filter(s => s.has(num)).length;
            if (c >= 3) votes.add(num, c * 5);
        }

        return votes.mostCommon(pool_size);
    }

    console.log("\n======================================================================");
    console.log("🧪 BẮT ĐẦU BACKTEST TOÀN BỘ LỊCH SỬ");
    console.log("======================================================================");

    const START = 60;
    const STEP = 1;
    const total_draws = data.length;
    let test_indices = [];
    for (let i = START; i < total_draws; i += STEP) test_indices.push(i);
    const n_test = test_indices.length;

    console.log(`📊 Test từ kỳ ${START} đến kỳ ${total_draws}`);
    console.log(`📊 Tổng số kỳ test: ${n_test}`);
    
    let counts6 = Array(7).fill(0);
    let counts10 = Array(7).fill(0);
    let counts15 = Array(7).fill(0);
    let counts20 = Array(7).fill(0);
    
    let detail_rows = [];
    let t0 = Date.now();
    let errors = 0;

    for (let step_i = 0; step_i < test_indices.length; step_i++) {
        let cur_idx = test_indices[step_i];
        let hist = data.slice(0, cur_idx);
        let actual = new Set(data[cur_idx]);
        
        if (step_i % 20 === 0) {
            process.stdout.write(`\r  ⏳ Kỳ ${cur_idx}/${total_draws} (${step_i+1}/${n_test}) — ${Math.floor((step_i+1)/n_test*100)}% `);
        }

        try {
            let ranked = run_ensemble(hist, 20);
            let top6 = new Set(ranked.slice(0, 6));
            let top10 = new Set(ranked.slice(0, 10));
            let top15 = new Set(ranked.slice(0, 15));
            let top20 = new Set(ranked.slice(0, 20));

            let hit6 = setIntersection(top6, actual).size;
            let hit10 = setIntersection(top10, actual).size;
            let hit15 = setIntersection(top15, actual).size;
            let hit20 = setIntersection(top20, actual).size;

            counts6[hit6]++;
            counts10[hit10]++;
            counts15[hit15]++;
            counts20[hit20]++;
            
            if (hit6 >= 5) {
                detail_rows.push({
                    draw: cur_idx,
                    actual: [...actual].sort((a,b)=>a-b),
                    top6: [...top6].sort((a,b)=>a-b),
                    hit6
                });
            }
        } catch (e) {
            errors++;
        }
    }

    let elapsed = (Date.now() - t0) / 1000;

    console.log("\n\n======================================================================");
    console.log("📊 KẾT QUẢ BACKTEST TOÀN BỘ LỊCH SỬ");
    console.log("======================================================================");
    console.log(`Tổng kỳ test: ${n_test} | Lỗi: ${errors} | Thời gian: ${elapsed.toFixed(0)}s\n`);

    function pct(c, t) { return (t > 0 ? (c / t * 100).toFixed(2) : "0.00") + "%"; }

    console.log("┌─────────────┬────────────┬────────────┬────────────┬────────────┐");
    console.log("│  Số trúng   │  Top-6 AI  │ Top-10 AI  │ Top-15 AI  │ Top-20 AI  │");
    console.log("├─────────────┼────────────┼────────────┼────────────┼────────────┤");
    
    for (let k = 6; k >= 0; k--) {
        let c6 = counts6[k];
        let c10 = 0, c15 = 0, c20 = 0;
        for (let i = k; i <= 6; i++) {
            c10 += counts10[i];
            c15 += counts15[i];
            c20 += counts20[i];
        }
        
        let emoji = {6:"🏆",5:"🥇",4:"🥈",3:"🥉",2:"  ",1:"  ",0:"  "}[k] || "";
        let label = k <= 6 ? `${emoji} ${k}/6`.padStart(11) : `   ${k}/6`.padStart(11);
        
        let s6 = `${c6}`.padStart(4) + ` (${pct(c6, n_test).padStart(6)})`;
        let s10 = (k===6 ? `${c10}`.padStart(4) : `≥${k}: ${c10}`.padStart(7)) + ` (${pct(c10, n_test).padStart(6)})`;
        let s15 = (k===6 ? `${c15}`.padStart(4) : `≥${k}: ${c15}`.padStart(7)) + ` (${pct(c15, n_test).padStart(6)})`;
        let s20 = (k===6 ? `${c20}`.padStart(4) : `≥${k}: ${c20}`.padStart(7)) + ` (${pct(c20, n_test).padStart(6)})`;
        
        console.log(`│ ${label} │ ${s6} │ ${s10} │ ${s15} │ ${s20} │`);
    }
    console.log("└─────────────┴────────────┴────────────┴────────────┴────────────┘");
    
    let v6_66 = counts6[6];
    console.log(`\n🔑 KẾT QUẢ ĐỈNH CAO NHẤT: Trúng 6/6 (Jackpot) trong Top-6: ${v6_66} kỳ (${pct(v6_66, n_test)})`);
    
    if (detail_rows.length > 0) {
        console.log(`\n🏆 CÁC KỲ TRÚNG LỚN (≥5/6 trong Top-6):`);
        for (let d of detail_rows) {
            console.log(`  Kỳ ${d.draw}: Thật=${d.actual} | AI Top-6=${d.top6} | Trúng: ${d.hit6}/6`);
        }
    }
}
main();
