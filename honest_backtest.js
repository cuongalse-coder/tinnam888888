/**
 * HONEST BACKTEST V600 — Node.js
 * ================================
 * Tải dữ liệu thật từ GitHub, chạy backtest trung thực.
 * Không dùng ML (vì Node), chỉ dùng core signals (frequency, gap, momentum, conditional prob)
 * để kiểm tra tỷ lệ trúng THỰC TẾ của Pool 6/10/15 số.
 * 
 * Đây là bài test TRUNG THỰC — mỗi step chỉ dùng dữ liệu TRƯỚC thời điểm đó.
 */

const https = require('https');

function fetchData(url) {
    return new Promise((resolve, reject) => {
        https.get(url, { headers: { 'User-Agent': 'Mozilla/5.0' } }, (res) => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => resolve(data));
        }).on('error', reject);
    });
}

async function loadHistory() {
    // Try GitHub dataset first
    try {
        const raw = await fetchData('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl');
        const lines = raw.trim().split('\n');
        const history = [];
        for (const line of lines) {
            try {
                const obj = JSON.parse(line);
                if (obj.result && obj.result.length >= 6) {
                    const draw = obj.result.slice(0, 6).map(Number).sort((a, b) => a - b);
                    if (draw.length === 6 && new Set(draw).size === 6 && draw.every(n => n >= 1 && n <= 45)) {
                        history.push(draw);
                    }
                }
            } catch (e) { }
        }
        return history;
    } catch (e) {
        console.error('Failed to load from GitHub:', e.message);
        return [];
    }
}

// ================================================================
// CORE SIGNAL ENGINES (Pure math, no ML)
// ================================================================

function sigMultiWindowFreq(data, maxNum, pickCount) {
    const scores = {};
    const windows = [5, 10, 20, 40, 80];
    const weights = [5, 3, 2, 1, 0.5];
    const expected = pickCount / maxNum;

    for (let num = 1; num <= maxNum; num++) {
        let score = 0;
        for (let wi = 0; wi < windows.length; wi++) {
            const w = windows[wi];
            if (data.length < w) continue;
            const recent = data.slice(-w);
            let count = 0;
            for (const d of recent) if (d.includes(num)) count++;
            const freq = count / w;
            score += ((freq - expected) / (expected + 0.001)) * weights[wi];
        }
        scores[num] = score;
    }
    return scores;
}

function sigGapOverdue(data, maxNum, pickCount) {
    const n = data.length;
    const scores = {};
    for (let num = 1; num <= maxNum; num++) {
        const appearances = [];
        for (let i = 0; i < n; i++) {
            if (data[i].includes(num)) appearances.push(i);
        }
        if (appearances.length < 3) { scores[num] = 0; continue; }

        const gaps = [];
        for (let j = 0; j < appearances.length - 1; j++) {
            gaps.push(appearances[j + 1] - appearances[j]);
        }
        const meanGap = gaps.reduce((s, g) => s + g, 0) / gaps.length;
        const currentGap = n - appearances[appearances.length - 1];
        const overdue = currentGap / (meanGap + 0.1);

        // Gap z-score
        const stdGap = Math.sqrt(gaps.reduce((s, g) => s + (g - meanGap) ** 2, 0) / gaps.length) || 1;
        const gapZ = (currentGap - meanGap) / stdGap;

        scores[num] = overdue > 1 ? overdue * 2 + Math.max(0, gapZ) : gapZ * 0.5;
    }
    return scores;
}

function sigMomentum(data, maxNum, pickCount) {
    const scores = {};
    const n = data.length;
    for (let num = 1; num <= maxNum; num++) {
        if (n < 30) { scores[num] = 0; continue; }
        const f5 = data.slice(-5).filter(d => d.includes(num)).length / 5;
        const f10 = data.slice(-10).filter(d => d.includes(num)).length / 10;
        const f20 = data.slice(-20).filter(d => d.includes(num)).length / 20;
        const velocity = f5 - f10;
        const accel = (f5 - f10) - (f10 - f20);
        scores[num] = velocity * 10 + accel * 5;
    }
    return scores;
}

function sigConditionalProb(data, maxNum, pickCount) {
    const scores = {};
    if (data.length < 20) {
        for (let num = 1; num <= maxNum; num++) scores[num] = 0;
        return scores;
    }
    const last = new Set(data[data.length - 1]);

    // Count transitions
    const condCounts = {}; // given -> next -> count
    const totalGiven = {};
    for (let i = 0; i < data.length - 1; i++) {
        for (const given of data[i]) {
            totalGiven[given] = (totalGiven[given] || 0) + 1;
            for (const next of data[i + 1]) {
                const key = `${given}_${next}`;
                condCounts[key] = (condCounts[key] || 0) + 1;
            }
        }
    }

    for (let num = 1; num <= maxNum; num++) {
        let probSum = 0;
        for (const given of last) {
            const total = totalGiven[given] || 0;
            if (total > 0) {
                probSum += (condCounts[`${given}_${num}`] || 0) / total;
            }
        }
        scores[num] = probSum * 3;
    }
    return scores;
}

function sigTemporalDecay(data, maxNum, pickCount) {
    const scores = {};
    const n = data.length;
    const lam = 0.05;
    for (let num = 1; num <= maxNum; num++) scores[num] = 0;
    for (let i = 0; i < n; i++) {
        const age = n - 1 - i;
        const w = Math.exp(-lam * age);
        for (const num of data[i]) {
            scores[num] = (scores[num] || 0) + w;
        }
    }
    const maxS = Math.max(...Object.values(scores)) || 1;
    for (let num = 1; num <= maxNum; num++) {
        scores[num] = (scores[num] / maxS) * 4;
    }
    return scores;
}

function sigPairBoost(data, maxNum) {
    const scores = {};
    const last = new Set(data[data.length - 1]);
    const pairCounts = {};
    const recent = data.slice(-150);
    for (const d of recent) {
        const s = [...d].sort((a, b) => a - b);
        for (let i = 0; i < s.length; i++) {
            for (let j = i + 1; j < s.length; j++) {
                const key = `${s[i]}_${s[j]}`;
                pairCounts[key] = (pairCounts[key] || 0) + 1;
            }
        }
    }
    for (let num = 1; num <= maxNum; num++) {
        let score = 0;
        for (const p of last) {
            const key = p < num ? `${p}_${num}` : `${num}_${p}`;
            const cnt = pairCounts[key] || 0;
            if (cnt > 2) score += cnt * 0.08;
        }
        scores[num] = score;
    }
    return scores;
}

// ================================================================
// ENSEMBLE: Combine all signals
// ================================================================

function predictPool(data, maxNum, pickCount, poolSize) {
    const s1 = sigMultiWindowFreq(data, maxNum, pickCount);
    const s2 = sigGapOverdue(data, maxNum, pickCount);
    const s3 = sigMomentum(data, maxNum, pickCount);
    const s4 = sigConditionalProb(data, maxNum, pickCount);
    const s5 = sigTemporalDecay(data, maxNum, pickCount);
    const s6 = sigPairBoost(data, maxNum);

    const signalSets = [
        { s: s1, w: 3.0 },
        { s: s2, w: 4.0 },
        { s: s3, w: 2.5 },
        { s: s4, w: 3.5 },
        { s: s5, w: 2.0 },
        { s: s6, w: 2.0 },
    ];

    // Normalize each signal and combine
    const finalScores = {};
    for (let num = 1; num <= maxNum; num++) finalScores[num] = 0;

    for (const { s, w } of signalSets) {
        const vals = Object.values(s);
        const maxV = Math.max(...vals.map(Math.abs)) || 1;
        for (let num = 1; num <= maxNum; num++) {
            finalScores[num] += (s[num] / maxV) * w;
        }
    }

    // Rank and return top pool
    const ranked = Object.entries(finalScores)
        .map(([num, score]) => ({ num: parseInt(num), score }))
        .sort((a, b) => b.score - a.score);

    return ranked.slice(0, poolSize).map(r => r.num);
}

// ================================================================
// HONEST BACKTEST
// ================================================================

async function runBacktest() {
    console.log('Loading real lottery data from GitHub...');
    const history = await loadHistory();
    console.log(`Loaded ${history.length} draws (Mega 6/45)\n`);

    if (history.length < 100) {
        console.error('Not enough data for backtest');
        return;
    }

    const maxNum = 45;
    const pickCount = 6;
    const minTrainSize = 60;

    // Test on last 200 draws (or all available after train)
    const testSize = Math.min(200, history.length - minTrainSize);
    const testStart = history.length - testSize;

    console.log(`Testing ${testSize} draws (from draw #${testStart} to #${history.length - 1})`);
    console.log('Each step uses ONLY data BEFORE that draw (no future leak)\n');
    console.log('='.repeat(60));

    const counts6 = [0, 0, 0, 0, 0, 0, 0];
    const counts10 = [0, 0, 0, 0, 0, 0, 0];
    const counts15 = [0, 0, 0, 0, 0, 0, 0];

    for (let idx = testStart; idx < history.length; idx++) {
        const trainData = history.slice(0, idx); // ONLY past data
        const actual = new Set(history[idx]);

        const pool = predictPool(trainData, maxNum, pickCount, 15);
        const top6 = new Set(pool.slice(0, 6));
        const top10 = new Set(pool.slice(0, 10));
        const top15 = new Set(pool);

        let hit6 = 0, hit10 = 0, hit15 = 0;
        for (const n of actual) {
            if (top6.has(n)) hit6++;
            if (top10.has(n)) hit10++;
            if (top15.has(n)) hit15++;
        }

        counts6[hit6]++;
        counts10[hit10]++;
        counts15[hit15]++;

        const step = idx - testStart + 1;
        if (step % 50 === 0 || step === 1 || step === testSize) {
            console.log(`  Step ${step}/${testSize}: Draw #${idx} | Actual: [${[...actual].sort((a,b)=>a-b).join(',')}] | Hit6=${hit6} Hit10=${hit10} Hit15=${hit15}`);
        }
    }

    // ================================================================
    // RESULTS — 100% TRUNG THỰC
    // ================================================================
    console.log('\n' + '='.repeat(60));
    console.log('KẾT QUẢ BACKTEST TRUNG THỰC — V600 CORE SIGNALS');
    console.log('(Không ML, chỉ frequency + gap + momentum + conditional + decay + pair)');
    console.log('='.repeat(60));

    const pct = (c) => `${(c / testSize * 100).toFixed(1)}%`;

    // Random baseline for comparison
    // For Mega 6/45: P(exactly k matches from pool of N choosing 6)
    // Using hypergeometric distribution
    function hypergeoPMF(k, N, K, n) {
        // P(X=k) where N=45 total, K=pool size, n=6 drawn
        function choose(a, b) {
            if (b > a) return 0;
            if (b === 0 || b === a) return 1;
            let r = 1;
            for (let i = 0; i < b; i++) r = r * (a - i) / (i + 1);
            return r;
        }
        return choose(K, k) * choose(N - K, n - k) / choose(N, n);
    }

    console.log('\n--- TOP-6 DỰ ĐOÁN (Chọn đúng 6 số tốt nhất) ---');
    console.log('Trúng | Số kỳ | Tỷ lệ AI | Tỷ lệ Ngẫu Nhiên | So sánh');
    console.log('------|-------|----------|------------------|--------');
    for (let k = 6; k >= 0; k--) {
        const randomProb = hypergeoPMF(k, 45, 6, 6) * 100;
        const aiProb = counts6[k] / testSize * 100;
        const ratio = randomProb > 0 ? (aiProb / randomProb).toFixed(1) : '∞';
        console.log(`  ${k}/6 | ${String(counts6[k]).padStart(5)} | ${pct(counts6[k]).padStart(8)} | ${randomProb.toFixed(3).padStart(16)}% | x${ratio}`);
    }

    const gte3_6 = counts6[3] + counts6[4] + counts6[5] + counts6[6];
    const gte4_6 = counts6[4] + counts6[5] + counts6[6];
    console.log(`\n  ≥3/6: ${gte3_6} kỳ (${pct(gte3_6)})`);
    console.log(`  ≥4/6: ${gte4_6} kỳ (${pct(gte4_6)})`);

    console.log('\n--- POOL-10 SỐ (Bao 10) ---');
    console.log('≥Trúng | Số kỳ | Tỷ lệ AI | Tỷ lệ Ngẫu Nhiên');
    console.log('-------|-------|----------|------------------');
    for (let k = 6; k >= 0; k--) {
        let aiAbove = 0;
        for (let i = k; i <= 6; i++) aiAbove += counts10[i];
        let rndAbove = 0;
        for (let i = k; i <= 6; i++) rndAbove += hypergeoPMF(i, 45, 10, 6);
        console.log(`  ≥${k}/6 | ${String(aiAbove).padStart(5)} | ${pct(aiAbove).padStart(8)} | ${(rndAbove * 100).toFixed(1).padStart(16)}%`);
    }

    console.log('\n--- POOL-15 SỐ (Bao 15) ---');
    console.log('≥Trúng | Số kỳ | Tỷ lệ AI | Tỷ lệ Ngẫu Nhiên');
    console.log('-------|-------|----------|------------------');
    for (let k = 6; k >= 0; k--) {
        let aiAbove = 0;
        for (let i = k; i <= 6; i++) aiAbove += counts15[i];
        let rndAbove = 0;
        for (let i = k; i <= 6; i++) rndAbove += hypergeoPMF(i, 45, 15, 6);
        console.log(`  ≥${k}/6 | ${String(aiAbove).padStart(5)} | ${pct(aiAbove).padStart(8)} | ${(rndAbove * 100).toFixed(1).padStart(16)}%`);
    }

    // KEY METRIC
    const pool15_gte3 = counts15.slice(3).reduce((s, c) => s + c, 0);
    const pool10_gte3 = counts10.slice(3).reduce((s, c) => s + c, 0);
    const rnd15_gte3 = [3,4,5,6].reduce((s,k) => s + hypergeoPMF(k,45,15,6), 0) * 100;
    const rnd10_gte3 = [3,4,5,6].reduce((s,k) => s + hypergeoPMF(k,45,10,6), 0) * 100;

    console.log('\n' + '='.repeat(60));
    console.log('CHỈ SỐ QUAN TRỌNG NHẤT');
    console.log('='.repeat(60));
    console.log(`Pool-15 trúng ≥3/6: ${pool15_gte3}/${testSize} kỳ = ${pct(pool15_gte3)} (ngẫu nhiên: ${rnd15_gte3.toFixed(1)}%)`);
    console.log(`Pool-10 trúng ≥3/6: ${pool10_gte3}/${testSize} kỳ = ${pct(pool10_gte3)} (ngẫu nhiên: ${rnd10_gte3.toFixed(1)}%)`);
    console.log(`Top-6 trúng ≥3/6:   ${gte3_6}/${testSize} kỳ = ${pct(gte3_6)} (ngẫu nhiên: ${([3,4,5,6].reduce((s,k) => s + hypergeoPMF(k,45,6,6), 0) * 100).toFixed(2)}%)`);

    console.log('\n⚠️  LƯU Ý: Đây là kết quả THẬT, KHÔNG CÓ DATA LEAKAGE.');
    console.log('    Mỗi dự đoán chỉ sử dụng dữ liệu TRƯỚC thời điểm đó.');
    console.log('    So sánh với "ngẫu nhiên" để thấy AI có hơn random không.');
}

runBacktest().catch(console.error);
