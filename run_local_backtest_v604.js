/**
 * V604.0 LOCAL BACKTEST — 7-Model Ensemble
 * Chạy offline, không cần sklearn/Streamlit
 * 
 * Models: KNN Mirror V2, Pair Matrix, Delta Momentum, Gap Overdue, Neural Momentum, Markov, ML-Proxy
 */
const fs = require('fs');
const https = require('https');

function fetchData() {
    return new Promise((resolve, reject) => {
        const url = 'https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl';
        https.get(url, res => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => {
                const draws = [];
                for (const line of data.trim().split('\n')) {
                    if (!line) continue;
                    const obj = JSON.parse(line);
                    if (obj.result && obj.result.length >= 6) {
                        draws.push(obj.result.slice(0, 6).map(Number).sort((a,b) => a-b));
                    }
                }
                resolve(draws);
            });
            res.on('error', reject);
        });
    });
}

// ========== AI MODELS ==========

function modelGapOverdue(data, maxNum, topN = 15) {
    const lastSeen = {};
    for (let i = 0; i < data.length; i++) {
        for (const n of data[i]) lastSeen[n] = i;
    }
    const current = data.length;
    const avgGaps = {};
    const lastIdx = {};
    const gapLists = {};
    for (let i = 0; i < data.length; i++) {
        for (const n of data[i]) {
            if (!gapLists[n]) gapLists[n] = [];
            if (lastIdx[n] !== undefined) gapLists[n].push(i - lastIdx[n]);
            lastIdx[n] = i;
        }
    }
    const scores = [];
    for (let n = 1; n <= maxNum; n++) {
        const gap = lastSeen[n] !== undefined ? current - lastSeen[n] : current;
        const meanGap = gapLists[n] && gapLists[n].length ? gapLists[n].reduce((a,b)=>a+b,0) / gapLists[n].length : current;
        scores.push([n, gap / (meanGap + 0.1)]);
    }
    scores.sort((a,b) => b[1] - a[1]);
    return scores.slice(0, topN).map(x => x[0]);
}

function modelMomentumNeural(data, maxNum) {
    const weights = {};
    for (let n = 1; n <= maxNum; n++) weights[n] = 0;
    const total = data.length;
    for (let i = 0; i < total; i++) {
        const decay = 1 / (1 + Math.exp(-(i - total + 20) / 5));
        for (const n of data[i]) weights[n] += decay;
    }
    const sorted = Object.entries(weights).sort((a,b) => b[1] - a[1]);
    return sorted.slice(0, 6).map(x => parseInt(x[0]));
}

function modelMarkov(data) {
    if (data.length < 2) return [];
    const transitions = {};
    for (let i = 0; i < data.length - 1; i++) {
        const key = data[i].join(',');
        if (!transitions[key]) transitions[key] = {};
        for (const n of data[i+1]) {
            transitions[key][n] = (transitions[key][n] || 0) + 1;
        }
    }
    const lastKey = data[data.length - 1].join(',');
    if (transitions[lastKey]) {
        const entries = Object.entries(transitions[lastKey]).sort((a,b) => b[1] - a[1]);
        return entries.slice(0, 6).map(x => parseInt(x[0]));
    }
    return modelMomentumNeural(data, 45);
}

function modelKnnMirrorV2(data, maxNum) {
    if (data.length < 15) return modelMomentumNeural(data, maxNum);
    
    const pattern = new Set([...data[data.length-1], ...data[data.length-2], ...data[data.length-3]]);
    const n = data.length;
    const similarities = [];
    
    for (let i = 2; i < n - 3; i++) {
        const pastPattern = new Set([...data[i], ...data[i-1], ...data[i-2]]);
        let intersect = 0;
        for (const x of pattern) if (pastPattern.has(x)) intersect++;
        const recency = 1.0 + 0.3 * (i / n);
        similarities.push([intersect * recency, i + 1]);
    }
    similarities.sort((a,b) => b[0] - a[0]);
    
    const votes = {};
    for (const [score, nextIdx] of similarities.slice(0, 20)) {
        if (score >= 2.5 && nextIdx < data.length) {
            for (const num of data[nextIdx]) {
                votes[num] = (votes[num] || 0) + score;
            }
        }
    }
    if (Object.keys(votes).length === 0) return modelMomentumNeural(data, maxNum);
    const sorted = Object.entries(votes).sort((a,b) => b[1] - a[1]);
    return sorted.slice(0, 20).map(x => parseInt(x[0]));
}

function modelPairMatrix(data, maxNum) {
    if (data.length < 30) return modelGapOverdue(data, maxNum);
    
    const n = data.length;
    const pairScores = {};
    for (let idx = 0; idx < n; idx++) {
        const decay = 0.3 + 0.7 * (idx / n);
        const draw = data[idx].slice(0, 6).sort((a,b) => a-b);
        for (let i = 0; i < draw.length; i++) {
            for (let j = i+1; j < draw.length; j++) {
                const key = `${draw[i]},${draw[j]}`;
                pairScores[key] = (pairScores[key] || 0) + decay;
            }
        }
    }
    
    const lastDraw = new Set(data[n-1].slice(0, 6));
    const candidateScores = {};
    for (let num = 1; num <= maxNum; num++) {
        if (lastDraw.has(num)) continue;
        candidateScores[num] = 0;
        for (const anchor of lastDraw) {
            const key = Math.min(num, anchor) + ',' + Math.max(num, anchor);
            candidateScores[num] += pairScores[key] || 0;
        }
    }
    
    // Triplet bonus
    const tripletBonus = {};
    for (let idx = Math.max(0, n - 100); idx < n; idx++) {
        const draw = data[idx].slice(0, 6).sort((a,b) => a-b);
        for (let i = 0; i < draw.length; i++) {
            for (let j = i+1; j < draw.length; j++) {
                for (let k = j+1; k < draw.length; k++) {
                    const tripSet = new Set([draw[i], draw[j], draw[k]]);
                    let overlap = 0;
                    for (const x of tripSet) if (lastDraw.has(x)) overlap++;
                    if (overlap >= 2) {
                        for (const x of tripSet) {
                            if (!lastDraw.has(x)) {
                                tripletBonus[x] = (tripletBonus[x] || 0) + 1.5;
                            }
                        }
                    }
                }
            }
        }
    }
    for (const [num, bonus] of Object.entries(tripletBonus)) {
        candidateScores[num] = (candidateScores[num] || 0) + bonus;
    }
    
    const sorted = Object.entries(candidateScores).sort((a,b) => b[1] - a[1]);
    return sorted.slice(0, 15).map(x => parseInt(x[0]));
}

function modelDeltaMomentum(data, maxNum) {
    if (data.length < 30) return modelMomentumNeural(data, maxNum);
    
    const scores = {};
    for (let num = 1; num <= maxNum; num++) {
        const f5 = data.slice(-5).filter(d => d.includes(num)).length / 5;
        const f5p = data.slice(-10, -5).filter(d => d.includes(num)).length / 5;
        const f15 = data.slice(-15).filter(d => d.includes(num)).length / 15;
        const f15p = data.slice(-30, -15).filter(d => d.includes(num)).length / 15;
        
        const deltaShort = f5 - f5p;
        const deltaMid = f15 - f15p;
        let momentum = deltaShort * 3 + deltaMid * 2;
        
        if (data[data.length-1].includes(num)) momentum += 0.5;
        if (data.length >= 2 && data[data.length-2].includes(num)) momentum += 0.3;
        
        scores[num] = momentum;
    }
    const sorted = Object.entries(scores).sort((a,b) => b[1] - a[1]);
    return sorted.slice(0, 15).map(x => parseInt(x[0]));
}

// Simple frequency-based ML proxy (since we can't use sklearn in Node)
function modelMLProxy(data, maxNum) {
    if (data.length < 20) return modelGapOverdue(data, maxNum);
    
    const windowSize = 10;
    // Count frequency in last window
    const freqs = {};
    for (let n = 1; n <= maxNum; n++) freqs[n] = 0;
    for (const draw of data.slice(-windowSize)) {
        for (const n of draw) freqs[n]++;
    }
    // Weight by position in window
    const scores = {};
    for (let n = 1; n <= maxNum; n++) {
        scores[n] = freqs[n] / windowSize;
    }
    const sorted = Object.entries(scores).sort((a,b) => b[1] - a[1]);
    return sorted.slice(0, 15).map(x => parseInt(x[0]));
}

// ========== ENSEMBLE ==========

function ensembleV604(data, maxNum) {
    const m1 = modelMarkov(data);
    const m2 = modelGapOverdue(data, maxNum, 15);
    const m3 = modelMomentumNeural(data, maxNum);
    const m4 = modelMLProxy(data, maxNum);
    const m5 = modelKnnMirrorV2(data, maxNum);
    const m6 = modelPairMatrix(data, maxNum);
    const m7 = modelDeltaMomentum(data, maxNum);
    
    const votes = {};
    for (const n of m5.slice(0, 15)) votes[n] = (votes[n] || 0) + 8;
    for (const n of m6.slice(0, 15)) votes[n] = (votes[n] || 0) + 6;
    for (const n of m4.slice(0, 15)) votes[n] = (votes[n] || 0) + 5;
    for (const n of m7.slice(0, 15)) votes[n] = (votes[n] || 0) + 4;
    for (const n of m2.slice(0, 15)) votes[n] = (votes[n] || 0) + 3;
    for (const n of m3.slice(0, 6))  votes[n] = (votes[n] || 0) + 2;
    for (const n of m1.slice(0, 6))  votes[n] = (votes[n] || 0) + 1;
    
    const ranked = Object.entries(votes).sort((a,b) => b[1] - a[1]).map(x => parseInt(x[0]));
    return ranked;
}

// ========== BACKTEST ==========

async function main() {
    console.log('🧬 V604.0 LOCAL BACKTEST — 7-Model Ensemble');
    console.log('='.repeat(60));
    console.log('Đang tải dữ liệu từ GitHub...');
    
    const allData = await fetchData();
    console.log(`✅ Đã tải ${allData.length} kỳ quay lịch sử (Mega 6/45)`);
    
    const maxNum = 45;
    const startIdx = 60;
    const step = 1; // Test mỗi kỳ
    
    const counts6 = {0:0,1:0,2:0,3:0,4:0,5:0,6:0};
    const counts10 = {0:0,1:0,2:0,3:0,4:0,5:0,6:0};
    const counts15 = {0:0,1:0,2:0,3:0,4:0,5:0,6:0};
    const counts20 = {0:0,1:0,2:0,3:0,4:0,5:0,6:0};
    let nTest = 0;
    
    const total = Math.floor((allData.length - startIdx) / step);
    
    for (let curIdx = startIdx; curIdx < allData.length; curIdx += step) {
        const hist = allData.slice(0, curIdx);
        const actual = new Set(allData[curIdx]);
        
        const ranked = ensembleV604(hist, maxNum);
        
        const top6 = new Set(ranked.slice(0, 6));
        const top10 = new Set(ranked.slice(0, 10));
        const top15 = new Set(ranked.slice(0, 15));
        const top20 = new Set(ranked.slice(0, 20));
        
        let hit6 = 0, hit10 = 0, hit15 = 0, hit20 = 0;
        for (const n of actual) {
            if (top6.has(n)) hit6++;
            if (top10.has(n)) hit10++;
            if (top15.has(n)) hit15++;
            if (top20.has(n)) hit20++;
        }
        
        counts6[hit6]++;
        counts10[hit10]++;
        counts15[hit15]++;
        counts20[hit20]++;
        nTest++;
        
        if (nTest % 100 === 0) {
            process.stdout.write(`\r⏳ Đã test ${nTest}/${total} kỳ (${Math.round(nTest/total*100)}%)`);
        }
    }
    
    console.log(`\n\n${'='.repeat(60)}`);
    console.log(`✅ HOÀN TẤT! Đã test ${nTest} kỳ`);
    console.log(`${'='.repeat(60)}\n`);
    
    const pct = (v) => `${(v/nTest*100).toFixed(1)}%`;
    
    console.log('📊 TOP-6 (Chỉ 6 số dự đoán):');
    for (let k = 6; k >= 0; k--) {
        const emoji = {6:'🏆', 5:'🥇', 4:'🥈', 3:'🥉'}[k] || '  ';
        console.log(`  ${emoji} ${k}/6: ${counts6[k]} kỳ (${pct(counts6[k])})`);
    }
    
    console.log('\n🔟 TOP-10 (Pool 10 số):');
    for (let k = 6; k >= 0; k--) {
        let above = 0;
        for (let i = k; i <= 6; i++) above += counts10[i];
        const emoji = {6:'🏆', 5:'🥇', 4:'🥈', 3:'🥉'}[k] || '  ';
        console.log(`  ${emoji} ≥${k}/6: ${above} kỳ (${pct(above)})`);
    }
    
    console.log('\n🎱 TOP-15 (Pool 15 số):');
    for (let k = 6; k >= 0; k--) {
        let above = 0;
        for (let i = k; i <= 6; i++) above += counts15[i];
        const emoji = {6:'🏆', 5:'🥇', 4:'🥈', 3:'🥉'}[k] || '  ';
        console.log(`  ${emoji} ≥${k}/6: ${above} kỳ (${pct(above)})`);
    }
    
    console.log('\n🚀 TOP-20 (Mở rộng):');
    for (let k = 6; k >= 0; k--) {
        let above = 0;
        for (let i = k; i <= 6; i++) above += counts20[i];
        const emoji = {6:'🏆', 5:'🥇', 4:'🥈', 3:'🥉'}[k] || '  ';
        console.log(`  ${emoji} ≥${k}/6: ${above} kỳ (${pct(above)})`);
    }
    
    // Key metrics
    console.log(`\n${'='.repeat(60)}`);
    console.log('🔑 CHỈ SỐ QUAN TRỌNG NHẤT:');
    const hit5_6_top15 = counts15[5] + counts15[6];
    const hit5_6_top20 = counts20[5] + counts20[6];
    const hit4_6_top15 = counts15[4] + counts15[5] + counts15[6];
    const hit4_6_top20 = counts20[4] + counts20[5] + counts20[6];
    console.log(`  💎 Top-15 trúng ≥5/6: ${hit5_6_top15} kỳ (${pct(hit5_6_top15)})`);
    console.log(`  💎 Top-20 trúng ≥5/6: ${hit5_6_top20} kỳ (${pct(hit5_6_top20)})`);
    console.log(`  🥈 Top-15 trúng ≥4/6: ${hit4_6_top15} kỳ (${pct(hit4_6_top15)})`);
    console.log(`  🥈 Top-20 trúng ≥4/6: ${hit4_6_top20} kỳ (${pct(hit4_6_top20)})`);
    console.log(`${'='.repeat(60)}`);
}

main().catch(console.error);
