/**
 * V700.0 BACKTEST — 7-Model + Consensus Bonus + 4 New Signals
 */
const https = require('https');

function fetchData() {
    return new Promise((resolve, reject) => {
        https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', res => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => {
                const draws = [];
                for (const line of data.trim().split('\n')) {
                    if (!line) continue;
                    const obj = JSON.parse(line);
                    if (obj.result && obj.result.length >= 6)
                        draws.push(obj.result.slice(0, 6).map(Number).sort((a,b) => a-b));
                }
                resolve(draws);
            });
            res.on('error', reject);
        });
    });
}

function modelGapOverdue(data, maxNum, topN = 15) {
    const lastSeen = {}, gapLists = {}, lastIdx = {};
    for (let i = 0; i < data.length; i++) {
        for (const n of data[i]) {
            if (!gapLists[n]) gapLists[n] = [];
            if (lastIdx[n] !== undefined) gapLists[n].push(i - lastIdx[n]);
            lastIdx[n] = i; lastSeen[n] = i;
        }
    }
    const scores = [];
    for (let n = 1; n <= maxNum; n++) {
        const gap = lastSeen[n] !== undefined ? data.length - lastSeen[n] : data.length;
        const meanGap = gapLists[n]?.length ? gapLists[n].reduce((a,b)=>a+b,0) / gapLists[n].length : data.length;
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
    return Object.entries(weights).sort((a,b) => b[1] - a[1]).slice(0, 6).map(x => parseInt(x[0]));
}

function modelMarkov(data) {
    if (data.length < 2) return [];
    const trans = {};
    for (let i = 0; i < data.length - 1; i++) {
        const key = data[i].join(',');
        if (!trans[key]) trans[key] = {};
        for (const n of data[i+1]) trans[key][n] = (trans[key][n] || 0) + 1;
    }
    const lastKey = data[data.length - 1].join(',');
    if (trans[lastKey]) return Object.entries(trans[lastKey]).sort((a,b) => b[1] - a[1]).slice(0, 6).map(x => parseInt(x[0]));
    return modelMomentumNeural(data, 45);
}

function modelKnnMirrorV2(data, maxNum) {
    if (data.length < 15) return modelMomentumNeural(data, maxNum);
    const pattern = new Set([...data[data.length-1], ...data[data.length-2], ...data[data.length-3]]);
    const n = data.length, sims = [];
    for (let i = 2; i < n - 3; i++) {
        const pp = new Set([...data[i], ...data[i-1], ...data[i-2]]);
        let inter = 0; for (const x of pattern) if (pp.has(x)) inter++;
        const rec = 1.0 + 0.3 * (i / n);
        sims.push([inter * rec, i + 1]);
    }
    sims.sort((a,b) => b[0] - a[0]);
    const votes = {};
    for (const [score, nextIdx] of sims.slice(0, 20)) {
        if (score >= 2.5 && nextIdx < data.length) for (const num of data[nextIdx]) votes[num] = (votes[num] || 0) + score;
    }
    if (!Object.keys(votes).length) return modelMomentumNeural(data, maxNum);
    return Object.entries(votes).sort((a,b) => b[1] - a[1]).slice(0, 20).map(x => parseInt(x[0]));
}

function modelPairMatrix(data, maxNum) {
    if (data.length < 30) return modelGapOverdue(data, maxNum);
    const n = data.length, pairScores = {};
    for (let idx = 0; idx < n; idx++) {
        const decay = 0.3 + 0.7 * (idx / n), draw = data[idx].slice(0,6).sort((a,b)=>a-b);
        for (let i = 0; i < draw.length; i++) for (let j = i+1; j < draw.length; j++) {
            const key = `${draw[i]},${draw[j]}`; pairScores[key] = (pairScores[key] || 0) + decay;
        }
    }
    const lastDraw = new Set(data[n-1].slice(0,6)), cs = {};
    for (let num = 1; num <= maxNum; num++) {
        if (lastDraw.has(num)) continue; cs[num] = 0;
        for (const anchor of lastDraw) { const key = Math.min(num,anchor)+','+Math.max(num,anchor); cs[num] += pairScores[key] || 0; }
    }
    for (let idx = Math.max(0,n-100); idx < n; idx++) {
        const draw = data[idx].slice(0,6).sort((a,b)=>a-b);
        for (let i = 0; i < draw.length; i++) for (let j = i+1; j < draw.length; j++) for (let k = j+1; k < draw.length; k++) {
            const ts = new Set([draw[i],draw[j],draw[k]]); let ov = 0;
            for (const x of ts) if (lastDraw.has(x)) ov++;
            if (ov >= 2) for (const x of ts) if (!lastDraw.has(x)) cs[x] = (cs[x]||0) + 1.5;
        }
    }
    return Object.entries(cs).sort((a,b) => b[1] - a[1]).slice(0, 15).map(x => parseInt(x[0]));
}

function modelDeltaMomentum(data, maxNum) {
    if (data.length < 30) return modelMomentumNeural(data, maxNum);
    const scores = {};
    for (let num = 1; num <= maxNum; num++) {
        const f5 = data.slice(-5).filter(d => d.includes(num)).length / 5;
        const f5p = data.slice(-10, -5).filter(d => d.includes(num)).length / 5;
        const f15 = data.slice(-15).filter(d => d.includes(num)).length / 15;
        const f15p = data.slice(-30, -15).filter(d => d.includes(num)).length / 15;
        let mom = (f5-f5p)*3 + (f15-f15p)*2;
        if (data[data.length-1].includes(num)) mom += 0.5;
        if (data.length >= 2 && data[data.length-2].includes(num)) mom += 0.3;
        scores[num] = mom;
    }
    return Object.entries(scores).sort((a,b) => b[1] - a[1]).slice(0, 15).map(x => parseInt(x[0]));
}

function modelMLProxy(data, maxNum) {
    if (data.length < 20) return modelGapOverdue(data, maxNum);
    const freqs = {}; for (let n = 1; n <= maxNum; n++) freqs[n] = 0;
    for (const draw of data.slice(-10)) for (const n of draw) freqs[n]++;
    return Object.entries(freqs).sort((a,b) => b[1] - a[1]).slice(0, 15).map(x => parseInt(x[0]));
}

// === V700 NEW SIGNALS ===

function sigRegimeDetector(data, maxNum) {
    const scores = {};
    if (data.length < 40) return {};
    const expected = 6 / maxNum;
    for (let num = 1; num <= maxNum; num++) {
        const f5 = data.slice(-5).filter(d => d.includes(num)).length / 5;
        const f20 = data.slice(-20).filter(d => d.includes(num)).length / 20;
        const f50 = data.slice(-Math.min(50, data.length)).filter(d => d.includes(num)).length / Math.min(50, data.length);
        if (f5 > expected * 1.3 && f20 > expected * 1.1) scores[num] = (f5 + f20) * 4;
        else if (f5 < expected * 0.5 && f50 > expected * 1.2) scores[num] = (f50 - f5) * 3;
        else if (f5 > expected * 1.5 && f20 < expected * 0.8) scores[num] = f5 * 5;
        else scores[num] = 0;
    }
    return scores;
}

function sigLagCorrelation(data, maxNum) {
    const scores = {};
    if (data.length < 30) return {};
    const lags = [2,3,4,5,7], lw = [3,2.5,2,1.5,1], expected = 6/maxNum;
    for (let num = 1; num <= maxNum; num++) {
        scores[num] = 0;
        for (let li = 0; li < lags.length; li++) {
            const lag = lags[li]; let count = 0, total = 0;
            for (let i = lag; i < data.length; i++) {
                if (data[i-lag].includes(num)) { total++; if (data[i].includes(num)) count++; }
            }
            if (total > 5) {
                const ratio = count / total;
                if (ratio > expected * 1.2) {
                    for (let lg = 1; lg <= lag; lg++) {
                        if (data.length - lg >= 0 && data[data.length - lg].includes(num)) {
                            scores[num] += (ratio - expected) * lw[li] * 3; break;
                        }
                    }
                }
            }
        }
    }
    return scores;
}

// ========== ENSEMBLES ==========

function ensembleV604(data, maxNum) {
    const m1=modelMarkov(data), m2=modelGapOverdue(data,maxNum,15), m3=modelMomentumNeural(data,maxNum);
    const m4=modelMLProxy(data,maxNum), m5=modelKnnMirrorV2(data,maxNum), m6=modelPairMatrix(data,maxNum), m7=modelDeltaMomentum(data,maxNum);
    const votes = {};
    for (const n of m5.slice(0,15)) votes[n] = (votes[n]||0) + 8;
    for (const n of m6.slice(0,15)) votes[n] = (votes[n]||0) + 6;
    for (const n of m4.slice(0,15)) votes[n] = (votes[n]||0) + 5;
    for (const n of m7.slice(0,15)) votes[n] = (votes[n]||0) + 4;
    for (const n of m2.slice(0,15)) votes[n] = (votes[n]||0) + 3;
    for (const n of m3.slice(0,6))  votes[n] = (votes[n]||0) + 2;
    for (const n of m1.slice(0,6))  votes[n] = (votes[n]||0) + 1;
    return Object.entries(votes).sort((a,b) => b[1] - a[1]).map(x => parseInt(x[0]));
}

function ensembleV700(data, maxNum) {
    const m1=modelMarkov(data), m2=modelGapOverdue(data,maxNum,15), m3=modelMomentumNeural(data,maxNum);
    const m4=modelMLProxy(data,maxNum), m5=modelKnnMirrorV2(data,maxNum), m6=modelPairMatrix(data,maxNum), m7=modelDeltaMomentum(data,maxNum);
    
    const votes = {};
    // V700 boosted weights
    for (const n of m5.slice(0,15)) votes[n] = (votes[n]||0) + 10;
    for (const n of m6.slice(0,15)) votes[n] = (votes[n]||0) + 8;
    for (const n of m4.slice(0,15)) votes[n] = (votes[n]||0) + 6;
    for (const n of m7.slice(0,15)) votes[n] = (votes[n]||0) + 5;
    for (const n of m2.slice(0,15)) votes[n] = (votes[n]||0) + 4;
    for (const n of m3.slice(0,6))  votes[n] = (votes[n]||0) + 3;
    for (const n of m1.slice(0,6))  votes[n] = (votes[n]||0) + 2;
    
    // V700: Consensus bonus
    const mLists = [new Set(m1.slice(0,10)), new Set(m2.slice(0,10)), new Set(m3.slice(0,6)), 
                    new Set(m4.slice(0,10)), new Set(m5.slice(0,10)), new Set(m6.slice(0,10)), new Set(m7.slice(0,10))];
    for (let num = 1; num <= maxNum; num++) {
        const consensus = mLists.filter(ml => ml.has(num)).length;
        if (consensus >= 5) votes[num] = (votes[num]||0) + consensus * 3;
    }
    
    // V700: Regime Detector bonus
    const regime = sigRegimeDetector(data, maxNum);
    for (const [num, score] of Object.entries(regime)) {
        if (score > 0) votes[num] = (votes[num]||0) + Math.min(score * 2, 8);
    }
    
    // V700: Lag Correlation bonus
    const lag = sigLagCorrelation(data, maxNum);
    for (const [num, score] of Object.entries(lag)) {
        if (score > 0) votes[num] = (votes[num]||0) + Math.min(score * 2, 6);
    }
    
    return Object.entries(votes).sort((a,b) => b[1] - a[1]).map(x => parseInt(x[0]));
}

// ========== BACKTEST ==========

function runBacktest(allData, maxNum, ensembleFn, label) {
    const startIdx = 60, step = 1;
    const counts = {6:{},10:{},15:{},20:{}};
    for (const pool of [6,10,15,20]) for (let k = 0; k <= 6; k++) counts[pool][k] = 0;
    let nTest = 0;
    
    for (let curIdx = startIdx; curIdx < allData.length; curIdx += step) {
        const hist = allData.slice(0, curIdx);
        const actual = new Set(allData[curIdx]);
        const ranked = ensembleFn(hist, maxNum);
        
        for (const pool of [6,10,15,20]) {
            const topSet = new Set(ranked.slice(0, pool));
            let hits = 0; for (const n of actual) if (topSet.has(n)) hits++;
            counts[pool][hits]++;
        }
        nTest++;
        if (nTest % 200 === 0) process.stdout.write(`\r  ${label}: ${nTest} kỳ...`);
    }
    process.stdout.write(`\r  ${label}: ${nTest} kỳ — DONE\n`);
    return { counts, nTest };
}

async function main() {
    console.log('🧬 V604 vs V700 BACKTEST COMPARISON');
    console.log('='.repeat(60));
    const allData = await fetchData();
    console.log(`✅ Loaded ${allData.length} draws\n`);
    
    const r604 = runBacktest(allData, 45, ensembleV604, 'V604');
    const r700 = runBacktest(allData, 45, ensembleV700, 'V700');
    
    const pct = (v, t) => `${(v/t*100).toFixed(1)}%`;
    
    console.log('\n' + '='.repeat(70));
    console.log('📊 SO SÁNH V604 vs V700');
    console.log('='.repeat(70));
    
    for (const pool of [6, 10, 15, 20]) {
        console.log(`\n${'─'.repeat(40)}`);
        console.log(`Pool-${pool}:`);
        for (let k = 6; k >= 3; k--) {
            let a604 = 0, a700 = 0;
            for (let i = k; i <= 6; i++) { a604 += r604.counts[pool][i]; a700 += r700.counts[pool][i]; }
            const diff = a700 - a604;
            const arrow = diff > 0 ? `⬆️ +${diff}` : diff < 0 ? `⬇️ ${diff}` : '➡️  0';
            console.log(`  ≥${k}/6: V604=${a604}(${pct(a604,r604.nTest)}) | V700=${a700}(${pct(a700,r700.nTest)}) | ${arrow}`);
        }
    }
    
    // Key metrics
    console.log('\n' + '='.repeat(70));
    console.log('🔑 KEY METRICS:');
    const k56_604_15 = r604.counts[15][5]+r604.counts[15][6];
    const k56_700_15 = r700.counts[15][5]+r700.counts[15][6];
    const k56_604_20 = r604.counts[20][5]+r604.counts[20][6];
    const k56_700_20 = r700.counts[20][5]+r700.counts[20][6];
    console.log(`  Pool-15 ≥5/6: V604=${k56_604_15}(${pct(k56_604_15,r604.nTest)}) → V700=${k56_700_15}(${pct(k56_700_15,r700.nTest)})`);
    console.log(`  Pool-20 ≥5/6: V604=${k56_604_20}(${pct(k56_604_20,r604.nTest)}) → V700=${k56_700_20}(${pct(k56_700_20,r700.nTest)})`);
    console.log(`  Pool-6 exact 6/6: V604=${r604.counts[6][6]} → V700=${r700.counts[6][6]}`);
    console.log('='.repeat(70));
}

main().catch(console.error);
