/**
 * FULL BACKTEST — TINNAM888888 V700 QUANTUM SUPREME
 * ==================================================
 * Test all prediction methods across ALL historical draws
 * Reports hit rates for 6/6, 5/6, 4/6, 3/6 etc.
 * 
 * Uses Node.js since Python is broken on this machine.
 * Reimplements core prediction algorithms in JS for testing.
 */

const https = require('https');
const http = require('http');

// ============================================================
// DATA FETCHER
// ============================================================
function fetchData(gameType = "Mega 6/45") {
    return new Promise((resolve, reject) => {
        const maxNum = gameType === "Mega 6/45" ? 45 : 55;
        const today = new Date();
        const dateStr = `${String(today.getDate()).padStart(2,'0')}-${String(today.getMonth()+1).padStart(2,'0')}-${today.getFullYear()}`;
        
        const url = gameType === "Mega 6/45"
            ? `https://www.ketquadientoan.com/tat-ca-ky-xo-so-mega-6-45.html?datef=18-07-2016&datet=${dateStr}`
            : `https://www.ketquadientoan.com/tat-ca-ky-xo-so-power-655.html?datef=01-01-2018&datet=${dateStr}`;

        console.log(`📡 Fetching ${gameType} data from ketquadientoan.com...`);
        
        https.get(url, { headers: { 'User-Agent': 'Mozilla/5.0' } }, (res) => {
            let data = '';
            res.on('data', c => data += c);
            res.on('end', () => {
                const history = [];
                const regex = /class="home-mini-whiteball">\s*(\d{2})\s*</g;
                const rows = data.split(/<tr/gi);
                
                for (const row of rows) {
                    const nums = [];
                    let m;
                    const ballRegex = /class="home-mini-whiteball">\s*(\d{2})\s*</g;
                    while ((m = ballRegex.exec(row)) !== null) {
                        nums.push(parseInt(m[1]));
                    }
                    if (nums.length >= 6) {
                        const chunk = nums.slice(0, 6);
                        const sorted = [...chunk].sort((a,b) => a-b);
                        const unique = new Set(sorted);
                        if (unique.size === 6 && sorted.every(n => n >= 1 && n <= maxNum)) {
                            const key = sorted.join(',');
                            if (!history.some(h => h.join(',') === key)) {
                                history.push(sorted);
                            }
                        }
                    }
                }
                
                history.reverse(); // Oldest first
                console.log(`  ✅ Loaded ${history.length} draws`);
                resolve(history);
            });
        }).on('error', (e) => {
            console.log(`  ⚠️ Error: ${e.message}, trying GitHub fallback...`);
            // GitHub fallback
            const ghUrl = gameType === "Mega 6/45"
                ? 'https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl'
                : 'https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power655.jsonl';
            
            https.get(ghUrl, (res) => {
                let data = '';
                res.on('data', c => data += c);
                res.on('end', () => {
                    const history = [];
                    for (const line of data.trim().split('\n')) {
                        try {
                            const obj = JSON.parse(line);
                            if (obj.result && obj.result.length >= 6) {
                                const draw = obj.result.slice(0, 6).map(Number).sort((a,b) => a-b);
                                if (draw.every(n => n >= 1 && n <= maxNum)) {
                                    history.push(draw);
                                }
                            }
                        } catch(e) {}
                    }
                    console.log(`  ✅ Loaded ${history.length} draws from GitHub`);
                    resolve(history);
                });
            }).on('error', reject);
        });
    });
}

// ============================================================
// PREDICTION METHODS (Reimplemented in JS)
// ============================================================

// 1. Markov Chain
function predictMarkov(data, maxNum) {
    const transitions = {};
    for (let i = 0; i < data.length - 1; i++) {
        const key = data[i].join(',');
        if (!transitions[key]) transitions[key] = {};
        for (const num of data[i+1]) {
            transitions[key][num] = (transitions[key][num] || 0) + 1;
        }
    }
    const lastKey = data[data.length - 1].join(',');
    if (transitions[lastKey]) {
        const sorted = Object.entries(transitions[lastKey]).sort((a,b) => b[1] - a[1]);
        return sorted.slice(0, 6).map(e => parseInt(e[0]));
    }
    return predictFrequency(data, maxNum, 20);
}

// 2. Gap/Overdue
function predictGapOverdue(data, maxNum, topN = 6) {
    const lastSeen = {};
    for (let i = 0; i < data.length; i++) {
        for (const num of data[i]) lastSeen[num] = i;
    }
    const n = data.length;
    const avgGaps = {};
    const lastIdx = {};
    const gapLists = {};
    
    for (let i = 0; i < data.length; i++) {
        for (const num of data[i]) {
            if (!gapLists[num]) gapLists[num] = [];
            if (lastIdx[num] !== undefined) {
                gapLists[num].push(i - lastIdx[num]);
            }
            lastIdx[num] = i;
        }
    }
    
    const scores = {};
    for (let num = 1; num <= maxNum; num++) {
        const gap = n - (lastSeen[num] !== undefined ? lastSeen[num] : -1);
        const gaps = gapLists[num] || [];
        const meanGap = gaps.length > 0 ? gaps.reduce((a,b) => a+b, 0) / gaps.length : n;
        scores[num] = gap / (meanGap + 0.1);
    }
    
    return Object.entries(scores).sort((a,b) => b[1] - a[1]).slice(0, topN).map(e => parseInt(e[0]));
}

// 3. Momentum Neural (Sigmoid decay)
function predictMomentum(data, maxNum) {
    const weights = {};
    for (let num = 1; num <= maxNum; num++) weights[num] = 0;
    const td = data.length;
    for (let i = 0; i < data.length; i++) {
        const decay = 1 / (1 + Math.exp(-(i - td + 20) / 5));
        for (const num of data[i]) weights[num] += decay;
    }
    return Object.entries(weights).sort((a,b) => b[1] - a[1]).slice(0, 6).map(e => parseInt(e[0]));
}

// 4. KNN Mirror
function predictKNN(data, maxNum) {
    const n = data.length;
    if (n < 20) return predictMomentum(data, maxNum);
    
    let pattern = new Set([...data[n-1], ...data[n-2], ...data[n-3]]);
    if (n > 3) data[n-4].forEach(x => pattern.add(x));
    
    const similarities = [];
    for (let i = 3; i < n - 3; i++) {
        const past = new Set([...data[i], ...data[i-1], ...data[i-2], ...data[i-3]]);
        let intersect = 0;
        for (const x of pattern) if (past.has(x)) intersect++;
        const recency = 1.0 + 0.5 * (i / n);
        if (intersect >= 5) similarities.push([intersect * recency, i + 1]);
    }
    
    similarities.sort((a,b) => b[0] - a[0]);
    const votes = {};
    for (const [score, nextIdx] of similarities.slice(0, 30)) {
        if (nextIdx < n) {
            for (const num of data[nextIdx]) {
                votes[num] = (votes[num] || 0) + score;
            }
        }
    }
    
    if (Object.keys(votes).length === 0) return predictMomentum(data, maxNum);
    return Object.entries(votes).sort((a,b) => b[1] - a[1]).slice(0, 20).map(e => parseInt(e[0]));
}

// 5. Frequency (simple)
function predictFrequency(data, maxNum, lookback = null) {
    const subset = lookback ? data.slice(-lookback) : data;
    const freq = {};
    for (const draw of subset) {
        for (const num of draw) freq[num] = (freq[num] || 0) + 1;
    }
    return Object.entries(freq).sort((a,b) => b[1] - a[1]).slice(0, 6).map(e => parseInt(e[0]));
}

// 6. Conditional Probability
function predictCondProb(data, maxNum) {
    if (data.length < 30) return [];
    const last = new Set(data[data.length - 1]);
    const condCounts = {};
    const totalGiven = {};
    
    for (let i = 0; i < data.length - 1; i++) {
        for (const given of data[i]) {
            totalGiven[given] = (totalGiven[given] || 0) + 1;
            if (!condCounts[given]) condCounts[given] = {};
            for (const next of data[i+1]) {
                condCounts[given][next] = (condCounts[given][next] || 0) + 1;
            }
        }
    }
    
    const scores = {};
    for (let num = 1; num <= maxNum; num++) {
        scores[num] = 0;
        for (const given of last) {
            if (totalGiven[given] > 0 && condCounts[given] && condCounts[given][num]) {
                scores[num] += condCounts[given][num] / totalGiven[given];
            }
        }
    }
    
    return Object.entries(scores).sort((a,b) => b[1] - a[1]).slice(0, 15).map(e => parseInt(e[0]));
}

// 7. Delta Momentum
function predictDeltaMomentum(data, maxNum) {
    if (data.length < 30) return predictMomentum(data, maxNum);
    const scores = {};
    for (let num = 1; num <= maxNum; num++) {
        const f5 = data.slice(-5).filter(d => d.includes(num)).length / 5;
        const f5prev = data.slice(-10, -5).filter(d => d.includes(num)).length / 5;
        const f15 = data.slice(-15).filter(d => d.includes(num)).length / 15;
        const f15prev = data.slice(-30, -15).filter(d => d.includes(num)).length / 15;
        scores[num] = (f5 - f5prev) * 3 + (f15 - f15prev) * 2;
        if (data[data.length-1].includes(num)) scores[num] += 0.5;
    }
    return Object.entries(scores).sort((a,b) => b[1] - a[1]).slice(0, 15).map(e => parseInt(e[0]));
}

// 8. Freq-Gap Hybrid
function predictFreqGapHybrid(data, maxNum) {
    if (data.length < 30) return predictGapOverdue(data, maxNum);
    const expected = 6 / maxNum;
    const scores = {};
    for (let num = 1; num <= maxNum; num++) {
        const f5 = data.slice(-5).filter(d => d.includes(num)).length / 5;
        const f15 = data.slice(-15).filter(d => d.includes(num)).length / 15;
        const freqScore = (f5 / (expected + 0.01)) * 0.6 + (f15 / (expected + 0.01)) * 0.4;
        
        let lastSeen = -1;
        for (let i = data.length - 1; i >= 0; i--) {
            if (data[i].includes(num)) { lastSeen = i; break; }
        }
        const gap = lastSeen >= 0 ? data.length - lastSeen : data.length;
        
        const apps = [];
        for (let i = 0; i < data.length; i++) {
            if (data[i].includes(num)) apps.push(i);
        }
        let meanGap = maxNum / 6;
        if (apps.length >= 2) {
            const gaps = [];
            for (let j = 0; j < apps.length - 1; j++) gaps.push(apps[j+1] - apps[j]);
            meanGap = gaps.reduce((a,b) => a+b, 0) / gaps.length;
        }
        const overdue = gap / (meanGap + 0.1);
        
        if (freqScore > 0.8 && overdue > 0.7) scores[num] = freqScore * overdue * 3;
        else if (overdue > 1.5) scores[num] = overdue * 1.5;
        else if (freqScore > 1.3) scores[num] = freqScore * 2;
        else scores[num] = freqScore * 0.5 + overdue * 0.5;
    }
    return Object.entries(scores).sort((a,b) => b[1] - a[1]).slice(0, 15).map(e => parseInt(e[0]));
}

// 9. Pair Matrix
function predictPairMatrix(data, maxNum) {
    if (data.length < 30) return predictGapOverdue(data, maxNum);
    const pairScores = {};
    const n = data.length;
    for (let idx = 0; idx < n; idx++) {
        const decay = 0.3 + 0.7 * (idx / n);
        const d = data[idx].slice(0, 6).sort((a,b) => a-b);
        for (let i = 0; i < d.length; i++) {
            for (let j = i+1; j < d.length; j++) {
                const key = `${d[i]},${d[j]}`;
                pairScores[key] = (pairScores[key] || 0) + decay;
            }
        }
    }
    const lastDraw = new Set(data[n-1].slice(0, 6));
    const candScores = {};
    for (let num = 1; num <= maxNum; num++) {
        if (lastDraw.has(num)) continue;
        candScores[num] = 0;
        for (const anchor of lastDraw) {
            const key = [num, anchor].sort((a,b) => a-b).join(',');
            candScores[num] += pairScores[key] || 0;
        }
    }
    return Object.entries(candScores).sort((a,b) => b[1] - a[1]).slice(0, 15).map(e => parseInt(e[0]));
}

// 10. 9-Model Ensemble (simplified)
function predict9ModelEnsemble(data, maxNum) {
    const m1 = predictMarkov(data, maxNum);
    const m2 = predictGapOverdue(data, maxNum, 15);
    const m3 = predictMomentum(data, maxNum);
    const m5 = predictKNN(data, maxNum);
    const m6 = predictPairMatrix(data, maxNum);
    const m7 = predictDeltaMomentum(data, maxNum);
    const m8 = predictCondProb(data, maxNum);
    const m9 = predictFreqGapHybrid(data, maxNum);
    
    const votes = {};
    const addVotes = (arr, w, limit = 15) => {
        for (const num of arr.slice(0, limit)) votes[num] = (votes[num] || 0) + w;
    };
    
    addVotes(m5, 12); addVotes(m6, 8); addVotes(m8, 6);
    addVotes(m9, 5);  addVotes(m7, 4); addVotes(m2, 3);
    addVotes(m3, 2, 6); addVotes(m1, 1, 6);
    
    return Object.entries(votes).sort((a,b) => b[1] - a[1]).slice(0, 20).map(e => parseInt(e[0]));
}

// ============================================================
// BACKTEST
// ============================================================
function runBacktest(allData, gameType) {
    const maxNum = gameType === "Mega 6/45" ? 45 : 55;
    const minHistory = 80;
    
    const methods = {
        'Markov Chain (top 6)':          { fn: (d) => predictMarkov(d, maxNum).slice(0, 6), pool: 6 },
        'Gap Overdue (top 6)':           { fn: (d) => predictGapOverdue(d, maxNum, 6), pool: 6 },
        'Momentum Neural (top 6)':       { fn: (d) => predictMomentum(d, maxNum), pool: 6 },
        'KNN Mirror (top 6)':            { fn: (d) => predictKNN(d, maxNum).slice(0, 6), pool: 6 },
        'KNN Mirror (top 10)':           { fn: (d) => predictKNN(d, maxNum).slice(0, 10), pool: 10 },
        'KNN Mirror (top 15)':           { fn: (d) => predictKNN(d, maxNum).slice(0, 15), pool: 15 },
        'KNN Mirror (top 20)':           { fn: (d) => predictKNN(d, maxNum).slice(0, 20), pool: 20 },
        'Pair Matrix (top 6)':           { fn: (d) => predictPairMatrix(d, maxNum).slice(0, 6), pool: 6 },
        'Cond Probability (top 6)':      { fn: (d) => predictCondProb(d, maxNum).slice(0, 6), pool: 6 },
        'Delta Momentum (top 6)':        { fn: (d) => predictDeltaMomentum(d, maxNum).slice(0, 6), pool: 6 },
        'Freq-Gap Hybrid (top 6)':       { fn: (d) => predictFreqGapHybrid(d, maxNum).slice(0, 6), pool: 6 },
        'Frequency 20 (top 6)':          { fn: (d) => predictFrequency(d, maxNum, 20), pool: 6 },
        '9-Model Ensemble (top 6)':      { fn: (d) => predict9ModelEnsemble(d, maxNum).slice(0, 6), pool: 6 },
        '9-Model Ensemble (top 10)':     { fn: (d) => predict9ModelEnsemble(d, maxNum).slice(0, 10), pool: 10 },
        '9-Model Ensemble (top 15)':     { fn: (d) => predict9ModelEnsemble(d, maxNum).slice(0, 15), pool: 15 },
        '9-Model Ensemble (top 20)':     { fn: (d) => predict9ModelEnsemble(d, maxNum).slice(0, 20), pool: 20 },
    };
    
    const results = {};
    for (const name in methods) {
        results[name] = { counts: {0:0,1:0,2:0,3:0,4:0,5:0,6:0}, tested: 0 };
    }
    
    const totalTests = allData.length - minHistory;
    console.log(`\n${'='.repeat(80)}`);
    console.log(`BACKTEST: ${gameType} — ${totalTests} kỳ (từ kỳ ${minHistory + 1} đến ${allData.length})`);
    console.log(`${'='.repeat(80)}\n`);
    
    const t0 = Date.now();
    
    for (let testIdx = minHistory; testIdx < allData.length; testIdx++) {
        const history = allData.slice(0, testIdx);
        const actual = new Set(allData[testIdx].slice(0, 6));
        const progress = testIdx - minHistory + 1;
        
        if (progress % 100 === 0) {
            const elapsed = (Date.now() - t0) / 1000;
            const pct = (progress / totalTests * 100).toFixed(1);
            const eta = ((elapsed / progress) * (totalTests - progress)).toFixed(0);
            console.log(`  Progress: ${progress}/${totalTests} (${pct}%) — ${elapsed.toFixed(0)}s elapsed — ETA: ${eta}s`);
        }
        
        for (const [name, method] of Object.entries(methods)) {
            try {
                const pred = method.fn(history);
                let hits = 0;
                for (const num of pred) {
                    if (actual.has(num)) hits++;
                }
                hits = Math.min(hits, 6); // Cap at 6 for larger pools
                results[name].counts[hits]++;
                results[name].tested++;
            } catch(e) {
                // Skip errors silently
            }
        }
    }
    
    const elapsed = (Date.now() - t0) / 1000;
    
    // ============================================================
    // PRINT RESULTS
    // ============================================================
    console.log(`\n${'='.repeat(120)}`);
    console.log(`RESULTS — ${gameType} (${totalTests} kỳ tested, ${elapsed.toFixed(1)}s)`);
    console.log(`${'='.repeat(120)}`);
    
    const header = [
        'Method'.padEnd(35),
        '0/6'.padStart(6), '1/6'.padStart(6), '2/6'.padStart(6),
        '3/6'.padStart(6), '4/6'.padStart(6), '5/6'.padStart(6), '6/6'.padStart(6),
        'Total'.padStart(7), '≥3/6%'.padStart(8), '≥4/6%'.padStart(8)
    ].join(' ');
    console.log(`\n${header}`);
    console.log('-'.repeat(120));
    
    for (const [name, result] of Object.entries(results)) {
        const c = result.counts;
        const total = result.tested;
        if (total === 0) continue;
        
        const ge3 = c[3] + c[4] + c[5] + c[6];
        const ge4 = c[4] + c[5] + c[6];
        const pct3 = (ge3 / total * 100).toFixed(2);
        const pct4 = (ge4 / total * 100).toFixed(2);
        
        const row = [
            name.padEnd(35),
            String(c[0]).padStart(6), String(c[1]).padStart(6), String(c[2]).padStart(6),
            String(c[3]).padStart(6), String(c[4]).padStart(6), String(c[5]).padStart(6), String(c[6]).padStart(6),
            String(total).padStart(7), `${pct3}%`.padStart(8), `${pct4}%`.padStart(8)
        ].join(' ');
        console.log(row);
    }
    
    // JACKPOT highlight
    console.log(`\n${'='.repeat(80)}`);
    console.log('🎯 HIGHLIGHT: 6/6 JACKPOT HITS');
    console.log(`${'='.repeat(80)}`);
    
    let anyJackpot = false;
    for (const [name, result] of Object.entries(results)) {
        if (result.counts[6] > 0) {
            console.log(`  🏆 ${name}: ${result.counts[6]} times out of ${result.tested} (${(result.counts[6]/result.tested*100).toFixed(6)}%)`);
            anyJackpot = true;
        }
    }
    if (!anyJackpot) {
        console.log('  ❌ Không có phương pháp nào trúng 6/6 trong backtest.');
        const totalCombos = combination(maxNum, 6);
        console.log(`  📊 Đây là BÌNH THƯỜNG — xác suất random 6/6 = 1/${totalCombos.toLocaleString()} ≈ ${(1/totalCombos*100).toFixed(7)}%`);
    }
    
    // Expected random baseline
    console.log(`\n${'='.repeat(80)}`);
    console.log('📊 SO SÁNH VỚI RANDOM BASELINE');
    console.log(`${'='.repeat(80)}`);
    const C = combination;
    const totalC = C(maxNum, 6);
    for (const k of [0, 1, 2, 3, 4, 5, 6]) {
        const prob = C(6, k) * C(maxNum - 6, 6 - k) / totalC;
        console.log(`  Random ${k}/6: ${(prob * 100).toFixed(4)}%`);
    }
    
    return results;
}

function combination(n, k) {
    if (k > n) return 0;
    let result = 1;
    for (let i = 0; i < k; i++) {
        result = result * (n - i) / (i + 1);
    }
    return Math.round(result);
}

// ============================================================
// MAIN
// ============================================================
async function main() {
    console.log('🧬'.repeat(40));
    console.log('  FULL BACKTEST — TINNAM888888 V700 QUANTUM SUPREME');
    console.log('🧬'.repeat(40));
    console.log(`  Time: ${new Date().toISOString()}\n`);
    
    try {
        const data = await fetchData("Mega 6/45");
        
        if (!data || data.length < 100) {
            console.log('❌ Không đủ dữ liệu! Chỉ có ' + (data ? data.length : 0) + ' kỳ');
            return;
        }
        
        console.log(`\n📊 Dữ liệu: ${data.length} kỳ`);
        console.log(`   Kỳ đầu: [${data[0].join(', ')}]`);
        console.log(`   Kỳ cuối: [${data[data.length-1].join(', ')}]`);
        
        runBacktest(data, "Mega 6/45");
        
    } catch(e) {
        console.error('❌ Error:', e);
    }
}

main();
