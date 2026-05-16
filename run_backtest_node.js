/**
 * TINNAM AI V200 — QUICK LOCAL BACKTEST (Node.js)
 * Tests Top-6, Top-10, Top-15 accuracy on all historical draws.
 * Uses the same statistical signals as the Python NexusEngine.
 */
const https = require('https');
const http = require('http');
const fs = require('fs');

function fetchData() {
    return new Promise((resolve, reject) => {
        const url = 'https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl';
        https.get(url, (res) => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => {
                const history = [];
                data.trim().split('\n').forEach(line => {
                    try {
                        const obj = JSON.parse(line);
                        if (obj.result && obj.result.length >= 6) {
                            const draw = obj.result.slice(0, 6).map(Number).sort((a, b) => a - b);
                            if (draw.length === 6 && new Set(draw).size === 6) {
                                history.push(draw);
                            }
                        }
                    } catch (e) {}
                });
                resolve(history);
            });
            res.on('error', reject);
        }).on('error', reject);
    });
}

// ===== SIGNAL IMPLEMENTATIONS =====

function sigFrequency(data, maxNum) {
    const scores = {};
    for (let n = 1; n <= maxNum; n++) scores[n] = 0;
    const all = data.flat();
    const freq = {};
    all.forEach(n => freq[n] = (freq[n] || 0) + 1);
    const ep = 6 / maxNum;
    const ec = data.length * ep;
    const std = Math.sqrt(data.length * ep * (1 - ep));
    for (let n = 1; n <= maxNum; n++) {
        scores[n] = std > 0 ? ((freq[n] || 0) - ec) / std * 2.0 : 0;
    }
    return scores;
}

function sigGapTiming(data, maxNum) {
    const scores = {};
    const n = data.length;
    for (let num = 1; num <= maxNum; num++) {
        const apps = [];
        data.forEach((d, i) => { if (d.includes(num)) apps.push(i); });
        if (apps.length < 5) { scores[num] = 0; continue; }
        const gaps = [];
        for (let j = 0; j < apps.length - 1; j++) gaps.push(apps[j + 1] - apps[j]);
        const mg = gaps.reduce((a, b) => a + b, 0) / gaps.length;
        const sg = Math.sqrt(gaps.reduce((a, b) => a + (b - mg) ** 2, 0) / gaps.length);
        const cur = n - apps[apps.length - 1];
        const z = sg > 0 ? (cur - mg) / sg : 0;
        const pa = gaps.filter(g => g <= cur).length / gaps.length;
        scores[num] = z > 0.5 ? z * 1.5 + pa * 2 : (z < -1 ? -1 : 0);
    }
    return scores;
}

function sigMomentum(data, maxNum) {
    const scores = {};
    const n = data.length;
    for (let num = 1; num <= maxNum; num++) {
        if (n < 50) { scores[num] = 0; continue; }
        const f5 = data.slice(-5).filter(d => d.includes(num)).length / 5;
        const f10 = data.slice(-10).filter(d => d.includes(num)).length / 10;
        const f20 = data.slice(-20).filter(d => d.includes(num)).length / 20;
        const f50 = data.slice(-50).filter(d => d.includes(num)).length / 50;
        scores[num] = (f5 - f10) * 15 + (f10 - f20) * 8 + (f20 - f50) * 4;
    }
    return scores;
}

function sigStreak(data, maxNum) {
    const scores = {};
    const eg = maxNum / 6;
    for (let num = 1; num <= maxNum; num++) {
        let cold = 0;
        for (let i = data.length - 1; i >= 0; i--) {
            if (!data[i].includes(num)) cold++;
            else break;
        }
        scores[num] = cold > 0 ? 1 / (1 + Math.exp(-3 * (cold / eg - 0.8))) * 2 : 0;
    }
    return scores;
}

function sigTransition(data, maxNum) {
    const scores = {};
    const follow = {};
    const pc = {};
    for (let i = 0; i < data.length - 1; i++) {
        for (const p of data[i]) {
            pc[p] = (pc[p] || 0) + 1;
            if (!follow[p]) follow[p] = {};
            for (const nx of data[i + 1]) {
                follow[p][nx] = (follow[p][nx] || 0) + 1;
            }
        }
    }
    const last = new Set(data[data.length - 1]);
    const base = 6 / maxNum;
    for (let num = 1; num <= maxNum; num++) {
        let tf = 0, tp = 0;
        for (const p of last) {
            tf += (follow[p] && follow[p][num]) || 0;
            tp += pc[p] || 0;
        }
        tp = Math.max(tp, 1);
        scores[num] = (tf / tp / base - 1) * 3;
    }
    return scores;
}

function sigSlidingWindow(data, maxNum) {
    const scores = {};
    for (let n = 1; n <= maxNum; n++) scores[n] = 0;
    const windows = [5, 10, 20, 40, 80];
    const wWeights = [5.0, 3.0, 2.0, 1.0, 0.5];
    const expected = 6 / maxNum;
    for (let wi = 0; wi < windows.length; wi++) {
        const w = windows[wi];
        if (data.length < w) continue;
        const recent = data.slice(-w);
        const freq = {};
        recent.forEach(d => d.forEach(n => freq[n] = (freq[n] || 0) + 1));
        for (let n = 1; n <= maxNum; n++) {
            const observed = (freq[n] || 0) / w;
            scores[n] += ((observed - expected) / (expected + 0.001)) * wWeights[wi];
        }
    }
    return scores;
}

function sigConditionalProb(data, maxNum) {
    const scores = {};
    for (let n = 1; n <= maxNum; n++) scores[n] = 0;
    if (data.length < 30) return scores;
    const last = new Set(data[data.length - 1]);
    const condCounts = {};
    const totalGiven = {};
    for (let i = 0; i < data.length - 1; i++) {
        for (const g of data[i]) {
            totalGiven[g] = (totalGiven[g] || 0) + 1;
            if (!condCounts[g]) condCounts[g] = {};
            for (const nx of data[i + 1]) {
                condCounts[g][nx] = (condCounts[g][nx] || 0) + 1;
            }
        }
    }
    for (let n = 1; n <= maxNum; n++) {
        let probSum = 0;
        for (const g of last) {
            if (totalGiven[g] > 0 && condCounts[g]) {
                probSum += (condCounts[g][n] || 0) / totalGiven[g];
            }
        }
        scores[n] = probSum * 3.0;
    }
    return scores;
}

function sigGapAcceleration(data, maxNum) {
    const scores = {};
    const n = data.length;
    for (let num = 1; num <= maxNum; num++) {
        scores[num] = 0;
        const apps = [];
        data.forEach((d, i) => { if (d.includes(num)) apps.push(i); });
        if (apps.length < 4) continue;
        const gaps = [];
        for (let j = 0; j < apps.length - 1; j++) gaps.push(apps[j + 1] - apps[j]);
        if (gaps.length < 3) continue;
        const recent = gaps.slice(-5);
        if (recent.length < 2) continue;
        const diffs = [];
        for (let i = 1; i < recent.length; i++) diffs.push(recent[i] - recent[i - 1]);
        const avgAccel = diffs.reduce((a, b) => a + b, 0) / diffs.length;
        const curGap = n - apps[apps.length - 1];
        const meanGap = gaps.reduce((a, b) => a + b, 0) / gaps.length;
        const overdueRatio = curGap / (meanGap + 0.1);
        if (avgAccel < 0 && overdueRatio > 0.8) {
            scores[num] = Math.abs(avgAccel) * overdueRatio * 2.0;
        } else if (overdueRatio > 1.5) {
            scores[num] = overdueRatio * 1.5;
        }
    }
    return scores;
}

function sigDeltaMomentum(data, maxNum) {
    const scores = {};
    for (let n = 1; n <= maxNum; n++) scores[n] = 0;
    if (data.length < 30) return scores;
    for (let num = 1; num <= maxNum; num++) {
        const f1 = data.slice(-10).filter(d => d.includes(num)).length / 10;
        const f2 = data.slice(-20, -10).filter(d => d.includes(num)).length / 10;
        const f3 = data.slice(-30, -20).filter(d => d.includes(num)).length / 10;
        const v1 = f1 - f2, v2 = f2 - f3, accel = v1 - v2;
        if (accel > 0 && v1 > 0) scores[num] = accel * 15.0 + v1 * 5.0;
        else if (accel > 0) scores[num] = accel * 8.0;
    }
    return scores;
}

function sigHotColdCross(data, maxNum) {
    const scores = {};
    for (let n = 1; n <= maxNum; n++) scores[n] = 0;
    if (data.length < 50) return scores;
    const expected = 6 / maxNum;
    for (let num = 1; num <= maxNum; num++) {
        const s = data.slice(-10).filter(d => d.includes(num)).length / 10;
        const m = data.slice(-30, -10).filter(d => d.includes(num)).length / 20;
        const l = data.slice(-80).filter(d => d.includes(num)).length / 80;
        if (s > expected * 1.3 && m < expected * 0.7) scores[num] = (s - m) * 8.0;
        else if (s < expected * 0.5 && l > expected * 1.2) scores[num] = (l - s) * 3.0;
    }
    return scores;
}

// ===== MAIN PREDICTION =====

function predictTopPool(data, maxNum) {
    const signals = [
        sigFrequency(data, maxNum),
        sigGapTiming(data, maxNum),
        sigMomentum(data, maxNum),
        sigStreak(data, maxNum),
        sigTransition(data, maxNum),
        sigSlidingWindow(data, maxNum),
        sigConditionalProb(data, maxNum),
        sigGapAcceleration(data, maxNum),
        sigDeltaMomentum(data, maxNum),
        sigHotColdCross(data, maxNum),
    ];
    
    const scores = {};
    for (let n = 1; n <= maxNum; n++) scores[n] = 0;
    
    for (const sig of signals) {
        const vals = Object.values(sig);
        const maxS = Math.max(...vals.map(Math.abs));
        if (maxS < 0.001) continue;
        for (let n = 1; n <= maxNum; n++) {
            scores[n] += (sig[n] || 0) / maxS;
        }
    }
    
    const ranked = Object.entries(scores).sort((a, b) => b[1] - a[1]);
    return ranked.map(x => parseInt(x[0]));
}

// ===== BACKTEST =====

async function main() {
    console.log('='.repeat(70));
    console.log('  TINNAM AI V200.0 — FULL HISTORY BACKTEST (Node.js)');
    console.log('='.repeat(70));
    
    console.log('\n[1/3] Fetching ALL historical data...');
    const data = await fetchData();
    console.log(`  => Got ${data.length} draws`);
    
    const maxNum = 45;
    const startIdx = 60;
    const total = data.length;
    
    const counts6 = {0:0,1:0,2:0,3:0,4:0,5:0,6:0};
    const counts10 = {0:0,1:0,2:0,3:0,4:0,5:0,6:0};
    const counts15 = {0:0,1:0,2:0,3:0,4:0,5:0,6:0};
    
    console.log(`\n[2/3] Backtesting ${total - startIdx} draws (draw ${startIdx} to ${total})...\n`);
    const t0 = Date.now();
    
    for (let idx = startIdx; idx < total; idx++) {
        const hist = data.slice(0, idx);
        const actual = new Set(data[idx]);
        
        const pool = predictTopPool(hist, maxNum);
        const top6 = new Set(pool.slice(0, 6));
        const top10 = new Set(pool.slice(0, 10));
        const top15 = new Set(pool.slice(0, 15));
        
        let hit6 = 0, hit10 = 0, hit15 = 0;
        for (const n of actual) {
            if (top6.has(n)) hit6++;
            if (top10.has(n)) hit10++;
            if (top15.has(n)) hit15++;
        }
        
        counts6[hit6]++;
        counts10[hit10]++;
        counts15[hit15]++;
        
        if ((idx - startIdx + 1) % 100 === 0) {
            const elapsed = (Date.now() - t0) / 1000;
            const eta = (elapsed / (idx - startIdx + 1)) * (total - idx - 1);
            console.log(`  Progress: ${idx - startIdx + 1}/${total - startIdx} (${((idx - startIdx + 1)/(total - startIdx)*100).toFixed(1)}%) | ${elapsed.toFixed(0)}s | ETA: ${eta.toFixed(0)}s`);
        }
    }
    
    const elapsed = (Date.now() - t0) / 1000;
    const nTest = total - startIdx;
    
    console.log(`\n${'='.repeat(70)}`);
    console.log(`  RESULTS — ${nTest} draws tested in ${elapsed.toFixed(1)}s`);
    console.log(`${'='.repeat(70)}`);
    
    const pct = (c, t) => t > 0 ? `${(c/t*100).toFixed(1)}%` : '0%';
    
    console.log(`\n--- TOP-6 (6 so chinh xac nhat) ---`);
    for (let k = 6; k >= 0; k--) {
        const tag = {6:'JACKPOT',5:'GIAI 1',4:'GIAI 2',3:'GIAI 3'}[k] || '';
        console.log(`  Trung ${k}/6: ${String(counts6[k]).padStart(5)} ky  (${pct(counts6[k], nTest).padStart(6)})  ${tag}`);
    }
    
    const ge3_6 = counts6[3] + counts6[4] + counts6[5] + counts6[6];
    const ge4_6 = counts6[4] + counts6[5] + counts6[6];
    console.log(`\n  => Top-6 trung >=3/6: ${pct(ge3_6, nTest)} (${ge3_6}/${nTest})`);
    console.log(`  => Top-6 trung >=4/6: ${pct(ge4_6, nTest)} (${ge4_6}/${nTest})`);
    
    console.log(`\n--- TOP-10 (Ho 10 so) ---`);
    for (let k = 6; k >= 0; k--) {
        const above = Object.keys(counts10).filter(i => parseInt(i) >= k).reduce((s, i) => s + counts10[i], 0);
        console.log(`  >= ${k}/6: ${String(above).padStart(5)} ky  (${pct(above, nTest).padStart(6)})`);
    }
    
    console.log(`\n--- TOP-15 (Ho 15 so) ---`);
    for (let k = 6; k >= 0; k--) {
        const above = Object.keys(counts15).filter(i => parseInt(i) >= k).reduce((s, i) => s + counts15[i], 0);
        console.log(`  >= ${k}/6: ${String(above).padStart(5)} ky  (${pct(above, nTest).padStart(6)})`);
    }
    
    // Save
    const results = { version: 'V200.0', total_draws: total, tested: nTest, elapsed_s: elapsed.toFixed(1), top6: counts6, top10: counts10, top15: counts15, ge3_6_pct: (ge3_6/nTest*100).toFixed(2), ge4_6_pct: (ge4_6/nTest*100).toFixed(2) };
    fs.writeFileSync('nexus_backtest_results.json', JSON.stringify(results, null, 2));
    console.log(`\n=> Saved to nexus_backtest_results.json`);
}

main().catch(e => console.error(e));
