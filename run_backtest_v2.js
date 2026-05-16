/**
 * TINNAM AI V200.1 — ENHANCED BACKTEST WITH OPTIMIZED SIGNALS
 * Key improvements:
 * 1. Walk-forward calibration (dynamic signal weighting per draw)
 * 2. Anti-repeat signal (penalize recent numbers)
 * 3. Pair co-occurrence boost
 * 4. Position frequency
 * 5. Better normalization
 */
const https = require('https');
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
                            if (draw.length === 6 && new Set(draw).size === 6) history.push(draw);
                        }
                    } catch (e) {}
                });
                resolve(history);
            });
            res.on('error', reject);
        }).on('error', reject);
    });
}

const MAX = 45;

// --- SIGNAL 1: Weighted Recency Frequency ---
function sigRecency(data) {
    const s = {}; for (let n=1;n<=MAX;n++) s[n]=0;
    const len = data.length;
    for (let i = 0; i < len; i++) {
        const w = Math.exp(-(len - 1 - i) * 0.03); // half-life ~23 draws
        for (const n of data[i]) s[n] += w;
    }
    return s;
}

// --- SIGNAL 2: Gap Overdue with Z-score ---
function sigGapOverdue(data) {
    const s = {}; const n = data.length;
    for (let num=1;num<=MAX;num++) {
        const apps = [];
        data.forEach((d, i) => { if (d.includes(num)) apps.push(i); });
        if (apps.length < 3) { s[num] = 0; continue; }
        const gaps = [];
        for (let j=0;j<apps.length-1;j++) gaps.push(apps[j+1]-apps[j]);
        const mg = gaps.reduce((a,b)=>a+b,0)/gaps.length;
        const cur = n - apps[apps.length-1];
        s[num] = Math.max(0, (cur/mg - 1) * 3);
    }
    return s;
}

// --- SIGNAL 3: Momentum (multi-scale) ---
function sigMomentum(data) {
    const s = {};
    for (let num=1;num<=MAX;num++) {
        const f5 = data.slice(-5).filter(d=>d.includes(num)).length/5;
        const f15 = data.slice(-15).filter(d=>d.includes(num)).length/15;
        const f40 = data.slice(-40).filter(d=>d.includes(num)).length/40;
        s[num] = (f5-f15)*12 + (f15-f40)*5;
    }
    return s;
}

// --- SIGNAL 4: Transition (Markov bigram) ---
function sigTransition(data) {
    const s = {};
    const follow = {}, total = {};
    for (let i=0;i<data.length-1;i++) {
        for (const p of data[i]) {
            total[p] = (total[p]||0)+1;
            if (!follow[p]) follow[p] = {};
            for (const nx of data[i+1]) follow[p][nx]=(follow[p][nx]||0)+1;
        }
    }
    const last = data[data.length-1];
    for (let num=1;num<=MAX;num++) {
        let tf=0, tp=0;
        for (const p of last) { tf+=(follow[p]&&follow[p][num])||0; tp+=total[p]||0; }
        tp = Math.max(tp, 1);
        s[num] = (tf/tp/(6/MAX)-1)*3;
    }
    return s;
}

// --- SIGNAL 5: Anti-repeat ---
function sigAntiRepeat(data) {
    const s = {};
    const last = new Set(data[data.length-1]);
    const prev = data.length > 1 ? new Set(data[data.length-2]) : new Set();
    for (let num=1;num<=MAX;num++) {
        if (last.has(num) && prev.has(num)) s[num] = -3;
        else if (last.has(num)) s[num] = -1;
        else if (prev.has(num) && !last.has(num)) s[num] = 1.5;
        else s[num] = 0.3;
    }
    return s;
}

// --- SIGNAL 6: Pair boost (co-occurrence with last draw) ---
function sigPairBoost(data) {
    const s = {}; for (let n=1;n<=MAX;n++) s[n]=0;
    const last = data[data.length-1];
    const recent = data.slice(-100);
    const pairs = {};
    for (const d of recent) {
        for (let i=0;i<d.length;i++) for (let j=i+1;j<d.length;j++) {
            const k = `${d[i]}_${d[j]}`;
            pairs[k] = (pairs[k]||0)+1;
        }
    }
    for (let num=1;num<=MAX;num++) {
        for (const p of last) {
            const k = p < num ? `${p}_${num}` : `${num}_${p}`;
            if ((pairs[k]||0) > 2) s[num] += (pairs[k]-2)*0.3;
        }
    }
    return s;
}

// --- SIGNAL 7: Sliding Window ---
function sigSlidingWindow(data) {
    const s = {}; for (let n=1;n<=MAX;n++) s[n]=0;
    const wins = [5,10,20,40]; const ww = [5,3,2,1];
    const exp = 6/MAX;
    for (let wi=0;wi<wins.length;wi++) {
        if (data.length<wins[wi]) continue;
        const r = data.slice(-wins[wi]);
        const freq = {};
        r.forEach(d=>d.forEach(n=>freq[n]=(freq[n]||0)+1));
        for (let n=1;n<=MAX;n++) {
            const obs = (freq[n]||0)/wins[wi];
            s[n] += ((obs-exp)/(exp+0.001))*ww[wi];
        }
    }
    return s;
}

// --- SIGNAL 8: Conditional Probability ---
function sigCondProb(data) {
    const s = {}; for (let n=1;n<=MAX;n++) s[n]=0;
    if (data.length<30) return s;
    const last = data[data.length-1];
    const cond = {}, tot = {};
    for (let i=0;i<data.length-1;i++) {
        for (const g of data[i]) {
            tot[g]=(tot[g]||0)+1;
            if (!cond[g]) cond[g]={};
            for (const nx of data[i+1]) cond[g][nx]=(cond[g][nx]||0)+1;
        }
    }
    for (let n=1;n<=MAX;n++) {
        let ps = 0;
        for (const g of last) {
            if (tot[g]>0 && cond[g]) ps += (cond[g][n]||0)/tot[g];
        }
        s[n] = ps*3;
    }
    return s;
}

// --- SIGNAL 9: Gap Acceleration ---
function sigGapAccel(data) {
    const s = {}; const n = data.length;
    for (let num=1;num<=MAX;num++) {
        s[num]=0;
        const apps = [];
        data.forEach((d,i)=>{if(d.includes(num))apps.push(i);});
        if (apps.length<4) continue;
        const gaps = [];
        for (let j=0;j<apps.length-1;j++) gaps.push(apps[j+1]-apps[j]);
        if (gaps.length<3) continue;
        const rg = gaps.slice(-5);
        if (rg.length<2) continue;
        const diffs = [];
        for (let i=1;i<rg.length;i++) diffs.push(rg[i]-rg[i-1]);
        const aa = diffs.reduce((a,b)=>a+b,0)/diffs.length;
        const cg = n-apps[apps.length-1];
        const mg = gaps.reduce((a,b)=>a+b,0)/gaps.length;
        const or_ = cg/(mg+0.1);
        if (aa<0 && or_>0.8) s[num]=Math.abs(aa)*or_*2;
        else if (or_>1.5) s[num]=or_*1.5;
    }
    return s;
}

// --- SIGNAL 10: Delta Momentum (2nd derivative) ---
function sigDelta(data) {
    const s = {}; for (let n=1;n<=MAX;n++) s[n]=0;
    if (data.length<30) return s;
    for (let num=1;num<=MAX;num++) {
        const f1=data.slice(-10).filter(d=>d.includes(num)).length/10;
        const f2=data.slice(-20,-10).filter(d=>d.includes(num)).length/10;
        const f3=data.slice(-30,-20).filter(d=>d.includes(num)).length/10;
        const v1=f1-f2, v2=f2-f3, a=v1-v2;
        if (a>0&&v1>0) s[num]=a*15+v1*5;
        else if (a>0) s[num]=a*8;
    }
    return s;
}

// --- SIGNAL 11: Hot-Cold Intersection ---
function sigHotCold(data) {
    const s = {}; for (let n=1;n<=MAX;n++) s[n]=0;
    if (data.length<50) return s;
    const exp = 6/MAX;
    for (let num=1;num<=MAX;num++) {
        const sh=data.slice(-10).filter(d=>d.includes(num)).length/10;
        const m=data.slice(-30,-10).filter(d=>d.includes(num)).length/20;
        const l=data.slice(-80).filter(d=>d.includes(num)).length/80;
        if (sh>exp*1.3&&m<exp*0.7) s[num]=(sh-m)*8;
        else if (sh<exp*0.5&&l>exp*1.2) s[num]=(l-sh)*3;
    }
    return s;
}

// --- SIGNAL 12: Streak-based Sigmoid ---
function sigStreak(data) {
    const s = {}; const eg = MAX/6;
    for (let num=1;num<=MAX;num++) {
        let cold=0;
        for (let i=data.length-1;i>=0;i--) {
            if (!data[i].includes(num)) cold++; else break;
        }
        s[num] = cold>0 ? 1/(1+Math.exp(-3*(cold/eg-0.8)))*2 : 0;
    }
    return s;
}

// ===== WALK-FORWARD CALIBRATION =====
function calibrateSignals(data, signalFuncs) {
    const testSize = Math.min(30, data.length - 70);
    if (testSize < 5) return signalFuncs.map(() => 1.0);
    
    const hits = signalFuncs.map(() => 0);
    let totalWeight = 0;
    
    for (let idx = data.length - testSize; idx < data.length; idx++) {
        const hist = data.slice(0, idx);
        const actual = new Set(data[idx]);
        const recency = Math.exp((idx - (data.length - testSize)) / 6);
        totalWeight += recency;
        
        signalFuncs.forEach((fn, si) => {
            const sig = fn(hist);
            const ranked = Object.entries(sig).sort((a,b)=>b[1]-a[1]).slice(0,10).map(x=>parseInt(x[0]));
            let matchCnt = 0;
            for (const n of ranked) { if (actual.has(n)) matchCnt++; }
            hits[si] += matchCnt * recency;
        });
    }
    
    const baseMatch = 10 * (6/MAX); // top-10 expected matches
    const expected = totalWeight * baseMatch;
    
    return hits.map(h => {
        if (expected > 0 && h > 0) return Math.max(h / expected, 0.1);
        return 0.1;
    });
}

// ===== PREDICTION WITH CALIBRATION =====
function predictCalibrated(data) {
    const signalFuncs = [
        sigRecency, sigGapOverdue, sigMomentum, sigTransition,
        sigAntiRepeat, sigPairBoost, sigSlidingWindow, sigCondProb,
        sigGapAccel, sigDelta, sigHotCold, sigStreak
    ];
    
    // Calibrate every 50 draws to save time
    const weights = calibrateSignals(data, signalFuncs);
    
    const scores = {}; for (let n=1;n<=MAX;n++) scores[n]=0;
    
    signalFuncs.forEach((fn, si) => {
        const sig = fn(data);
        const vals = Object.values(sig);
        const maxS = Math.max(...vals.map(Math.abs));
        if (maxS < 0.001) return;
        for (let n=1;n<=MAX;n++) {
            scores[n] += ((sig[n]||0) / maxS) * weights[si];
        }
    });
    
    return Object.entries(scores).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));
}

// ===== BACKTEST =====
async function main() {
    console.log('='.repeat(70));
    console.log('  TINNAM AI V200.1 — CALIBRATED BACKTEST (12 Signals + Walk-Forward)');
    console.log('='.repeat(70));
    
    console.log('\n[1/3] Fetching data...');
    const data = await fetchData();
    console.log(`  => ${data.length} draws`);
    
    const startIdx = 80; // Need more data for calibration
    const total = data.length;
    const nTest = total - startIdx;
    
    const c6={0:0,1:0,2:0,3:0,4:0,5:0,6:0};
    const c10={0:0,1:0,2:0,3:0,4:0,5:0,6:0};
    const c15={0:0,1:0,2:0,3:0,4:0,5:0,6:0};
    
    console.log(`\n[2/3] Testing ${nTest} draws with walk-forward calibration...\n`);
    const t0 = Date.now();
    
    for (let idx=startIdx; idx<total; idx++) {
        const hist = data.slice(0, idx);
        const actual = new Set(data[idx]);
        
        // Use simple signals for speed (calibration is expensive)
        const pool = predictCalibrated(hist);
        const top6=new Set(pool.slice(0,6));
        const top10=new Set(pool.slice(0,10));
        const top15=new Set(pool.slice(0,15));
        
        let h6=0,h10=0,h15=0;
        for (const n of actual) { if(top6.has(n))h6++; if(top10.has(n))h10++; if(top15.has(n))h15++; }
        c6[h6]++; c10[h10]++; c15[h15]++;
        
        if ((idx-startIdx+1)%100===0) {
            const el=(Date.now()-t0)/1000;
            const eta=(el/(idx-startIdx+1))*(total-idx-1);
            console.log(`  ${idx-startIdx+1}/${nTest} (${((idx-startIdx+1)/nTest*100).toFixed(1)}%) | ${el.toFixed(0)}s | ETA: ${eta.toFixed(0)}s`);
        }
    }
    
    const elapsed = (Date.now()-t0)/1000;
    const pct = (c,t) => t>0?`${(c/t*100).toFixed(1)}%`:'0%';
    
    console.log(`\n${'='.repeat(70)}`);
    console.log(`  RESULTS — ${nTest} draws in ${elapsed.toFixed(1)}s`);
    console.log(`${'='.repeat(70)}`);
    
    console.log(`\n--- TOP-6 ---`);
    for (let k=6;k>=0;k--) console.log(`  ${k}/6: ${String(c6[k]).padStart(5)} (${pct(c6[k],nTest).padStart(6)})`);
    
    const ge3=c6[3]+c6[4]+c6[5]+c6[6], ge4=c6[4]+c6[5]+c6[6];
    console.log(`\n  Top-6 >=3: ${pct(ge3,nTest)} (${ge3}/${nTest})`);
    console.log(`  Top-6 >=4: ${pct(ge4,nTest)} (${ge4}/${nTest})`);
    
    console.log(`\n--- TOP-10 ---`);
    for (let k=6;k>=0;k--) {
        const ab=Object.keys(c10).filter(i=>parseInt(i)>=k).reduce((s,i)=>s+c10[i],0);
        console.log(`  >=${k}: ${String(ab).padStart(5)} (${pct(ab,nTest).padStart(6)})`);
    }
    
    console.log(`\n--- TOP-15 ---`);
    for (let k=6;k>=0;k--) {
        const ab=Object.keys(c15).filter(i=>parseInt(i)>=k).reduce((s,i)=>s+c15[i],0);
        console.log(`  >=${k}: ${String(ab).padStart(5)} (${pct(ab,nTest).padStart(6)})`);
    }
    
    fs.writeFileSync('nexus_backtest_v2.json', JSON.stringify({v:'V200.1',n:nTest,c6,c10,c15,ge3_pct:(ge3/nTest*100).toFixed(2),ge4_pct:(ge4/nTest*100).toFixed(2)},null,2));
    console.log(`\n=> Saved to nexus_backtest_v2.json`);
}

main().catch(console.error);
