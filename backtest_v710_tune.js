/**
 * V710 TUNING BACKTEST — Test nhiều cấu hình trọng số để tìm optimal
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

// ========== ALL MODELS ==========

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
        const mg = gapLists[n]?.length ? gapLists[n].reduce((a,b)=>a+b,0) / gapLists[n].length : data.length;
        scores.push([n, gap / (mg + 0.1)]);
    }
    scores.sort((a,b) => b[1] - a[1]);
    return scores.slice(0, topN).map(x => x[0]);
}

function modelMomentumNeural(data, maxNum) {
    const w = {}; for (let n = 1; n <= maxNum; n++) w[n] = 0;
    const t = data.length;
    for (let i = 0; i < t; i++) { const d = 1/(1+Math.exp(-(i-t+20)/5)); for (const n of data[i]) w[n] += d; }
    return Object.entries(w).sort((a,b) => b[1]-a[1]).slice(0,6).map(x => parseInt(x[0]));
}

function modelMarkov(data) {
    if (data.length < 2) return [];
    const tr = {};
    for (let i = 0; i < data.length-1; i++) { const k = data[i].join(','); if (!tr[k]) tr[k] = {}; for (const n of data[i+1]) tr[k][n] = (tr[k][n]||0)+1; }
    const lk = data[data.length-1].join(',');
    if (tr[lk]) return Object.entries(tr[lk]).sort((a,b) => b[1]-a[1]).slice(0,6).map(x => parseInt(x[0]));
    return modelMomentumNeural(data, 45);
}

function modelKnnMirrorV2(data, maxNum) {
    if (data.length < 15) return modelMomentumNeural(data, maxNum);
    const pat = new Set([...data[data.length-1], ...data[data.length-2], ...data[data.length-3]]);
    const n = data.length, sims = [];
    for (let i = 2; i < n-3; i++) {
        const pp = new Set([...data[i], ...data[i-1], ...data[i-2]]);
        let inter = 0; for (const x of pat) if (pp.has(x)) inter++;
        sims.push([inter * (1+0.3*(i/n)), i+1]);
    }
    sims.sort((a,b) => b[0]-a[0]);
    const v = {};
    for (const [sc, ni] of sims.slice(0,20)) if (sc >= 2.5 && ni < data.length) for (const num of data[ni]) v[num] = (v[num]||0)+sc;
    if (!Object.keys(v).length) return modelMomentumNeural(data, maxNum);
    return Object.entries(v).sort((a,b) => b[1]-a[1]).slice(0,20).map(x => parseInt(x[0]));
}

function modelPairMatrix(data, maxNum) {
    if (data.length < 30) return modelGapOverdue(data, maxNum);
    const n = data.length, ps = {};
    for (let idx = 0; idx < n; idx++) {
        const dc = 0.3+0.7*(idx/n), dr = data[idx].slice(0,6).sort((a,b)=>a-b);
        for (let i = 0; i < dr.length; i++) for (let j = i+1; j < dr.length; j++) { const k = `${dr[i]},${dr[j]}`; ps[k] = (ps[k]||0)+dc; }
    }
    const ld = new Set(data[n-1].slice(0,6)), cs = {};
    for (let num = 1; num <= maxNum; num++) { if (ld.has(num)) continue; cs[num] = 0; for (const a of ld) { const k = Math.min(num,a)+','+Math.max(num,a); cs[num] += ps[k]||0; } }
    for (let idx = Math.max(0,n-100); idx < n; idx++) {
        const dr = data[idx].slice(0,6).sort((a,b)=>a-b);
        for (let i=0;i<dr.length;i++) for (let j=i+1;j<dr.length;j++) for (let k=j+1;k<dr.length;k++) { const ts = new Set([dr[i],dr[j],dr[k]]); let ov=0; for (const x of ts) if (ld.has(x)) ov++; if (ov >= 2) for (const x of ts) if (!ld.has(x)) cs[x] = (cs[x]||0)+1.5; }
    }
    return Object.entries(cs).sort((a,b) => b[1]-a[1]).slice(0,15).map(x => parseInt(x[0]));
}

function modelDeltaMomentum(data, maxNum) {
    if (data.length < 30) return modelMomentumNeural(data, maxNum);
    const sc = {};
    for (let num = 1; num <= maxNum; num++) {
        const f5 = data.slice(-5).filter(d => d.includes(num)).length/5, f5p = data.slice(-10,-5).filter(d => d.includes(num)).length/5;
        const f15 = data.slice(-15).filter(d => d.includes(num)).length/15, f15p = data.slice(-30,-15).filter(d => d.includes(num)).length/15;
        let m = (f5-f5p)*3 + (f15-f15p)*2;
        if (data[data.length-1].includes(num)) m += 0.5;
        if (data.length >= 2 && data[data.length-2].includes(num)) m += 0.3;
        sc[num] = m;
    }
    return Object.entries(sc).sort((a,b) => b[1]-a[1]).slice(0,15).map(x => parseInt(x[0]));
}

function modelMLProxy(data, maxNum) {
    if (data.length < 20) return modelGapOverdue(data, maxNum);
    const f = {}; for (let n = 1; n <= maxNum; n++) f[n] = 0;
    for (const d of data.slice(-10)) for (const n of d) f[n]++;
    return Object.entries(f).sort((a,b) => b[1]-a[1]).slice(0,15).map(x => parseInt(x[0]));
}

function sigRegime(data, maxNum) {
    const sc = {}; if (data.length < 40) return {};
    const exp = 6/maxNum;
    for (let num = 1; num <= maxNum; num++) {
        const f5 = data.slice(-5).filter(d=>d.includes(num)).length/5;
        const f20 = data.slice(-20).filter(d=>d.includes(num)).length/20;
        const f50 = data.slice(-Math.min(50,data.length)).filter(d=>d.includes(num)).length/Math.min(50,data.length);
        if (f5 > exp*1.3 && f20 > exp*1.1) sc[num] = (f5+f20)*4;
        else if (f5 < exp*0.5 && f50 > exp*1.2) sc[num] = (f50-f5)*3;
        else if (f5 > exp*1.5 && f20 < exp*0.8) sc[num] = f5*5;
        else sc[num] = 0;
    }
    return sc;
}

function sigLag(data, maxNum) {
    const sc = {}; if (data.length < 30) return {};
    const lags=[2,3,4,5,7], lw=[3,2.5,2,1.5,1], exp=6/maxNum;
    for (let num = 1; num <= maxNum; num++) {
        sc[num] = 0;
        for (let li=0; li<lags.length; li++) {
            const lag=lags[li]; let cnt=0, tot=0;
            for (let i=lag; i<data.length; i++) { if (data[i-lag].includes(num)) { tot++; if (data[i].includes(num)) cnt++; } }
            if (tot > 5) { const r = cnt/tot; if (r > exp*1.2) for (let lg=1; lg<=lag; lg++) if (data.length-lg >= 0 && data[data.length-lg].includes(num)) { sc[num] += (r-exp)*lw[li]*3; break; } }
        }
    }
    return sc;
}

// ========== CONFIGS TO TEST ==========

function makeEnsemble(config) {
    return function(data, maxNum) {
        const m1=modelMarkov(data), m2=modelGapOverdue(data,maxNum,15), m3=modelMomentumNeural(data,maxNum);
        const m4=modelMLProxy(data,maxNum), m5=modelKnnMirrorV2(data,maxNum), m6=modelPairMatrix(data,maxNum), m7=modelDeltaMomentum(data,maxNum);
        
        const votes = {};
        for (const n of m5.slice(0,15)) votes[n] = (votes[n]||0) + config.w5;
        for (const n of m6.slice(0,15)) votes[n] = (votes[n]||0) + config.w6;
        for (const n of m4.slice(0,15)) votes[n] = (votes[n]||0) + config.w4;
        for (const n of m7.slice(0,15)) votes[n] = (votes[n]||0) + config.w7;
        for (const n of m2.slice(0,15)) votes[n] = (votes[n]||0) + config.w2;
        for (const n of m3.slice(0,6))  votes[n] = (votes[n]||0) + config.w3;
        for (const n of m1.slice(0,6))  votes[n] = (votes[n]||0) + config.w1;
        
        if (config.consensus) {
            const mL = [new Set(m1.slice(0,10)), new Set(m2.slice(0,10)), new Set(m3.slice(0,6)), 
                        new Set(m4.slice(0,10)), new Set(m5.slice(0,10)), new Set(m6.slice(0,10)), new Set(m7.slice(0,10))];
            for (let num = 1; num <= maxNum; num++) {
                const c = mL.filter(ml => ml.has(num)).length;
                if (c >= config.consensusThresh) votes[num] = (votes[num]||0) + c * config.consensusMult;
            }
        }
        
        if (config.regime) {
            const reg = sigRegime(data, maxNum);
            for (const [num, sc] of Object.entries(reg)) if (sc > 0) votes[num] = (votes[num]||0) + Math.min(sc * config.regimeMult, config.regimeCap);
        }
        
        if (config.lag) {
            const lg = sigLag(data, maxNum);
            for (const [num, sc] of Object.entries(lg)) if (sc > 0) votes[num] = (votes[num]||0) + Math.min(sc * config.lagMult, config.lagCap);
        }
        
        return Object.entries(votes).sort((a,b) => b[1]-a[1]).map(x => parseInt(x[0]));
    };
}

const configs = [
    { name: 'V604_baseline', w5:8, w6:6, w4:5, w7:4, w2:3, w3:2, w1:1, consensus:false, regime:false, lag:false },
    { name: 'V710a_consensus_only', w5:10, w6:8, w4:6, w7:5, w2:4, w3:3, w1:2, consensus:true, consensusThresh:4, consensusMult:4, regime:false, lag:false },
    { name: 'V710b_regime+lag', w5:10, w6:8, w4:6, w7:5, w2:4, w3:3, w1:2, consensus:false, regime:true, regimeMult:1.5, regimeCap:5, lag:true, lagMult:1.5, lagCap:4 },
    { name: 'V710c_all_moderate', w5:10, w6:8, w4:6, w7:5, w2:4, w3:3, w1:2, consensus:true, consensusThresh:4, consensusMult:3, regime:true, regimeMult:1, regimeCap:4, lag:true, lagMult:1, lagCap:3 },
    { name: 'V710d_knn_dominant', w5:14, w6:8, w4:5, w7:4, w2:3, w3:2, w1:1, consensus:true, consensusThresh:5, consensusMult:2, regime:true, regimeMult:0.8, regimeCap:3, lag:true, lagMult:0.8, lagCap:3 },
    { name: 'V710e_balanced_high', w5:12, w6:10, w4:7, w7:6, w2:5, w3:3, w1:2, consensus:true, consensusThresh:4, consensusMult:5, regime:true, regimeMult:2, regimeCap:6, lag:true, lagMult:2, lagCap:5 },
];

function runTest(allData, maxNum, fn, label) {
    const startIdx = 60;
    const c = {6:{},10:{},15:{},20:{}};
    for (const p of [6,10,15,20]) for (let k=0; k<=6; k++) c[p][k] = 0;
    let n = 0;
    for (let ci = startIdx; ci < allData.length; ci++) {
        const hist = allData.slice(0, ci), actual = new Set(allData[ci]), ranked = fn(hist, maxNum);
        for (const p of [6,10,15,20]) { const ts = new Set(ranked.slice(0,p)); let h=0; for (const x of actual) if (ts.has(x)) h++; c[p][h]++; }
        n++;
    }
    return { c, n };
}

async function main() {
    console.log('🧬 V710 TUNING — Testing 6 configurations');
    console.log('='.repeat(70));
    const allData = await fetchData();
    console.log(`✅ Loaded ${allData.length} draws\n`);
    
    const results = [];
    for (const cfg of configs) {
        process.stdout.write(`  Testing ${cfg.name}...`);
        const fn = makeEnsemble(cfg);
        const r = runTest(allData, 45, fn, cfg.name);
        results.push({ name: cfg.name, ...r });
        // Key metric: Pool-15 ≥5/6
        const h56 = r.c[15][5]+r.c[15][6];
        const h46 = r.c[15][4]+r.c[15][5]+r.c[15][6];
        const h56_20 = r.c[20][5]+r.c[20][6];
        console.log(` Pool15≥5: ${h56}(${(h56/r.n*100).toFixed(1)}%) | Pool15≥4: ${h46}(${(h46/r.n*100).toFixed(1)}%) | Pool20≥5: ${h56_20}(${(h56_20/r.n*100).toFixed(1)}%)`);
    }
    
    console.log('\n' + '='.repeat(90));
    console.log('📊 FULL COMPARISON TABLE');
    console.log('='.repeat(90));
    console.log('Config'.padEnd(25) + 'P6≥3'.padEnd(10) + 'P10≥4'.padEnd(10) + 'P15≥5'.padEnd(10) + 'P15≥4'.padEnd(10) + 'P20≥5'.padEnd(10) + 'P20≥6'.padEnd(10));
    console.log('-'.repeat(85));
    
    for (const r of results) {
        const p6_3 = r.c[6][3]+r.c[6][4]+r.c[6][5]+r.c[6][6];
        const p10_4 = r.c[10][4]+r.c[10][5]+r.c[10][6];
        const p15_5 = r.c[15][5]+r.c[15][6];
        const p15_4 = r.c[15][4]+r.c[15][5]+r.c[15][6];
        const p20_5 = r.c[20][5]+r.c[20][6];
        const p20_6 = r.c[20][6];
        const pct = v => `${v}(${(v/r.n*100).toFixed(1)}%)`;
        console.log(r.name.padEnd(25) + pct(p6_3).padEnd(10) + pct(p10_4).padEnd(10) + pct(p15_5).padEnd(10) + pct(p15_4).padEnd(10) + pct(p20_5).padEnd(10) + pct(p20_6).padEnd(10));
    }
    
    // Find best config
    let best = null, bestScore = -1;
    for (const r of results) {
        const score = (r.c[15][5]+r.c[15][6])*100 + (r.c[15][4])*10 + (r.c[20][5]+r.c[20][6])*50 + r.c[20][6]*500;
        if (score > bestScore) { bestScore = score; best = r.name; }
    }
    console.log(`\n🏆 BEST CONFIG: ${best} (score: ${bestScore})`);
}

main().catch(console.error);
