/**
 * V730 — MAXIMIZE 6/6 JACKPOT RATE
 * Focus: get all 6 winning numbers into pool-20
 * New strategies: Adaptive KNN, Cluster Boost, Rolling Model Selection, Freq-Gap Hybrid
 */
const https = require('https');
function fetchData() { return new Promise((res, rej) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', r => { let d='';r.on('data',c=>d+=c);r.on('end',()=>{const dr=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)dr.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}res(dr);});r.on('error',rej);}); }); }

// === BASE MODELS ===
function mGap(data,mx,topN=15){const ls={},gl={},li={};for(let i=0;i<data.length;i++)for(const n of data[i]){if(!gl[n])gl[n]=[];if(li[n]!==undefined)gl[n].push(i-li[n]);li[n]=i;ls[n]=i;}const sc=[];for(let n=1;n<=mx;n++){const g=ls[n]!==undefined?data.length-ls[n]:data.length;const mg=gl[n]?.length?gl[n].reduce((a,b)=>a+b,0)/gl[n].length:data.length;sc.push([n,g/(mg+0.1)]);}sc.sort((a,b)=>b[1]-a[1]);return sc.slice(0,topN).map(x=>x[0]);}
function mMom(data,mx){const w={};for(let n=1;n<=mx;n++)w[n]=0;const t=data.length;for(let i=0;i<t;i++){const d=1/(1+Math.exp(-(i-t+20)/5));for(const n of data[i])w[n]+=d;}return Object.entries(w).sort((a,b)=>b[1]-a[1]).slice(0,6).map(x=>parseInt(x[0]));}
function mMarkov(data){if(data.length<2)return[];const tr={};for(let i=0;i<data.length-1;i++){const k=data[i].join(',');if(!tr[k])tr[k]={};for(const n of data[i+1])tr[k][n]=(tr[k][n]||0)+1;}const lk=data[data.length-1].join(',');if(tr[lk])return Object.entries(tr[lk]).sort((a,b)=>b[1]-a[1]).slice(0,6).map(x=>parseInt(x[0]));return mMom(data,45);}
function mKnnV3(data,mx){if(data.length<20)return mMom(data,mx);const pat=new Set([...data[data.length-1],...data[data.length-2],...data[data.length-3],...(data.length>3?data[data.length-4]:[])]);const n=data.length,sims=[];for(let i=3;i<n-3;i++){const pp=new Set([...data[i],...data[i-1],...data[i-2],...data[i-3]]);let inter=0;for(const x of pat)if(pp.has(x))inter++;const rec=1+0.5*(i/n);if(inter>=5)sims.push([inter*rec,i+1]);}sims.sort((a,b)=>b[0]-a[0]);const v={};for(const[sc,ni]of sims.slice(0,30))if(ni<data.length)for(const num of data[ni])v[num]=(v[num]||0)+sc;if(!Object.keys(v).length)return mMom(data,mx);return Object.entries(v).sort((a,b)=>b[1]-a[1]).slice(0,20).map(x=>parseInt(x[0]));}
function mPair(data,mx){if(data.length<30)return mGap(data,mx);const n=data.length,ps={};for(let idx=0;idx<n;idx++){const dc=0.3+0.7*(idx/n),dr=data[idx].slice(0,6).sort((a,b)=>a-b);for(let i=0;i<dr.length;i++)for(let j=i+1;j<dr.length;j++){const k=`${dr[i]},${dr[j]}`;ps[k]=(ps[k]||0)+dc;}}const ld=new Set(data[n-1].slice(0,6)),cs={};for(let num=1;num<=mx;num++){if(ld.has(num))continue;cs[num]=0;for(const a of ld){const k=Math.min(num,a)+','+Math.max(num,a);cs[num]+=ps[k]||0;}}for(let idx=Math.max(0,n-100);idx<n;idx++){const dr=data[idx].slice(0,6).sort((a,b)=>a-b);for(let i=0;i<dr.length;i++)for(let j=i+1;j<dr.length;j++)for(let k=j+1;k<dr.length;k++){const ts=new Set([dr[i],dr[j],dr[k]]);let ov=0;for(const x of ts)if(ld.has(x))ov++;if(ov>=2)for(const x of ts)if(!ld.has(x))cs[x]=(cs[x]||0)+1.5;}}return Object.entries(cs).sort((a,b)=>b[1]-a[1]).slice(0,15).map(x=>parseInt(x[0]));}
function mDelta(data,mx){if(data.length<30)return mMom(data,mx);const sc={};for(let num=1;num<=mx;num++){const f5=data.slice(-5).filter(d=>d.includes(num)).length/5,f5p=data.slice(-10,-5).filter(d=>d.includes(num)).length/5,f15=data.slice(-15).filter(d=>d.includes(num)).length/15,f15p=data.slice(-30,-15).filter(d=>d.includes(num)).length/15;let m=(f5-f5p)*3+(f15-f15p)*2;if(data[data.length-1].includes(num))m+=0.5;if(data.length>=2&&data[data.length-2].includes(num))m+=0.3;sc[num]=m;}return Object.entries(sc).sort((a,b)=>b[1]-a[1]).slice(0,15).map(x=>parseInt(x[0]));}
function mML(data,mx){if(data.length<20)return mGap(data,mx);const f={};for(let n=1;n<=mx;n++)f[n]=0;for(const d of data.slice(-10))for(const n of d)f[n]++;return Object.entries(f).sort((a,b)=>b[1]-a[1]).slice(0,15).map(x=>parseInt(x[0]));}
function mCond(data,mx){if(data.length<30)return[];const last=new Set(data[data.length-1]);const cc={},tg={};for(let i=0;i<data.length-1;i++)for(const g of data[i]){tg[g]=(tg[g]||0)+1;for(const nx of data[i+1]){const k=`${g}-${nx}`;cc[k]=(cc[k]||0)+1;}}const sc={};for(let num=1;num<=mx;num++){sc[num]=0;for(const g of last)if(tg[g]>0)sc[num]+=(cc[`${g}-${num}`]||0)/tg[g];}return Object.entries(sc).sort((a,b)=>b[1]-a[1]).slice(0,15).map(x=>parseInt(x[0]));}

// === NEW V730 MODELS ===

// 1. Freq-Gap Hybrid: multiplicative score (hot + overdue = explosive)
function mFreqGap(data, mx) {
    if (data.length < 30) return mGap(data, mx);
    const sc = {};
    for (let num = 1; num <= mx; num++) {
        // Frequency component (multi-window)
        const f5 = data.slice(-5).filter(d => d.includes(num)).length / 5;
        const f15 = data.slice(-15).filter(d => d.includes(num)).length / 15;
        const expected = 6 / mx;
        const freqScore = (f5 / (expected + 0.01)) * 0.6 + (f15 / (expected + 0.01)) * 0.4;
        
        // Gap component
        let lastSeen = -1;
        for (let i = data.length - 1; i >= 0; i--) { if (data[i].includes(num)) { lastSeen = i; break; } }
        const gap = lastSeen >= 0 ? data.length - lastSeen : data.length;
        const appearances = [];
        for (let i = 0; i < data.length; i++) if (data[i].includes(num)) appearances.push(i);
        let meanGap = mx / 6;
        if (appearances.length >= 2) {
            const gaps = []; for (let j = 1; j < appearances.length; j++) gaps.push(appearances[j] - appearances[j-1]);
            meanGap = gaps.reduce((a,b) => a+b, 0) / gaps.length;
        }
        const overdueRatio = gap / (meanGap + 0.1);
        
        // Hybrid: numbers that are BOTH frequent AND overdue are explosive
        if (freqScore > 0.8 && overdueRatio > 0.7) sc[num] = freqScore * overdueRatio * 3;
        else if (overdueRatio > 1.5) sc[num] = overdueRatio * 1.5;
        else if (freqScore > 1.3) sc[num] = freqScore * 2;
        else sc[num] = freqScore * 0.5 + overdueRatio * 0.5;
    }
    return Object.entries(sc).sort((a,b) => b[1] - a[1]).slice(0, 15).map(x => parseInt(x[0]));
}

// 2. Cluster Boost: find groups of numbers that appear together frequently
function mCluster(data, mx) {
    if (data.length < 50) return [];
    const n = data.length;
    const last = data[n-1];
    // For each number, find its "best friends" (most co-occurring numbers)
    const coOccur = {};
    for (let num = 1; num <= mx; num++) coOccur[num] = {};
    for (let idx = Math.max(0, n-200); idx < n; idx++) {
        const w = 1 + (idx - Math.max(0, n-200)) / 200;
        const d = data[idx];
        for (let i = 0; i < d.length; i++) for (let j = i+1; j < d.length; j++) {
            coOccur[d[i]][d[j]] = (coOccur[d[i]][d[j]] || 0) + w;
            coOccur[d[j]][d[i]] = (coOccur[d[j]][d[i]] || 0) + w;
        }
    }
    // Score: for each candidate, sum co-occurrence with ALL numbers in last draw
    const sc = {};
    for (let num = 1; num <= mx; num++) {
        sc[num] = 0;
        for (const anchor of last) {
            sc[num] += coOccur[num][anchor] || 0;
        }
        // Bonus: if num has strong co-occurrence with 3+ of last draw numbers
        let strongLinks = 0;
        for (const anchor of last) if ((coOccur[num][anchor] || 0) > 3) strongLinks++;
        if (strongLinks >= 3) sc[num] *= 1.5;
    }
    return Object.entries(sc).sort((a,b) => b[1] - a[1]).slice(0, 15).map(x => parseInt(x[0]));
}

// 3. Rolling Performance Selector: track recent accuracy of each model and weight accordingly
function mRollingSelect(data, mx, modelFns) {
    if (data.length < 70) return {};
    // Test each model on last 20 draws to get performance weights
    const perfScores = {};
    for (const [name, fn] of Object.entries(modelFns)) perfScores[name] = 0;
    
    for (let testIdx = data.length - 20; testIdx < data.length; testIdx++) {
        const hist = data.slice(0, testIdx);
        const actual = new Set(data[testIdx]);
        for (const [name, fn] of Object.entries(modelFns)) {
            try {
                const pred = fn(hist, mx);
                const top15 = new Set(pred.slice(0, 15));
                let hits = 0; for (const n of actual) if (top15.has(n)) hits++;
                perfScores[name] += hits;
            } catch(e) {}
        }
    }
    
    // Normalize to weights
    const maxPerf = Math.max(...Object.values(perfScores), 1);
    const weights = {};
    for (const [name, score] of Object.entries(perfScores)) {
        weights[name] = Math.max(0.5, score / maxPerf * 3);
    }
    return weights;
}

// === ENSEMBLES ===
function ensV720B(data, mx) {
    const m1=mMarkov(data),m2=mGap(data,mx,15),m3=mMom(data,mx),m4=mML(data,mx),m5=mKnnV3(data,mx),m6=mPair(data,mx),m7=mDelta(data,mx),m8=mCond(data,mx);
    const v={};
    for(const n of m5.slice(0,15))v[n]=(v[n]||0)+12;for(const n of m6.slice(0,15))v[n]=(v[n]||0)+8;for(const n of m8.slice(0,15))v[n]=(v[n]||0)+6;
    for(const n of m4.slice(0,15))v[n]=(v[n]||0)+5;for(const n of m7.slice(0,15))v[n]=(v[n]||0)+4;for(const n of m2.slice(0,15))v[n]=(v[n]||0)+3;
    for(const n of m3.slice(0,6))v[n]=(v[n]||0)+2;for(const n of m1.slice(0,6))v[n]=(v[n]||0)+1;
    const strong=[new Set(m5.slice(0,12)),new Set(m6.slice(0,12)),new Set(m8.slice(0,12)),new Set(m7.slice(0,12))];
    for(let num=1;num<=mx;num++){const c=strong.filter(s=>s.has(num)).length;if(c>=3)v[num]=(v[num]||0)+c*5;}
    return Object.entries(v).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));
}

// V730A: Add FreqGap + Cluster to V720B
function ensV730A(data, mx) {
    const m1=mMarkov(data),m2=mGap(data,mx,15),m3=mMom(data,mx),m4=mML(data,mx),m5=mKnnV3(data,mx),m6=mPair(data,mx),m7=mDelta(data,mx),m8=mCond(data,mx);
    const m9=mFreqGap(data,mx),m10=mCluster(data,mx);
    const v={};
    for(const n of m5.slice(0,15))v[n]=(v[n]||0)+12;for(const n of m6.slice(0,15))v[n]=(v[n]||0)+8;for(const n of m8.slice(0,15))v[n]=(v[n]||0)+6;
    for(const n of m9.slice(0,15))v[n]=(v[n]||0)+5;for(const n of m10.slice(0,15))v[n]=(v[n]||0)+5;
    for(const n of m4.slice(0,15))v[n]=(v[n]||0)+4;for(const n of m7.slice(0,15))v[n]=(v[n]||0)+4;for(const n of m2.slice(0,15))v[n]=(v[n]||0)+3;
    for(const n of m3.slice(0,6))v[n]=(v[n]||0)+2;for(const n of m1.slice(0,6))v[n]=(v[n]||0)+1;
    const strong=[new Set(m5.slice(0,12)),new Set(m6.slice(0,12)),new Set(m8.slice(0,12)),new Set(m9.slice(0,12)),new Set(m10.slice(0,12))];
    for(let num=1;num<=mx;num++){const c=strong.filter(s=>s.has(num)).length;if(c>=3)v[num]=(v[num]||0)+c*4;}
    return Object.entries(v).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));
}

// V730B: Rolling adaptive weights
function ensV730B(data, mx) {
    const modelFns = {knn:mKnnV3, pair:mPair, cond:mCond, delta:mDelta, gap:mGap, freqgap:mFreqGap, cluster:mCluster};
    const m1=mMarkov(data),m2=mGap(data,mx,15),m3=mMom(data,mx),m4=mML(data,mx),m5=mKnnV3(data,mx),m6=mPair(data,mx),m7=mDelta(data,mx),m8=mCond(data,mx),m9=mFreqGap(data,mx),m10=mCluster(data,mx);
    
    // Get adaptive weights from recent performance
    const perf = mRollingSelect(data, mx, modelFns);
    const w5 = (perf.knn || 1) * 4;
    const w6 = (perf.pair || 1) * 3;
    const w8 = (perf.cond || 1) * 2.5;
    const w9 = (perf.freqgap || 1) * 2;
    const w10 = (perf.cluster || 1) * 2;
    const w7 = (perf.delta || 1) * 1.5;
    const w2 = (perf.gap || 1) * 1.5;
    
    const v={};
    for(const n of m5.slice(0,15))v[n]=(v[n]||0)+w5;for(const n of m6.slice(0,15))v[n]=(v[n]||0)+w6;for(const n of m8.slice(0,15))v[n]=(v[n]||0)+w8;
    for(const n of m9.slice(0,15))v[n]=(v[n]||0)+w9;for(const n of m10.slice(0,15))v[n]=(v[n]||0)+w10;
    for(const n of m4.slice(0,15))v[n]=(v[n]||0)+3;for(const n of m7.slice(0,15))v[n]=(v[n]||0)+w7;for(const n of m2.slice(0,15))v[n]=(v[n]||0)+w2;
    for(const n of m3.slice(0,6))v[n]=(v[n]||0)+2;for(const n of m1.slice(0,6))v[n]=(v[n]||0)+1;
    return Object.entries(v).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));
}

// V730C: V720B + FreqGap only (simpler, less noise)
function ensV730C(data, mx) {
    const m1=mMarkov(data),m2=mGap(data,mx,15),m3=mMom(data,mx),m4=mML(data,mx),m5=mKnnV3(data,mx),m6=mPair(data,mx),m7=mDelta(data,mx),m8=mCond(data,mx),m9=mFreqGap(data,mx);
    const v={};
    for(const n of m5.slice(0,15))v[n]=(v[n]||0)+12;for(const n of m6.slice(0,15))v[n]=(v[n]||0)+8;for(const n of m8.slice(0,15))v[n]=(v[n]||0)+6;
    for(const n of m9.slice(0,15))v[n]=(v[n]||0)+6;  // FreqGap same weight as CondProb
    for(const n of m4.slice(0,15))v[n]=(v[n]||0)+5;for(const n of m7.slice(0,15))v[n]=(v[n]||0)+4;for(const n of m2.slice(0,15))v[n]=(v[n]||0)+3;
    for(const n of m3.slice(0,6))v[n]=(v[n]||0)+2;for(const n of m1.slice(0,6))v[n]=(v[n]||0)+1;
    const strong=[new Set(m5.slice(0,12)),new Set(m6.slice(0,12)),new Set(m8.slice(0,12)),new Set(m9.slice(0,12))];
    for(let num=1;num<=mx;num++){const c=strong.filter(s=>s.has(num)).length;if(c>=3)v[num]=(v[num]||0)+c*5;}
    return Object.entries(v).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));
}

function runTest(allData, mx, fn) {
    const c={6:{},10:{},15:{},20:{}}; for(const p of[6,10,15,20])for(let k=0;k<=6;k++)c[p][k]=0; let n=0;
    for(let ci=60;ci<allData.length;ci++){const hist=allData.slice(0,ci),actual=new Set(allData[ci]),ranked=fn(hist,mx);for(const p of[6,10,15,20]){const ts=new Set(ranked.slice(0,p));let h=0;for(const x of actual)if(ts.has(x))h++;c[p][h]++;}n++;}
    return{c,n};
}

async function main() {
    console.log('🧬 V730 JACKPOT MAXIMIZER — Testing 4 configs');
    console.log('='.repeat(70));
    const allData = await fetchData();
    console.log(`✅ ${allData.length} draws\n`);
    
    const tests = [
        ['V720B_current', ensV720B],
        ['V730A_+freqgap+cluster', ensV730A],
        ['V730B_rolling_adaptive', ensV730B],
        ['V730C_+freqgap_only', ensV730C],
    ];
    
    const pct = (v,t) => `${(v/t*100).toFixed(1)}%`;
    console.log('Config'.padEnd(25) + 'P20≥6'.padEnd(10) + 'P20≥5'.padEnd(12) + 'P15≥5'.padEnd(12) + 'P15≥4'.padEnd(12) + 'P10≥4'.padEnd(12));
    console.log('-'.repeat(75));
    
    for (const [name, fn] of tests) {
        process.stdout.write(`  ${name}...`);
        const r = runTest(allData, 45, fn);
        const p20_6=r.c[20][6], p20_5=r.c[20][5]+r.c[20][6], p15_5=r.c[15][5]+r.c[15][6], p15_4=r.c[15][4]+r.c[15][5]+r.c[15][6], p10_4=r.c[10][4]+r.c[10][5]+r.c[10][6];
        console.log(`\r${name.padEnd(25)}${(p20_6+'('+pct(p20_6,r.n)+')').padEnd(10)}${(p20_5+'('+pct(p20_5,r.n)+')').padEnd(12)}${(p15_5+'('+pct(p15_5,r.n)+')').padEnd(12)}${(p15_4+'('+pct(p15_4,r.n)+')').padEnd(12)}${(p10_4+'('+pct(p10_4,r.n)+')').padEnd(12)}`);
    }
    console.log('='.repeat(75));
}
main().catch(console.error);
