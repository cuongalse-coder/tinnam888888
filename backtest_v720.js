/**
 * V720 FOCUS — Strategy: maximize Pool-15 ≥5/6 and Pool-20 ≥6/6
 * Approach: KNN super-dominant + eliminate noise from weak models
 */
const https = require('https');
function fetchData() { return new Promise((resolve, reject) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', res => { let d='';res.on('data',c=>d+=c);res.on('end',()=>{const draws=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)draws.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}resolve(draws);});res.on('error',reject);}); }); }

function modelGapOverdue(data,mx,topN=15){const ls={},gl={},li={};for(let i=0;i<data.length;i++)for(const n of data[i]){if(!gl[n])gl[n]=[];if(li[n]!==undefined)gl[n].push(i-li[n]);li[n]=i;ls[n]=i;}const sc=[];for(let n=1;n<=mx;n++){const g=ls[n]!==undefined?data.length-ls[n]:data.length;const mg=gl[n]?.length?gl[n].reduce((a,b)=>a+b,0)/gl[n].length:data.length;sc.push([n,g/(mg+0.1)]);}sc.sort((a,b)=>b[1]-a[1]);return sc.slice(0,topN).map(x=>x[0]);}
function modelMomentum(data,mx){const w={};for(let n=1;n<=mx;n++)w[n]=0;const t=data.length;for(let i=0;i<t;i++){const d=1/(1+Math.exp(-(i-t+20)/5));for(const n of data[i])w[n]+=d;}return Object.entries(w).sort((a,b)=>b[1]-a[1]).slice(0,6).map(x=>parseInt(x[0]));}
function modelMarkov(data){if(data.length<2)return[];const tr={};for(let i=0;i<data.length-1;i++){const k=data[i].join(',');if(!tr[k])tr[k]={};for(const n of data[i+1])tr[k][n]=(tr[k][n]||0)+1;}const lk=data[data.length-1].join(',');if(tr[lk])return Object.entries(tr[lk]).sort((a,b)=>b[1]-a[1]).slice(0,6).map(x=>parseInt(x[0]));return modelMomentum(data,45);}

// KNN Mirror V3 — enhanced with 4-draw fingerprint + stronger recency
function modelKnnV3(data, mx) {
    if (data.length < 20) return modelMomentum(data, mx);
    // 4-draw fingerprint (wider pattern)
    const pat = new Set([...data[data.length-1], ...data[data.length-2], ...data[data.length-3], ...(data.length>3?data[data.length-4]:[])]);
    const n = data.length, sims = [];
    for (let i = 3; i < n - 3; i++) {
        const pp = new Set([...data[i], ...data[i-1], ...data[i-2], ...data[i-3]]);
        let inter = 0; for (const x of pat) if (pp.has(x)) inter++;
        const recency = 1.0 + 0.5 * (i / n); // Stronger recency
        if (inter >= 5) sims.push([inter * recency, i + 1]); // Higher threshold
    }
    sims.sort((a,b) => b[0] - a[0]);
    const v = {};
    for (const [sc, ni] of sims.slice(0, 30)) { // More neighbors
        if (ni < data.length) for (const num of data[ni]) v[num] = (v[num] || 0) + sc;
    }
    if (!Object.keys(v).length) return modelMomentum(data, mx);
    return Object.entries(v).sort((a,b) => b[1] - a[1]).slice(0, 20).map(x => parseInt(x[0]));
}

function modelPair(data,mx){if(data.length<30)return modelGapOverdue(data,mx);const n=data.length,ps={};for(let idx=0;idx<n;idx++){const dc=0.3+0.7*(idx/n),dr=data[idx].slice(0,6).sort((a,b)=>a-b);for(let i=0;i<dr.length;i++)for(let j=i+1;j<dr.length;j++){const k=`${dr[i]},${dr[j]}`;ps[k]=(ps[k]||0)+dc;}}const ld=new Set(data[n-1].slice(0,6)),cs={};for(let num=1;num<=mx;num++){if(ld.has(num))continue;cs[num]=0;for(const a of ld){const k=Math.min(num,a)+','+Math.max(num,a);cs[num]+=ps[k]||0;}}for(let idx=Math.max(0,n-100);idx<n;idx++){const dr=data[idx].slice(0,6).sort((a,b)=>a-b);for(let i=0;i<dr.length;i++)for(let j=i+1;j<dr.length;j++)for(let k=j+1;k<dr.length;k++){const ts=new Set([dr[i],dr[j],dr[k]]);let ov=0;for(const x of ts)if(ld.has(x))ov++;if(ov>=2)for(const x of ts)if(!ld.has(x))cs[x]=(cs[x]||0)+1.5;}}return Object.entries(cs).sort((a,b)=>b[1]-a[1]).slice(0,15).map(x=>parseInt(x[0]));}
function modelDelta(data,mx){if(data.length<30)return modelMomentum(data,mx);const sc={};for(let num=1;num<=mx;num++){const f5=data.slice(-5).filter(d=>d.includes(num)).length/5,f5p=data.slice(-10,-5).filter(d=>d.includes(num)).length/5,f15=data.slice(-15).filter(d=>d.includes(num)).length/15,f15p=data.slice(-30,-15).filter(d=>d.includes(num)).length/15;let m=(f5-f5p)*3+(f15-f15p)*2;if(data[data.length-1].includes(num))m+=0.5;if(data.length>=2&&data[data.length-2].includes(num))m+=0.3;sc[num]=m;}return Object.entries(sc).sort((a,b)=>b[1]-a[1]).slice(0,15).map(x=>parseInt(x[0]));}
function modelML(data,mx){if(data.length<20)return modelGapOverdue(data,mx);const f={};for(let n=1;n<=mx;n++)f[n]=0;for(const d of data.slice(-10))for(const n of d)f[n]++;return Object.entries(f).sort((a,b)=>b[1]-a[1]).slice(0,15).map(x=>parseInt(x[0]));}

// Conditional probability model
function modelCondProb(data, mx) {
    if (data.length < 30) return [];
    const last = new Set(data[data.length-1]);
    const condC = {}, totG = {};
    for (let i = 0; i < data.length-1; i++) {
        for (const g of data[i]) { totG[g] = (totG[g]||0)+1; for (const nx of data[i+1]) { const k = `${g}-${nx}`; condC[k] = (condC[k]||0)+1; } }
    }
    const sc = {};
    for (let num = 1; num <= mx; num++) {
        sc[num] = 0;
        for (const g of last) if (totG[g] > 0) sc[num] += (condC[`${g}-${num}`]||0) / totG[g];
    }
    return Object.entries(sc).sort((a,b) => b[1]-a[1]).slice(0,15).map(x => parseInt(x[0]));
}

// ========== ENSEMBLES ==========
function ensembleV604(data, mx) {
    const m1=modelMarkov(data),m2=modelGapOverdue(data,mx,15),m3=modelMomentum(data,mx),m4=modelML(data,mx),m5=modelKnnV3(data,mx),m6=modelPair(data,mx),m7=modelDelta(data,mx);
    const v={}; for(const n of m5.slice(0,15))v[n]=(v[n]||0)+8;for(const n of m6.slice(0,15))v[n]=(v[n]||0)+6;for(const n of m4.slice(0,15))v[n]=(v[n]||0)+5;for(const n of m7.slice(0,15))v[n]=(v[n]||0)+4;for(const n of m2.slice(0,15))v[n]=(v[n]||0)+3;for(const n of m3.slice(0,6))v[n]=(v[n]||0)+2;for(const n of m1.slice(0,6))v[n]=(v[n]||0)+1;
    return Object.entries(v).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));
}

// V720A: KNN V3 super dominant (16) + CondProb new signal
function ensembleV720A(data, mx) {
    const m1=modelMarkov(data),m2=modelGapOverdue(data,mx,15),m3=modelMomentum(data,mx),m4=modelML(data,mx),m5=modelKnnV3(data,mx),m6=modelPair(data,mx),m7=modelDelta(data,mx),m8=modelCondProb(data,mx);
    const v={};
    for(const n of m5.slice(0,15))v[n]=(v[n]||0)+16; // KNN V3 ultra dominant
    for(const n of m6.slice(0,15))v[n]=(v[n]||0)+8;
    for(const n of m8.slice(0,15))v[n]=(v[n]||0)+7;  // CondProb NEW
    for(const n of m4.slice(0,15))v[n]=(v[n]||0)+5;
    for(const n of m7.slice(0,15))v[n]=(v[n]||0)+4;
    for(const n of m2.slice(0,15))v[n]=(v[n]||0)+3;
    for(const n of m3.slice(0,6))v[n]=(v[n]||0)+2;
    for(const n of m1.slice(0,6))v[n]=(v[n]||0)+1;
    return Object.entries(v).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));
}

// V720B: KNN V3 dominant (12) + agreement filtering (only keep if agreed by ≥3 models)
function ensembleV720B(data, mx) {
    const m1=modelMarkov(data),m2=modelGapOverdue(data,mx,15),m3=modelMomentum(data,mx),m4=modelML(data,mx),m5=modelKnnV3(data,mx),m6=modelPair(data,mx),m7=modelDelta(data,mx),m8=modelCondProb(data,mx);
    const v={};
    for(const n of m5.slice(0,15))v[n]=(v[n]||0)+12;
    for(const n of m6.slice(0,15))v[n]=(v[n]||0)+8;
    for(const n of m8.slice(0,15))v[n]=(v[n]||0)+6;
    for(const n of m4.slice(0,15))v[n]=(v[n]||0)+5;
    for(const n of m7.slice(0,15))v[n]=(v[n]||0)+4;
    for(const n of m2.slice(0,15))v[n]=(v[n]||0)+3;
    for(const n of m3.slice(0,6))v[n]=(v[n]||0)+2;
    for(const n of m1.slice(0,6))v[n]=(v[n]||0)+1;
    // Agreement filter: bonus only if voted by ≥3 strong models
    const strong = [new Set(m5.slice(0,12)),new Set(m6.slice(0,12)),new Set(m8.slice(0,12)),new Set(m7.slice(0,12))];
    for(let num=1;num<=mx;num++){const c=strong.filter(s=>s.has(num)).length;if(c>=3)v[num]=(v[num]||0)+c*5;}
    return Object.entries(v).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));
}

// V720C: Original KNN V2 (not V3) with boosted weights + CondProb
function modelKnnV2(data, mx) {
    if(data.length<15)return modelMomentum(data,mx);
    const pat=new Set([...data[data.length-1],...data[data.length-2],...data[data.length-3]]);
    const n=data.length,sims=[];
    for(let i=2;i<n-3;i++){const pp=new Set([...data[i],...data[i-1],...data[i-2]]);let inter=0;for(const x of pat)if(pp.has(x))inter++;sims.push([inter*(1+0.3*(i/n)),i+1]);}
    sims.sort((a,b)=>b[0]-a[0]);const v={};for(const[sc,ni]of sims.slice(0,20))if(sc>=2.5&&ni<data.length)for(const num of data[ni])v[num]=(v[num]||0)+sc;
    if(!Object.keys(v).length)return modelMomentum(data,mx);return Object.entries(v).sort((a,b)=>b[1]-a[1]).slice(0,20).map(x=>parseInt(x[0]));
}

function ensembleV720C(data, mx) {
    const m1=modelMarkov(data),m2=modelGapOverdue(data,mx,15),m3=modelMomentum(data,mx),m4=modelML(data,mx),m5=modelKnnV2(data,mx),m6=modelPair(data,mx),m7=modelDelta(data,mx),m8=modelCondProb(data,mx);
    const v={};
    for(const n of m5.slice(0,15))v[n]=(v[n]||0)+12;
    for(const n of m6.slice(0,15))v[n]=(v[n]||0)+8;
    for(const n of m8.slice(0,15))v[n]=(v[n]||0)+6;
    for(const n of m4.slice(0,15))v[n]=(v[n]||0)+5;
    for(const n of m7.slice(0,15))v[n]=(v[n]||0)+4;
    for(const n of m2.slice(0,15))v[n]=(v[n]||0)+3;
    for(const n of m3.slice(0,6))v[n]=(v[n]||0)+2;
    for(const n of m1.slice(0,6))v[n]=(v[n]||0)+1;
    return Object.entries(v).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));
}

function runTest(allData, mx, fn) {
    const c={6:{},10:{},15:{},20:{}}; for(const p of[6,10,15,20])for(let k=0;k<=6;k++)c[p][k]=0; let n=0;
    for(let ci=60;ci<allData.length;ci++){const hist=allData.slice(0,ci),actual=new Set(allData[ci]),ranked=fn(hist,mx);for(const p of[6,10,15,20]){const ts=new Set(ranked.slice(0,p));let h=0;for(const x of actual)if(ts.has(x))h++;c[p][h]++;}n++;}
    return{c,n};
}

async function main() {
    console.log('🧬 V720 FOCUS BACKTEST — Maximize ≥5/6 and 6/6');
    console.log('='.repeat(60));
    const allData = await fetchData();
    console.log(`✅ ${allData.length} draws\n`);
    
    const tests = [
        ['V604_base', ensembleV604],
        ['V720A_knn_ultra', ensembleV720A],
        ['V720B_agree_filt', ensembleV720B],
        ['V720C_knnV2+cond', ensembleV720C],
    ];
    
    const pct = (v,t) => `${(v/t*100).toFixed(1)}%`;
    console.log('Config'.padEnd(22) + 'P15≥5'.padEnd(12) + 'P15≥4'.padEnd(12) + 'P20≥6'.padEnd(12) + 'P20≥5'.padEnd(12) + 'P10≥4'.padEnd(12));
    console.log('-'.repeat(70));
    
    for (const [name, fn] of tests) {
        process.stdout.write(`  ${name}...`);
        const r = runTest(allData, 45, fn);
        const p15_5=r.c[15][5]+r.c[15][6], p15_4=r.c[15][4]+r.c[15][5]+r.c[15][6], p20_6=r.c[20][6], p20_5=r.c[20][5]+r.c[20][6], p10_4=r.c[10][4]+r.c[10][5]+r.c[10][6];
        console.log(`\r${name.padEnd(22)}${(p15_5+'('+pct(p15_5,r.n)+')').padEnd(12)}${(p15_4+'('+pct(p15_4,r.n)+')').padEnd(12)}${(p20_6+'('+pct(p20_6,r.n)+')').padEnd(12)}${(p20_5+'('+pct(p20_5,r.n)+')').padEnd(12)}${(p10_4+'('+pct(p10_4,r.n)+')').padEnd(12)}`);
    }
}
main().catch(console.error);
