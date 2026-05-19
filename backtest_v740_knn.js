/**
 * V740 — KNN EVOLUTION: Test multiple KNN variants to find optimal fingerprint
 * The KNN model has the highest weight, so optimizing it yields the most impact
 */
const https = require('https');
function fetchData() { return new Promise((res, rej) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', r => { let d='';r.on('data',c=>d+=c);r.on('end',()=>{const dr=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)dr.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}res(dr);});r.on('error',rej);}); }); }

function mGap(data,mx,topN=15){const ls={},gl={},li={};for(let i=0;i<data.length;i++)for(const n of data[i]){if(!gl[n])gl[n]=[];if(li[n]!==undefined)gl[n].push(i-li[n]);li[n]=i;ls[n]=i;}const sc=[];for(let n=1;n<=mx;n++){const g=ls[n]!==undefined?data.length-ls[n]:data.length;const mg=gl[n]?.length?gl[n].reduce((a,b)=>a+b,0)/gl[n].length:data.length;sc.push([n,g/(mg+0.1)]);}sc.sort((a,b)=>b[1]-a[1]);return sc.slice(0,topN).map(x=>x[0]);}
function mMom(data,mx){const w={};for(let n=1;n<=mx;n++)w[n]=0;const t=data.length;for(let i=0;i<t;i++){const d=1/(1+Math.exp(-(i-t+20)/5));for(const n of data[i])w[n]+=d;}return Object.entries(w).sort((a,b)=>b[1]-a[1]).slice(0,6).map(x=>parseInt(x[0]));}
function mMarkov(data){if(data.length<2)return[];const tr={};for(let i=0;i<data.length-1;i++){const k=data[i].join(',');if(!tr[k])tr[k]={};for(const n of data[i+1])tr[k][n]=(tr[k][n]||0)+1;}const lk=data[data.length-1].join(',');if(tr[lk])return Object.entries(tr[lk]).sort((a,b)=>b[1]-a[1]).slice(0,6).map(x=>parseInt(x[0]));return mMom(data,45);}
function mPair(data,mx){if(data.length<30)return mGap(data,mx);const n=data.length,ps={};for(let idx=0;idx<n;idx++){const dc=0.3+0.7*(idx/n),dr=data[idx].slice(0,6).sort((a,b)=>a-b);for(let i=0;i<dr.length;i++)for(let j=i+1;j<dr.length;j++){const k=`${dr[i]},${dr[j]}`;ps[k]=(ps[k]||0)+dc;}}const ld=new Set(data[n-1].slice(0,6)),cs={};for(let num=1;num<=mx;num++){if(ld.has(num))continue;cs[num]=0;for(const a of ld){const k=Math.min(num,a)+','+Math.max(num,a);cs[num]+=ps[k]||0;}}for(let idx=Math.max(0,n-100);idx<n;idx++){const dr=data[idx].slice(0,6).sort((a,b)=>a-b);for(let i=0;i<dr.length;i++)for(let j=i+1;j<dr.length;j++)for(let k=j+1;k<dr.length;k++){const ts=new Set([dr[i],dr[j],dr[k]]);let ov=0;for(const x of ts)if(ld.has(x))ov++;if(ov>=2)for(const x of ts)if(!ld.has(x))cs[x]=(cs[x]||0)+1.5;}}return Object.entries(cs).sort((a,b)=>b[1]-a[1]).slice(0,15).map(x=>parseInt(x[0]));}
function mDelta(data,mx){if(data.length<30)return mMom(data,mx);const sc={};for(let num=1;num<=mx;num++){const f5=data.slice(-5).filter(d=>d.includes(num)).length/5,f5p=data.slice(-10,-5).filter(d=>d.includes(num)).length/5,f15=data.slice(-15).filter(d=>d.includes(num)).length/15,f15p=data.slice(-30,-15).filter(d=>d.includes(num)).length/15;let m=(f5-f5p)*3+(f15-f15p)*2;if(data[data.length-1].includes(num))m+=0.5;if(data.length>=2&&data[data.length-2].includes(num))m+=0.3;sc[num]=m;}return Object.entries(sc).sort((a,b)=>b[1]-a[1]).slice(0,15).map(x=>parseInt(x[0]));}
function mML(data,mx){if(data.length<20)return mGap(data,mx);const f={};for(let n=1;n<=mx;n++)f[n]=0;for(const d of data.slice(-10))for(const n of d)f[n]++;return Object.entries(f).sort((a,b)=>b[1]-a[1]).slice(0,15).map(x=>parseInt(x[0]));}
function mCond(data,mx){if(data.length<30)return[];const last=new Set(data[data.length-1]);const cc={},tg={};for(let i=0;i<data.length-1;i++)for(const g of data[i]){tg[g]=(tg[g]||0)+1;for(const nx of data[i+1]){const k=`${g}-${nx}`;cc[k]=(cc[k]||0)+1;}}const sc={};for(let num=1;num<=mx;num++){sc[num]=0;for(const g of last)if(tg[g]>0)sc[num]+=(cc[`${g}-${num}`]||0)/tg[g];}return Object.entries(sc).sort((a,b)=>b[1]-a[1]).slice(0,15).map(x=>parseInt(x[0]));}

// === KNN VARIANTS ===

// KNN V3 original (4-draw, thresh 5, 30 neighbors)
function knnV3(data,mx){if(data.length<20)return mMom(data,mx);const pat=new Set([...data[data.length-1],...data[data.length-2],...data[data.length-3],...(data.length>3?data[data.length-4]:[])]);const n=data.length,sims=[];for(let i=3;i<n-3;i++){const pp=new Set([...data[i],...data[i-1],...data[i-2],...data[i-3]]);let inter=0;for(const x of pat)if(pp.has(x))inter++;const rec=1+0.5*(i/n);if(inter>=5)sims.push([inter*rec,i+1]);}sims.sort((a,b)=>b[0]-a[0]);const v={};for(const[sc,ni]of sims.slice(0,30))if(ni<data.length)for(const num of data[ni])v[num]=(v[num]||0)+sc;if(!Object.keys(v).length)return mMom(data,mx);return Object.entries(v).sort((a,b)=>b[1]-a[1]).slice(0,20).map(x=>parseInt(x[0]));}

// KNN V4a: 3-draw fingerprint (original V2 style) but with stronger recency
function knnV4a(data,mx){if(data.length<15)return mMom(data,mx);const pat=new Set([...data[data.length-1],...data[data.length-2],...data[data.length-3]]);const n=data.length,sims=[];for(let i=2;i<n-3;i++){const pp=new Set([...data[i],...data[i-1],...data[i-2]]);let inter=0;for(const x of pat)if(pp.has(x))inter++;const rec=1+0.5*(i/n);if(inter>=4)sims.push([inter*rec,i+1]);}sims.sort((a,b)=>b[0]-a[0]);const v={};for(const[sc,ni]of sims.slice(0,25))if(ni<data.length)for(const num of data[ni])v[num]=(v[num]||0)+sc;if(!Object.keys(v).length)return mMom(data,mx);return Object.entries(v).sort((a,b)=>b[1]-a[1]).slice(0,20).map(x=>parseInt(x[0]));}

// KNN V4b: 5-draw fingerprint (wider pattern)
function knnV4b(data,mx){if(data.length<25)return mMom(data,mx);const pat=new Set([...data[data.length-1],...data[data.length-2],...data[data.length-3],...data[data.length-4],...data[data.length-5]]);const n=data.length,sims=[];for(let i=4;i<n-3;i++){const pp=new Set([...data[i],...data[i-1],...data[i-2],...data[i-3],...data[i-4]]);let inter=0;for(const x of pat)if(pp.has(x))inter++;const rec=1+0.5*(i/n);if(inter>=7)sims.push([inter*rec,i+1]);}sims.sort((a,b)=>b[0]-a[0]);const v={};for(const[sc,ni]of sims.slice(0,30))if(ni<data.length)for(const num of data[ni])v[num]=(v[num]||0)+sc;if(!Object.keys(v).length)return mMom(data,mx);return Object.entries(v).sort((a,b)=>b[1]-a[1]).slice(0,20).map(x=>parseInt(x[0]));}

// KNN V4c: weighted fingerprint (more recent draws = higher weight in matching)
function knnV4c(data,mx){if(data.length<20)return mMom(data,mx);const n=data.length;
// Weighted fingerprint: last draw counts 3x, -2 counts 2x, -3 counts 1.5x, -4 counts 1x
const fpWeights={};for(const num of data[n-1])fpWeights[num]=(fpWeights[num]||0)+3;for(const num of data[n-2])fpWeights[num]=(fpWeights[num]||0)+2;for(const num of data[n-3])fpWeights[num]=(fpWeights[num]||0)+1.5;if(n>3)for(const num of data[n-4])fpWeights[num]=(fpWeights[num]||0)+1;
const sims=[];for(let i=3;i<n-3;i++){const pp={};for(const num of data[i])pp[num]=(pp[num]||0)+3;for(const num of data[i-1])pp[num]=(pp[num]||0)+2;for(const num of data[i-2])pp[num]=(pp[num]||0)+1.5;for(const num of data[i-3])pp[num]=(pp[num]||0)+1;
let simScore=0;for(const[num,w]of Object.entries(fpWeights)){if(pp[num])simScore+=Math.min(w,pp[num]);}const rec=1+0.5*(i/n);if(simScore>=8)sims.push([simScore*rec,i+1]);}
sims.sort((a,b)=>b[0]-a[0]);const v={};for(const[sc,ni]of sims.slice(0,30))if(ni<data.length)for(const num of data[ni])v[num]=(v[num]||0)+sc;if(!Object.keys(v).length)return mMom(data,mx);return Object.entries(v).sort((a,b)=>b[1]-a[1]).slice(0,20).map(x=>parseInt(x[0]));}

// === BUILD ENSEMBLE WITH EACH KNN ===
function makeEns(knnFn) {
    return function(data, mx) {
        const m1=mMarkov(data),m2=mGap(data,mx,15),m3=mMom(data,mx),m4=mML(data,mx),m5=knnFn(data,mx),m6=mPair(data,mx),m7=mDelta(data,mx),m8=mCond(data,mx);
        const v={};
        for(const n of m5.slice(0,15))v[n]=(v[n]||0)+12;for(const n of m6.slice(0,15))v[n]=(v[n]||0)+8;for(const n of m8.slice(0,15))v[n]=(v[n]||0)+6;
        for(const n of m4.slice(0,15))v[n]=(v[n]||0)+5;for(const n of m7.slice(0,15))v[n]=(v[n]||0)+4;for(const n of m2.slice(0,15))v[n]=(v[n]||0)+3;
        for(const n of m3.slice(0,6))v[n]=(v[n]||0)+2;for(const n of m1.slice(0,6))v[n]=(v[n]||0)+1;
        const strong=[new Set(m5.slice(0,12)),new Set(m6.slice(0,12)),new Set(m8.slice(0,12)),new Set(m7.slice(0,12))];
        for(let num=1;num<=mx;num++){const c=strong.filter(s=>s.has(num)).length;if(c>=3)v[num]=(v[num]||0)+c*5;}
        return Object.entries(v).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));
    };
}

function runTest(allData, mx, fn) {
    const c={6:{},10:{},15:{},20:{}}; for(const p of[6,10,15,20])for(let k=0;k<=6;k++)c[p][k]=0; let n=0;
    for(let ci=60;ci<allData.length;ci++){const hist=allData.slice(0,ci),actual=new Set(allData[ci]),ranked=fn(hist,mx);for(const p of[6,10,15,20]){const ts=new Set(ranked.slice(0,p));let h=0;for(const x of actual)if(ts.has(x))h++;c[p][h]++;}n++;}
    return{c,n};
}

async function main() {
    console.log('🧬 V740 — KNN EVOLUTION BACKTEST');
    console.log('='.repeat(75));
    const allData = await fetchData();
    console.log(`✅ ${allData.length} draws\n`);
    
    const tests = [
        ['KNN_V3_4draw_t5', makeEns(knnV3)],
        ['KNN_V4a_3draw_strongR', makeEns(knnV4a)],
        ['KNN_V4b_5draw_wide', makeEns(knnV4b)],
        ['KNN_V4c_weighted_fp', makeEns(knnV4c)],
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
