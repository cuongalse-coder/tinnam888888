/**
 * POOL-10 SNIPER BACKTEST
 * Goal: Maximize hit rate for 5/6 and 6/6 using exactly 10 numbers.
 * Strategy: Extremely aggressive signal overlap, discarding "broad net" models.
 */
const https = require('https');
function fetchData() { return new Promise((res, rej) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', r => { let d='';r.on('data',c=>d+=c);r.on('end',()=>{const dr=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)dr.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}res(dr);});r.on('error',rej);}); }); }

function mGap(data,mx){const ls={},gl={},li={};for(let i=0;i<data.length;i++)for(const n of data[i]){if(!gl[n])gl[n]=[];if(li[n]!==undefined)gl[n].push(i-li[n]);li[n]=i;ls[n]=i;}const sc=[];for(let n=1;n<=mx;n++){const g=ls[n]!==undefined?data.length-ls[n]:data.length;const mg=gl[n]?.length?gl[n].reduce((a,b)=>a+b,0)/gl[n].length:data.length;sc.push([n,g/(mg+0.1)]);}sc.sort((a,b)=>b[1]-a[1]);return sc.map(x=>x[0]);}
function mMom(data,mx){const w={};for(let n=1;n<=mx;n++)w[n]=0;const t=data.length;for(let i=0;i<t;i++){const d=1/(1+Math.exp(-(i-t+20)/5));for(const n of data[i])w[n]+=d;}return Object.entries(w).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));}
function mMarkov(data,mx){if(data.length<2)return mMom(data,mx);const tr={};for(let i=0;i<data.length-1;i++){const k=data[i].join(',');if(!tr[k])tr[k]={};for(const n of data[i+1])tr[k][n]=(tr[k][n]||0)+1;}const lk=data[data.length-1].join(',');if(tr[lk])return Object.entries(tr[lk]).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));return mMom(data,mx);}
function mKnn(data,mx){if(data.length<20)return mMom(data,mx);const pat=new Set([...data[data.length-1],...data[data.length-2],...data[data.length-3],...(data.length>3?data[data.length-4]:[])]);const n=data.length,sims=[];for(let i=3;i<n-3;i++){const pp=new Set([...data[i],...data[i-1],...data[i-2],...data[i-3]]);let inter=0;for(const x of pat)if(pp.has(x))inter++;const rec=1+0.5*(i/n);if(inter>=5)sims.push([inter*rec,i+1]);}sims.sort((a,b)=>b[0]-a[0]);const v={};for(const[sc,ni]of sims.slice(0,30))if(ni<data.length)for(const num of data[ni])v[num]=(v[num]||0)+sc;if(!Object.keys(v).length)return mMom(data,mx);return Object.entries(v).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));}
function mPair(data,mx){if(data.length<30)return mGap(data,mx);const n=data.length,ps={};for(let idx=0;idx<n;idx++){const dc=0.3+0.7*(idx/n),dr=data[idx].slice(0,6).sort((a,b)=>a-b);for(let i=0;i<dr.length;i++)for(let j=i+1;j<dr.length;j++){const k=`${dr[i]},${dr[j]}`;ps[k]=(ps[k]||0)+dc;}}const ld=new Set(data[n-1].slice(0,6)),cs={};for(let num=1;num<=mx;num++){if(ld.has(num))continue;cs[num]=0;for(const a of ld){const k=Math.min(num,a)+','+Math.max(num,a);cs[num]+=ps[k]||0;}}for(let idx=Math.max(0,n-100);idx<n;idx++){const dr=data[idx].slice(0,6).sort((a,b)=>a-b);for(let i=0;i<dr.length;i++)for(let j=i+1;j<dr.length;j++)for(let k=j+1;k<dr.length;k++){const ts=new Set([dr[i],dr[j],dr[k]]);let ov=0;for(const x of ts)if(ld.has(x))ov++;if(ov>=2)for(const x of ts)if(!ld.has(x))cs[x]=(cs[x]||0)+1.5;}}return Object.entries(cs).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));}
function mCond(data,mx){if(data.length<30)return mMom(data,mx);const last=new Set(data[data.length-1]);const cc={},tg={};for(let i=0;i<data.length-1;i++)for(const g of data[i]){tg[g]=(tg[g]||0)+1;for(const nx of data[i+1]){const k=`${g}-${nx}`;cc[k]=(cc[k]||0)+1;}}const sc={};for(let num=1;num<=mx;num++){sc[num]=0;for(const g of last)if(tg[g]>0)sc[num]+=(cc[`${g}-${num}`]||0)/tg[g];}return Object.entries(sc).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));}
function mML(data,mx){if(data.length<20)return mGap(data,mx);const f={};for(let n=1;n<=mx;n++)f[n]=0;for(const d of data.slice(-10))for(const n of d)f[n]++;return Object.entries(f).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));}

// SNIPER CONFIGURATIONS (Focus ONLY on top 10)

function ensV750_broad(data, mx) { // The current V750A (good for Pool-20, bad for Pool-10)
    const m5=mKnn(data,mx),m6=mPair(data,mx),m8=mCond(data,mx),m4=mML(data,mx),m2=mGap(data,mx);
    const v={};
    for(const n of m5.slice(0,15))v[n]=(v[n]||0)+12;
    for(const n of m6.slice(0,15))v[n]=(v[n]||0)+8;
    for(const n of m8.slice(0,15))v[n]=(v[n]||0)+6;
    for(const n of m4.slice(0,15))v[n]=(v[n]||0)+4;
    for(const n of m2.slice(0,15))v[n]=(v[n]||0)+3;
    const strong=[new Set(m5.slice(0,12)),new Set(m6.slice(0,12)),new Set(m8.slice(0,12))];
    for(let num=1;num<=mx;num++){const c=strong.filter(s=>s.has(num)).length;if(c>=3)v[num]=(v[num]||0)+5;}
    return Object.entries(v).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));
}

// Sniper 1: Exclude Overdue completely, super-heavy on KNN and strict overlap
function ensSniper1(data, mx) {
    const m5=mKnn(data,mx),m6=mPair(data,mx),m8=mCond(data,mx),m1=mMarkov(data,mx);
    const v={};
    for(const n of m5.slice(0,10))v[n]=(v[n]||0)+15; // Only top 10
    for(const n of m6.slice(0,10))v[n]=(v[n]||0)+10;
    for(const n of m8.slice(0,10))v[n]=(v[n]||0)+8;
    for(const n of m1.slice(0,5))v[n]=(v[n]||0)+5;
    // Strict overlap: Only boost if present in at least 2 models' top 10
    const top10s=[new Set(m5.slice(0,10)),new Set(m6.slice(0,10)),new Set(m8.slice(0,10))];
    for(let num=1;num<=mx;num++){const c=top10s.filter(s=>s.has(num)).length;if(c>=2)v[num]=(v[num]||0)+10; if(c>=3)v[num]=(v[num]||0)+20;}
    return Object.entries(v).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));
}

// Sniper 2: ML + Momentum focused
function ensSniper2(data, mx) {
    const m3=mMom(data,mx),m4=mML(data,mx),m5=mKnn(data,mx);
    const v={};
    for(const n of m3.slice(0,10))v[n]=(v[n]||0)+10;
    for(const n of m4.slice(0,10))v[n]=(v[n]||0)+12;
    for(const n of m5.slice(0,10))v[n]=(v[n]||0)+15;
    const top10s=[new Set(m3.slice(0,10)),new Set(m4.slice(0,10)),new Set(m5.slice(0,10))];
    for(let num=1;num<=mx;num++){const c=top10s.filter(s=>s.has(num)).length;if(c>=2)v[num]=(v[num]||0)+15;}
    return Object.entries(v).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));
}

// Sniper 3: Hyper-aggressive Pattern Matching (KNN ONLY + filtering)
function ensSniper3(data, mx) {
    const m5=mKnn(data,mx),m6=mPair(data,mx);
    const v={};
    for(const n of m5.slice(0,8))v[n]=(v[n]||0)+20; // Only take absolute best KNN
    for(const n of m6.slice(0,8))v[n]=(v[n]||0)+10;
    const top8s=[new Set(m5.slice(0,8)),new Set(m6.slice(0,8))];
    for(let num=1;num<=mx;num++){const c=top8s.filter(s=>s.has(num)).length;if(c>=2)v[num]=(v[num]||0)+25;}
    return Object.entries(v).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));
}

function runTest(allData, mx, fn) {
    const c={10:{}}; for(let k=0;k<=6;k++)c[10][k]=0; let n=0;
    for(let ci=60;ci<allData.length;ci++){const hist=allData.slice(0,ci),actual=new Set(allData[ci]),ranked=fn(hist,mx);const ts=new Set(ranked.slice(0,10));let h=0;for(const x of actual)if(ts.has(x))h++;c[10][h]++;n++;}
    return{c,n};
}

async function main() {
    console.log('🎯 POOL-10 SNIPER BACKTEST (Max 10 numbers only)');
    console.log('='.repeat(60));
    const allData = await fetchData();
    console.log(`✅ ${allData.length} draws\n`);
    
    const tests = [
        ['V750_broad (baseline)', ensV750_broad],
        ['Sniper_1_NoOverdue', ensSniper1],
        ['Sniper_2_Mom+ML', ensSniper2],
        ['Sniper_3_HyperKNN', ensSniper3],
    ];
    
    const pct = (v,t) => `${(v/t*100).toFixed(2)}%`;
    console.log('Config'.padEnd(25) + '10≥6'.padEnd(10) + '10≥5'.padEnd(10) + '10≥4'.padEnd(10) + '10≥3'.padEnd(10));
    console.log('-'.repeat(65));
    
    for (const [name, fn] of tests) {
        process.stdout.write(`  ${name}...`);
        const r = runTest(allData, 45, fn);
        const p10_6=r.c[10][6], p10_5=r.c[10][5]+r.c[10][6], p10_4=r.c[10][4]+r.c[10][5]+r.c[10][6], p10_3=r.c[10][3]+r.c[10][4]+r.c[10][5]+r.c[10][6];
        console.log(`\r${name.padEnd(25)}${(p10_6+'('+pct(p10_6,r.n)+')').padEnd(10)}${(p10_5+'('+pct(p10_5,r.n)+')').padEnd(10)}${(p10_4+'('+pct(p10_4,r.n)+')').padEnd(10)}${(p10_3+'('+pct(p10_3,r.n)+')').padEnd(10)}`);
    }
}
main().catch(console.error);
