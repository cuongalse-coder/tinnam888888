/**
 * V710 FINAL — Optimized hybrid config based on grid search results
 * Best of: consensus_only (P20≥6=10) + regime+lag (P20≥5=66) 
 */
const https = require('https');
function fetchData() {
    return new Promise((resolve, reject) => {
        https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', res => {
            let d = ''; res.on('data', c => d += c);
            res.on('end', () => { const draws = []; for (const l of d.trim().split('\n')) { if (!l) continue; const o = JSON.parse(l); if (o.result?.length >= 6) draws.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b)); } resolve(draws); });
            res.on('error', reject);
        });
    });
}
function modelGapOverdue(data, mx, topN=15) { const ls={},gl={},li={}; for (let i=0;i<data.length;i++) for (const n of data[i]) { if(!gl[n])gl[n]=[]; if(li[n]!==undefined)gl[n].push(i-li[n]); li[n]=i;ls[n]=i; } const sc=[]; for (let n=1;n<=mx;n++) { const g=ls[n]!==undefined?data.length-ls[n]:data.length; const mg=gl[n]?.length?gl[n].reduce((a,b)=>a+b,0)/gl[n].length:data.length; sc.push([n,g/(mg+0.1)]); } sc.sort((a,b)=>b[1]-a[1]); return sc.slice(0,topN).map(x=>x[0]); }
function modelMomentum(data, mx) { const w={}; for(let n=1;n<=mx;n++)w[n]=0; const t=data.length; for(let i=0;i<t;i++){const d=1/(1+Math.exp(-(i-t+20)/5));for(const n of data[i])w[n]+=d;} return Object.entries(w).sort((a,b)=>b[1]-a[1]).slice(0,6).map(x=>parseInt(x[0])); }
function modelMarkov(data) { if(data.length<2)return[]; const tr={}; for(let i=0;i<data.length-1;i++){const k=data[i].join(',');if(!tr[k])tr[k]={};for(const n of data[i+1])tr[k][n]=(tr[k][n]||0)+1;} const lk=data[data.length-1].join(','); if(tr[lk])return Object.entries(tr[lk]).sort((a,b)=>b[1]-a[1]).slice(0,6).map(x=>parseInt(x[0])); return modelMomentum(data,45); }
function modelKnn(data, mx) { if(data.length<15)return modelMomentum(data,mx); const pat=new Set([...data[data.length-1],...data[data.length-2],...data[data.length-3]]); const n=data.length,sims=[]; for(let i=2;i<n-3;i++){const pp=new Set([...data[i],...data[i-1],...data[i-2]]);let inter=0;for(const x of pat)if(pp.has(x))inter++;sims.push([inter*(1+0.3*(i/n)),i+1]);} sims.sort((a,b)=>b[0]-a[0]); const v={}; for(const[sc,ni]of sims.slice(0,20))if(sc>=2.5&&ni<data.length)for(const num of data[ni])v[num]=(v[num]||0)+sc; if(!Object.keys(v).length)return modelMomentum(data,mx); return Object.entries(v).sort((a,b)=>b[1]-a[1]).slice(0,20).map(x=>parseInt(x[0])); }
function modelPair(data, mx) { if(data.length<30)return modelGapOverdue(data,mx); const n=data.length,ps={}; for(let idx=0;idx<n;idx++){const dc=0.3+0.7*(idx/n),dr=data[idx].slice(0,6).sort((a,b)=>a-b);for(let i=0;i<dr.length;i++)for(let j=i+1;j<dr.length;j++){const k=`${dr[i]},${dr[j]}`;ps[k]=(ps[k]||0)+dc;}} const ld=new Set(data[n-1].slice(0,6)),cs={}; for(let num=1;num<=mx;num++){if(ld.has(num))continue;cs[num]=0;for(const a of ld){const k=Math.min(num,a)+','+Math.max(num,a);cs[num]+=ps[k]||0;}} for(let idx=Math.max(0,n-100);idx<n;idx++){const dr=data[idx].slice(0,6).sort((a,b)=>a-b);for(let i=0;i<dr.length;i++)for(let j=i+1;j<dr.length;j++)for(let k=j+1;k<dr.length;k++){const ts=new Set([dr[i],dr[j],dr[k]]);let ov=0;for(const x of ts)if(ld.has(x))ov++;if(ov>=2)for(const x of ts)if(!ld.has(x))cs[x]=(cs[x]||0)+1.5;}} return Object.entries(cs).sort((a,b)=>b[1]-a[1]).slice(0,15).map(x=>parseInt(x[0])); }
function modelDelta(data, mx) { if(data.length<30)return modelMomentum(data,mx); const sc={}; for(let num=1;num<=mx;num++){const f5=data.slice(-5).filter(d=>d.includes(num)).length/5,f5p=data.slice(-10,-5).filter(d=>d.includes(num)).length/5,f15=data.slice(-15).filter(d=>d.includes(num)).length/15,f15p=data.slice(-30,-15).filter(d=>d.includes(num)).length/15;let m=(f5-f5p)*3+(f15-f15p)*2;if(data[data.length-1].includes(num))m+=0.5;if(data.length>=2&&data[data.length-2].includes(num))m+=0.3;sc[num]=m;} return Object.entries(sc).sort((a,b)=>b[1]-a[1]).slice(0,15).map(x=>parseInt(x[0])); }
function modelML(data, mx) { if(data.length<20)return modelGapOverdue(data,mx); const f={}; for(let n=1;n<=mx;n++)f[n]=0; for(const d of data.slice(-10))for(const n of d)f[n]++; return Object.entries(f).sort((a,b)=>b[1]-a[1]).slice(0,15).map(x=>parseInt(x[0])); }
function sigRegime(data,mx){const sc={};if(data.length<40)return{};const exp=6/mx;for(let num=1;num<=mx;num++){const f5=data.slice(-5).filter(d=>d.includes(num)).length/5,f20=data.slice(-20).filter(d=>d.includes(num)).length/20,f50=data.slice(-Math.min(50,data.length)).filter(d=>d.includes(num)).length/Math.min(50,data.length);if(f5>exp*1.3&&f20>exp*1.1)sc[num]=(f5+f20)*4;else if(f5<exp*0.5&&f50>exp*1.2)sc[num]=(f50-f5)*3;else if(f5>exp*1.5&&f20<exp*0.8)sc[num]=f5*5;else sc[num]=0;}return sc;}
function sigLag(data,mx){const sc={};if(data.length<30)return{};const lags=[2,3,4,5,7],lw=[3,2.5,2,1.5,1],exp=6/mx;for(let num=1;num<=mx;num++){sc[num]=0;for(let li=0;li<lags.length;li++){const lag=lags[li];let cnt=0,tot=0;for(let i=lag;i<data.length;i++){if(data[i-lag].includes(num)){tot++;if(data[i].includes(num))cnt++;}}if(tot>5){const r=cnt/tot;if(r>exp*1.2)for(let lg=1;lg<=lag;lg++)if(data.length-lg>=0&&data[data.length-lg].includes(num)){sc[num]+=(r-exp)*lw[li]*3;break;}}}}return sc;}

// ========== FINAL CONFIGS ==========

function ensembleV604(data, mx) {
    const m1=modelMarkov(data),m2=modelGapOverdue(data,mx,15),m3=modelMomentum(data,mx),m4=modelML(data,mx),m5=modelKnn(data,mx),m6=modelPair(data,mx),m7=modelDelta(data,mx);
    const v={}; for(const n of m5.slice(0,15))v[n]=(v[n]||0)+8;for(const n of m6.slice(0,15))v[n]=(v[n]||0)+6;for(const n of m4.slice(0,15))v[n]=(v[n]||0)+5;for(const n of m7.slice(0,15))v[n]=(v[n]||0)+4;for(const n of m2.slice(0,15))v[n]=(v[n]||0)+3;for(const n of m3.slice(0,6))v[n]=(v[n]||0)+2;for(const n of m1.slice(0,6))v[n]=(v[n]||0)+1;
    return Object.entries(v).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));
}

// V710 FINAL: consensus(thresh=4,mult=4) + light regime(1,3) + light lag(1,3)
function ensembleV710Final(data, mx) {
    const m1=modelMarkov(data),m2=modelGapOverdue(data,mx,15),m3=modelMomentum(data,mx),m4=modelML(data,mx),m5=modelKnn(data,mx),m6=modelPair(data,mx),m7=modelDelta(data,mx);
    const v={};
    for(const n of m5.slice(0,15))v[n]=(v[n]||0)+10;
    for(const n of m6.slice(0,15))v[n]=(v[n]||0)+8;
    for(const n of m4.slice(0,15))v[n]=(v[n]||0)+6;
    for(const n of m7.slice(0,15))v[n]=(v[n]||0)+5;
    for(const n of m2.slice(0,15))v[n]=(v[n]||0)+4;
    for(const n of m3.slice(0,6))v[n]=(v[n]||0)+3;
    for(const n of m1.slice(0,6))v[n]=(v[n]||0)+2;
    // Consensus bonus (thresh=4, mult=4)
    const mL=[new Set(m1.slice(0,10)),new Set(m2.slice(0,10)),new Set(m3.slice(0,6)),new Set(m4.slice(0,10)),new Set(m5.slice(0,10)),new Set(m6.slice(0,10)),new Set(m7.slice(0,10))];
    for(let num=1;num<=mx;num++){const c=mL.filter(ml=>ml.has(num)).length;if(c>=4)v[num]=(v[num]||0)+c*4;}
    // Light regime
    const reg=sigRegime(data,mx); for(const[num,sc]of Object.entries(reg))if(sc>0)v[num]=(v[num]||0)+Math.min(sc,3);
    // Light lag
    const lg=sigLag(data,mx); for(const[num,sc]of Object.entries(lg))if(sc>0)v[num]=(v[num]||0)+Math.min(sc,3);
    return Object.entries(v).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));
}

function runTest(allData, mx, fn, label) {
    const c={6:{},10:{},15:{},20:{}}; for(const p of[6,10,15,20])for(let k=0;k<=6;k++)c[p][k]=0; let n=0;
    for(let ci=60;ci<allData.length;ci++){const hist=allData.slice(0,ci),actual=new Set(allData[ci]),ranked=fn(hist,mx);for(const p of[6,10,15,20]){const ts=new Set(ranked.slice(0,p));let h=0;for(const x of actual)if(ts.has(x))h++;c[p][h]++;}n++;if(n%200===0)process.stdout.write(`\r  ${label}: ${n} kỳ...`);}
    process.stdout.write(`\r  ${label}: ${n} kỳ — DONE\n`); return{c,n};
}

async function main() {
    console.log('🧬 V710 FINAL BACKTEST');
    console.log('='.repeat(60));
    const allData = await fetchData();
    console.log(`✅ Loaded ${allData.length} draws\n`);
    
    const r604 = runTest(allData, 45, ensembleV604, 'V604');
    const r710 = runTest(allData, 45, ensembleV710Final, 'V710F');
    
    const pct = (v,t) => `${(v/t*100).toFixed(1)}%`;
    console.log('\n' + '='.repeat(70));
    console.log('📊 V604 (OLD) vs V710-FINAL (NEW)');
    console.log('='.repeat(70));
    
    for (const pool of [6,10,15,20]) {
        console.log(`\nPool-${pool}:`);
        for (let k=6;k>=3;k--) {
            let a604=0,a710=0; for(let i=k;i<=6;i++){a604+=r604.c[pool][i];a710+=r710.c[pool][i];}
            const diff=a710-a604; const arrow=diff>0?`⬆️+${diff}`:diff<0?`⬇️${diff}`:'➡️ 0';
            console.log(`  ≥${k}/6: V604=${a604}(${pct(a604,r604.n)}) | V710=${a710}(${pct(a710,r710.n)}) | ${arrow}`);
        }
    }
    console.log('\n🔑 JACKPOT METRICS:');
    console.log(`  Pool-20 6/6: V604=${r604.c[20][6]} → V710=${r710.c[20][6]}`);
    console.log(`  Pool-15 6/6: V604=${r604.c[15][6]} → V710=${r710.c[15][6]}`);
    console.log(`  Pool-10 6/6: V604=${r604.c[10][6]} → V710=${r710.c[10][6]}`);
}
main().catch(console.error);
