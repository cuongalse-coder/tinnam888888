/**
 * TRUE 6/6 HIT RATE BACKTEST
 * This script tests the true 6/6 percentage of the two unified strategies:
 * 1. Bắn Tỉa Tối Thượng (Khóa 4 số)
 * 2. Lưới Quét Diện Rộng (Top 20 + Lọc Dây Thun + Lọc Cạn Kiệt Nhóm)
 */
const https = require('https');
function fetchData() { return new Promise((res, rej) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', r => { let d='';r.on('data',c=>d+=c);r.on('end',()=>{const dr=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)dr.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}res(dr);});r.on('error',rej);}); }); }

// AI logic mocks
function mGap(data,mx){const ls={},gl={},li={};for(let i=0;i<data.length;i++)for(const n of data[i]){if(!gl[n])gl[n]=[];if(li[n]!==undefined)gl[n].push(i-li[n]);li[n]=i;ls[n]=i;}const sc=[];for(let n=1;n<=mx;n++){const g=ls[n]!==undefined?data.length-ls[n]:data.length;const mg=gl[n]?.length?gl[n].reduce((a,b)=>a+b,0)/gl[n].length:data.length;sc.push([n,g/(mg+0.1)]);}sc.sort((a,b)=>b[1]-a[1]);return sc.map(x=>x[0]);}
function mMom(data,mx){const w={};for(let n=1;n<=mx;n++)w[n]=0;const t=data.length;for(let i=0;i<t;i++){const d=1/(1+Math.exp(-(i-t+20)/5));for(const n of data[i])w[n]+=d;}return Object.entries(w).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));}
function mKnn(data,mx){if(data.length<20)return mMom(data,mx);const pat=new Set([...data[data.length-1],...data[data.length-2],...data[data.length-3]]);const n=data.length,sims=[];for(let i=3;i<n-3;i++){const pp=new Set([...data[i],...data[i-1],...data[i-2],...data[i-3]]);let inter=0;for(const x of pat)if(pp.has(x))inter++;const rec=1+0.5*(i/n);if(inter>=5)sims.push([inter*rec,i+1]);}sims.sort((a,b)=>b[0]-a[0]);const v={};for(const[sc,ni]of sims.slice(0,30))if(ni<data.length)for(const num of data[ni])v[num]=(v[num]||0)+sc;if(!Object.keys(v).length)return mMom(data,mx);return Object.entries(v).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));}
function mPair(data,mx){if(data.length<30)return mGap(data,mx);const n=data.length,ps={};for(let idx=0;idx<n;idx++){const dc=0.3+0.7*(idx/n),dr=data[idx].slice(0,6).sort((a,b)=>a-b);for(let i=0;i<dr.length;i++)for(let j=i+1;j<dr.length;j++){const k=`${dr[i]},${dr[j]}`;ps[k]=(ps[k]||0)+dc;}}const ld=new Set(data[n-1].slice(0,6)),cs={};for(let num=1;num<=mx;num++){if(ld.has(num))continue;cs[num]=0;for(const a of ld){const k=Math.min(num,a)+','+Math.max(num,a);cs[num]+=ps[k]||0;}}return Object.entries(cs).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));}

function getDecades(combo) {
    let decs = [0, 0, 0, 0, 0];
    for (let n of combo) decs[Math.min(Math.floor((n-1)/10), 4)]++;
    return decs;
}

function percentile(arr, p) {
    if (arr.length === 0) return 0;
    const sorted = [...arr].sort((a,b)=>a-b);
    const index = (p/100) * (sorted.length - 1);
    return sorted[Math.round(index)];
}

async function main() {
    const data = await fetchData();
    const mx = 45;
    
    let hit66_Sniper = 0;
    let hit66_Broad = 0;
    
    let hit56_Sniper = 0;
    let hit56_Broad = 0;
    
    const numTickets = 100;
    const testDraws = 500;
    console.log(`⏳ Đang chạy Backtest THỰC TẾ trên ${testDraws} kỳ gần nhất (Ngân sách: 100 vé/kỳ)`);
    
    for (let ci = data.length - testDraws; ci < data.length; ci++) {
        const hist = data.slice(0, ci);
        const actual = new Set(data[ci].slice(0, 6));
        
        // --- AI POOL PREDICTION ---
        const mK = mKnn(hist, mx);
        const mM = mMom(hist, mx);
        const mP = mPair(hist, mx);
        const scores = {};
        for(let i=1; i<=mx; i++) scores[i] = 0;
        for(let i=0; i<20; i++) {
            if(mK[i]) scores[mK[i]] += (20-i)*1.5;
            if(mM[i]) scores[mM[i]] += (20-i)*1.0;
            if(mP[i]) scores[mP[i]] += (20-i)*1.0;
        }
        const pool20 = Object.entries(scores).sort((a,b)=>b[1]-a[1]).slice(0, 20).map(x=>parseInt(x[0]));
        
        // ==========================================
        // CHIẾN THUẬT 1: BẮN TỈA TỐI THƯỢNG (KHÓA 4)
        // ==========================================
        let h_t1 = hist[hist.length-1][0];
        let h_t2 = hist[hist.length-2][0];
        let t_t1 = hist[hist.length-1][5];
        let t_t2 = hist[hist.length-2][5];
        
        let validHeads = []; for(let n=1; n<=10; n++) validHeads.push(n);
        if (h_t1 > h_t2) validHeads = validHeads.filter(n => n < h_t1);
        else if (h_t1 < h_t2) validHeads = validHeads.filter(n => n > h_t1);
        if (!validHeads.length) for(let n=1; n<=10; n++) validHeads.push(n);
        let bestHead = validHeads.sort((a,b) => (scores[b]||0) - (scores[a]||0))[0];
        
        let validTails = []; for(let n=36; n<=45; n++) validTails.push(n);
        if (t_t1 > t_t2) validTails = validTails.filter(n => n < t_t1);
        else if (t_t1 < t_t2) validTails = validTails.filter(n => n > t_t1);
        if (!validTails.length) for(let n=36; n<=45; n++) validTails.push(n);
        let bestTail = validTails.sort((a,b) => (scores[b]||0) - (scores[a]||0))[0];
        
        let mid_pool = pool20.filter(n => n > 10 && n < mx - 10);
        let bestPair = [mid_pool[0]||12, mid_pool[1]||13];
        for(let j=0; j<mid_pool.length; j++) {
            for(let k=j+1; k<mid_pool.length; k++) {
                if (Math.abs(mid_pool[j] - mid_pool[k]) === 1) {
                    bestPair = [Math.min(mid_pool[j], mid_pool[k]), Math.max(mid_pool[j], mid_pool[k])];
                    break;
                }
            }
        }
        
        let locked4 = [bestHead, bestTail, bestPair[0], bestPair[1]];
        let remainSniper = pool20.filter(n => !locked4.includes(n));
        
        let maxHitsSniper = 0;
        for (let t=0; t<numTickets; t++) {
            let shuffled = [...remainSniper].sort(()=>0.5-Math.random());
            let ticket = [...locked4, ...shuffled.slice(0, 2)];
            let hits = ticket.filter(n => actual.has(n)).length;
            if (hits > maxHitsSniper) maxHitsSniper = hits;
        }
        if (maxHitsSniper === 6) hit66_Sniper++;
        if (maxHitsSniper >= 5) hit56_Sniper++;
        
        // ==========================================
        // CHIẾN THUẬT 2: LƯỚI QUÉT DIỆN RỘNG (NO LOCK)
        // ==========================================
        let recent = hist.slice(-50);
        let ranges = recent.map(d => Math.max(...d) - Math.min(...d));
        let r_lo = percentile(ranges, 8);
        let r_hi = percentile(ranges, 92);
        
        let s_t1 = hist[hist.length-1][5] - hist[hist.length-1][0];
        let s_t2 = hist[hist.length-2][5] - hist[hist.length-2][0];
        
        if (s_t1 >= 40) r_hi = Math.min(r_hi, 38);
        else if (s_t1 <= 25) r_lo = Math.max(r_lo, 28);
        else {
            if (s_t1 > s_t2) r_hi = Math.min(r_hi, s_t1 - 1);
            else if (s_t1 < s_t2) r_lo = Math.max(r_lo, s_t1 + 1);
        }
        if (r_lo > r_hi) { let t=r_lo; r_lo=r_hi; r_hi=t; }
        
        let prevDecs = getDecades(hist[hist.length-1]);
        
        let maxHitsBroad = 0;
        let validGenerated = 0;
        for (let t=0; t<2000; t++) {
            if (validGenerated >= numTickets) break;
            
            let ticket = [...pool20].sort(()=>0.5-Math.random()).slice(0,6).sort((a,b)=>a-b);
            let rng = ticket[5] - ticket[0];
            
            // Elastic Filter
            if (rng < r_lo || rng > r_hi) continue;
            
            // Exhaustion Filter
            let decs = getDecades(ticket);
            let badGroup = false;
            for (let d=0; d<5; d++) {
                if (decs[d] > 3) badGroup = true; // Max 3
                if (prevDecs[d] >= 3 && decs[d] > 2) badGroup = true; // Chết nhóm
            }
            if (badGroup) continue;
            
            validGenerated++;
            let hits = ticket.filter(n => actual.has(n)).length;
            if (hits > maxHitsBroad) maxHitsBroad = hits;
        }
        if (maxHitsBroad === 6) hit66_Broad++;
        if (maxHitsBroad >= 5) hit56_Broad++;
        
        if (ci % 50 === 0) process.stdout.write('.');
    }
    
    console.log('\n\n======================================================');
    console.log('🚀 TỶ LỆ TRÚNG 6/6 THỰC TẾ (MUA 100 VÉ/KỲ TRONG 500 KỲ)');
    console.log('======================================================');
    console.log(`Chiến thuật 1: Bắn Tỉa Tối Thượng (Khóa 4 số)`);
    console.log(`- Trúng 6/6: ${hit66_Sniper} lần (${(hit66_Sniper/testDraws*100).toFixed(2)}%)`);
    console.log(`- Trúng 5/6: ${hit56_Sniper} lần (${(hit56_Sniper/testDraws*100).toFixed(2)}%)`);
    
    console.log(`\nChiến thuật 2: Lưới Quét Diện Rộng (Top 20 + 2 Bộ Lọc)`);
    console.log(`- Trúng 6/6: ${hit66_Broad} lần (${(hit66_Broad/testDraws*100).toFixed(2)}%)`);
    console.log(`- Trúng 5/6: ${hit56_Broad} lần (${(hit56_Broad/testDraws*100).toFixed(2)}%)`);
    console.log('======================================================');
}

main().catch(console.error);
