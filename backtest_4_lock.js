/**
 * BACKTEST: 4-NUMBER LOCK (Chốt 4 Số: Đầu + Đuôi + Cặp Giữa)
 * Budget: 100 tickets (100 vé)
 */
const https = require('https');
function fetchData() { return new Promise((res, rej) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', r => { let d='';r.on('data',c=>d+=c);r.on('end',()=>{const dr=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)dr.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}res(dr);});r.on('error',rej);}); }); }

function mGap(data,mx){const ls={},gl={},li={};for(let i=0;i<data.length;i++)for(const n of data[i]){if(!gl[n])gl[n]=[];if(li[n]!==undefined)gl[n].push(i-li[n]);li[n]=i;ls[n]=i;}const sc=[];for(let n=1;n<=mx;n++){const g=ls[n]!==undefined?data.length-ls[n]:data.length;const mg=gl[n]?.length?gl[n].reduce((a,b)=>a+b,0)/gl[n].length:data.length;sc.push([n,g/(mg+0.1)]);}sc.sort((a,b)=>b[1]-a[1]);return sc.map(x=>x[0]);}
function mMom(data,mx){const w={};for(let n=1;n<=mx;n++)w[n]=0;const t=data.length;for(let i=0;i<t;i++){const d=1/(1+Math.exp(-(i-t+20)/5));for(const n of data[i])w[n]+=d;}return Object.entries(w).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));}
function mKnn(data,mx){if(data.length<20)return mMom(data,mx);const pat=new Set([...data[data.length-1],...data[data.length-2],...data[data.length-3]]);const n=data.length,sims=[];for(let i=3;i<n-3;i++){const pp=new Set([...data[i],...data[i-1],...data[i-2],...data[i-3]]);let inter=0;for(const x of pat)if(pp.has(x))inter++;const rec=1+0.5*(i/n);if(inter>=5)sims.push([inter*rec,i+1]);}sims.sort((a,b)=>b[0]-a[0]);const v={};for(const[sc,ni]of sims.slice(0,30))if(ni<data.length)for(const num of data[ni])v[num]=(v[num]||0)+sc;if(!Object.keys(v).length)return mMom(data,mx);return Object.entries(v).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));}
function mML(data,mx){if(data.length<20)return mGap(data,mx);const f={};for(let n=1;n<=mx;n++)f[n]=0;for(const d of data.slice(-10))for(const n of d)f[n]++;return Object.entries(f).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));}
function mFreqGap(data,mx){const v={};if(data.length<30)return mGap(data,mx);for(let num=1;num<=mx;num++){let f5=0,f15=0,ls=-1;for(let i=data.length-1;i>=0;i--){if(data[i].includes(num)){if(ls<0)ls=i;if(i>=data.length-5)f5++;if(i>=data.length-15)f15++;}}const fs=(f5/(6/mx))*0.6+(f15/(6/mx))*0.4;const gap=ls>=0?data.length-ls:data.length;const ov=gap/((mx/6)+0.1);if(fs>0.8&&ov>0.7)v[num]=fs*ov*3;else if(ov>1.5)v[num]=ov*1.5;else if(fs>1.3)v[num]=fs*2;else v[num]=fs*0.5+ov*0.5;}return Object.entries(v).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));}
function mPair(data,mx){if(data.length<30)return mGap(data,mx);const n=data.length,ps={};for(let idx=0;idx<n;idx++){const dc=0.3+0.7*(idx/n),dr=data[idx].slice(0,6).sort((a,b)=>a-b);for(let i=0;i<dr.length;i++)for(let j=i+1;j<dr.length;j++){const k=`${dr[i]},${dr[j]}`;ps[k]=(ps[k]||0)+dc;}}const ld=new Set(data[n-1].slice(0,6)),cs={};for(let num=1;num<=mx;num++){if(ld.has(num))continue;cs[num]=0;for(const a of ld){const k=Math.min(num,a)+','+Math.max(num,a);cs[num]+=ps[k]||0;}}return Object.entries(cs).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));}

function getPairMatrix(hist, max_number) {
    const pf = {};
    for (let i = Math.max(0, hist.length - 150); i < hist.length; i++) {
        const mid_d = hist[i].filter(n => n > 10 && n < max_number - 10);
        for(let j=0; j<mid_d.length; j++) {
            for(let k=j+1; k<mid_d.length; k++) {
                const kStr = `${mid_d[j]}-${mid_d[k]}`;
                pf[kStr] = (pf[kStr] || 0) + 1;
            }
        }
    }
    return pf;
}

function runBacktest(allData) {
    const mx = 45;
    let hitCounts = {6:0, 5:0, 4:0, 3:0, 2:0, 1:0, 0:0};
    let lockAccuracy = {locked_4_correct: 0, locked_3_correct: 0, locked_2_correct: 0, locked_1_correct: 0, locked_0_correct: 0};
    
    const numTickets = 100; // Ngân sách: 100 vé (1 triệu VNĐ)
    console.log(`\n⏳ Đang tiến hành Backtest Chốt 4 Số với ngân sách ${numTickets} vé/kỳ...`);
    
    for (let ci = 100; ci < allData.length; ci++) {
        const hist = allData.slice(0, ci);
        const actual = new Set(allData[ci]);
        
        // --- 1. Tạo Pool-20 (Mô phỏng AI chính) ---
        const mK = mKnn(hist, mx);
        const mF = mFreqGap(hist, mx);
        const mM = mML(hist, mx);
        const mP = mPair(hist, mx);
        
        const scores = {};
        for(let i=1; i<=mx; i++) scores[i] = 0;
        
        for(let i=0; i<20; i++) {
            if(mK[i]) scores[mK[i]] += (20-i)*1.5;
            if(mF[i]) scores[mF[i]] += (20-i)*1.0;
            if(mM[i]) scores[mM[i]] += (20-i)*0.5;
            if(mP[i]) scores[mP[i]] += (20-i)*0.8;
        }
        
        const pool20 = Object.entries(scores).sort((a,b)=>b[1]-a[1]).slice(0, 20).map(x=>parseInt(x[0]));
        
        // --- 2. Tìm Đầu - Đuôi Tự Động ---
        let h_t1 = hist[hist.length-1][0];
        let h_t2 = hist[hist.length-2][0];
        let t_t1 = hist[hist.length-1][5];
        let t_t2 = hist[hist.length-2][5];
        
        let validHeads = [];
        for(let n=1; n<=10; n++) validHeads.push(n);
        if (h_t1 > h_t2) validHeads = validHeads.filter(n => n < h_t1);
        else if (h_t1 < h_t2) validHeads = validHeads.filter(n => n > h_t1);
        if (validHeads.length === 0) for(let n=1; n<=10; n++) validHeads.push(n);
        
        let bestHead = -1, bestHeadScore = -1;
        for(const n of validHeads) {
            if(scores[n] > bestHeadScore) { bestHeadScore = scores[n]; bestHead = n; }
        }
        
        let validTails = [];
        for(let n=36; n<=45; n++) validTails.push(n);
        if (t_t1 > t_t2) validTails = validTails.filter(n => n < t_t1);
        else if (t_t1 < t_t2) validTails = validTails.filter(n => n > t_t1);
        if (validTails.length === 0) for(let n=36; n<=45; n++) validTails.push(n);
        
        let bestTail = -1, bestTailScore = -1;
        for(const n of validTails) {
            if(scores[n] > bestTailScore) { bestTailScore = scores[n]; bestTail = n; }
        }

        // --- 3. Tìm Cặp Giữa ---
        let mid_pool = pool20.filter(n => n > 10 && n < mx - 10);
        let pf = getPairMatrix(hist, mx);
        
        let bestPair = [];
        let bestPairScore = -1;
        for(let j=0; j<mid_pool.length; j++) {
            for(let k=j+1; k<mid_pool.length; k++) {
                const p1 = Math.min(mid_pool[j], mid_pool[k]);
                const p2 = Math.max(mid_pool[j], mid_pool[k]);
                // CHỈ CHỌN CẶP LIỀN KỀ THEO INSIGHT "DÍNH CHÙM" CỦA USER
                if (p2 - p1 !== 1) continue; 
                
                const key = `${p1}-${p2}`;
                let sc = pf[key] || 0;
                if (sc > bestPairScore) {
                    bestPairScore = sc;
                    bestPair = [p1, p2];
                }
            }
        }
        
        // Nếu không có cặp liền kề nào, lấy mặc định 2 số mạnh nhất vùng giữa
        if (bestPair.length === 0 && mid_pool.length >= 2) {
            bestPair = [mid_pool[0], mid_pool[1]];
        }
        
        let locked4 = [bestHead, bestTail];
        if (bestPair.length === 2) locked4.push(bestPair[0], bestPair[1]);
        
        // Measure locked accuracy
        let lockedHits = 0;
        for(const n of locked4) if(actual.has(n)) lockedHits++;
        lockAccuracy[`locked_${lockedHits}_correct`]++;
        
        // --- 4. Tạo Pool cuối cùng ---
        let finalPool = new Set(pool20);
        for(const n of locked4) finalPool.add(n);
        let remainingPool = Array.from(finalPool).filter(n => !locked4.includes(n));
        
        // --- 5. Cắt vé ---
        let maxHitInDraw = 0;
        for (let t = 0; t < numTickets; t++) {
            let shuffled = [...remainingPool].sort(() => 0.5 - Math.random());
            let ticket = [...locked4, ...shuffled.slice(0, 6 - locked4.length)];
            let hits = 0;
            for(const n of ticket) if(actual.has(n)) hits++;
            if (hits > maxHitInDraw) maxHitInDraw = hits;
        }
        hitCounts[maxHitInDraw]++;
        
        if (ci % 100 === 0) process.stdout.write('.');
    }
    
    const totalDraws = allData.length - 100;
    
    console.log('\n\n======================================================');
    console.log('🚀 KẾT QUẢ BACKTEST: ÉP SIÊU CẤP (KHÓA 4/6 SỐ) - 100 VÉ');
    console.log('======================================================');
    console.log(`Hiệu suất dự đoán 4 số lõi (Đầu + Đuôi + Cặp Giữa):`);
    console.log(`- Trúng phóc cả 4/4 số lõi: ${lockAccuracy.locked_4_correct} lần (${(lockAccuracy.locked_4_correct/totalDraws*100).toFixed(2)}%) -> Cơ hội vàng 6/6`);
    console.log(`- Trúng 3/4 số lõi      : ${lockAccuracy.locked_3_correct} lần (${(lockAccuracy.locked_3_correct/totalDraws*100).toFixed(2)}%)`);
    console.log(`- Trúng 2/4 số lõi      : ${lockAccuracy.locked_2_correct} lần (${(lockAccuracy.locked_2_correct/totalDraws*100).toFixed(2)}%)`);
    console.log('------------------------------------------------------');
    console.log('Thành tích cao nhất trên 1 kỳ quay (Mua 100 vé):');
    console.log(`🏆 Trúng 6/6: ${hitCounts[6]} kỳ (${(hitCounts[6]/totalDraws*100).toFixed(2)}%)`);
    console.log(`🥇 Trúng 5/6: ${hitCounts[5]} kỳ (${(hitCounts[5]/totalDraws*100).toFixed(2)}%)`);
    console.log(`🥈 Trúng 4/6: ${hitCounts[4]} kỳ (${(hitCounts[4]/totalDraws*100).toFixed(2)}%)`);
    console.log(`🥉 Trúng 3/6: ${hitCounts[3]} kỳ (${(hitCounts[3]/totalDraws*100).toFixed(2)}%)`);
    console.log('======================================================');
}

async function main() {
    const data = await fetchData();
    runBacktest(data);
}
main().catch(console.error);
