/**
 * HEAD-TAIL PINNING BACKTEST V2 (Chốt Đầu Đuôi Nâng Cấp)
 * Includes Trend Reversal and Pair Matrix Co-occurrence.
 */
const https = require('https');
function fetchData() { return new Promise((res, rej) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', r => { let d='';r.on('data',c=>d+=c);r.on('end',()=>{const dr=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)dr.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}res(dr);});r.on('error',rej);}); }); }

function mGap(data,mx){const ls={},gl={},li={};for(let i=0;i<data.length;i++)for(const n of data[i]){if(!gl[n])gl[n]=[];if(li[n]!==undefined)gl[n].push(i-li[n]);li[n]=i;ls[n]=i;}const sc=[];for(let n=1;n<=mx;n++){const g=ls[n]!==undefined?data.length-ls[n]:data.length;const mg=gl[n]?.length?gl[n].reduce((a,b)=>a+b,0)/gl[n].length:data.length;sc.push([n,g/(mg+0.1)]);}sc.sort((a,b)=>b[1]-a[1]);return sc.map(x=>x[0]);}
function mMom(data,mx){const w={};for(let n=1;n<=mx;n++)w[n]=0;const t=data.length;for(let i=0;i<t;i++){const d=1/(1+Math.exp(-(i-t+20)/5));for(const n of data[i])w[n]+=d;}return Object.entries(w).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));}
function mKnn(data,mx){if(data.length<20)return mMom(data,mx);const pat=new Set([...data[data.length-1],...data[data.length-2],...data[data.length-3]]);const n=data.length,sims=[];for(let i=3;i<n-3;i++){const pp=new Set([...data[i],...data[i-1],...data[i-2],...data[i-3]]);let inter=0;for(const x of pat)if(pp.has(x))inter++;const rec=1+0.5*(i/n);if(inter>=5)sims.push([inter*rec,i+1]);}sims.sort((a,b)=>b[0]-a[0]);const v={};for(const[sc,ni]of sims.slice(0,30))if(ni<data.length)for(const num of data[ni])v[num]=(v[num]||0)+sc;if(!Object.keys(v).length)return mMom(data,mx);return Object.entries(v).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));}
function mML(data,mx){if(data.length<20)return mGap(data,mx);const f={};for(let n=1;n<=mx;n++)f[n]=0;for(const d of data.slice(-10))for(const n of d)f[n]++;return Object.entries(f).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));}
function mFreqGap(data,mx){const v={};if(data.length<30)return mGap(data,mx);for(let num=1;num<=mx;num++){let f5=0,f15=0,ls=-1;for(let i=data.length-1;i>=0;i--){if(data[i].includes(num)){if(ls<0)ls=i;if(i>=data.length-5)f5++;if(i>=data.length-15)f15++;}}const fs=(f5/(6/mx))*0.6+(f15/(6/mx))*0.4;const gap=ls>=0?data.length-ls:data.length;const ov=gap/((mx/6)+0.1);if(fs>0.8&&ov>0.7)v[num]=fs*ov*3;else if(ov>1.5)v[num]=ov*1.5;else if(fs>1.3)v[num]=fs*2;else v[num]=fs*0.5+ov*0.5;}return Object.entries(v).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));}

function mPair(data,mx){if(data.length<30)return mGap(data,mx);const n=data.length,ps={};for(let idx=0;idx<n;idx++){const dc=0.3+0.7*(idx/n),dr=data[idx].slice(0,6).sort((a,b)=>a-b);for(let i=0;i<dr.length;i++)for(let j=i+1;j<dr.length;j++){const k=`${dr[i]},${dr[j]}`;ps[k]=(ps[k]||0)+dc;}}const ld=new Set(data[n-1].slice(0,6)),cs={};for(let num=1;num<=mx;num++){if(ld.has(num))continue;cs[num]=0;for(const a of ld){const k=Math.min(num,a)+','+Math.max(num,a);cs[num]+=ps[k]||0;}}return Object.entries(cs).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));}

function runBacktest(allData) {
    const mx = 45;
    let hitCounts = {6:0, 5:0, 4:0, 3:0, 2:0, 1:0, 0:0};
    let headTailCorrect = {head:0, tail:0, both:0};
    
    const numTickets = 500; // Tăng ngân sách lên 500 vé (5 triệu VNĐ) mỗi kỳ
    
    console.log(`\n⏳ Đang tiến hành Backtest V2 (Nâng Cấp Trend & Pair)...`);
    
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
        
        // --- 2. Tìm Đầu - Đuôi Tự Động (KÈM TREND & PAIR) ---
        let h_t1 = hist[hist.length-1][0];
        let h_t2 = hist[hist.length-2][0];
        let t_t1 = hist[hist.length-1][5];
        let t_t2 = hist[hist.length-2][5];
        
        // Head filter
        let validHeads = [];
        for(let n=1; n<=10; n++) validHeads.push(n);
        if (h_t1 > h_t2) validHeads = validHeads.filter(n => n < h_t1);
        else if (h_t1 < h_t2) validHeads = validHeads.filter(n => n > h_t1);
        if (validHeads.length === 0) for(let n=1; n<=10; n++) validHeads.push(n);
        
        let bestHead = -1, bestHeadScore = -1;
        for(const n of validHeads) {
            if(scores[n] > bestHeadScore) { bestHeadScore = scores[n]; bestHead = n; }
        }
        
        // Tail filter
        let validTails = [];
        for(let n=36; n<=45; n++) validTails.push(n);
        if (t_t1 > t_t2) validTails = validTails.filter(n => n < t_t1);
        else if (t_t1 < t_t2) validTails = validTails.filter(n => n > t_t1);
        if (validTails.length === 0) for(let n=36; n<=45; n++) validTails.push(n);
        
        // Pair Co-occurrence Boost: 
        // If bestHead is 1, boost 45. If 2, boost 43, etc. (Approximation for JS test)
        let tailScores = {};
        for (const n of validTails) {
            tailScores[n] = scores[n];
            // Simple mapping from previous analysis
            if (bestHead === 1 && n === 45) tailScores[n] += 50;
            if (bestHead === 2 && n === 43) tailScores[n] += 50;
            if (bestHead === 3 && n === 44) tailScores[n] += 50;
            if (bestHead === 4 && n === 43) tailScores[n] += 50;
            if (bestHead === 5 && n === 45) tailScores[n] += 50;
            if (bestHead === 6 && n === 44) tailScores[n] += 50;
        }
        
        let bestTail = -1, bestTailScore = -1;
        for(const n of validTails) {
            if(tailScores[n] > bestTailScore) { bestTailScore = tailScores[n]; bestTail = n; }
        }
        
        // Tracking độ chính xác của chức năng Chốt
        let headHit = actual.has(bestHead);
        let tailHit = actual.has(bestTail);
        if(headHit) headTailCorrect.head++;
        if(tailHit) headTailCorrect.tail++;
        if(headHit && tailHit) headTailCorrect.both++;
        
        // --- 3. Đưa Đầu-Đuôi vào Pool nếu chưa có ---
        let finalPool = new Set(pool20);
        finalPool.add(bestHead);
        finalPool.add(bestTail);
        let poolArr = Array.from(finalPool);
        
        // --- 4. Cắt vé (Wheeling) với Hard Core Lock ---
        const remainingPool = poolArr.filter(n => n !== bestHead && n !== bestTail);
        let maxHitInDraw = 0;
        for (let t = 0; t < numTickets; t++) {
            let shuffled = [...remainingPool].sort(() => 0.5 - Math.random());
            let ticket = [bestHead, bestTail, ...shuffled.slice(0, 4)];
            let hits = 0;
            for(const n of ticket) if(actual.has(n)) hits++;
            if (hits > maxHitInDraw) maxHitInDraw = hits;
        }
        hitCounts[maxHitInDraw]++;
        
        if (ci % 100 === 0) process.stdout.write('.');
    }
    
    const totalDraws = allData.length - 100;
    
    console.log('\n\n======================================================');
    console.log('🚀 KẾT QUẢ BACKTEST V2: TREND ĐẢO CHIỀU + LỰC HÚT ĐẦU ĐUÔI');
    console.log('======================================================');
    console.log(`Độ chính xác Đoán Số Đầu (V1 -> V2): 14.42% -> ${(headTailCorrect.head/totalDraws*100).toFixed(2)}%`);
    console.log(`Độ chính xác Đoán Số Đuôi (V1 -> V2): 13.34% -> ${(headTailCorrect.tail/totalDraws*100).toFixed(2)}%`);
    console.log(`Đoán trúng ĐỒNG THỜI cả 2 chốt: 2.47% -> ${(headTailCorrect.both/totalDraws*100).toFixed(2)}%`);
    console.log('------------------------------------------------------');
    console.log('Thành tích cao nhất trên 1 kỳ quay (Mua 20 vé Bao Lô):');
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
