/**
 * HEAD-TAIL PINNING BACKTEST (Chốt Đầu Đuôi)
 * Test the efficiency of locking the top Head (1-10) and top Tail (36-45)
 * into a 20-ticket Bao 20 strategy.
 */
const https = require('https');
function fetchData() { return new Promise((res, rej) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', r => { let d='';r.on('data',c=>d+=c);r.on('end',()=>{const dr=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)dr.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}res(dr);});r.on('error',rej);}); }); }

function mGap(data,mx){const ls={},gl={},li={};for(let i=0;i<data.length;i++)for(const n of data[i]){if(!gl[n])gl[n]=[];if(li[n]!==undefined)gl[n].push(i-li[n]);li[n]=i;ls[n]=i;}const sc=[];for(let n=1;n<=mx;n++){const g=ls[n]!==undefined?data.length-ls[n]:data.length;const mg=gl[n]?.length?gl[n].reduce((a,b)=>a+b,0)/gl[n].length:data.length;sc.push([n,g/(mg+0.1)]);}sc.sort((a,b)=>b[1]-a[1]);return sc.map(x=>x[0]);}
function mMom(data,mx){const w={};for(let n=1;n<=mx;n++)w[n]=0;const t=data.length;for(let i=0;i<t;i++){const d=1/(1+Math.exp(-(i-t+20)/5));for(const n of data[i])w[n]+=d;}return Object.entries(w).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));}
function mKnn(data,mx){if(data.length<20)return mMom(data,mx);const pat=new Set([...data[data.length-1],...data[data.length-2],...data[data.length-3]]);const n=data.length,sims=[];for(let i=3;i<n-3;i++){const pp=new Set([...data[i],...data[i-1],...data[i-2],...data[i-3]]);let inter=0;for(const x of pat)if(pp.has(x))inter++;const rec=1+0.5*(i/n);if(inter>=5)sims.push([inter*rec,i+1]);}sims.sort((a,b)=>b[0]-a[0]);const v={};for(const[sc,ni]of sims.slice(0,30))if(ni<data.length)for(const num of data[ni])v[num]=(v[num]||0)+sc;if(!Object.keys(v).length)return mMom(data,mx);return Object.entries(v).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));}
function mML(data,mx){if(data.length<20)return mGap(data,mx);const f={};for(let n=1;n<=mx;n++)f[n]=0;for(const d of data.slice(-10))for(const n of d)f[n]++;return Object.entries(f).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));}
function mFreqGap(data,mx){const v={};if(data.length<30)return mGap(data,mx);for(let num=1;num<=mx;num++){let f5=0,f15=0,ls=-1;for(let i=data.length-1;i>=0;i--){if(data[i].includes(num)){if(ls<0)ls=i;if(i>=data.length-5)f5++;if(i>=data.length-15)f15++;}}const fs=(f5/(6/mx))*0.6+(f15/(6/mx))*0.4;const gap=ls>=0?data.length-ls:data.length;const ov=gap/((mx/6)+0.1);if(fs>0.8&&ov>0.7)v[num]=fs*ov*3;else if(ov>1.5)v[num]=ov*1.5;else if(fs>1.3)v[num]=fs*2;else v[num]=fs*0.5+ov*0.5;}return Object.entries(v).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0]));}

function runBacktest(allData) {
    const mx = 45;
    let hitCounts = {6:0, 5:0, 4:0, 3:0, 2:0, 1:0, 0:0};
    let headTailCorrect = {head:0, tail:0, both:0};
    
    const numTickets = 20; // Giả lập ngân sách mua 20 vé (200k) mỗi kỳ
    
    console.log(`\n⏳ Đang tiến hành Backtest ${allData.length - 100} kỳ với ngân sách ${numTickets} vé/kỳ...`);
    
    for (let ci = 100; ci < allData.length; ci++) {
        const hist = allData.slice(0, ci);
        const actual = new Set(allData[ci]);
        
        // --- 1. Tạo Pool-20 (Mô phỏng AI chính) ---
        const mK = mKnn(hist, mx);
        const mF = mFreqGap(hist, mx);
        const mM = mML(hist, mx);
        
        const scores = {};
        for(let i=1; i<=mx; i++) scores[i] = 0;
        
        for(let i=0; i<20; i++) {
            if(mK[i]) scores[mK[i]] += (20-i)*1.5;
            if(mF[i]) scores[mF[i]] += (20-i)*1.0;
            if(mM[i]) scores[mM[i]] += (20-i)*0.5;
        }
        
        // Lấy 20 số mạnh nhất
        const pool20 = Object.entries(scores).sort((a,b)=>b[1]-a[1]).slice(0, 20).map(x=>parseInt(x[0]));
        
        // --- 2. Tìm Đầu - Đuôi Tự Động ---
        let bestHead = -1, bestHeadScore = -1;
        for(let n=1; n<=10; n++) {
            if(scores[n] > bestHeadScore) { bestHeadScore = scores[n]; bestHead = n; }
        }
        
        let bestTail = -1, bestTailScore = -1;
        for(let n=36; n<=45; n++) {
            if(scores[n] > bestTailScore) { bestTailScore = scores[n]; bestTail = n; }
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
        // Ép cứng bestHead và bestTail vào TẤT CẢ 20 vé
        const remainingPool = poolArr.filter(n => n !== bestHead && n !== bestTail);
        
        // Giả lập thuật toán random sampling để lấy vé
        let maxHitInDraw = 0;
        for (let t = 0; t < numTickets; t++) {
            // Shuffle remaining pool
            let shuffled = [...remainingPool].sort(() => 0.5 - Math.random());
            let ticket = [bestHead, bestTail, ...shuffled.slice(0, 4)];
            
            // Check ticket against actual
            let hits = 0;
            for(const n of ticket) if(actual.has(n)) hits++;
            if (hits > maxHitInDraw) maxHitInDraw = hits;
        }
        
        hitCounts[maxHitInDraw]++;
        
        if (ci % 100 === 0) process.stdout.write('.');
    }
    
    const totalDraws = allData.length - 100;
    
    console.log('\n\n======================================================');
    console.log('🚀 KẾT QUẢ BACKTEST: CHỐT ĐẦU-ĐUÔI + BAO 20 SỐ (20 VÉ)');
    console.log('======================================================');
    console.log(`Độ chính xác Đoán Số Đầu (1-10): ${(headTailCorrect.head/totalDraws*100).toFixed(2)}%`);
    console.log(`Độ chính xác Đoán Số Đuôi (36-45): ${(headTailCorrect.tail/totalDraws*100).toFixed(2)}%`);
    console.log(`Đoán trúng ĐỒNG THỜI cả 2 chốt: ${(headTailCorrect.both/totalDraws*100).toFixed(2)}% (Cơ hội JackPot cực đại)`);
    console.log('------------------------------------------------------');
    console.log('Thành tích cao nhất trên 1 kỳ quay (chỉ mua 20 vé/kỳ):');
    console.log(`🏆 Trúng 6/6: ${hitCounts[6]} kỳ (${(hitCounts[6]/totalDraws*100).toFixed(2)}%)`);
    console.log(`🥇 Trúng 5/6: ${hitCounts[5]} kỳ (${(hitCounts[5]/totalDraws*100).toFixed(2)}%)`);
    console.log(`🥈 Trúng 4/6: ${hitCounts[4]} kỳ (${(hitCounts[4]/totalDraws*100).toFixed(2)}%)`);
    console.log(`🥉 Trúng 3/6: ${hitCounts[3]} kỳ (${(hitCounts[3]/totalDraws*100).toFixed(2)}%)`);
    console.log('======================================================');
    
    // Tính toán Random để so sánh
    // Mua 20 vé ngẫu nhiên, xác suất trúng 6/6 là 20 / 8,145,060 = 0.000245%
    const rand_6_pct = (20 / 8145060) * 100;
    console.log(`💡 So với chơi ngẫu nhiên (chỉ đạt ${rand_6_pct.toFixed(5)}% 6/6), `);
    if(hitCounts[6]>0) {
        console.log(`chiến thuật Chốt Đầu-Đuôi ĐÃ GIÚP BẠN MẠNH GẤP ${(hitCounts[6]/totalDraws*100 / rand_6_pct).toFixed(0)} LẦN!`);
    } else {
        console.log(`hãy tập trung vào giải 5/6 (Vì 6/6 vẫn cần lưới vé rộng hơn 20 vé).`);
    }
}

async function main() {
    const data = await fetchData();
    runBacktest(data);
}
main().catch(console.error);
