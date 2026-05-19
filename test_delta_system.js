/**
 * DELTA SYSTEM ANALYSIS (Global Lottery Strategy)
 * Tests the maximum allowed gap (Delta) between adjacent numbers.
 */
const https = require('https');
function fetchData() { return new Promise((res, rej) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', r => { let d='';r.on('data',c=>d+=c);r.on('end',()=>{const dr=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)dr.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}res(dr);});r.on('error',rej);}); }); }

async function main() {
    const data = await fetchData();
    
    let maxDeltas = [];
    
    for (let d of data) {
        let maxD = d[0]; // First delta is just n1 - 0
        for (let i = 1; i < 6; i++) {
            let gap = d[i] - d[i-1];
            if (gap > maxD) maxD = gap;
        }
        maxDeltas.push(maxD);
    }
    
    maxDeltas.sort((a,b)=>a-b);
    let p90 = maxDeltas[Math.floor(maxDeltas.length * 0.90)];
    let p95 = maxDeltas[Math.floor(maxDeltas.length * 0.95)];
    let p99 = maxDeltas[Math.floor(maxDeltas.length * 0.99)];
    
    console.log('======================================================');
    console.log('🌐 NGHIÊN CỨU TOÀN CẦU: HỆ THỐNG DELTA (DELTA SYSTEM)');
    console.log('======================================================');
    console.log(`Khoảng cách lớn nhất (Max Delta) giữa 2 quả bóng kề nhau trong 1 vé:`);
    console.log(`- 90% số kỳ quay, khoảng cách lớn nhất KHÔNG VƯỢT QUÁ: ${p90}`);
    console.log(`- 95% số kỳ quay, khoảng cách lớn nhất KHÔNG VƯỢT QUÁ: ${p95}`);
    console.log(`- 99% số kỳ quay, khoảng cách lớn nhất KHÔNG VƯỢT QUÁ: ${p99}`);
    
    // Test Elimination Power
    let totalRandom = 100000;
    let passed = 0;
    
    for (let i = 0; i < totalRandom; i++) {
        let pool = [];
        while(pool.length < 6) {
            let r = Math.floor(Math.random() * 45) + 1;
            if (!pool.includes(r)) pool.push(r);
        }
        pool.sort((a,b)=>a-b);
        
        let maxGap = pool[0];
        for (let j = 1; j < 6; j++) {
            if (pool[j] - pool[j-1] > maxGap) maxGap = pool[j] - pool[j-1];
        }
        
        if (maxGap <= p95) passed++;
    }
    
    console.log('\n======================================================');
    console.log('🛑 SỨC MẠNH CỦA BỘ LỌC DELTA TOÀN CẦU');
    console.log('======================================================');
    console.log(`- Lọc bỏ tất cả các vé có khoảng cách giữa 2 bóng > ${p95}`);
    console.log(`- Phát sinh ngẫu nhiên ${totalRandom} vé:`);
    console.log(`- Số vé lọt qua : ${passed} vé`);
    console.log(`- Số vé bị CHÉM BAY : ${totalRandom - passed} vé`);
    console.log(`=> Bộ lọc Delta tiêu diệt thêm ${((totalRandom - passed) / totalRandom * 100).toFixed(2)}% tổ hợp rác!`);
}

main().catch(console.error);
