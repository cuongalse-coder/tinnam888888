/**
 * PHÂN TÍCH TỪNG CỘT (COLUMN-WISE BOUNDS)
 * Analyze the statistical ranges of each position (1 to 6) in the sorted ticket.
 */
const https = require('https');
function fetchData() { return new Promise((res, rej) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', r => { let d='';r.on('data',c=>d+=c);r.on('end',()=>{const dr=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)dr.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}res(dr);});r.on('error',rej);}); }); }

function percentile(arr, p) {
    if (arr.length === 0) return 0;
    const sorted = [...arr].sort((a,b)=>a-b);
    const index = (p/100) * (sorted.length - 1);
    return sorted[Math.round(index)];
}

async function main() {
    const data = await fetchData();
    
    // Arrays to hold the values for each column
    let columns = [[], [], [], [], [], []];
    
    for (let d of data) {
        for (let i = 0; i < 6; i++) {
            columns[i].push(d[i]);
        }
    }
    
    console.log('======================================================');
    console.log('📉 PHÂN TÍCH BIÊN ĐỘ TỪNG CỘT (VỊ TRÍ BÓNG TỪ 1 ĐẾN 6)');
    console.log('======================================================');
    
    let bounds = [];
    
    for (let i = 0; i < 6; i++) {
        let col = columns[i];
        let p5 = percentile(col, 5);
        let p95 = percentile(col, 95);
        let min = Math.min(...col);
        let max = Math.max(...col);
        let avg = (col.reduce((a,b)=>a+b, 0) / col.length).toFixed(1);
        
        bounds.push({ min: p5, max: p95 });
        
        console.log(`Cột ${i+1}: Trung bình [${avg.padStart(4)}] | Vùng an toàn 90%: [${p5.toString().padStart(2)} -> ${p95.toString().padStart(2)}] | Dị biệt từng thấy: ${min}->${max}`);
    }
    
    console.log('\n======================================================');
    console.log('🛑 KIỂM TRA SỨC MẠNH LỌC CỦA "GIỚI HẠN CỘT"');
    console.log('======================================================');
    // Test how many randomly generated combinations pass this filter
    let totalRandom = 100000;
    let passed = 0;
    
    for (let i = 0; i < totalRandom; i++) {
        let pool = [];
        while(pool.length < 6) {
            let r = Math.floor(Math.random() * 45) + 1;
            if (!pool.includes(r)) pool.push(r);
        }
        pool.sort((a,b)=>a-b);
        
        let isValid = true;
        for (let j = 0; j < 6; j++) {
            if (pool[j] < bounds[j].min || pool[j] > bounds[j].max) {
                isValid = false;
                break;
            }
        }
        if (isValid) passed++;
    }
    
    console.log(`Phát sinh ngẫu nhiên ${totalRandom} vé (đại diện cho người mua mù quáng ngoài quầy):`);
    console.log(`- Số vé lọt qua bộ lọc Cột : ${passed} vé`);
    console.log(`- Số vé bị CHÉM BAY        : ${totalRandom - passed} vé`);
    console.log(`=> Bộ lọc "Giới Hạn Cột" tự động tiêu diệt ${((totalRandom - passed) / totalRandom * 100).toFixed(2)}% các tổ hợp phi thực tế!`);
}

main().catch(console.error);
