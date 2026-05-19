/**
 * PHÂN TÍCH SÓNG HÀNG ĐƠN VỊ VÀ ĐIỂM NGẮT (WAVE INFLECTION POINTS)
 * Phân tích 6 chữ số hàng đơn vị như một đoạn sóng.
 */
const https = require('https');
function fetchData() { return new Promise((res, arrayRej) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', r => { let d='';r.on('data',c=>d+=c);r.on('end',()=>{const dr=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)dr.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}res(dr);});r.on('error',arrayRej);}); }); }

async function main() {
    const data = await fetchData();
    
    let inflectionCounts = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0};
    
    for (let d of data) {
        // Extract Ones digits
        let ones = d.map(n => n % 10);
        
        // Calculate slopes: 1 for UP, -1 for DOWN, 0 for FLAT
        let slopes = [];
        for (let i = 1; i < 6; i++) {
            if (ones[i] > ones[i-1]) slopes.push(1);
            else if (ones[i] < ones[i-1]) slopes.push(-1);
            else slopes.push(0);
        }
        
        // Count inflection points (Breaking points / Điểm ngắt)
        // A break is when the slope changes direction (ignoring flat 0s)
        let breaks = 0;
        let currentDir = 0;
        for (let s of slopes) {
            if (s !== 0) {
                if (currentDir !== 0 && currentDir !== s) {
                    breaks++; // Direction changed!
                }
                currentDir = s;
            }
        }
        
        inflectionCounts[breaks] = (inflectionCounts[breaks] || 0) + 1;
    }
    
    console.log('======================================================');
    console.log('🌊 PHÂN TÍCH SÓNG HÀNG ĐƠN VỊ & ĐIỂM NGẮT');
    console.log('======================================================');
    console.log('Số lượng "Điểm Ngắt Sóng" (Đảo chiều Tăng/Giảm) trong 1 vé:');
    let cumulative = 0;
    for (let i = 0; i <= 4; i++) {
        let count = inflectionCounts[i] || 0;
        let pct = (count / data.length * 100).toFixed(2);
        cumulative += parseFloat(pct);
        console.log(`- Có ${i} điểm ngắt sóng: ${pct}%`);
    }
    
    console.log('\n=> KẾT LUẬN: Một vé tự nhiên PHẢI DAO ĐỘNG. Vé có 0 điểm ngắt (Chỉ tăng hoặc chỉ giảm liên tục) là cực hiếm!');
    
    // Test sức mạnh lọc
    let totalRandom = 100000;
    let passed = 0;
    for (let i = 0; i < totalRandom; i++) {
        let pool = [];
        while(pool.length < 6) {
            let r = Math.floor(Math.random() * 45) + 1;
            if (!pool.includes(r)) pool.push(r);
        }
        pool.sort((a,b)=>a-b);
        let ones = pool.map(n => n % 10);
        
        let breaks = 0;
        let currentDir = 0;
        for (let j = 1; j < 6; j++) {
            let s = 0;
            if (ones[j] > ones[j-1]) s = 1;
            else if (ones[j] < ones[j-1]) s = -1;
            
            if (s !== 0) {
                if (currentDir !== 0 && currentDir !== s) breaks++;
                currentDir = s;
            }
        }
        
        if (breaks >= 1) passed++;
    }
    
    console.log('\n======================================================');
    console.log('🛑 SỨC MẠNH CỦA BỘ LỌC ĐIỂM NGẮT SÓNG');
    console.log('======================================================');
    console.log(`Giới hạn: Tiêu diệt TẤT CẢ các vé có 0 điểm ngắt sóng (vd: Hàng đơn vị tăng dần đều 1, 2, 4, 5, 8, 9)`);
    console.log(`Phát sinh ngẫu nhiên ${totalRandom} vé:`);
    console.log(`- Số vé CÒN SỐNG   : ${passed} vé`);
    console.log(`- Số vé BỊ CHÉM BAY: ${totalRandom - passed} vé`);
    console.log(`=> Tiêu diệt thêm ${((totalRandom - passed) / totalRandom * 100).toFixed(2)}% tổng lượng vé rác!`);
}

main().catch(console.error);
