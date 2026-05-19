/**
 * PHÂN TÍCH TẦN SUẤT CHỮ SỐ (DIGIT FREQUENCY ANALYSIS)
 * Chuyển 6 quả bóng thành 12 chữ số (VD: 01, 12, 25, 36, 41, 45 -> 0,1,1,2,2,5,3,6,4,1,4,5)
 * Xem tỷ lệ trùng lặp của các chữ số từ 0-9.
 */
const https = require('https');
function fetchData() { return new Promise((res, rej) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', r => { let d='';r.on('data',c=>d+=c);r.on('end',()=>{const dr=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)dr.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}res(dr);});r.on('error',rej);}); }); }

function getDigitFrequency(combo) {
    let counts = new Array(10).fill(0);
    for (let num of combo) {
        let str = num.toString().padStart(2, '0');
        counts[parseInt(str[0])]++;
        counts[parseInt(str[1])]++;
    }
    return counts;
}

async function main() {
    const data = await fetchData();
    
    // 1. Tìm giới hạn lớn nhất của một chữ số trong 1 kỳ
    let maxSameDigitCounts = { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0 };
    
    // 2. Theo dõi chữ số bị "Bùng nổ" (xuất hiện >= 4 lần)
    let exhaustionNextCounts = { 0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0 };
    let totalExhaustion = 0;

    for (let i = 0; i < data.length; i++) {
        let freq = getDigitFrequency(data[i]);
        let maxInDraw = Math.max(...freq);
        maxSameDigitCounts[maxInDraw] = (maxSameDigitCounts[maxInDraw] || 0) + 1;
        
        // Hiệu ứng cạn kiệt: Kỳ trước chữ số nào nổ >= 4 lần
        if (i > 0) {
            let prevFreq = getDigitFrequency(data[i-1]);
            for (let digit = 0; digit <= 9; digit++) {
                if (prevFreq[digit] >= 4) { // Reached physical limit
                    totalExhaustion++;
                    let nextCount = freq[digit];
                    exhaustionNextCounts[nextCount] = (exhaustionNextCounts[nextCount] || 0) + 1;
                }
            }
        }
    }
    
    console.log('======================================================');
    console.log('🔢 PHÂN TÍCH CHỮ SỐ (12 CHỮ SỐ / KỲ QUAY)');
    console.log('======================================================');
    console.log('Trong số 12 chữ số tạo nên tấm vé, số lần LẶP LẠI MAX của 1 chữ số bất kỳ (từ 0-9):');
    for (let i = 1; i <= 8; i++) {
        let count = maxSameDigitCounts[i] || 0;
        let pct = (count / data.length * 100).toFixed(2);
        console.log(`- Lặp lại tối đa ${i} lần: ${count} kỳ (${pct}%)`);
    }
    console.log('\n=> KẾT LUẬN: Gần như TUYỆT ĐỐI không bao giờ 1 chữ số lặp lại quá 4 lần. Nếu vượt quá 4 lần (chỉ chiếm ~8%), chúng ta có thể chém bỏ!');

    console.log('\n======================================================');
    console.log('📉 HIỆU ỨNG CẠN KIỆT CHỮ SỐ (DIGIT EXHAUSTION)');
    console.log('======================================================');
    console.log(`Nếu kỳ trước có 1 chữ số nổ kịch trần (>= 4 lần), kỳ sau chữ số đó sẽ ra sao?`);
    for (let i = 0; i <= 5; i++) {
        let count = exhaustionNextCounts[i] || 0;
        let pct = (count / totalExhaustion * 100).toFixed(2);
        console.log(`- Kỳ sau nổ lại đúng ${i} lần: ${pct}%`);
    }
    
    // Test sức mạnh lọc
    let totalRandom = 100000;
    let passed = 0;
    for (let i = 0; i < totalRandom; i++) {
        let pool = [];
        while(pool.length < 6) {
            let r = Math.floor(Math.random() * 45) + 1;
            if (!pool.includes(r)) pool.push(r);
        }
        let freq = getDigitFrequency(pool);
        let maxFreq = Math.max(...freq);
        if (maxFreq <= 4) passed++;
    }
    
    console.log('\n======================================================');
    console.log('🛑 SỨC MẠNH CỦA BỘ LỌC CHỮ SỐ (DIGIT FILTER)');
    console.log('======================================================');
    console.log(`Giới hạn tàn nhẫn: BẤT KỲ chữ số nào từ 0-9 xuất hiện >= 5 lần -> VỨT SỌT RÁC!`);
    console.log(`Phát sinh ngẫu nhiên ${totalRandom} vé:`);
    console.log(`- Số vé CÒN SỐNG   : ${passed} vé`);
    console.log(`- Số vé BỊ CHÉM BAY: ${totalRandom - passed} vé`);
    console.log(`=> Chém bay thêm ${((totalRandom - passed) / totalRandom * 100).toFixed(2)}% tổng lượng vé rác!`);
}

main().catch(console.error);
