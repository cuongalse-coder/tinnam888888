/**
 * PHÂN TÍCH CHU KỲ CỬA SỔ TRƯỢT (SLIDING WINDOW ANALYSIS)
 * Quan sát 10 và 20 kỳ quay gần nhất để tìm Nhóm Vắng Mặt (Lô gan) 
 * và Nhóm Lặp Lại (Hot numbers). Đoán xem kỳ tiếp theo sẽ lấy bao nhiêu số từ mỗi nhóm.
 */
const https = require('https');
function fetchData() { return new Promise((res, rej) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', r => { let d='';r.on('data',c=>d+=c);r.on('end',()=>{const dr=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)dr.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}res(dr);});r.on('error',rej);}); }); }

async function analyzeWindow(data, windowSize) {
    let stats = {
        missing_picked: {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0},
        hot_picked: {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0},
        avg_missing_pool: 0
    };
    
    let validDraws = 0;
    
    for (let i = windowSize; i < data.length; i++) {
        let window = data.slice(i - windowSize, i);
        let target = data[i];
        
        // Đếm tần suất xuất hiện của 45 số trong Window
        let counts = Array(46).fill(0);
        for (let draw of window) {
            for (let num of draw) counts[num]++;
        }
        
        let missingPool = [];
        let warmPool = [];
        let hotPool = [];
        
        for (let n = 1; n <= 45; n++) {
            if (counts[n] === 0) missingPool.push(n);
            else if (counts[n] === 1) warmPool.push(n);
            else hotPool.push(n); // Xuất hiện >= 2 lần
        }
        
        stats.avg_missing_pool += missingPool.length;
        
        // Phân tích Kỳ mục tiêu (Target)
        let missingHit = 0;
        let hotHit = 0;
        
        for (let num of target) {
            if (counts[num] === 0) missingHit++;
            else if (counts[num] >= 2) hotHit++;
        }
        
        stats.missing_picked[missingHit] = (stats.missing_picked[missingHit] || 0) + 1;
        stats.hot_picked[hotHit] = (stats.hot_picked[hotHit] || 0) + 1;
        
        validDraws++;
    }
    
    console.log(`\n======================================================`);
    console.log(`🔍 PHÂN TÍCH CỬA SỔ ${windowSize} KỲ GẦN NHẤT`);
    console.log(`======================================================`);
    console.log(`- Dữ liệu quét: ${validDraws} kỳ quay.`);
    console.log(`- Trung bình có: ${(stats.avg_missing_pool / validDraws).toFixed(1)} số "Vắng bóng hoàn toàn" (Count = 0).`);
    
    console.log(`\n1. SỐ VẮNG BÓNG CÓ RỚT VÀO KỲ TIẾP THEO KHÔNG? (Missing Numbers)`);
    for (let i = 0; i <= 6; i++) {
        let pct = (stats.missing_picked[i] / validDraws * 100).toFixed(2);
        console.log(`   - Rớt trúng ${i} số vắng bóng: ${pct}%`);
    }
    
    console.log(`\n2. SỐ HOT (Xuất hiện >= 2 lần) CÓ LẶP LẠI TIẾP KHÔNG? (Hot Numbers)`);
    for (let i = 0; i <= 6; i++) {
        let pct = (stats.hot_picked[i] / validDraws * 100).toFixed(2);
        console.log(`   - Rớt trúng ${i} số Hot: ${pct}%`);
    }
    
    return {
        missing_picked: stats.missing_picked,
        hot_picked: stats.hot_picked,
        validDraws: validDraws
    };
}

async function main() {
    const data = await fetchData();
    
    let res10 = await analyzeWindow(data, 10);
    let res20 = await analyzeWindow(data, 20);
    
    console.log('\n======================================================');
    console.log('🛑 SỨC MẠNH BỘ LỌC CỬA SỔ TRƯỢT (SLIDING WINDOW FILTER)');
    console.log('======================================================');
    
    let totalRandom = 100000;
    let passed = 0;
    
    // Giả lập Window là 10 kỳ gần nhất của thực tế
    let window = data.slice(data.length - 10, data.length);
    let counts = Array(46).fill(0);
    for (let draw of window) {
        for (let num of draw) counts[num]++;
    }
    
    for (let i = 0; i < totalRandom; i++) {
        let pool = [];
        while(pool.length < 6) {
            let r = Math.floor(Math.random() * 45) + 1;
            if (!pool.includes(r)) pool.push(r);
        }
        
        let missingHit = 0;
        let hotHit = 0;
        for (let num of pool) {
            if (counts[num] === 0) missingHit++;
            else if (counts[num] >= 2) hotHit++;
        }
        
        // LUẬT: Dựa theo xác suất 10 kỳ, KHÔNG BAO GIỜ có vé nào lấy toàn bộ 5-6 số từ Vắng bóng hoặc toàn bộ 5-6 số từ Hot.
        if (missingHit <= 3 && hotHit <= 3) {
            passed++;
        }
    }
    
    console.log(`Luật đề xuất: Tấm vé không được phép lấy > 3 số Vắng mặt HOẶC > 3 số Hot.`);
    console.log(`Phát sinh ngẫu nhiên ${totalRandom} vé (để test lọc):`);
    console.log(`- Số vé CÒN SỐNG   : ${passed} vé`);
    console.log(`- Số vé BỊ CHÉM BAY: ${totalRandom - passed} vé`);
    console.log(`=> Bộ Lọc Chu Kỳ (Sliding Window) tự động diệt thêm ${((totalRandom - passed) / totalRandom * 100).toFixed(2)}% vé rác!`);
}

main().catch(console.error);
