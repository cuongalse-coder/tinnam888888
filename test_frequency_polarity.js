/**
 * PHÂN CỰC TẦN SUẤT (FREQUENCY POLARIZATION ANALYSIS)
 * Phân chia 45 con số thành 2 nửa: 
 * - Nửa Về Nhiều Nhất (Top Frequent)
 * - Nửa Về Ít Nhất (Bottom Frequent)
 * Xem tỷ lệ phân bổ của 6 quả bóng trúng giải nằm ở 2 nửa này như thế nào.
 */
const https = require('https');
function fetchData() { return new Promise((res, rej) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', r => { let d='';r.on('data',c=>d+=c);r.on('end',()=>{const dr=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)dr.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}res(dr);});r.on('error',rej);}); }); }

async function main() {
    const data = await fetchData();
    
    // Ta dùng cửa sổ 100 kỳ (Khoảng 8-9 tháng) để xác định độ Hot/Cold dài hạn
    let windowSize = 100; 
    let validDraws = 0;
    
    // Thống kê số lượng bóng rơi vào "Nửa trên" (Top 22)
    let topCounts = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0};

    for (let i = windowSize; i < data.length; i++) {
        let window = data.slice(i - windowSize, i);
        let target = data[i].slice(0, 6);
        
        // Đếm tần suất 45 số trong 100 kỳ
        let freqs = Array(46).fill(0);
        for (let draw of window) {
            for (let num of draw.slice(0, 6)) {
                freqs[num]++;
            }
        }
        
        // Tạo mảng object để sort
        let arr = [];
        for (let n = 1; n <= 45; n++) {
            arr.push({num: n, count: freqs[n]});
        }
        // Sort giảm dần (Về nhiều nhất lên đầu)
        arr.sort((a, b) => b.count - a.count);
        
        // Lấy Top 22 số về nhiều nhất
        let top22 = new Set();
        for (let j = 0; j < 22; j++) {
            top22.add(arr[j].num);
        }
        
        // Xem kỳ Target có bao nhiêu số nằm trong Top 22 này
        let hitTop = 0;
        for (let num of target) {
            if (top22.has(num)) hitTop++;
        }
        
        topCounts[hitTop]++;
        validDraws++;
    }
    
    console.log(`======================================================`);
    console.log(`⚖️ PHÂN CỰC TẦN SUẤT: NHIỀU NHẤT vs ÍT NHẤT`);
    console.log(`======================================================`);
    console.log(`Quét lịch sử bằng Cửa sổ ${windowSize} kỳ để xác định Hot/Cold zone dài hạn.`);
    console.log(`Tổng số kỳ quét: ${validDraws}`);
    
    console.log(`\nTỶ LỆ PHÂN BỔ CỦA 6 QUẢ BÓNG TRÚNG JACKPOT:`);
    console.log(`(Tỷ lệ: Số lượng quả bóng lấy từ TOP 22 Về Nhiều / Số lượng từ BOTTOM 23 Về Ít)`);
    
    let combinations = [
        {name: "6 Nhiều - 0 Ít", hits: topCounts[6]},
        {name: "5 Nhiều - 1 Ít", hits: topCounts[5]},
        {name: "4 Nhiều - 2 Ít", hits: topCounts[4]},
        {name: "3 Nhiều - 3 Ít", hits: topCounts[3]},
        {name: "2 Nhiều - 4 Ít", hits: topCounts[2]},
        {name: "1 Nhiều - 5 Ít", hits: topCounts[1]},
        {name: "0 Nhiều - 6 Ít", hits: topCounts[0]},
    ];
    
    let cum = 0;
    for (let c of combinations) {
        let pct = (c.hits / validDraws * 100).toFixed(2);
        cum += parseFloat(pct);
        console.log(`- ${c.name}: ${pct}% (${c.hits} lần)`);
    }
    
    // ĐO SỨC MẠNH LỌC
    let totalRandom = 100000;
    let passed = 0;
    
    // Tính tần suất của 100 kỳ cuối cùng hiện tại
    let window = data.slice(data.length - 100, data.length);
    let freqs = Array(46).fill(0);
    for (let draw of window) {
        for (let num of draw.slice(0, 6)) freqs[num]++;
    }
    let arr = [];
    for (let n = 1; n <= 45; n++) arr.push({num: n, count: freqs[n]});
    arr.sort((a, b) => b.count - a.count);
    let currentTop22 = new Set();
    for (let j = 0; j < 22; j++) currentTop22.add(arr[j].num);
    
    for (let i = 0; i < totalRandom; i++) {
        let pool = [];
        while(pool.length < 6) {
            let r = Math.floor(Math.random() * 45) + 1;
            if (!pool.includes(r)) pool.push(r);
        }
        
        let hitTop = 0;
        for (let num of pool) if (currentTop22.has(num)) hitTop++;
        
        // Luật 1: Loại bỏ sự phân cực tuyệt đối (6-0 và 0-6 và 5-1 và 1-5 nếu hiếm)
        // Dựa vào KQ test, ta sẽ chỉ cho phép 2-4, 3-3, 4-2
        if (hitTop >= 2 && hitTop <= 4) {
            passed++;
        }
    }
    
    console.log(`\n======================================================`);
    console.log(`🛑 SỨC MẠNH BỘ LỌC CÂN BẰNG TẦN SUẤT (FREQUENCY POLARITY FILTER)`);
    console.log(`======================================================`);
    console.log(`Luật áp dụng: Tấm vé PHẢI lấy sự pha trộn cân bằng (Từ 2 đến 4 số Về Nhiều, còn lại là Về Ít).`);
    console.log(`CẤM Tấm vé thiên vị tuyệt đối (Lấy 5-6 số từ 1 phe).`);
    console.log(`Phát sinh ngẫu nhiên ${totalRandom} vé (để test lọc):`);
    console.log(`- Số vé CÒN SỐNG   : ${passed} vé`);
    console.log(`- Số vé BỊ CHÉM BAY: ${totalRandom - passed} vé`);
    console.log(`=> Lọc Cân bằng Tần suất diệt thêm được ${((totalRandom - passed) / totalRandom * 100).toFixed(2)}% vé rác!`);
}

main().catch(console.error);
