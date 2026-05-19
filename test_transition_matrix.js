/**
 * PHÂN TÍCH MA TRẬN CHUYỂN TRẠNG THÁI (MARKOV TRANSITION MATRIX)
 * Cho mỗi cột (0-5): Nếu số kỳ trước là X, số kỳ này là Y.
 * Lưu lại toàn bộ các cặp chuyển đổi (X -> Y) đã từng xảy ra trong lịch sử.
 */
const https = require('https');
function fetchData() { return new Promise((res, rej) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', r => { let d='';r.on('data',c=>d+=c);r.on('end',()=>{const dr=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)dr.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}res(dr);});r.on('error',rej);}); }); }

async function main() {
    const data = await fetchData();
    
    console.log(`======================================================`);
    console.log(`🔗 PHÂN TÍCH MA TRẬN CHUYỂN TRẠNG THÁI (MARKOV CHAIN)`);
    console.log(`======================================================`);
    
    // Bước 1: Xây dựng Ma trận Chuyển đổi (Train on first N draws)
    // Để backtest chính xác, ta sẽ test trên 200 kỳ cuối cùng.
    let testSize = 200;
    let trainData = data.slice(0, data.length - testSize);
    let testData = data.slice(data.length - testSize);
    
    // Transitions[col][prev_num] = Set of next_nums
    let transitions = Array(6).fill().map(() => ({}));
    
    for (let i = 1; i < trainData.length; i++) {
        let prev = trainData[i-1];
        let curr = trainData[i];
        for (let c = 0; c < 6; c++) {
            let pVal = prev[c];
            let cVal = curr[c];
            if (!transitions[c][pVal]) transitions[c][pVal] = new Set();
            transitions[c][pVal].add(cVal);
        }
    }
    
    // Thống kê độ lớn của tập hợp tiếp theo
    let avgFollowers = [];
    for (let c = 0; c < 6; c++) {
        let sumSizes = 0;
        let countKeys = 0;
        for (let k in transitions[c]) {
            sumSizes += transitions[c][k].size;
            countKeys++;
        }
        avgFollowers.push((sumSizes / countKeys).toFixed(1));
    }
    console.log(`- Trung bình số lượng kết quả nối tiếp từng cột: ${avgFollowers.join(', ')}`);
    
    // Bước 2: Backtest trên 200 kỳ cuối cùng
    // Xem tỷ lệ trúng thực tế nếu ta ÉP BUỘC các cột phải đi theo đường dẫn lịch sử
    let strictPass = 0; // Cả 6 cột đều nằm trong lịch sử
    let passCount = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}; // Số lượng cột pass
    
    // Update transitions as we go (Walk-forward)
    let currentTransitions = JSON.parse(JSON.stringify(transitions));
    // Convert arrays back to sets for the copy (since JSON doesn't handle Sets)
    for(let c=0; c<6; c++) {
        let newDict = {};
        for(let k in transitions[c]) newDict[k] = new Set([...transitions[c][k]]);
        currentTransitions[c] = newDict;
    }
    
    for (let i = 0; i < testData.length; i++) {
        let prev = i === 0 ? trainData[trainData.length-1] : testData[i-1];
        let curr = testData[i];
        
        let colsPassed = 0;
        for (let c = 0; c < 6; c++) {
            let pVal = prev[c];
            let cVal = curr[c];
            
            let allowedNext = currentTransitions[c][pVal] || new Set();
            if (allowedNext.has(cVal)) {
                colsPassed++;
            }
            
            // Add to history for next step
            if (!currentTransitions[c][pVal]) currentTransitions[c][pVal] = new Set();
            currentTransitions[c][pVal].add(cVal);
        }
        
        passCount[colsPassed]++;
        if (colsPassed === 6) strictPass++;
    }
    
    console.log(`\n🔙 KẾT QUẢ BACKTEST 200 KỲ THỰC TẾ GẦN NHẤT:`);
    console.log(`(Nếu bắt buộc cột tiếp theo phải nằm trong các số đã từng xuất hiện sau nó)`);
    for (let i = 0; i <= 6; i++) {
        let pct = (passCount[i] / testSize * 100).toFixed(2);
        console.log(`- Số kỳ có ${i}/6 cột đi theo lối cũ: ${pct}%`);
    }
    
    console.log(`\n=> CẢNH BÁO: Tỷ lệ trúng nếu Ép Buộc 6/6 cột tuân theo lịch sử chỉ là ${(strictPass/testSize*100).toFixed(2)}%!`);
    console.log(`(Điều này có nghĩa là "Đường Rẽ Mới" - New Transitions liên tục xuất hiện).`);
    console.log(`=> Giải pháp tối ưu: Chỉ yêu cầu TỐI THIỂU 3 cột HOẶC 4 cột đi theo lối cũ, cho phép 2-3 cột rẽ nhánh mới.`);
    
    // Bước 3: Tính sức mạnh Lọc (Test trên ngẫu nhiên)
    let totalRandom = 100000;
    let passed3 = 0;
    let passed4 = 0;
    
    let lastDraw = data[data.length-1];
    
    for (let i = 0; i < totalRandom; i++) {
        let pool = [];
        while(pool.length < 6) {
            let r = Math.floor(Math.random() * 45) + 1;
            if (!pool.includes(r)) pool.push(r);
        }
        pool.sort((a,b)=>a-b);
        
        let colsPassed = 0;
        for (let c = 0; c < 6; c++) {
            let pVal = lastDraw[c];
            let cVal = pool[c];
            let allowedNext = currentTransitions[c][pVal] || new Set();
            if (allowedNext.has(cVal)) colsPassed++;
        }
        
        if (colsPassed >= 3) passed3++;
        if (colsPassed >= 4) passed4++;
    }
    
    console.log(`\n======================================================`);
    console.log(`🛑 SỨC MẠNH BỘ LỌC ĐƯỜNG RẼ MARKOV (MARKOV TRANSITION FILTER)`);
    console.log(`======================================================`);
    console.log(`Phát sinh ngẫu nhiên ${totalRandom} vé:`);
    console.log(`Luật 1: Đòi hỏi ÍT NHẤT 3 CỘT phải tuân theo lịch sử (Pass >= 3)`);
    console.log(`- Số vé CÒN SỐNG   : ${passed3} vé`);
    console.log(`- Số vé BỊ CHÉM BAY: ${totalRandom - passed3} vé`);
    console.log(`=> Lọc thêm được ${((totalRandom - passed3) / totalRandom * 100).toFixed(2)}% vé rác!`);
    
    console.log(`\nLuật 2: Đòi hỏi ÍT NHẤT 4 CỘT phải tuân theo lịch sử (Pass >= 4)`);
    console.log(`- Số vé CÒN SỐNG   : ${passed4} vé`);
    console.log(`- Số vé BỊ CHÉM BAY: ${totalRandom - passed4} vé`);
    console.log(`=> Lọc thêm được ${((totalRandom - passed4) / totalRandom * 100).toFixed(2)}% vé rác!`);
}

main().catch(console.error);
