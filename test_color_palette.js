/**
 * PHÂN TÍCH BẢNG MÀU NGŨ HÀNH (5-COLOR PALETTE ANALYSIS)
 * Chia 45 số thành 5 nhóm Màu sắc/Ngũ hành dựa trên chữ số tận cùng.
 * Mỗi nhóm có chính xác 9 con số.
 * Màu 1 (Thủy): Tận cùng 1, 6
 * Màu 2 (Hỏa): Tận cùng 2, 7
 * Màu 3 (Mộc): Tận cùng 3, 8
 * Màu 4 (Kim): Tận cùng 4, 9
 * Màu 5 (Thổ): Tận cùng 5, 0
 */
const https = require('https');
function fetchData() { return new Promise((res, rej) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', r => { let d='';r.on('data',c=>d+=c);r.on('end',()=>{const dr=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)dr.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}res(dr);});r.on('error',rej);}); }); }

async function main() {
    const data = await fetchData();
    
    let maxColorCounts = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0};
    let uniqueColorCounts = {1:0, 2:0, 3:0, 4:0, 5:0};
    
    for (let d of data) {
        let colors = [0, 0, 0, 0, 0]; // 5 colors
        for (let n of d) {
            let lastDigit = n % 10;
            if (lastDigit === 1 || lastDigit === 6) colors[0]++;
            else if (lastDigit === 2 || lastDigit === 7) colors[1]++;
            else if (lastDigit === 3 || lastDigit === 8) colors[2]++;
            else if (lastDigit === 4 || lastDigit === 9) colors[3]++;
            else if (lastDigit === 5 || lastDigit === 0) colors[4]++;
        }
        
        let maxOfAnyColor = Math.max(...colors);
        maxColorCounts[maxOfAnyColor] = (maxColorCounts[maxOfAnyColor] || 0) + 1;
        
        let uniqueColors = colors.filter(c => c > 0).length;
        uniqueColorCounts[uniqueColors] = (uniqueColorCounts[uniqueColors] || 0) + 1;
    }
    
    console.log('======================================================');
    console.log('🎨 PHÂN TÍCH BẢNG MÀU XỔ SỐ (5-COLOR PALETTE)');
    console.log('======================================================');
    console.log(`Dữ liệu: ${data.length} kỳ quay thực tế.`);
    
    console.log('\n1. Số lượng MÀU KHÁC NHAU hội tụ trong 1 tấm vé (Max 5 màu):');
    for (let i = 1; i <= 5; i++) {
        let pct = ((uniqueColorCounts[i] || 0) / data.length * 100).toFixed(2);
        console.log(`- Vé chứa ${i} màu: ${pct}%`);
    }
    
    console.log('\n2. Số lượng bóng NHIỀU NHẤT cùng rớt vào 1 màu (Color Overload):');
    let cumulative = 0;
    for (let i = 1; i <= 6; i++) {
        let pct = parseFloat(((maxColorCounts[i] || 0) / data.length * 100).toFixed(2));
        cumulative += pct;
        console.log(`- Có tối đa ${i} bóng cùng màu: ${pct.toFixed(2)}% (Lũy kế: ${cumulative.toFixed(2)}%)`);
    }
    
    // TEST LỌC
    let totalRandom = 100000;
    let passed = 0;
    for (let i = 0; i < totalRandom; i++) {
        let pool = [];
        while(pool.length < 6) {
            let r = Math.floor(Math.random() * 45) + 1;
            if (!pool.includes(r)) pool.push(r);
        }
        pool.sort((a,b)=>a-b);
        
        let colors = [0, 0, 0, 0, 0];
        for (let n of pool) {
            let lastDigit = n % 10;
            if (lastDigit === 1 || lastDigit === 6) colors[0]++;
            else if (lastDigit === 2 || lastDigit === 7) colors[1]++;
            else if (lastDigit === 3 || lastDigit === 8) colors[2]++;
            else if (lastDigit === 4 || lastDigit === 9) colors[3]++;
            else if (lastDigit === 5 || lastDigit === 0) colors[4]++;
        }
        
        let unique = colors.filter(c => c > 0).length;
        let maxColor = Math.max(...colors);
        
        // LUẬT LỌC:
        // 1. Phải có từ 3 đến 5 màu khác nhau (Vé có 1 hoặc 2 màu quá cực đoan -> Vứt).
        // 2. Không có màu nào chiếm > 3 bóng (Nếu có 4 bóng cùng 1 màu -> Vứt).
        if (unique >= 3 && maxColor <= 3) {
            passed++;
        }
    }
    
    console.log('\n======================================================');
    console.log('🛑 SỨC MẠNH CỦA BỘ LỌC MÀU SẮC (COLOR OVERLOAD FILTER)');
    console.log('======================================================');
    console.log(`Luật: TẤT CẢ các vé chỉ có <= 2 màu (đơn điệu), HOẶC có >= 4 bóng tụ vào 1 màu -> LẬP TỨC CHÉM!`);
    console.log(`Phát sinh ngẫu nhiên ${totalRandom} vé:`);
    console.log(`- Số vé CÒN SỐNG   : ${passed} vé`);
    console.log(`- Số vé BỊ CHÉM BAY: ${totalRandom - passed} vé`);
    console.log(`=> Lọc thêm được ${((totalRandom - passed) / totalRandom * 100).toFixed(2)}% tổng lượng vé rác!`);
}

main().catch(console.error);
