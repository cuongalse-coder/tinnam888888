/**
 * PHÂN TÍCH BÀN CỜ VÂY (GO BOARD ANALYSIS)
 * Ánh xạ không gian 45 số thành bàn cờ 5x9.
 * Kỳ trước (N-1) là 6 quân ĐEN.
 * Kỳ hiện tại (N) là 6 quân TRẮNG.
 * Đo lường số quân Trắng rơi vào "Khí" (Liberties) của quân Đen.
 */
const https = require('https');
function fetchData() { return new Promise((res, rej) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', r => { let d='';r.on('data',c=>d+=c);r.on('end',()=>{const dr=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)dr.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}res(dr);});r.on('error',rej);}); }); }

async function main() {
    const data = await fetchData();
    
    // Board is 5 rows, 9 cols. Numbers 1 to 45.
    // row = Math.floor((n-1)/9)
    // col = (n-1) % 9
    
    let stats = {
        overlap: {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}, // White on Black
        contact: {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}, // White on Liberties of Black
        empty: {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}    // White on Empty Space
    };

    for (let i = 1; i < data.length; i++) {
        let blackStones = data[i-1];
        let whiteStones = data[i];
        
        // Calculate Liberties (Khí) of Black Stones
        let liberties = new Set();
        let blackSet = new Set(blackStones);
        
        for (let b of blackStones) {
            let r = Math.floor((b-1)/9);
            let c = (b-1)%9;
            
            // Up (r-1, c)
            if (r > 0) liberties.add((r-1)*9 + c + 1);
            // Down (r+1, c)
            if (r < 4) liberties.add((r+1)*9 + c + 1);
            // Left (r, c-1)
            if (c > 0) liberties.add(r*9 + (c-1) + 1);
            // Right (r, c+1)
            if (c < 8) liberties.add(r*9 + (c+1) + 1);
        }
        
        // Remove liberties that are actually Black stones (stones can't be played on occupied points)
        for (let b of blackStones) liberties.delete(b);
        
        let overlapCount = 0;
        let contactCount = 0;
        let emptyCount = 0;
        
        for (let w of whiteStones) {
            if (blackSet.has(w)) overlapCount++;
            else if (liberties.has(w)) contactCount++;
            else emptyCount++;
        }
        
        stats.overlap[overlapCount] = (stats.overlap[overlapCount] || 0) + 1;
        stats.contact[contactCount] = (stats.contact[contactCount] || 0) + 1;
        stats.empty[emptyCount] = (stats.empty[emptyCount] || 0) + 1;
    }
    
    let totalDraws = data.length - 1;
    
    console.log('======================================================');
    console.log('⚪⚫ PHÂN TÍCH BÀN CỜ VÂY LỒNG CẦU (GO BOARD DYNAMICS) ⚫⚪');
    console.log('======================================================');
    console.log(`Dữ liệu: ${totalDraws} ván cờ (kỳ quay).`);
    
    console.log('\n1. ĐÁNH TRÙNG QUÂN (Overlap - Rơi lại số kỳ trước):');
    for (let i = 0; i <= 6; i++) {
        let pct = (stats.overlap[i] / totalDraws * 100).toFixed(2);
        console.log(`- Có ${i} quân rơi lại: ${pct}%`);
    }
    
    console.log('\n2. ĐÁNH ÁP SÁT (Contact Play - Rơi vào "Khí" của quân kỳ trước):');
    for (let i = 0; i <= 6; i++) {
        let pct = (stats.contact[i] / totalDraws * 100).toFixed(2);
        console.log(`- Có ${i} quân đánh áp sát: ${pct}%`);
    }
    
    console.log('\n3. ĐÁNH KHOẢNG TRỐNG (Empty Space - Không liên quan quân kỳ trước):');
    for (let i = 0; i <= 6; i++) {
        let pct = (stats.empty[i] / totalDraws * 100).toFixed(2);
        console.log(`- Có ${i} quân đánh ra khoảng trống: ${pct}%`);
    }
    
    // TÍNH TOÁN SỨC MẠNH LỌC
    let totalRandom = 100000;
    let passed = 0;
    
    // Giả lập kỳ trước là 1 bộ số cố định (ví dụ kỳ vừa rồi)
    let lastDraw = data[data.length-1];
    let blackSet = new Set(lastDraw);
    let liberties = new Set();
    for (let b of lastDraw) {
        let r = Math.floor((b-1)/9);
        let c = (b-1)%9;
        if (r > 0) liberties.add((r-1)*9 + c + 1);
        if (r < 4) liberties.add((r+1)*9 + c + 1);
        if (c > 0) liberties.add(r*9 + (c-1) + 1);
        if (c < 8) liberties.add(r*9 + (c+1) + 1);
    }
    for (let b of lastDraw) liberties.delete(b);
    
    for (let i = 0; i < totalRandom; i++) {
        let pool = [];
        while(pool.length < 6) {
            let r = Math.floor(Math.random() * 45) + 1;
            if (!pool.includes(r)) pool.push(r);
        }
        
        let overlap = 0;
        let contact = 0;
        for (let w of pool) {
            if (blackSet.has(w)) overlap++;
            else if (liberties.has(w)) contact++;
        }
        
        // LUẬT CỜ VÂY (GO RULES):
        // 1. Không bao giờ rơi lại >= 3 quân của kỳ trước.
        // 2. Không bao giờ đánh áp sát >= 5 quân vào Khí của kỳ trước.
        if (overlap <= 2 && contact <= 4) {
            passed++;
        }
    }
    
    console.log('\n======================================================');
    console.log('🛑 SỨC MẠNH BỘ LỌC ĐỊA BÀN CỜ VÂY (GO BOARD FILTER)');
    console.log('======================================================');
    console.log(`Phát sinh ngẫu nhiên ${totalRandom} vé:`);
    console.log(`- Số vé CÒN SỐNG   : ${passed} vé`);
    console.log(`- Số vé BỊ CHÉM BAY: ${totalRandom - passed} vé`);
    console.log(`=> Lọc thêm được ${((totalRandom - passed) / totalRandom * 100).toFixed(2)}% tổ hợp vi phạm luật Tương Tác Địa Bàn!`);
}

main().catch(console.error);
