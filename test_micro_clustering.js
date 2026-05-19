/**
 * BÀN CỜ CẤP 2 (SUB-GRID MATRIX): ZOOM VÀO 1 Ô ĐỂ TÌM 20 VÉ
 * - Cấp 1: 8x8 (Modulo X, Modulo Y) -> Giới hạn xuống còn ~30k vé.
 * - Cấp 2: Chia nhỏ ô 30k vé này bằng:
 *   + Trục Ngang Phụ (Sub-X): Độ giãn cách 2 đầu (B6 - B1).
 *   + Trục Dọc Phụ (Sub-Y): Tổng 2 quả bóng lõi giữa (B3 + B4).
 */
const https = require('https');
function fetchData() { return new Promise((res, rej) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', r => { let d='';r.on('data',c=>d+=c);r.on('end',()=>{const dr=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)dr.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}res(dr);});r.on('error',rej);}); }); }

async function main() {
    const data = await fetchData();
    
    // Giả sử ta nhắm vào Ô A4 (X=0, Y=3) - Là Ô chứa nhiều Jackpot nhất
    let target_mod_x = 0;
    let target_mod_y = 3;
    
    let subGrid = {};
    let totalInSquare = 0;
    
    for (let draw of data) {
        let first3 = draw[0] + draw[1] + draw[2];
        let last3 = draw[3] + draw[4] + draw[5];
        
        let x = first3 % 8;
        let y = last3 % 8;
        
        if (x === target_mod_x && y === target_mod_y) {
            totalInSquare++;
            
            // Tính Tọa độ Cấp 2 (Micro-Coordinates)
            let delta = draw[5] - draw[0]; // Độ giãn cách (Sub-X)
            let midSum = draw[2] + draw[3]; // Tổng 2 bóng giữa (Sub-Y)
            
            let subCoord = `${delta}-${midSum}`;
            if (!subGrid[subCoord]) subGrid[subCoord] = [];
            subGrid[subCoord].push(draw);
        }
    }
    
    console.log(`======================================================`);
    console.log(`🔬 BÀN CỜ CẤP 2 (MICRO-GRID): ZOOM VÀO Ô A4`);
    console.log(`======================================================`);
    console.log(`Có tổng cộng ${totalInSquare} Jackpot lịch sử đã nổ trong Ô A4.`);
    
    let keys = Object.keys(subGrid);
    console.log(`Bằng cách dùng (Độ giãn cách) và (Tổng Lõi), ta đã chẻ Ô A4 này thành ${keys.length} Ô CON (Sub-Squares).`);
    
    let avg = totalInSquare / keys.length;
    console.log(`Trung bình mỗi Ô Con chứa khoảng: ${avg.toFixed(2)} vé.`);
    
    // Thử tạo ra 2 triệu vé giả lập để xem mỗi Ô Con chứa chính xác bao nhiêu vé trong thực tế
    console.log(`\n⏳ Đang mô phỏng quét 2.000.000 vé ngẫu nhiên để đo sức chứa thực tế của các Ô Con...`);
    
    let simSubGrid = {};
    let simTotal = 0;
    // Lấy 2 triệu tổ hợp ngẫu nhiên (hoặc duyệt nhanh 2 triệu)
    for(let i=0; i<1000000; i++) { // Chạy 1M cho lẹ
        let combo = [];
        let pool = Array.from({length: 45}, (_, i) => i + 1);
        for(let j=0; j<6; j++) {
            let idx = Math.floor(Math.random() * pool.length);
            combo.push(pool[idx]);
            pool.splice(idx, 1);
        }
        combo.sort((a,b) => a-b);
        
        let x = (combo[0]+combo[1]+combo[2]) % 8;
        let y = (combo[3]+combo[4]+combo[5]) % 8;
        
        if (x === target_mod_x && y === target_mod_y) {
            simTotal++;
            let delta = combo[5] - combo[0];
            let midSum = combo[2] + combo[3];
            let subCoord = `${delta}-${midSum}`;
            if (!simSubGrid[subCoord]) simSubGrid[subCoord] = 0;
            simSubGrid[subCoord]++;
        }
    }
    
    console.log(`Trong 1.000.000 vé ngẫu nhiên, có ${simTotal} vé lọt vào Ô A4.`);
    let simKeys = Object.keys(simSubGrid);
    console.log(`Ô A4 được chẻ thành ${simKeys.length} Ô CON.`);
    
    let simArr = [];
    for(let k in simSubGrid) {
        simArr.push({coord: k, count: simSubGrid[k]});
    }
    simArr.sort((a,b) => b.count - a.count);
    
    console.log(`\n🔥 KÍCH THƯỚC CỦA CÁC Ô CON (Số vé bị nhốt trong mỗi Ô Con):`);
    for(let i=0; i<10; i++) {
        if(i >= simArr.length) break;
        let [delta, midSum] = simArr[i].coord.split('-');
        console.log(`  Ô Con [Giãn cách: ${delta}, Tổng giữa: ${midSum}]: Nhốt chính xác ${simArr[i].count * 2} vé (Ước tính quy đổi cho 2M vé)`);
    }
}

main().catch(console.error);
