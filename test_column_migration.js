/**
 * PHÂN TÍCH SỰ DỊCH CHUYỂN CỘT (COLUMN MIGRATION ANALYSIS)
 * Phân tích hành vi của các "Con số Rơi lại" (Overlap Numbers) từ kỳ trước.
 * Xem chúng thường đứng yên tại cột cũ, hay dịch chuyển sang các cột khác như thế nào.
 */
const https = require('https');
function fetchData() { return new Promise((res, rej) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', r => { let d='';r.on('data',c=>d+=c);r.on('end',()=>{const dr=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)dr.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}res(dr);});r.on('error',rej);}); }); }

async function main() {
    const data = await fetchData();
    
    let shiftStats = {
        '-5': 0, '-4': 0, '-3': 0, '-2': 0, '-1': 0,
        '0': 0,
        '1': 0, '2': 0, '3': 0, '4': 0, '5': 0
    };
    
    let totalOverlaps = 0;
    
    for (let i = 1; i < data.length; i++) {
        let prevDraw = data[i-1].slice(0,6);
        let currDraw = data[i].slice(0,6);
        
        for (let oldCol = 0; oldCol < 6; oldCol++) {
            let num = prevDraw[oldCol];
            let newCol = currDraw.indexOf(num);
            if (newCol !== -1) { // Có rơi lại
                let shift = newCol - oldCol;
                shiftStats[shift.toString()]++;
                totalOverlaps++;
            }
        }
    }
    
    console.log(`======================================================`);
    console.log(`🔄 PHÂN TÍCH SỰ DỊCH CHUYỂN CỘT (NUMBER COLUMN MIGRATION)`);
    console.log(`======================================================`);
    console.log(`Tổng số trường hợp "Con số rơi lại" (Overlap) trong lịch sử: ${totalOverlaps}`);
    
    console.log(`\nQUY LUẬT DỊCH CHUYỂN CỘT CỦA CÁC SỐ RƠI LẠI:`);
    console.log(`(Số âm: Dịch sang Trái | Số 0: Đứng im | Số dương: Dịch sang Phải)`);
    
    let shifts = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5];
    for (let s of shifts) {
        let count = shiftStats[s.toString()];
        let pct = (count / totalOverlaps * 100).toFixed(2);
        console.log(`- Dịch chuyển ${s} cột: ${pct.padStart(5)}% (${count} lần)`);
    }
    
    let stayOr1 = shiftStats['-1'] + shiftStats['0'] + shiftStats['1'];
    let stayOr2 = stayOr1 + shiftStats['-2'] + shiftStats['2'];
    
    console.log(`\n=> CÁC PHÁT HIỆN LỚN:`);
    console.log(`- Tỷ lệ Đứng im hoặc dịch +/- 1 cột: ${(stayOr1 / totalOverlaps * 100).toFixed(2)}%`);
    console.log(`- Tỷ lệ Đứng im hoặc dịch tối đa +/- 2 cột: ${(stayOr2 / totalOverlaps * 100).toFixed(2)}%`);
    console.log(`- Tỷ lệ Dịch chuyển cực đoan (>= 3 cột): ${((totalOverlaps - stayOr2) / totalOverlaps * 100).toFixed(2)}%`);
    
    // ĐO SỨC MẠNH LỌC
    let totalRandom = 100000;
    let passed = 0;
    
    let lastDraw = data[data.length-1].slice(0,6);
    
    for (let i = 0; i < totalRandom; i++) {
        let pool = [];
        while(pool.length < 6) {
            let r = Math.floor(Math.random() * 45) + 1;
            if (!pool.includes(r)) pool.push(r);
        }
        pool.sort((a,b)=>a-b);
        
        let valid = true;
        for (let newCol = 0; newCol < 6; newCol++) {
            let num = pool[newCol];
            let oldCol = lastDraw.indexOf(num);
            if (oldCol !== -1) { // Số rơi lại
                let shift = Math.abs(newCol - oldCol);
                if (shift >= 3) {
                    valid = false;
                    break;
                }
            }
        }
        
        if (valid) passed++;
    }
    
    console.log(`\n======================================================`);
    console.log(`🛑 SỨC MẠNH BỘ LỌC DỊCH CHUYỂN CỘT (COLUMN SHIFT FILTER)`);
    console.log(`======================================================`);
    console.log(`Luật áp dụng: Nếu có bất kỳ con số nào rơi lại từ kỳ trước, nó KHÔNG ĐƯỢC PHÉP nhảy cách >= 3 cột!`);
    console.log(`(Bởi vì xác suất xảy ra điều này là siêu hiếm)`);
    console.log(`Phát sinh ngẫu nhiên ${totalRandom} vé (để test lọc):`);
    console.log(`- Số vé CÒN SỐNG   : ${passed} vé`);
    console.log(`- Số vé BỊ CHÉM BAY: ${totalRandom - passed} vé`);
    console.log(`=> Lọc Dịch Chuyển Cột sẽ diệt thêm được ${((totalRandom - passed) / totalRandom * 100).toFixed(2)}% vé rác!`);
}

main().catch(console.error);
