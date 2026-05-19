/**
 * MA TRẬN BÀN CỜ 8x8 CÂN BẰNG TUYỆT ĐỐI (UNIFORM 64-SQUARE CHESSBOARD)
 * Để đảm bảo KHÔNG CÓ Ô NÀO BỊ TRỐNG và số lượng vé chia đều:
 * - Trục Ngang (X): (Tổng 3 bóng ĐẦU) Modulo 8 -> Sinh ra 8 cột (0-7)
 * - Trục Dọc (Y): (Tổng 3 bóng CUỐI) Modulo 8 -> Sinh ra 8 hàng (0-7)
 */
const https = require('https');
function fetchData() { return new Promise((res, rej) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', r => { let d='';r.on('data',c=>d+=c);r.on('end',()=>{const dr=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)dr.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}res(dr);});r.on('error',rej);}); }); }

function getChessNotation(x, y) {
    let cols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'];
    return `${cols[x]}${y + 1}`; // A1 đến H8
}

async function main() {
    const data = await fetchData();
    
    let board = Array(8).fill(0).map(() => Array(8).fill(0));
    
    for (let draw of data) {
        let first3 = draw[0] + draw[1] + draw[2];
        let last3 = draw[3] + draw[4] + draw[5];
        
        let x = first3 % 8;
        let y = last3 % 8;
        
        board[x][y]++;
    }
    
    console.log(`======================================================`);
    console.log(`♟️ BÀN CỜ 64 Ô CÂN BẰNG TỔNG (MODULO MATRIX)`);
    console.log(`======================================================`);
    console.log(`Tiêu chí: Trục Ngang = (Tổng 3 bóng Đầu) chia dư 8.`);
    console.log(`         Trục Dọc = (Tổng 3 bóng Cuối) chia dư 8.`);
    
    console.log(`\n📊 BẢN ĐỒ NHIỆT (HEATMAP) PHÂN BỔ JACKPOT THỰC TẾ:`);
    
    let header = "      |";
    for(let i=0; i<8; i++) header += ` Cột ${String.fromCharCode(65+i)} |`;
    console.log(header);
    console.log("-".repeat(header.length));
    
    let emptySquares = 0;
    for(let y=0; y<8; y++) {
        let rowStr = `Hàng ${y+1} |`;
        for(let x=0; x<8; x++) {
            let countStr = board[x][y].toString().padStart(5, ' ');
            rowStr += `${countStr} |`;
            if (board[x][y] === 0) emptySquares++;
        }
        console.log(rowStr);
    }
    
    let arr = [];
    for(let x=0; x<8; x++) {
        for(let y=0; y<8; y++) {
            arr.push({ notation: getChessNotation(x, y), count: board[x][y] });
        }
    }
    arr.sort((a,b) => b.count - a.count);
    
    console.log(`\n=> CÓ BAO NHIÊU Ô TRỐNG KHÔNG CÓ VÉ NÀO? : ${emptySquares} Ô.`);
    if (emptySquares === 0) {
        console.log(`Tuyệt vời! Toàn bộ không gian được chia RẤT ĐỀU cho cả 64 Ô.`);
    }
    
    console.log(`\n🔥 TOP 5 Ô CHỨA NHIỀU JACKPOT NHẤT:`);
    for(let i=0; i<5; i++) {
        console.log(`  ${i+1}. Ô ${arr[i].notation} -> Nổ ${arr[i].count} lần`);
    }
}

main().catch(console.error);
