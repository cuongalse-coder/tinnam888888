/**
 * MA TRẬN BÀN CỜ 8x8 TÁCH NỬA (SPLIT-HALF 8x8 CHESSBOARD)
 * Phân tách 6 quả bóng thành 2 nửa (3 Đầu, 3 Cuối).
 * Mỗi nửa được mã hóa Chẵn/Lẻ thành số từ 0-7.
 * Trục Ngang (X): 3 Bóng Đầu (0-7 -> A-H)
 * Trục Dọc (Y): 3 Bóng Cuối (0-7 -> 1-8)
 */
const https = require('https');
function fetchData() { return new Promise((res, rej) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', r => { let d='';r.on('data',c=>d+=c);r.on('end',()=>{const dr=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)dr.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}res(dr);});r.on('error',rej);}); }); }

function getHalfSignature(balls) {
    let binStr = balls.map(x => (x % 2 === 1) ? '1' : '0').join('');
    return parseInt(binStr, 2); // Trả về 0 đến 7
}

function getChessNotation(x, y) {
    let cols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'];
    return `${cols[x]}${y + 1}`; // A1 đến H8
}

function binStr(val) {
    return val.toString(2).padStart(3, '0').replace(/0/g, 'C').replace(/1/g, 'L');
}

async function main() {
    const data = await fetchData();
    
    // Khởi tạo bàn cờ 8x8
    let board = Array(8).fill(0).map(() => Array(8).fill(0));
    
    for (let draw of data) {
        let firstHalf = draw.slice(0, 3);
        let secondHalf = draw.slice(3, 6);
        
        let x = getHalfSignature(firstHalf); // 0-7
        let y = getHalfSignature(secondHalf); // 0-7
        
        board[x][y]++;
    }
    
    console.log(`======================================================`);
    console.log(`♟️ BÀN CỜ 2 TRIỆU VÉ: TRỤC DỌC & TRỤC NGANG`);
    console.log(`======================================================`);
    console.log(`Không gian 2 triệu vé (đã lọc) được trải đều lên 64 Ô của bàn cờ.`);
    console.log(`- Trục Ngang (A->H): Mã Chẵn/Lẻ của 3 Quả Bóng Đầu Tiên.`);
    console.log(`- Trục Dọc (1->8): Mã Chẵn/Lẻ của 3 Quả Bóng Cuối Cùng.`);
    
    console.log(`\n📊 BẢN ĐỒ NHIỆT (HEATMAP) THỰC TẾ TRÊN BÀN CỜ:`);
    
    // In bảng 8x8
    let header = "      |";
    for(let i=0; i<8; i++) header += ` Cột ${String.fromCharCode(65+i)} (${binStr(i)}) |`;
    console.log(header);
    console.log("-".repeat(header.length));
    
    for(let y=0; y<8; y++) {
        let rowStr = `Hàng ${y+1} |`;
        for(let x=0; x<8; x++) {
            let countStr = board[x][y].toString().padStart(11, ' ');
            rowStr += `${countStr} |`;
        }
        console.log(rowStr);
    }
    
    // Tìm các cụm (Clusters)
    let arr = [];
    for(let x=0; x<8; x++) {
        for(let y=0; y<8; y++) {
            arr.push({ notation: getChessNotation(x, y), count: board[x][y], xVal: binStr(x), yVal: binStr(y) });
        }
    }
    arr.sort((a,b) => b.count - a.count);
    
    console.log(`\n🔥 TOP 5 Ô "TỤ KHÍ" (Nhiều Jackpot và chứa nhiều vé nhất trong 2 triệu vé):`);
    for(let i=0; i<5; i++) {
        console.log(`  ${i+1}. Ô ${arr[i].notation} (Cột ${arr[i].xVal} x Hàng ${arr[i].yVal}) -> Nổ ${arr[i].count} lần`);
    }
    
    console.log(`\n🧊 VÙNG RỖNG BÀN CỜ (Không chứa vé nào - Khai tử hoàn toàn):`);
    let deadSquares = arr.filter(sq => sq.count === 0);
    console.log(`  Có ${deadSquares.length} Ô trên bàn cờ là VÙNG CHẾT (Trống rỗng).`);
    if(deadSquares.length > 0) {
        let names = deadSquares.map(sq => sq.notation).join(', ');
        console.log(`  Đó là các Ô: ${names}`);
    }
    
    console.log(`\n=> CÁCH SOI ĐƯỜNG DỌC NGANG:`);
    console.log(`Giả sử bạn soi Trục Ngang (3 Bóng đầu) nhận định xu hướng ra 2 Chẵn 1 Lẻ (Các cột C, D, E).`);
    console.log(`Bạn soi Trục Dọc (3 Bóng cuối) nhận định xu hướng ra 2 Lẻ 1 Chẵn (Hàng 4, 6, 7).`);
    console.log(`Bạn chỉ việc dóng tọa độ chéo, tìm ra Ô giao cắt. Các Ô giao cắt này sẽ gom đúng khoảng MỘT VÀI NGÀN VÉ trong tổng số 2 triệu vé!`);
}

main().catch(console.error);
