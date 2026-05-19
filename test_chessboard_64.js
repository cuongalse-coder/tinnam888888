/**
 * MA TRẬN BÀN CỜ 64 Ô (64-SQUARE CHESSBOARD MATRIX)
 * Biến mỗi tấm vé thành 1 chuỗi nhị phân 6-bit dựa trên tính Chẵn/Lẻ của từng vị trí.
 * Chẵn (E) = 0, Lẻ (O) = 1.
 * Ví dụ: E-O-E-O-O-E = 010110 (Hệ nhị phân) = 22 (Hệ thập phân).
 * 64 ô sẽ có ID từ 0 đến 63.
 */
const https = require('https');
function fetchData() { return new Promise((res, rej) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', r => { let d='';r.on('data',c=>d+=c);r.on('end',()=>{const dr=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)dr.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}res(dr);});r.on('error',rej);}); }); }

function getSquareId(combo) {
    let binStr = combo.map(x => (x % 2 === 1) ? '1' : '0').join('');
    return parseInt(binStr, 2); // Chuyển chuỗi nhị phân thành số thập phân (0 - 63)
}

function getChessNotation(squareId) {
    // Chuyển 0-63 thành tọa độ bàn cờ (A1 đến H8)
    let cols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'];
    let col = cols[squareId % 8];
    let row = Math.floor(squareId / 8) + 1;
    return `${col}${row}`;
}

async function main() {
    const data = await fetchData();
    
    let squareCounts = Array(64).fill(0);
    let transitionMatrix = Array.from({length: 64}, () => ({})); // Để tính xác suất di chuyển cờ
    
    let lastSquare = -1;
    
    for (let draw of data) {
        let sq = getSquareId(draw);
        squareCounts[sq]++;
        
        if (lastSquare !== -1) {
            if (!transitionMatrix[lastSquare][sq]) transitionMatrix[lastSquare][sq] = 0;
            transitionMatrix[lastSquare][sq]++;
        }
        lastSquare = sq;
    }
    
    console.log(`======================================================`);
    console.log(`♟️ MA TRẬN BÀN CỜ 64 Ô LƯỢNG TỬ (QUANTUM CHESSBOARD)`);
    console.log(`======================================================`);
    console.log(`Bằng cách gán Chẵn=0, Lẻ=1 cho 6 vị trí, ta tạo ra mã Nhị phân 6-bit.`);
    console.log(`Mỗi vé sẽ được quy đổi thành một số từ 0 đến 63, tương ứng với 64 Ô trên Bàn Cờ Vua.\n`);
    
    // Sort and find hottest squares
    let arr = [];
    for(let i=0; i<64; i++) {
        arr.push({id: i, count: squareCounts[i], notation: getChessNotation(i)});
    }
    arr.sort((a,b) => b.count - a.count);
    
    console.log(`🔥 TOP 10 Ô VÀNG (HOT SQUARES) - Nơi Jackpot rơi nhiều nhất:`);
    let top10Sum = 0;
    for(let i=0; i<10; i++) {
        let pct = (arr[i].count / data.length * 100).toFixed(2);
        let binStr = arr[i].id.toString(2).padStart(6, '0').replace(/0/g, 'C').replace(/1/g, 'L');
        console.log(`  Hạng ${i+1}: Ô ${arr[i].notation} (ID: ${arr[i].id}) - Cấu trúc: [${binStr}] -> Trúng ${arr[i].count} lần (${pct}%)`);
        top10Sum += arr[i].count;
    }
    console.log(`=> Nếu chỉ đặt cược vào TOP 10 Ô này, bạn bao phủ ${(top10Sum/data.length*100).toFixed(2)}% cơ hội trúng thưởng!\n`);
    
    console.log(`🧊 TOP 10 Ô CHẾT (DEAD SQUARES) - Nơi Jackpot gần như KHÔNG BAO GIỜ đến:`);
    let bottom10 = arr.slice(-10).reverse();
    for(let i=0; i<10; i++) {
        let binStr = bottom10[i].id.toString(2).padStart(6, '0').replace(/0/g, 'C').replace(/1/g, 'L');
        console.log(`  Chót ${i+1}: Ô ${bottom10[i].notation} (ID: ${bottom10[i].id}) - Cấu trúc: [${binStr}] -> Trúng ${bottom10[i].count} lần`);
    }
    
    console.log(`\n======================================================`);
    console.log(`🎯 DỰ ĐOÁN ĐƯỜNG ĐI CỦA JACKPOT KỲ TIẾP THEO`);
    console.log(`======================================================`);
    let currentDraw = data[data.length - 1];
    let currentSquare = getSquareId(currentDraw);
    let binStr = currentSquare.toString(2).padStart(6, '0').replace(/0/g, 'C').replace(/1/g, 'L');
    console.log(`Kỳ trước vừa nổ tại Ô: ${getChessNotation(currentSquare)} (ID: ${currentSquare}, Cấu trúc: [${binStr}])`);
    
    // Tìm các ô có khả năng cao nhất mà Jackpot sẽ nhảy tới
    let nextMoves = transitionMatrix[currentSquare];
    let movesArr = [];
    for(let nextSq in nextMoves) {
        movesArr.push({id: parseInt(nextSq), count: nextMoves[nextSq]});
    }
    movesArr.sort((a,b) => b.count - a.count);
    
    console.log(`Dựa trên dữ liệu chuỗi Markov lịch sử, Jackpot thường xuyên từ Ô ${getChessNotation(currentSquare)} nhảy sang các Ô sau:`);
    for(let i=0; i<5 && i<movesArr.length; i++) {
        let targetId = movesArr[i].id;
        let tBinStr = targetId.toString(2).padStart(6, '0').replace(/0/g, 'C').replace(/1/g, 'L');
        console.log(`  -> Ưu tiên ${i+1}: Nhảy sang Ô ${getChessNotation(targetId)} (ID: ${targetId}, Cấu trúc: [${tBinStr}]) - Tần suất: ${movesArr[i].count} lần`);
    }
}

main().catch(console.error);
