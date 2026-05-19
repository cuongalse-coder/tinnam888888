/**
 * PHÂN TÍCH MA TRẬN RUBIK (RUBIK'S MATRIX ANALYSIS)
 * Ánh xạ 6 quả bóng lên lưới ma trận 5x10 (Hàng chục x Hàng đơn vị)
 * Tìm kiếm các hình khối (Topology) bất thường như Khối vuông 2x2, Đường chéo dài.
 */
const https = require('https');
function fetchData() { return new Promise((res, rej) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', r => { let d='';r.on('data',c=>d+=c);r.on('end',()=>{const dr=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)dr.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}res(dr);});r.on('error',rej);}); }); }

async function main() {
    const data = await fetchData();
    
    let stats = {
        square_2x2: 0,
        L_shape: 0,
        diagonal_3: 0
    };

    for (let d of data) {
        // Tạo ma trận 5x10, điền số 1 vào vị trí có bóng
        let matrix = Array(5).fill().map(() => Array(10).fill(0));
        for (let n of d) {
            let row = Math.floor(n / 10);
            let col = n % 10;
            if (row < 5 && col < 10) {
                matrix[row][col] = 1;
            }
        }
        
        // 1. Khối Vuông 2x2 (Rubik's Block 2x2)
        // Yêu cầu 4 số: x, x+1, x+10, x+11
        let has_2x2 = false;
        for (let r = 0; r < 4; r++) {
            for (let c = 0; c < 9; c++) {
                if (matrix[r][c] && matrix[r][c+1] && matrix[r+1][c] && matrix[r+1][c+1]) {
                    has_2x2 = true;
                }
            }
        }
        if (has_2x2) stats.square_2x2++;
        
        // 2. Hình chữ L (L-shape 3 cells)
        // Ví dụ: x, x+1, x+10
        let has_L = false;
        for (let r = 0; r < 4; r++) {
            for (let c = 0; c < 9; c++) {
                let count = matrix[r][c] + matrix[r][c+1] + matrix[r+1][c] + matrix[r+1][c+1];
                if (count === 3) has_L = true;
            }
        }
        if (has_L) stats.L_shape++;
        
        // 3. Đường chéo 3 ô (Diagonal 3)
        // Ví dụ: x, x+11, x+22 hoặc x, x+9, x+18
        let has_diag3 = false;
        for (let r = 0; r < 3; r++) {
            for (let c = 0; c < 8; c++) {
                // Chéo xuống phải
                if (matrix[r][c] && matrix[r+1][c+1] && matrix[r+2][c+2]) has_diag3 = true;
            }
            for (let c = 2; c < 10; c++) {
                // Chéo xuống trái
                if (matrix[r][c] && matrix[r+1][c-1] && matrix[r+2][c-2]) has_diag3 = true;
            }
        }
        if (has_diag3) stats.diagonal_3++;
    }
    
    console.log('======================================================');
    console.log('🧩 PHÂN TÍCH MA TRẬN RUBIK (RUBIK MATRIX TOPOLOGY)');
    console.log('======================================================');
    console.log(`Kiểm tra trên tổng số ${data.length} kỳ quay thực tế:`);
    console.log(`1. Khối Vuông 2x2 (Gồm 4 số chụm lại thành hình vuông): ${stats.square_2x2} kỳ (${(stats.square_2x2 / data.length * 100).toFixed(2)}%)`);
    console.log(`2. Khối Chữ L (Gồm 3 số chụm lại góc vuông): ${stats.L_shape} kỳ (${(stats.L_shape / data.length * 100).toFixed(2)}%)`);
    console.log(`3. Khối Chéo 3 (Gồm 3 số xếp thành đường chéo thẳng): ${stats.diagonal_3} kỳ (${(stats.diagonal_3 / data.length * 100).toFixed(2)}%)`);
    
    console.log('\n=> NẾU LOẠI BỎ KHỐI VUÔNG 2X2 VÀ KHỐI CHÉM 3: CHÚNG TA SẼ GIẾT ĐƯỢC BAO NHIÊU VÉ RÁC?');
    
    let totalRandom = 100000;
    let passed = 0;
    for (let i = 0; i < totalRandom; i++) {
        let pool = [];
        while(pool.length < 6) {
            let r = Math.floor(Math.random() * 45) + 1;
            if (!pool.includes(r)) pool.push(r);
        }
        pool.sort((a,b)=>a-b);
        
        let matrix = Array(5).fill().map(() => Array(10).fill(0));
        for (let n of pool) {
            let row = Math.floor(n / 10);
            let col = n % 10;
            if (row < 5 && col < 10) matrix[row][col] = 1;
        }
        
        let has_2x2 = false;
        let has_diag3 = false;
        
        for (let r = 0; r < 4; r++) {
            for (let c = 0; c < 9; c++) {
                if (matrix[r][c] && matrix[r][c+1] && matrix[r+1][c] && matrix[r+1][c+1]) has_2x2 = true;
            }
        }
        
        for (let r = 0; r < 3; r++) {
            for (let c = 0; c < 8; c++) {
                if (matrix[r][c] && matrix[r+1][c+1] && matrix[r+2][c+2]) has_diag3 = true;
            }
            for (let c = 2; c < 10; c++) {
                if (matrix[r][c] && matrix[r+1][c-1] && matrix[r+2][c-2]) has_diag3 = true;
            }
        }
        
        // CHỈ LỌC KHỐI 2X2 VÀ CHÉO 3 (KHÔNG LỌC CHỮ L VÌ NÓ XẢY RA KHÁ NHIỀU ~27%)
        if (!has_2x2 && !has_diag3) {
            passed++;
        }
    }
    
    console.log('\n======================================================');
    console.log(`Phát sinh ngẫu nhiên ${totalRandom} vé (để test lọc):`);
    console.log(`- Số vé CÒN SỐNG   : ${passed} vé`);
    console.log(`- Số vé BỊ CHÉM BAY: ${totalRandom - passed} vé`);
    console.log(`=> Sức mạnh Màng Lọc Rubik: Lọc thêm được ${((totalRandom - passed) / totalRandom * 100).toFixed(2)}% tổng lượng vé rác!`);
}

main().catch(console.error);
