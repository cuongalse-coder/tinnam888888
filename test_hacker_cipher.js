/**
 * PHÂN TÍCH MẬT MÃ HACKER 12-BIT (12-BIT HACKER CIPHER ANALYSIS)
 * Chuyển đổi 6 quả bóng thành chuỗi 12 chữ số.
 * Ánh xạ mỗi chữ số thành nhị phân (Binary 0/1) dựa trên tính Chẵn/Lẻ (Even=0, Odd=1).
 * Tìm kiếm các ranh giới cấu trúc nhị phân bất thường.
 */
const https = require('https');
function fetchData() { return new Promise((res, rej) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', r => { let d='';r.on('data',c=>d+=c);r.on('end',()=>{const dr=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)dr.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}res(dr);});r.on('error',rej);}); }); }

async function main() {
    const data = await fetchData();
    
    let stats = {
        total_zeros: {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0},
        max_consec_0: {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0},
        max_consec_1: {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0},
        palindromes: 0,
        alternating_runs: 0 // Chuỗi dao động liên tục kiểu 10101010
    };

    for (let draw of data) {
        // Build 12-digit string
        let s = "";
        for (let num of draw) {
            s += num.toString().padStart(2, '0');
        }
        
        // Convert to 12-bit string (Even=0, Odd=1)
        let bin = "";
        let zeros = 0;
        for (let i = 0; i < 12; i++) {
            let b = parseInt(s[i]) % 2 === 0 ? "0" : "1";
            bin += b;
            if (b === "0") zeros++;
        }
        
        stats.total_zeros[zeros]++;
        
        // Max consecutive 0s
        let max0 = 0;
        let c0 = 0;
        for (let i = 0; i < 12; i++) {
            if (bin[i] === "0") { c0++; if (c0 > max0) max0 = c0; }
            else c0 = 0;
        }
        stats.max_consec_0[max0] = (stats.max_consec_0[max0] || 0) + 1;
        
        // Max consecutive 1s
        let max1 = 0;
        let c1 = 0;
        for (let i = 0; i < 12; i++) {
            if (bin[i] === "1") { c1++; if (c1 > max1) max1 = c1; }
            else c1 = 0;
        }
        stats.max_consec_1[max1] = (stats.max_consec_1[max1] || 0) + 1;
        
        // Check Palindrome (Đối xứng hoàn hảo e.g 011001100110)
        let rev = bin.split('').reverse().join('');
        if (bin === rev) stats.palindromes++;
        
        // Check alternating oscillation (e.g., 010101010101 or 101010101010)
        if (bin === "010101010101" || bin === "101010101010") stats.alternating_runs++;
    }
    
    let totalDraws = data.length;
    
    console.log(`======================================================`);
    console.log(`🕵️ MÃ HÓA HACKER 12-BIT (12-BIT CIPHER ANALYSIS)`);
    console.log(`======================================================`);
    console.log(`Dữ liệu quét: ${totalDraws} mã 12-bit từ lịch sử thực tế.\n`);
    
    console.log(`1. Chuỗi BIT 0 liên tiếp (Kéo dài các chữ số CHẴN liên tục):`);
    let cum0 = 0;
    for (let i = 1; i <= 12; i++) {
        if (!stats.max_consec_0[i]) continue;
        let pct = (stats.max_consec_0[i] / totalDraws * 100).toFixed(2);
        cum0 += parseFloat(pct);
        console.log(`- Tối đa ${i} bit 0 liên tiếp: ${pct}% (Lũy kế: ${cum0.toFixed(2)}%)`);
    }
    
    console.log(`\n2. Chuỗi BIT 1 liên tiếp (Kéo dài các chữ số LẺ liên tục):`);
    let cum1 = 0;
    for (let i = 1; i <= 12; i++) {
        if (!stats.max_consec_1[i]) continue;
        let pct = (stats.max_consec_1[i] / totalDraws * 100).toFixed(2);
        cum1 += parseFloat(pct);
        console.log(`- Tối đa ${i} bit 1 liên tiếp: ${pct}% (Lũy kế: ${cum1.toFixed(2)}%)`);
    }
    
    console.log(`\n3. Cấu trúc Hình Học Nhị Phân Bất Thường:`);
    console.log(`- Chuỗi Đối Xứng Hoàn Hảo (Palindrome): ${stats.palindromes} lần (${(stats.palindromes/totalDraws*100).toFixed(2)}%)`);
    console.log(`- Chuỗi Dao Động Cực Đoan (101010...): ${stats.alternating_runs} lần (${(stats.alternating_runs/totalDraws*100).toFixed(2)}%)`);
    
    // ĐO SỨC MẠNH LỌC NHỊ PHÂN
    let totalRandom = 100000;
    let passed = 0;
    
    for (let i = 0; i < totalRandom; i++) {
        let pool = [];
        while(pool.length < 6) {
            let r = Math.floor(Math.random() * 45) + 1;
            if (!pool.includes(r)) pool.push(r);
        }
        pool.sort((a,b)=>a-b);
        
        let s = "";
        for (let num of pool) s += num.toString().padStart(2, '0');
        
        let bin = "";
        for (let j = 0; j < 12; j++) bin += (parseInt(s[j]) % 2 === 0 ? "0" : "1");
        
        let max0 = 0, c0 = 0;
        for (let j = 0; j < 12; j++) { if (bin[j] === "0") { c0++; if (c0 > max0) max0 = c0; } else c0 = 0; }
        
        let max1 = 0, c1 = 0;
        for (let j = 0; j < 12; j++) { if (bin[j] === "1") { c1++; if (c1 > max1) max1 = c1; } else c1 = 0; }
        
        let rev = bin.split('').reverse().join('');
        let isPal = (bin === rev);
        let isAlt = (bin === "010101010101" || bin === "101010101010");
        
        // LUẬT LỌC 12-BIT HACKER:
        // Cấm kéo dài >= 7 bit 0 liên tiếp (100% lịch sử ko có)
        // Cấm kéo dài >= 7 bit 1 liên tiếp (100% lịch sử ko có)
        // Cấm cấu trúc đối xứng (Palindrome)
        // Cấm cấu trúc dao động cực đoan (010101)
        if (max0 <= 6 && max1 <= 6 && !isPal && !isAlt) {
            passed++;
        }
    }
    
    console.log(`\n======================================================`);
    console.log(`🛑 SỨC MẠNH MẬT MÃ HACKER 12-BIT (12-BIT CIPHER FILTER)`);
    console.log(`======================================================`);
    console.log(`Luật áp dụng: Tiêu diệt các vé có >= 7 bit chẵn/lẻ liên tiếp, vé đối xứng hoàn toàn, dao động cực đoan.`);
    console.log(`Phát sinh ngẫu nhiên ${totalRandom} vé 12-bit (để test lọc):`);
    console.log(`- Số vé CÒN SỐNG   : ${passed} vé`);
    console.log(`- Số vé BỊ CHÉM BAY: ${totalRandom - passed} vé`);
    console.log(`=> Mật mã Hacker loại bỏ thêm được ${((totalRandom - passed) / totalRandom * 100).toFixed(2)}% vé rác!`);
}

main().catch(console.error);
