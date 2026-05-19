/**
 * PHÂN TÍCH MẬT MÃ CHỮ CÁI (ALPHABET DECADE CIPHER)
 * 1-9 -> A, 10-19 -> B, 20-29 -> C, 30-39 -> D, 40-45 -> E
 * Tìm cấu trúc xếp chữ của các giải Jackpot và đếm tần suất trùng lặp.
 */
const https = require('https');
function fetchData() { return new Promise((res, rej) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', r => { let d='';r.on('data',c=>d+=c);r.on('end',()=>{const dr=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)dr.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}res(dr);});r.on('error',rej);}); }); }

function getCipher(combo) {
    let word = "";
    for (let x of combo) {
        if (x <= 9) word += "A";
        else if (x <= 19) word += "B";
        else if (x <= 29) word += "C";
        else if (x <= 39) word += "D";
        else word += "E";
    }
    return word; // Vì mảng đã sort nên chữ cái cũng tự sort thành ABCDE
}

async function main() {
    const data = await fetchData();
    
    let patternCounts = {};
    let totalDraws = data.length;
    
    for (let draw of data) {
        let word = getCipher(draw);
        patternCounts[word] = (patternCounts[word] || 0) + 1;
    }
    
    // Sort patterns by frequency
    let arr = [];
    for (let p in patternCounts) {
        arr.push({ word: p, count: patternCounts[p] });
    }
    arr.sort((a, b) => b.count - a.count);
    
    console.log(`======================================================`);
    console.log(`🔠 MÃ HÓA THẬP KỶ (ALPHABET CIPHER ANALYSIS)`);
    console.log(`======================================================`);
    console.log(`Tổng số giải Jackpot đã quét: ${totalDraws}`);
    console.log(`Theo lý thuyết có tối đa 210 Cách Xếp Chữ (Word Configurations).`);
    console.log(`Thực tế, hệ thống ghi nhận được ${arr.length} cách xếp chữ khác nhau từng trúng giải.\n`);
    
    let top10 = 0;
    for (let i = 0; i < 10; i++) top10 += arr[i].count;
    let top20 = 0;
    for (let i = 0; i < 20; i++) top20 += arr[i].count;
    let top40 = 0;
    for (let i = 0; i < 40 && i < arr.length; i++) top40 += arr[i].count;
    let top60 = 0;
    for (let i = 0; i < 60 && i < arr.length; i++) top60 += arr[i].count;
    
    console.log(`ĐỘ TẬP TRUNG CỦA MẬT MÃ (OVERLAP CONCENTRATION):`);
    console.log(`- Top 10 Mật mã phổ biến nhất   : Bao phủ ${(top10/totalDraws*100).toFixed(2)}% toàn bộ giải thưởng.`);
    console.log(`- Top 20 Mật mã phổ biến nhất   : Bao phủ ${(top20/totalDraws*100).toFixed(2)}% toàn bộ giải thưởng.`);
    console.log(`- Top 40 Mật mã phổ biến nhất   : Bao phủ ${(top40/totalDraws*100).toFixed(2)}% toàn bộ giải thưởng.`);
    console.log(`- Top 60 Mật mã phổ biến nhất   : Bao phủ ${(top60/totalDraws*100).toFixed(2)}% toàn bộ giải thưởng.`);
    
    console.log(`\n📋 DANH SÁCH TOP 15 MẬT MÃ "VUA" TRÚNG NHIỀU NHẤT:`);
    for (let i = 0; i < 15; i++) {
        let pct = (arr[i].count / totalDraws * 100).toFixed(2);
        console.log(`  ${i+1}. Mật mã [${arr[i].word}] -> Trúng ${arr[i].count} lần (${pct}%)`);
    }
    
    // ĐO SỨC MẠNH LỌC
    let totalRandom = 100000;
    let passedTop60 = 0;
    let passedHistorical = 0; // Bất kỳ mẫu nào đã từng ra
    
    // Lưu tập các mẫu Top 60
    let top60Set = new Set();
    for(let i=0; i<60 && i<arr.length; i++) top60Set.add(arr[i].word);
    let historicalSet = new Set(arr.map(x => x.word));
    
    for (let i = 0; i < totalRandom; i++) {
        let pool = [];
        while(pool.length < 6) {
            let r = Math.floor(Math.random() * 45) + 1;
            if (!pool.includes(r)) pool.push(r);
        }
        pool.sort((a,b)=>a-b);
        
        let word = getCipher(pool);
        if (top60Set.has(word)) passedTop60++;
        if (historicalSet.has(word)) passedHistorical++;
    }
    
    console.log(`\n======================================================`);
    console.log(`🛑 SỨC MẠNH BỘ LỌC MẬT MÃ CHỮ CÁI (ALPHABET CIPHER FILTER)`);
    console.log(`======================================================`);
    console.log(`Phát sinh ngẫu nhiên ${totalRandom} vé (để test lọc):`);
    
    console.log(`\nLUẬT 1: Chỉ cho phép vé có Mật mã nằm trong TOP 60 mẫu phổ biến nhất`);
    console.log(`(Hy sinh ${(100 - (top60/totalDraws*100)).toFixed(2)}% cơ hội trúng để bóp nghẹt Không gian)`);
    console.log(`- Số vé CÒN SỐNG   : ${passedTop60} vé`);
    console.log(`- Số vé BỊ CHÉM BAY: ${totalRandom - passedTop60} vé => Diệt được ${((totalRandom - passedTop60) / totalRandom * 100).toFixed(2)}% vé rác!`);
    
    console.log(`\nLUẬT 2: Chỉ cho phép vé có Mật mã ĐÃ TỪNG XUẤT HIỆN trong lịch sử`);
    console.log(`(Bảo toàn 100% tỷ lệ trúng thực tế)`);
    console.log(`- Số vé CÒN SỐNG   : ${passedHistorical} vé`);
    console.log(`- Số vé BỊ CHÉM BAY: ${totalRandom - passedHistorical} vé => Diệt được ${((totalRandom - passedHistorical) / totalRandom * 100).toFixed(2)}% vé rác!`);
}

main().catch(console.error);
