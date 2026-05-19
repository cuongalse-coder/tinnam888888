/**
 * PHÂN TÍCH MẬT MÃ VỊ TRÍ CHẴN LẺ / CAO THẤP / TĂNG GIẢM
 * Phân tích các mô hình chính xác theo từng vị trí (Positional Signatures).
 */
const https = require('https');
function fetchData() { return new Promise((res, rej) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', r => { let d='';r.on('data',c=>d+=c);r.on('end',()=>{const dr=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)dr.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}res(dr);});r.on('error',rej);}); }); }

function getOddEvenSig(combo) {
    return combo.map(x => x % 2 === 1 ? 'O' : 'E').join(''); // Odd=O, Even=E
}

function getHighLowSig(combo) {
    return combo.map(x => x >= 23 ? 'H' : 'L').join(''); // High=H, Low=L
}

function getDeltaTrendSig(combo) {
    let deltas = [];
    for(let i=1; i<6; i++) deltas.push(combo[i] - combo[i-1]);
    let trend = "";
    for(let i=1; i<5; i++) {
        if(deltas[i] > deltas[i-1]) trend += "U"; // Lớn hơn -> Tăng (Up)
        else if(deltas[i] < deltas[i-1]) trend += "D"; // Nhỏ hơn -> Giảm (Down)
        else trend += "S"; // Bằng nhau -> Đứng im (Same)
    }
    return trend;
}

async function main() {
    const data = await fetchData();
    
    let oeCounts = {};
    let hlCounts = {};
    let dtCounts = {};
    let totalDraws = data.length;
    
    for (let draw of data) {
        let oe = getOddEvenSig(draw);
        let hl = getHighLowSig(draw);
        let dt = getDeltaTrendSig(draw);
        
        oeCounts[oe] = (oeCounts[oe] || 0) + 1;
        hlCounts[hl] = (hlCounts[hl] || 0) + 1;
        dtCounts[dt] = (dtCounts[dt] || 0) + 1;
    }
    
    let sortStats = (obj) => {
        let arr = [];
        for(let k in obj) arr.push({key: k, count: obj[k]});
        arr.sort((a,b) => b.count - a.count);
        return arr;
    };
    
    let oeArr = sortStats(oeCounts);
    let hlArr = sortStats(hlCounts);
    let dtArr = sortStats(dtCounts);
    
    console.log(`======================================================`);
    console.log(`☯️ MẬT MÃ VỊ TRÍ CHẴN LẺ / CAO THẤP / TĂNG GIẢM`);
    console.log(`======================================================`);
    console.log(`Tổng số giải Jackpot đã quét: ${totalDraws}`);
    
    console.log(`\n1. MẬT MÃ CHẴN LẺ THEO VỊ TRÍ (Odd/Even Positional Signature):`);
    console.log(`- Lý thuyết: Tối đa 64 Mẫu`);
    console.log(`- Thực tế  : Xuất hiện ${oeArr.length}/64 mẫu`);
    let oeBottom10 = oeArr.slice(-10);
    console.log(`- Các mẫu cực kỳ hiếm (Gần như không bao giờ ra): ${oeBottom10.map(x => x.key).join(', ')}`);
    
    console.log(`\n2. MẬT MÃ CAO THẤP THEO VỊ TRÍ (High/Low Positional Signature):`);
    console.log(`- Lý thuyết: Tối đa 64 Mẫu`);
    console.log(`- Thực tế  : Xuất hiện ${hlArr.length}/64 mẫu`);
    console.log(`- Danh sách TOP 10 mẫu thống trị:`);
    for(let i=0; i<10 && i<hlArr.length; i++) {
        let pct = (hlArr[i].count / totalDraws * 100).toFixed(2);
        console.log(`  ${i+1}. [${hlArr[i].key}] -> ${pct}%`);
    }
    console.log(`- LƯU Ý KỲ LẠ: Do bóng xếp tăng dần, các mẫu HHHLLL, HLHLHL là KHÔNG THỂ XẢY RA!`);
    
    console.log(`\n3. MẬT MÃ TĂNG GIẢM KHOẢNG CÁCH (Delta Trend Signature):`);
    console.log(`- Lý thuyết: Tối đa 81 Mẫu (U/D/S)`);
    console.log(`- Thực tế  : Xuất hiện ${dtArr.length}/81 mẫu`);
    let dtTop5 = dtArr.slice(0,5);
    console.log(`- TOP 5 Nhịp đập Tăng/Giảm phổ biến nhất: ${dtTop5.map(x => `${x.key} (${(x.count/totalDraws*100).toFixed(1)}%)`).join(', ')}`);
    
    // ĐO SỨC MẠNH LỌC BẰNG CHỮ KÝ LỊCH SỬ (HISTORICAL SIGNATURE FILTER)
    let totalRandom = 100000;
    let passed = 0;
    
    // Tạo tập Set các mẫu đã từng xuất hiện (để không giết nhầm vé thật)
    let oeHist = new Set(oeArr.map(x => x.key));
    let hlHist = new Set(hlArr.map(x => x.key));
    let dtHist = new Set(dtArr.map(x => x.key));
    
    for (let i = 0; i < totalRandom; i++) {
        let pool = [];
        while(pool.length < 6) {
            let r = Math.floor(Math.random() * 45) + 1;
            if (!pool.includes(r)) pool.push(r);
        }
        pool.sort((a,b)=>a-b);
        
        let oe = getOddEvenSig(pool);
        let hl = getHighLowSig(pool);
        let dt = getDeltaTrendSig(pool);
        
        // LUẬT LỌC: Mã định danh phải ĐÃ TỪNG xuất hiện trong lịch sử
        if (oeHist.has(oe) && hlHist.has(hl) && dtHist.has(dt)) {
            passed++;
        }
    }
    
    console.log(`\n======================================================`);
    console.log(`🛑 SỨC MẠNH BỘ LỌC CHỮ KÝ POSITIONAL (SIGNATURE FILTER)`);
    console.log(`======================================================`);
    console.log(`Luật áp dụng: Tấm vé PHẢI mang mã Chẵn/Lẻ, Cao/Thấp, và Nhịp Tăng/Giảm đã từng tồn tại.`);
    console.log(`Phát sinh ngẫu nhiên ${totalRandom} vé (để test lọc):`);
    console.log(`- Số vé CÒN SỐNG   : ${passed} vé`);
    console.log(`- Số vé BỊ CHÉM BAY: ${totalRandom - passed} vé`);
    console.log(`=> Lọc Chữ Ký Vị Trí cắt bỏ thêm được ${((totalRandom - passed) / totalRandom * 100).toFixed(2)}% Không gian vô lý!`);
}

main().catch(console.error);
