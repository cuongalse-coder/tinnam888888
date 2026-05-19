/**
 * PHÂN TÍCH KHOẢNG CÁCH CHỮ SỐ (DIGIT DISTANCE ANALYSIS)
 * 12 Cột chữ số. Đo lường khoảng cách giữa các chữ số trùng nhau.
 */
const https = require('https');
function fetchData() { return new Promise((res, rej) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', r => { let d='';r.on('data',c=>d+=c);r.on('end',()=>{const dr=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)dr.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}res(dr);});r.on('error',rej);}); }); }

async function main() {
    const data = await fetchData();
    
    let maxAdjacentPairsCount = { 0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0 };
    let maxDistanceDistribution = {};

    for (let d of data) {
        let str = "";
        for (let num of d) {
            str += num.toString().padStart(2, '0');
        }
        // str has length 12
        
        // 1. Đếm số lượng cặp chữ số đứng liền kề giống nhau (Khoảng cách = 1)
        let adjacentCount = 0;
        for (let i = 1; i < 12; i++) {
            if (str[i] === str[i-1]) adjacentCount++;
        }
        maxAdjacentPairsCount[adjacentCount] = (maxAdjacentPairsCount[adjacentCount] || 0) + 1;
        
        // 2. Tìm khoảng cách xa nhất giữa 2 chữ số giống nhau trong cùng 1 vé
        let maxDist = 0;
        for (let digit = 0; digit <= 9; digit++) {
            let char = digit.toString();
            let firstIdx = str.indexOf(char);
            let lastIdx = str.lastIndexOf(char);
            if (firstIdx !== -1 && lastIdx !== firstIdx) {
                let dist = lastIdx - firstIdx;
                if (dist > maxDist) maxDist = dist;
            }
        }
        maxDistanceDistribution[maxDist] = (maxDistanceDistribution[maxDist] || 0) + 1;
    }
    
    console.log('======================================================');
    console.log('📏 KHOẢNG CÁCH CHỮ SỐ (12 CỘT)');
    console.log('======================================================');
    
    console.log('1. SỐ LẦN CÁC CHỮ SỐ TRÙNG NHAU ĐỨNG LIỀN KỀ (Adjacent Pairs):');
    for (let i = 0; i <= 6; i++) {
        let count = maxAdjacentPairsCount[i] || 0;
        let pct = (count / data.length * 100).toFixed(2);
        console.log(`- Vé có ${i} cặp liền kề: ${pct}%`);
    }
    console.log('=> Rất hiếm vé có >= 3 cặp chữ số đứng liền kề giống nhau (vd: 11, 22, 33).');
    
    console.log('\n2. KHOẢNG CÁCH XA NHẤT GIỮA 2 CHỮ SỐ GIỐNG NHAU (Max Distance):');
    let maxDistKeys = Object.keys(maxDistanceDistribution).map(Number).sort((a,b)=>a-b);
    let cumulative = 0;
    for (let k of maxDistKeys) {
        let count = maxDistanceDistribution[k];
        let pct = (count / data.length * 100).toFixed(2);
        cumulative += parseFloat(pct);
        console.log(`- Khoảng cách lớn nhất = ${k} ô: ${pct}% (Lũy kế: ${cumulative.toFixed(2)}%)`);
    }
    
    // TEST LỌC
    let totalRandom = 100000;
    let passed = 0;
    for (let i = 0; i < totalRandom; i++) {
        let pool = [];
        while(pool.length < 6) {
            let r = Math.floor(Math.random() * 45) + 1;
            if (!pool.includes(r)) pool.push(r);
        }
        pool.sort((a,b)=>a-b);
        let str = "";
        for (let num of pool) str += num.toString().padStart(2, '0');
        
        let adjacentCount = 0;
        for (let j = 1; j < 12; j++) {
            if (str[j] === str[j-1]) adjacentCount++;
        }
        
        if (adjacentCount <= 2) passed++;
    }
    
    console.log('\n======================================================');
    console.log('🛑 SỨC MẠNH CỦA BỘ LỌC KHOẢNG CÁCH CHỮ SỐ');
    console.log('======================================================');
    console.log(`Giới hạn: BẤT KỲ vé nào có > 2 cặp chữ số giống nhau ĐỨNG LIỀN KỀ -> VỨT!`);
    console.log(`Phát sinh ngẫu nhiên ${totalRandom} vé:`);
    console.log(`- Số vé CÒN SỐNG   : ${passed} vé`);
    console.log(`- Số vé BỊ CHÉM BAY: ${totalRandom - passed} vé`);
    console.log(`=> Tiêu diệt thêm ${((totalRandom - passed) / totalRandom * 100).toFixed(2)}% tổng lượng vé rác!`);
}

main().catch(console.error);
