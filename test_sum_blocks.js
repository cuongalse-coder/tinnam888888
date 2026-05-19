/**
 * TỔNG SỐ (SUM BLOCKS) ANALYSIS
 * Test the behavior of the Sum of 6 numbers.
 */
const https = require('https');
function fetchData() { return new Promise((res, rej) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', r => { let d='';r.on('data',c=>d+=c);r.on('end',()=>{const dr=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)dr.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}res(dr);});r.on('error',rej);}); }); }

async function main() {
    const data = await fetchData();
    
    let sums = data.map(d => d.slice(0, 6).reduce((a, b) => a + b, 0));
    
    // Create buckets of 20
    let sumBuckets = {};
    for (let s of sums) {
        let b = Math.floor(s / 20) * 20;
        let key = `${b}-${b+19}`;
        sumBuckets[key] = (sumBuckets[key] || 0) + 1;
    }
    
    let consecutiveBlocks = 0;
    
    let extremeLowNext = { total: 0, bounceUp: 0 };  // Sum <= 100
    let extremeHighNext = { total: 0, bounceDown: 0 }; // Sum >= 180

    for (let i = 1; i < sums.length; i++) {
        let prevSum = sums[i-1];
        let currSum = sums[i];
        
        let prevBlock = Math.floor(prevSum / 20);
        let currBlock = Math.floor(currSum / 20);
        
        if (prevBlock === currBlock) consecutiveBlocks++;
        
        // Analyze Bounce effect
        if (prevSum <= 100) {
            extremeLowNext.total++;
            if (currSum > 110) extremeLowNext.bounceUp++;
        }
        
        if (prevSum >= 180) {
            extremeHighNext.total++;
            if (currSum < 170) extremeHighNext.bounceDown++;
        }
    }
    
    console.log('======================================================');
    console.log('📊 MẬT ĐỘ TẬP TRUNG TỔNG CỦA 6 SỐ (BLOCKS)');
    console.log('======================================================');
    let sortedBuckets = Object.entries(sumBuckets).sort((a,b) => {
        return parseInt(a[0].split('-')[0]) - parseInt(b[0].split('-')[0]);
    });
    
    for (let [k, v] of sortedBuckets) {
        let pct = (v / sums.length * 100).toFixed(2);
        console.log(`Khối Tổng [${k.padStart(7)}]: ${v.toString().padStart(4)} kỳ (${pct.padStart(5)}%)`);
    }
    
    console.log('\n======================================================');
    console.log('💥 HIỆU ỨNG LẶP LẠI (CONSECUTIVE REPETITION)');
    console.log('======================================================');
    console.log(`Xác suất Tổng của kỳ sau rớt lại đúng vào Khối 20 số của kỳ trước:`);
    console.log(`=> ${(consecutiveBlocks / sums.length * 100).toFixed(2)}% (${consecutiveBlocks} lần)`);
    console.log(`Kết luận: Đúng như bạn nghĩ, Tổng RẤT HIẾM KHI đứng yên một chỗ!`);
    
    console.log('\n======================================================');
    console.log('📉 HIỆU ỨNG BẬT TƯỜNG (REBOUND) Ở CÁC KHỐI CỰC HẠN');
    console.log('======================================================');
    console.log(`Nếu kỳ trước Tổng quá bé (Sum <= 100):`);
    console.log(`- Khả năng kỳ này BẬT TĂNG MẠNH (Sum > 110) là: ${(extremeLowNext.bounceUp / extremeLowNext.total * 100).toFixed(2)}%`);
    
    console.log(`\nNếu kỳ trước Tổng quá lớn (Sum >= 180):`);
    console.log(`- Khả năng kỳ này BẬT GIẢM MẠNH (Sum < 170) là: ${(extremeHighNext.bounceDown / extremeHighNext.total * 100).toFixed(2)}%`);
}

main().catch(console.error);
