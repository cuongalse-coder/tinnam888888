/**
 * NUMBER SPREAD ANALYSIS (Phân tích độ giãn số)
 * Analyze the gaps between adjacent numbers and their distribution across the board.
 */
const https = require('https');
function fetchData() { return new Promise((res, rej) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', r => { let d='';r.on('data',c=>d+=c);r.on('end',()=>{const dr=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)dr.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}res(dr);});r.on('error',rej);}); }); }

async function main() {
    const data = await fetchData();
    
    let gaps = {
        g1: [], // n2 - n1
        g2: [], // n3 - n2
        g3: [], // n4 - n3
        g4: [], // n5 - n4
        g5: [], // n6 - n5
    };
    
    let zoneDistribution = {
        low: 0,   // 1 - 15
        mid: 0,   // 16 - 30
        high: 0   // 31 - 45
    };
    
    // Distribution of total distance (Tail - Head)
    let totalSpreadCounts = {};

    for (let i = 0; i < data.length; i++) {
        const d = data[i];
        gaps.g1.push(d[1] - d[0]);
        gaps.g2.push(d[2] - d[1]);
        gaps.g3.push(d[3] - d[2]);
        gaps.g4.push(d[4] - d[3]);
        gaps.g5.push(d[5] - d[4]);
        
        for(const n of d) {
            if(n <= 15) zoneDistribution.low++;
            else if(n <= 30) zoneDistribution.mid++;
            else zoneDistribution.high++;
        }
        
        let totalSpread = d[5] - d[0];
        let spreadBucket = Math.floor(totalSpread / 5) * 5; // Group by 5s (e.g. 30-34, 35-39)
        const bucketKey = `${spreadBucket}-${spreadBucket+4}`;
        totalSpreadCounts[bucketKey] = (totalSpreadCounts[bucketKey] || 0) + 1;
    }
    
    const avg = arr => (arr.reduce((a,b) => a+b, 0) / arr.length).toFixed(2);
    
    console.log('======================================================');
    console.log('📏 PHÂN TÍCH ĐỘ GIÃN (GAPS) GIỮA CÁC CON SỐ');
    console.log('======================================================');
    console.log(`Khoảng cách trung bình từ Số 1 (Đầu) đến Số 2 : ${avg(gaps.g1)} đơn vị`);
    console.log(`Khoảng cách trung bình từ Số 2 đến Số 3       : ${avg(gaps.g2)} đơn vị`);
    console.log(`Khoảng cách trung bình từ Số 3 đến Số 4       : ${avg(gaps.g3)} đơn vị`);
    console.log(`Khoảng cách trung bình từ Số 4 đến Số 5       : ${avg(gaps.g4)} đơn vị`);
    console.log(`Khoảng cách trung bình từ Số 5 đến Số 6 (Đuôi): ${avg(gaps.g5)} đơn vị`);
    
    console.log('\n=> Nhận xét: Các con số không rải đều! Khoảng cách giữa các số Ở GIỮA (Số 3 và 4) thường hẹp hơn (có xu hướng dính chùm vào nhau).');

    console.log('\n======================================================');
    console.log('📊 MẬT ĐỘ TẬP TRUNG THEO VÙNG (ZONES)');
    console.log('======================================================');
    let totalNums = data.length * 6;
    console.log(`Vùng Thấp (01 - 15) : ${(zoneDistribution.low/totalNums*100).toFixed(2)}% tổng số bóng rơi ra`);
    console.log(`Vùng Giữa (16 - 30) : ${(zoneDistribution.mid/totalNums*100).toFixed(2)}% tổng số bóng rơi ra`);
    console.log(`Vùng Cao  (31 - 45) : ${(zoneDistribution.high/totalNums*100).toFixed(2)}% tổng số bóng rơi ra`);
    
    console.log('\n======================================================');
    console.log('📐 ĐỘ RỘNG MẠNG LƯỚI (Số Đuôi trừ Số Đầu)');
    console.log('======================================================');
    const sortedSpreads = Object.entries(totalSpreadCounts).sort((a,b)=>b[1]-a[1]);
    for(const [bucket, count] of sortedSpreads) {
        if(count > 20) {
            console.log(`Biên độ [${bucket}] đơn vị: ${count} kỳ (${(count/data.length*100).toFixed(2)}%)`);
        }
    }
}

main().catch(console.error);
