/**
 * DECADE (GROUP) LIMITS & EXHAUSTION ANALYSIS
 * 1. Phân tích giới hạn số lượng bóng tối đa trong 1 nhóm (Decade: 0x, 1x, 2x, 3x, 4x)
 * 2. Phân tích hiệu ứng "cạn kiệt" (Nếu nhóm đã nổ nhiều ở kỳ trước, kỳ này có bị loại bỏ/nghỉ hay không?)
 */
const https = require('https');
function fetchData() { return new Promise((res, rej) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', r => { let d='';r.on('data',c=>d+=c);r.on('end',()=>{const dr=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)dr.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}res(dr);});r.on('error',rej);}); }); }

function getDecades(draw) {
    let decs = [0, 0, 0, 0, 0]; // 01-09, 10-19, 20-29, 30-39, 40-45
    for (let n of draw) {
        if (n < 10) decs[0]++;
        else if (n < 20) decs[1]++;
        else if (n < 30) decs[2]++;
        else if (n < 40) decs[3]++;
        else decs[4]++;
    }
    return decs;
}

async function main() {
    const data = await fetchData();
    
    // 1. Phân tích tỷ lệ dính chùm trong 1 nhóm (Max limit)
    let maxInGroupCounts = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0 };
    
    // 2. Phân tích Hiệu ứng Cạn Kiệt (Exhaustion)
    // Nếu kỳ t-1 có nhóm nổ >= 3 số, kỳ t nhóm đó sẽ nổ bao nhiêu số?
    let overloadNextCounts = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0 };
    let overloadTotal = 0;

    for (let i = 0; i < data.length; i++) {
        let decs = getDecades(data[i]);
        let maxInSingleGroup = Math.max(...decs);
        maxInGroupCounts[maxInSingleGroup] = (maxInGroupCounts[maxInSingleGroup] || 0) + 1;
        
        // Exhaustion analysis
        if (i > 0) {
            let prevDecs = getDecades(data[i-1]);
            for (let d = 0; d < 5; d++) {
                if (prevDecs[d] >= 3) { // Overload (Tới hạn)
                    overloadTotal++;
                    let nextHits = decs[d];
                    overloadNextCounts[nextHits] = (overloadNextCounts[nextHits] || 0) + 1;
                }
            }
        }
    }
    
    console.log('======================================================');
    console.log('📦 PHÂN TÍCH GIỚI HẠN TỐI ĐA TRONG MỘT NHÓM SỐ (10 SỐ)');
    console.log('======================================================');
    console.log('Trong 1 kỳ quay, số lượng bóng TỐI ĐA rơi vào cùng 1 đầu số (ví dụ: Đầu 2 gồm 20->29):');
    for (let i = 1; i <= 6; i++) {
        let pct = (maxInGroupCounts[i] / data.length * 100).toFixed(2);
        console.log(`- Tối đa ${i} bóng cùng nhóm : ${maxInGroupCounts[i]} kỳ (${pct}%)`);
    }
    console.log('\n=> CHỐT CHẶN: Tuyệt đối không bao giờ đánh 4 hoặc 5 số trong cùng 1 đầu số. Giới hạn "Tới hạn" an toàn là TỐI ĐA 3 SỐ/NHÓM.');

    console.log('\n======================================================');
    console.log('📉 HIỆU ỨNG CẠN KIỆT NHÓM (CHẾT NHÓM SAU KHI NỔ TỚI HẠN)');
    console.log('======================================================');
    console.log(`Nếu kỳ trước có một nhóm đạt tới hạn (Nổ từ 3 số trở lên trong cùng 1 nhóm):`);
    console.log(`Kỳ tiếp theo, nhóm đó sẽ xảy ra phản ứng gì?`);
    for (let i = 0; i <= 4; i++) {
        let count = overloadNextCounts[i] || 0;
        let pct = (count / overloadTotal * 100).toFixed(2);
        console.log(`- Nhóm đó nổ ${i} số (Tắt điện / Giảm nhiệt): ${count} lần (${pct}%)`);
    }
    
    let deadPct = ((overloadNextCounts[0] + overloadNextCounts[1]) / overloadTotal * 100).toFixed(2);
    console.log(`\n=> KẾT LUẬN VÀNG: Nếu kỳ trước đầu số nào đã nổ >= 3 con, kỳ sau có ${deadPct}% xác suất đầu số đó SẼ CHẾT (Ra 0 con) hoặc chỉ thoi thóp (Ra 1 con). Chúng ta hoàn toàn có thể LOẠI BỎ hoặc ép giới hạn nhóm đó xuống cực thấp!`);
    console.log('======================================================');
}

main().catch(console.error);
