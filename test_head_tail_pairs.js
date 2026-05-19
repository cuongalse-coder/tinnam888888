/**
 * HEAD-TAIL PAIR CO-OCCURRENCE ANALYSIS
 * Find which Head numbers and Tail numbers most frequently appear TOGETHER in the same draw.
 */
const https = require('https');
function fetchData() { return new Promise((res, rej) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', r => { let d='';r.on('data',c=>d+=c);r.on('end',()=>{const dr=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)dr.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}res(dr);});r.on('error',rej);}); }); }

async function main() {
    const data = await fetchData();
    const pairCounts = {};
    
    // Matrix for Head-Tail pairs
    for (let i = 0; i < data.length; i++) {
        const head = data[i][0];
        const tail = data[i][5];
        
        const key = `${head.toString().padStart(2, '0')} - ${tail.toString().padStart(2, '0')}`;
        pairCounts[key] = (pairCounts[key] || 0) + 1;
    }
    
    const sortedPairs = Object.entries(pairCounts).sort((a, b) => b[1] - a[1]);
    
    console.log('======================================================');
    console.log('🔗 TOP 20 CẶP ĐẦU-ĐUÔI HAY ĐI CHUNG NHẤT LỊCH SỬ');
    console.log('======================================================');
    console.log('Cặp [Đầu - Đuôi] | Số lần đi chung | Tỷ lệ xuất hiện');
    console.log('------------------------------------------------------');
    
    for (let i = 0; i < 20; i++) {
        if (!sortedPairs[i]) break;
        const [pair, count] = sortedPairs[i];
        const pct = ((count / data.length) * 100).toFixed(2);
        console.log(`Cặp [${pair}]    | ${count.toString().padStart(15)} | ${pct.padStart(14)}%`);
    }

    console.log('\n======================================================');
    console.log('🔗 TOP ĐUÔI TƯƠNG ỨNG VỚI MỖI SỐ ĐẦU (1-5)');
    console.log('======================================================');
    for (let head = 1; head <= 5; head++) {
        const hStr = head.toString().padStart(2, '0');
        const specificTails = {};
        for (let i = 0; i < data.length; i++) {
            if (data[i][0] === head) {
                const tail = data[i][5];
                specificTails[tail] = (specificTails[tail] || 0) + 1;
            }
        }
        const topTail = Object.entries(specificTails).sort((a,b)=>b[1]-a[1])[0];
        if (topTail) {
            console.log(`Nếu Đầu là ${hStr} => Đuôi hay đi kèm nhất là ${topTail[0].padStart(2, '0')} (gặp ${topTail[1]} lần)`);
        }
    }
}

main().catch(console.error);
