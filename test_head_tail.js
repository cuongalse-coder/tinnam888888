/**
 * HEAD-TAIL ANALYSIS (Số Đầu - Số Đuôi)
 * Analyzing the distribution of the first and last numbers in Mega 6/45.
 */
const https = require('https');
function fetchData() { return new Promise((res, rej) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', r => { let d='';r.on('data',c=>d+=c);r.on('end',()=>{const dr=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)dr.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}res(dr);});r.on('error',rej);}); }); }

async function main() {
    const data = await fetchData();
    const headCounts = {};
    const tailCounts = {};
    
    for (let i = 0; i < data.length; i++) {
        const head = data[i][0];
        const tail = data[i][5];
        
        headCounts[head] = (headCounts[head] || 0) + 1;
        tailCounts[tail] = (tailCounts[tail] || 0) + 1;
    }
    
    console.log('=' . repeat(50));
    console.log('📊 PHÂN BỐ SỐ ĐẦU (HEAD) - 10 SỐ HAY RA NHẤT');
    console.log('=' . repeat(50));
    const sortedHeads = Object.entries(headCounts).sort((a,b)=>b[1]-a[1]);
    let top10HeadSum = 0;
    for (let i=0; i<10; i++) {
        const num = parseInt(sortedHeads[i][0]);
        const count = sortedHeads[i][1];
        const pct = (count / data.length * 100).toFixed(2);
        top10HeadSum += count / data.length;
        console.log(`Số ${num.toString().padStart(2, '0')}: ${count.toString().padStart(3)} lần (${pct}%)`);
    }
    console.log(`\n=> Top 10 số đầu tiên chiếm ${(top10HeadSum * 100).toFixed(2)}% tổng số kỳ quay.`);
    
    console.log('\n' + '=' . repeat(50));
    console.log('📊 PHÂN BỐ SỐ ĐUÔI (TAIL) - 10 SỐ HAY RA NHẤT');
    console.log('=' . repeat(50));
    const sortedTails = Object.entries(tailCounts).sort((a,b)=>b[1]-a[1]);
    let top10TailSum = 0;
    for (let i=0; i<10; i++) {
        const num = parseInt(sortedTails[i][0]);
        const count = sortedTails[i][1];
        const pct = (count / data.length * 100).toFixed(2);
        top10TailSum += count / data.length;
        console.log(`Số ${num.toString().padStart(2, '0')}: ${count.toString().padStart(3)} lần (${pct}%)`);
    }
    console.log(`\n=> Top 10 số đuôi chiếm ${(top10TailSum * 100).toFixed(2)}% tổng số kỳ quay.`);
}

main().catch(console.error);
