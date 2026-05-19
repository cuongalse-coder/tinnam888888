/**
 * MIDDLE PAIRS ANALYSIS (Cặp Số Giữa)
 * Find the most frequent pairs that occur in the middle 4 positions (Index 1, 2, 3, 4).
 */
const https = require('https');
function fetchData() { return new Promise((res, rej) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', r => { let d='';r.on('data',c=>d+=c);r.on('end',()=>{const dr=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)dr.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}res(dr);});r.on('error',rej);}); }); }

async function main() {
    const data = await fetchData();
    const pairCounts = {};
    const consecCounts = {};
    
    for (let i = 0; i < data.length; i++) {
        // Lấy 4 số ở giữa
        const middle = data[i].slice(1, 5); 
        
        for (let j = 0; j < middle.length; j++) {
            for (let k = j + 1; k < middle.length; k++) {
                const n1 = middle[j];
                const n2 = middle[k];
                const key = `${n1.toString().padStart(2, '0')} - ${n2.toString().padStart(2, '0')}`;
                pairCounts[key] = (pairCounts[key] || 0) + 1;
                
                // Track consecutive pairs (e.g. 15-16)
                if (n2 - n1 === 1) {
                    consecCounts[key] = (consecCounts[key] || 0) + 1;
                }
            }
        }
    }
    
    const sortedPairs = Object.entries(pairCounts).sort((a, b) => b[1] - a[1]);
    const sortedConsec = Object.entries(consecCounts).sort((a, b) => b[1] - a[1]);
    
    console.log('======================================================');
    console.log('🔗 TOP 20 CẶP SỐ GIỮA HAY ĐI CHUNG NHẤT (VÙNG LÕI)');
    console.log('======================================================');
    for (let i = 0; i < 20; i++) {
        if (!sortedPairs[i]) break;
        const [pair, count] = sortedPairs[i];
        const pct = ((count / data.length) * 100).toFixed(2);
        console.log(`Cặp [${pair}]    | ${count.toString().padStart(5)} lần | ${pct}%`);
    }

    console.log('\n======================================================');
    console.log('🔥 TOP 10 CẶP SỐ GIỮA "LIỀN KỀ" (Ví dụ 12-13)');
    console.log('======================================================');
    for (let i = 0; i < 10; i++) {
        if (!sortedConsec[i]) break;
        const [pair, count] = sortedConsec[i];
        const pct = ((count / data.length) * 100).toFixed(2);
        console.log(`Cặp [${pair}]    | ${count.toString().padStart(5)} lần | ${pct}%`);
    }
}

main().catch(console.error);
