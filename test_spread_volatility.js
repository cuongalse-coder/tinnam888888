/**
 * SPREAD VOLATILITY ANALYSIS (Hiệu ứng Co Giãn Mạng Lưới)
 * Test if the sequence of numbers stretches or shrinks over time.
 */
const https = require('https');
function fetchData() { return new Promise((res, rej) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', r => { let d='';r.on('data',c=>d+=c);r.on('end',()=>{const dr=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)dr.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}res(dr);});r.on('error',rej);}); }); }

async function main() {
    const data = await fetchData();
    
    // Spread S = Tail - Head
    let spreads = data.map(d => d[5] - d[0]);
    
    let expandThenContract = 0;
    let expandThenExpand = 0;
    
    let contractThenExpand = 0;
    let contractThenContract = 0;
    
    let extremeContracts = { total: 0, contract_next: 0 };
    let extremeExpands = { total: 0, expand_next: 0 }; // meaning spread was very narrow, will it expand?

    for (let i = 2; i < spreads.length; i++) {
        let s_t2 = spreads[i-2];
        let s_t1 = spreads[i-1];
        let s_t  = spreads[i];
        
        let prevAction = s_t1 - s_t2;
        let currAction = s_t - s_t1;
        
        if (prevAction > 0) { // Previous was EXPAND
            if (currAction < 0) expandThenContract++;
            else if (currAction > 0) expandThenExpand++;
        } else if (prevAction < 0) { // Previous was CONTRACT
            if (currAction > 0) contractThenExpand++;
            else if (currAction < 0) contractThenContract++;
        }
        
        // Extreme boundaries analysis
        if (s_t1 >= 40) { // Very stretched out
            extremeContracts.total++;
            if (currAction < 0) extremeContracts.contract_next++;
        }
        if (s_t1 <= 25) { // Very squished together
            extremeExpands.total++;
            if (currAction > 0) extremeExpands.expand_next++;
        }
    }
    
    console.log('======================================================');
    console.log('🎈 HIỆU ỨNG CO GIÃN MẠNG LƯỚI (DỰA TRÊN ĐỘ RỘNG ĐẦU-ĐUÔI)');
    console.log('======================================================');
    
    let totalE = expandThenContract + expandThenExpand;
    console.log(`Khi mạng lưới kỳ trước GIÃN RA (Spread tăng):`);
    console.log(`- Kỳ này sẽ CO LẠI   : ${expandThenContract} lần (${(expandThenContract/totalE*100).toFixed(2)}%)`);
    console.log(`- Kỳ này tiếp tục GIÃN : ${expandThenExpand} lần (${(expandThenExpand/totalE*100).toFixed(2)}%)`);
    
    console.log('');
    let totalC = contractThenExpand + contractThenContract;
    console.log(`Khi mạng lưới kỳ trước CO LẠI (Spread giảm):`);
    console.log(`- Kỳ này sẽ GIÃN RA  : ${contractThenExpand} lần (${(contractThenExpand/totalC*100).toFixed(2)}%)`);
    console.log(`- Kỳ này tiếp tục CO : ${contractThenContract} lần (${(contractThenContract/totalC*100).toFixed(2)}%)`);
    
    console.log('\n======================================================');
    console.log('💥 PHẢN ỨNG TẠI CÁC ĐIỂM CỰC HẠN (Dây thun bị kéo căng)');
    console.log('======================================================');
    console.log(`Khi mạng lưới bị KÉO GIÃN CỰC ĐẠI (Độ rộng >= 40 đơn vị):`);
    console.log(`=> Khả năng kỳ sau lập tức CO LẠI là: ${(extremeContracts.contract_next/extremeContracts.total*100).toFixed(2)}% (${extremeContracts.contract_next}/${extremeContracts.total} lần)`);
    
    console.log('');
    console.log(`Khi mạng lưới bị ÉP CO CỰC ĐẠI (Độ rộng <= 25 đơn vị):`);
    console.log(`=> Khả năng kỳ sau lập tức GIÃN BUNG RA là: ${(extremeExpands.expand_next/extremeExpands.total*100).toFixed(2)}% (${extremeExpands.expand_next}/${extremeExpands.total} lần)`);
}

main().catch(console.error);
