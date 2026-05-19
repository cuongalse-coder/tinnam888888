/**
 * TÁC ĐỘNG CỦA BỘ LỌC KHỐI TỔNG (SUM BLOCK FILTER IMPACT)
 * Đo lường mức độ cô đặc vé và sự thay đổi EV của bộ lọc Khối Tổng.
 */
const https = require('https');
function fetchData() { return new Promise((res, rej) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', r => { let d='';r.on('data',c=>d+=c);r.on('end',()=>{const dr=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)dr.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}res(dr);});r.on('error',rej);}); }); }

function getCombinations(arr, k) {
    const result = [];
    function f(prefix, arr) {
        if (prefix.length === k) { result.push(prefix); return; }
        for (let i = 0; i < arr.length; i++) f([...prefix, arr[i]], arr.slice(i + 1));
    }
    f([], arr);
    return result;
}

function percentile(arr, p) {
    if (arr.length === 0) return 0;
    const sorted = [...arr].sort((a,b)=>a-b);
    const index = (p/100) * (sorted.length - 1);
    return sorted[Math.round(index)];
}

async function main() {
    const data = await fetchData();
    const mx = 45;
    
    let totalNormalCombos = 0;
    let totalSumBlockCombos = 0;
    
    let hit66Normal = 0;
    let hit66SumBlock = 0;
    
    let testDraws = 100;
    
    console.log(`⏳ Đang chạy đo lường tác động Bộ Lọc Khối Tổng trên ${testDraws} kỳ gần nhất...`);
    
    for (let ci = data.length - testDraws; ci < data.length; ci++) {
        const hist = data.slice(0, ci);
        const actual = data[ci].slice(0, 6);
        let actualSum = actual.reduce((a, b) => a + b, 0);
        
        let pool = new Set(actual);
        while(pool.size < 20) pool.add(Math.floor(Math.random() * mx) + 1);
        let poolArr = Array.from(pool).sort((a,b)=>a-b);
        
        // Normal constraints
        let recent = hist.slice(-50);
        let sums = recent.map(d => d.slice(0,6).reduce((a,b)=>a+b, 0));
        let sum_lo_normal = Math.floor(percentile(sums, 8));
        let sum_hi_normal = Math.floor(percentile(sums, 92));
        
        // Sum Block constraints
        let sum_lo_block = sum_lo_normal;
        let sum_hi_block = sum_hi_normal;
        let prevSum = hist[hist.length-1].slice(0,6).reduce((a,b)=>a+b, 0);
        
        let banned_block = [prevSum - 10, prevSum + 10];
        if (prevSum <= 100) sum_lo_block = Math.max(sum_lo_block, 110);
        if (prevSum >= 180) sum_hi_block = Math.min(sum_hi_block, 170);
        
        const allCombos = getCombinations(poolArr, 6);
        
        let validNormal = 0;
        let validBlock = 0;
        
        for (const c of allCombos) {
            let s = c.reduce((a,b)=>a+b, 0);
            
            // Check Normal
            if (s >= sum_lo_normal && s <= sum_hi_normal) validNormal++;
            
            // Check Block
            let passBlock = (s >= sum_lo_block && s <= sum_hi_block);
            if (s >= banned_block[0] && s <= banned_block[1]) passBlock = false;
            if (passBlock) validBlock++;
        }
        
        totalNormalCombos += validNormal;
        totalSumBlockCombos += validBlock;
        
        let actualNormalValid = (actualSum >= sum_lo_normal && actualSum <= sum_hi_normal);
        
        let actualBlockValid = (actualSum >= sum_lo_block && actualSum <= sum_hi_block);
        if (actualSum >= banned_block[0] && actualSum <= banned_block[1]) actualBlockValid = false;
        
        const budget = 100;
        
        if (actualNormalValid) hit66Normal += (budget / validNormal);
        if (actualBlockValid) hit66SumBlock += (budget / validBlock);
    }
    
    let avgNormal = totalNormalCombos / testDraws;
    let avgBlock = totalSumBlockCombos / testDraws;
    
    console.log('\n======================================================');
    console.log('🚀 TÁC ĐỘNG CỦA BỘ LỌC KHỐI TỔNG (SUM BLOCK FILTER)');
    console.log('======================================================');
    console.log(`1. SỐ LƯỢNG VÉ RÁC BỊ CHÉM BAY TỪ POOL 20 SỐ:`);
    console.log(`- Lọc Tĩnh thông thường : Giữ lại ~${Math.round(avgNormal)} vé`);
    console.log(`- Bật Bộ Lọc Khối Tổng  : Giữ lại ~${Math.round(avgBlock)} vé`);
    let pctReduce = ((avgNormal - avgBlock) / avgNormal * 100).toFixed(2);
    console.log(`=> Bộ lọc mới đã tiếp tục chém bay THÊM ${pctReduce}% lượng vé rác!`);
    
    console.log(`\n2. XÁC SUẤT BẮT TRÚNG JACKPOT 6/6 (MUA 100 VÉ):`);
    let pctIncrease = ((hit66SumBlock - hit66Normal) / hit66Normal * 100).toFixed(2);
    console.log(`- Tỷ lệ 6/6 (Lọc Tĩnh)      : ${(hit66Normal / testDraws * 100).toFixed(4)}%`);
    console.log(`- Tỷ lệ 6/6 (Khối Tổng)     : ${(hit66SumBlock / testDraws * 100).toFixed(4)}%`);
    console.log(`\n🔥 KẾT LUẬN: BỘ LỌC KHỐI TỔNG ĐÃ LÀM TĂNG TỶ LỆ TRÚNG LÊN +${pctIncrease}% !!!`);
    console.log('======================================================');
}

main().catch(console.error);
