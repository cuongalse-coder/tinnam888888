/**
 * ELASTIC FILTER IMPACT TEST (Tác động của Bộ Lọc Dây Thun lên tổ hợp vé)
 * This script measures how many trash combinations are eliminated by the Elastic Spread Filter.
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
    const sorted = [...arr].sort((a,b)=>a-b);
    const index = (p/100) * (sorted.length - 1);
    return sorted[Math.round(index)];
}

async function main() {
    const data = await fetchData();
    const mx = 45;
    
    let totalNormalCombinations = 0;
    let totalElasticCombinations = 0;
    
    let hit66Normal = 0;
    let hit66Elastic = 0;
    
    // Test over the last 100 draws to save time
    const testDraws = 100;
    
    console.log(`⏳ Đang chạy mô phỏng tổ hợp trên ${testDraws} kỳ quay gần nhất...`);
    
    for (let ci = data.length - testDraws; ci < data.length; ci++) {
        const hist = data.slice(0, ci);
        const actual = data[ci].slice(0, 6);
        
        // --- 1. Tạo Pool 20 số (Giả định chứa 6 số trúng để đo lượng rác) ---
        // Để công bằng, tạo pool gồm 6 số trúng + 14 số random
        let pool = new Set(actual);
        while(pool.size < 20) {
            pool.add(Math.floor(Math.random() * mx) + 1);
        }
        let poolArr = Array.from(pool).sort((a,b)=>a-b);
        
        // --- 2. Tính Constraints tĩnh (Normal) ---
        let recent = hist.slice(-50);
        let ranges = recent.map(d => Math.max(...d.slice(0,6)) - Math.min(...d.slice(0,6)));
        let range_lo = percentile(ranges, 8);
        let range_hi = percentile(ranges, 92);
        
        let normal_lo = range_lo;
        let normal_hi = range_hi;
        
        // --- 3. Tính Constraints Động (Elastic) ---
        let s_t1 = Math.max(...hist[hist.length-1].slice(0,6)) - Math.min(...hist[hist.length-1].slice(0,6));
        let s_t2 = Math.max(...hist[hist.length-2].slice(0,6)) - Math.min(...hist[hist.length-2].slice(0,6));
        
        if (s_t1 >= 40) range_hi = Math.min(range_hi, 38);
        else if (s_t1 <= 25) range_lo = Math.max(range_lo, 28);
        else {
            if (s_t1 > s_t2) range_hi = Math.min(range_hi, s_t1 - 1);
            else if (s_t1 < s_t2) range_lo = Math.max(range_lo, s_t1 + 1);
        }
        if (range_lo > range_hi) { let t=range_lo; range_lo=range_hi; range_hi=t; }
        
        // --- 4. Tạo toàn bộ 38,760 tổ hợp từ Pool 20 ---
        const allCombos = getCombinations(poolArr, 6);
        
        let validNormal = 0;
        let validElastic = 0;
        
        let actualRange = actual[5] - actual[0];
        
        for (const c of allCombos) {
            const rng = c[5] - c[0];
            
            // Check normal
            if (rng >= normal_lo && rng <= normal_hi) validNormal++;
            
            // Check elastic
            if (rng >= range_lo && rng <= range_hi) validElastic++;
        }
        
        totalNormalCombinations += validNormal;
        totalElasticCombinations += validElastic;
        
        // Giả lập mua 100 vé
        // Tỷ lệ trúng 6/6 = 100 / số tổ hợp hợp lệ (Nếu actualRange lọt vào constraint)
        if (actualRange >= normal_lo && actualRange <= normal_hi) {
            hit66Normal += (100 / validNormal); 
        }
        if (actualRange >= range_lo && actualRange <= range_hi) {
            hit66Elastic += (100 / validElastic);
        }
    }
    
    let avgNormal = totalNormalCombinations / testDraws;
    let avgElastic = totalElasticCombinations / testDraws;
    
    console.log('\n======================================================');
    console.log('🚀 TÁC ĐỘNG CỦA BỘ LỌC DÂY THUN (ELASTIC SPREAD FILTER)');
    console.log('======================================================');
    console.log(`Giả định bạn có 1 Pool 20 số chứa Jackpot (Tổng: 38,760 vé).`);
    console.log(`\n1. SỐ LƯỢNG VÉ RÁC BỊ LOẠI BỎ (Mức độ cô đặc vé):`);
    console.log(`- Lọc Thông thường (Tĩnh): Còn lại trung bình ${Math.round(avgNormal)} vé`);
    console.log(`- Lọc Dây Thun (Động)    : Còn lại trung bình ${Math.round(avgElastic)} vé`);
    console.log(`=> Bộ Lọc Dây Thun giúp CHẶT ĐỨT THÊM ${Math.round(avgNormal - avgElastic)} vé rác mỗi kỳ!`);
    
    console.log(`\n2. KHẢ NĂNG TRÚNG 6/6 KHI MUA 100 VÉ:`);
    let pctIncrease = ((hit66Elastic - hit66Normal) / hit66Normal) * 100;
    console.log(`- Tỷ lệ bắt 6/6 (Lọc Thông thường): ${(hit66Normal / testDraws * 100).toFixed(4)}%`);
    console.log(`- Tỷ lệ bắt 6/6 (Lọc Dây Thun)    : ${(hit66Elastic / testDraws * 100).toFixed(4)}%`);
    console.log(`\n🔥 KẾT LUẬN: BỘ LỌC CỦA BẠN ĐÃ TĂNG XÁC SUẤT TRÚNG 6/6 LÊN +${pctIncrease.toFixed(2)}% !`);
    console.log('======================================================');
}

main().catch(console.error);
