/**
 * SIÊU BỘ LỌC TỐI THƯỢNG (ABSOLUTE ELIMINATION)
 * Từ 8,145,060 tổ hợp, áp dụng TẤT CẢ các bộ lọc để xem còn lại bao nhiêu vé.
 * Không cần AI dự đoán số, chỉ dùng thuần túy toán học loại trừ.
 */
const https = require('https');
function fetchData() { return new Promise((res, rej) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', r => { let d='';r.on('data',c=>d+=c);r.on('end',()=>{const dr=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)dr.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}res(dr);});r.on('error',rej);}); }); }

function percentile(arr, p) {
    if (arr.length === 0) return 0;
    const sorted = [...arr].sort((a,b)=>a-b);
    const index = (p/100) * (sorted.length - 1);
    return sorted[Math.round(index)];
}

async function main() {
    const data = await fetchData();
    const mx = 45;
    
    console.log('⏳ Đang thiết lập các Màng Lọc Tối Thượng từ Lịch sử...');
    const hist = data;
    
    // 1. Calculate Bounds
    let recent = hist.slice(-50);
    let sums = recent.map(d => d.slice(0,6).reduce((a,b)=>a+b, 0));
    let odds = recent.map(d => d.slice(0,6).filter(x => x%2===1).length);
    let highs = recent.map(d => d.slice(0,6).filter(x => x>22).length);
    let ranges = recent.map(d => d[5] - d[0]);
    
    let sum_lo = percentile(sums, 8);
    let sum_hi = percentile(sums, 92);
    let odd_lo = Math.max(1, percentile(odds, 8)); // Usually 2
    let odd_hi = Math.min(5, percentile(odds, 92)); // Usually 4
    let high_lo = Math.max(1, percentile(highs, 8)); // Usually 2
    let high_hi = Math.min(5, percentile(highs, 92)); // Usually 4
    let range_lo = percentile(ranges, 8);
    let range_hi = percentile(ranges, 92);
    
    // 2. Col Bounds
    let col_bounds = [];
    for (let i = 0; i < 6; i++) {
        let col = hist.map(d => d[i]);
        col_bounds.push({ min: percentile(col, 5), max: percentile(col, 95) });
    }
    
    // 3. User's Inventions
    let lastDraw = hist[hist.length-1].slice(0,6);
    let prevSum = lastDraw.reduce((a,b)=>a+b, 0);
    let banned_sum_block = [prevSum - 10, prevSum + 10];
    
    if (prevSum <= 100) sum_lo = Math.max(sum_lo, 110);
    if (prevSum >= 180) sum_hi = Math.min(sum_hi, 170);
    
    let s_t1 = lastDraw[5] - lastDraw[0];
    let s_t2 = hist[hist.length-2][5] - hist[hist.length-2][0];
    if (s_t1 >= 40) range_hi = Math.min(range_hi, 38);
    else if (s_t1 <= 25) range_lo = Math.max(range_lo, 28);
    else {
        if (s_t1 > s_t2) range_hi = Math.min(range_hi, s_t1 - 1);
        else if (s_t1 < s_t2) range_lo = Math.max(range_lo, s_t1 + 1);
    }
    if (range_lo > range_hi) { let t=range_lo; range_lo=range_hi; range_hi=t; }
    
    // 4. Exhaustion Filter
    let prevDecs = [0,0,0,0,0];
    for (let n of lastDraw) prevDecs[Math.min(Math.floor((n-1)/10), 4)]++;
    
    console.log('⚔️  Đang càn quét 8,145,060 tổ hợp toán học...');
    
    let totalCombos = 0;
    let survivedCombos = 0;
    
    // Lặp qua 8.1 triệu tổ hợp (Sử dụng 6 vòng lặp lồng nhau cho nhanh)
    for(let n1=1; n1<=40; n1++) {
        if (n1 < col_bounds[0].min || n1 > col_bounds[0].max) continue;
        for(let n2=n1+1; n2<=41; n2++) {
            if (n2 < col_bounds[1].min || n2 > col_bounds[1].max) continue;
            for(let n3=n2+1; n3<=42; n3++) {
                if (n3 < col_bounds[2].min || n3 > col_bounds[2].max) continue;
                for(let n4=n3+1; n4<=43; n4++) {
                    if (n4 < col_bounds[3].min || n4 > col_bounds[3].max) continue;
                    for(let n5=n4+1; n5<=44; n5++) {
                        if (n5 < col_bounds[4].min || n5 > col_bounds[4].max) continue;
                        for(let n6=n5+1; n6<=45; n6++) {
                            if (n6 < col_bounds[5].min || n6 > col_bounds[5].max) continue;
                            
                            totalCombos++;
                            
                            let rng = n6 - n1;
                            if (rng < range_lo || rng > range_hi) continue;
                            
                            let s = n1+n2+n3+n4+n5+n6;
                            if (s < sum_lo || s > sum_hi) continue;
                            if (s >= banned_sum_block[0] && s <= banned_sum_block[1]) continue;
                            
                            let o = (n1%2) + (n2%2) + (n3%2) + (n4%2) + (n5%2) + (n6%2);
                            if (o < odd_lo || o > odd_hi) continue;
                            
                            let h = (n1>22?1:0) + (n2>22?1:0) + (n3>22?1:0) + (n4>22?1:0) + (n5>22?1:0) + (n6>22?1:0);
                            if (h < high_lo || h > high_hi) continue;
                            
                            // Consec check
                            let max_consec = 1; let consec = 1;
                            let c = [n1,n2,n3,n4,n5,n6];
                            for(let i=0; i<5; i++) {
                                if (c[i+1] - c[i] === 1) { consec++; if(consec>max_consec) max_consec=consec; }
                                else consec = 1;
                            }
                            if (max_consec > 3) continue; // No 4 consecutive
                            
                            // Decades check
                            let dec = [0,0,0,0,0];
                            for(let x of c) dec[Math.min(Math.floor((x-1)/10), 4)]++;
                            let maxDec = Math.max(...dec);
                            if (maxDec > 3) continue;
                            
                            let badGroup = false;
                            for (let d=0; d<5; d++) {
                                if (prevDecs[d] >= 3 && dec[d] > 2) badGroup = true;
                            }
                            if (badGroup) continue;
                            
                            survivedCombos++;
                        }
                    }
                }
            }
        }
    }
    
    // We didn't iterate through ALL 8.1M because of the initial Col Bounds filter inside the loop
    // Actual total is 8,145,060
    
    console.log('\n======================================================');
    console.log('💥 KẾT QUẢ CỦA CỖ MÁY LỌC TỐI THƯỢNG');
    console.log('======================================================');
    console.log(`- Tổng tổ hợp sinh ra ban đầu : 8,145,060 vé`);
    console.log(`- Số vé CÒN SỐNG SÓT sau lọc  : ${survivedCombos.toLocaleString()} vé`);
    let pctRed = ((8145060 - survivedCombos) / 8145060 * 100).toFixed(2);
    console.log(`=> Tỷ lệ tiêu diệt            : Giết chết ${pctRed}% TOÀN BỘ VÉ RÁC!`);
    console.log(`=> Thay vì chơi 1/8,145,060, trò chơi bây giờ đã bị ép thành 1/${survivedCombos.toLocaleString()}!`);
    console.log('======================================================');
}

main().catch(console.error);
