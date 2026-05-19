/**
 * BÀI TEST TỔNG HÒA SIÊU MÀNG LỌC (THE ULTIMATE SYNERGY TEST)
 * Áp dụng toàn bộ 10 lớp khiên bảo vệ lên không gian 8,145,060 tổ hợp.
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
    console.log("⏳ Đang tải dữ liệu lịch sử để học các thông số giới hạn...");
    const data = await fetchData();
    const hist = data;
    
    // 1. Học giới hạn từ lịch sử
    let recent = hist.slice(-200); // Dùng 200 kỳ gần nhất cho chính xác
    let sums = recent.map(d => d.slice(0,6).reduce((a,b)=>a+b, 0));
    let odds = recent.map(d => d.slice(0,6).filter(x => x%2===1).length);
    let highs = recent.map(d => d.slice(0,6).filter(x => x>22).length);
    let ranges = recent.map(d => d[5] - d[0]);
    let deltas = recent.map(d => { let m=d[0]; for(let i=1;i<6;i++) if(d[i]-d[i-1]>m) m=d[i]-d[i-1]; return m; });
    
    let sum_lo = percentile(sums, 8); let sum_hi = percentile(sums, 92);
    let odd_lo = Math.max(0, percentile(odds, 8)); let odd_hi = Math.min(6, percentile(odds, 92));
    let high_lo = Math.max(0, percentile(highs, 8)); let high_hi = Math.min(6, percentile(highs, 92));
    let range_lo = percentile(ranges, 8); let range_hi = percentile(ranges, 92);
    let delta_hi = percentile(deltas, 95);
    
    let col_bounds = [];
    for (let i = 0; i < 6; i++) {
        let col = recent.map(d => d[i]);
        col_bounds.push({ min: percentile(col, 3), max: percentile(col, 97) });
    }
    
    let prevDraw = hist[hist.length-1].slice(0,6);
    let prevSum = prevDraw.reduce((a,b)=>a+b, 0);
    let banned_sum_block = [prevSum - 10, prevSum + 10];
    
    if (prevSum <= 100) sum_lo = Math.max(sum_lo, 110);
    if (prevSum >= 180) sum_hi = Math.min(sum_hi, 170);

    console.log("⚔️ Khởi động Cỗ Máy Quét 8,145,060 tổ hợp...");
    
    let stats = {
        total: 0,
        survive_col: 0,
        survive_sum: 0,
        survive_block: 0,
        survive_delta: 0,
        survive_digit: 0,
        survive_adj: 0,
        survive_wave: 0,
        survive_odd_high: 0,
        survive_consec_dec: 0,
        final_survivors: 0
    };

    // Generating all 8.1M combinations
    for(let n1=1; n1<=40; n1++) {
        for(let n2=n1+1; n2<=41; n2++) {
            for(let n3=n2+1; n3<=42; n3++) {
                for(let n4=n3+1; n4<=43; n4++) {
                    for(let n5=n4+1; n5<=44; n5++) {
                        for(let n6=n5+1; n6<=45; n6++) {
                            stats.total++;
                            let combo = [n1, n2, n3, n4, n5, n6];
                            
                            // 1. Lọc Cột (Col Bounds)
                            let passCol = true;
                            for (let i=0; i<6; i++) {
                                if (combo[i] < col_bounds[i].min || combo[i] > col_bounds[i].max) { passCol = false; break; }
                            }
                            if (!passCol) continue;
                            stats.survive_col++;
                            
                            // 2. Lọc Tổng (Sum Range)
                            let s = n1+n2+n3+n4+n5+n6;
                            if (s < sum_lo || s > sum_hi) continue;
                            stats.survive_sum++;
                            
                            // 3. Lọc Khối Tổng (Sum Block - Banned)
                            if (s >= banned_sum_block[0] && s <= banned_sum_block[1]) continue;
                            stats.survive_block++;
                            
                            // 4. Lọc Delta System (Global)
                            let md = n1;
                            for(let i=1; i<6; i++) { if (combo[i]-combo[i-1] > md) md = combo[i]-combo[i-1]; }
                            if (md > delta_hi) continue;
                            stats.survive_delta++;
                            
                            // 5. Lọc Tần suất Chữ số (Digit Frequency)
                            let dCounts = [0,0,0,0,0,0,0,0,0,0];
                            let cStr = "";
                            for(let x of combo) {
                                let st = x.toString().padStart(2, '0');
                                cStr += st;
                                dCounts[parseInt(st[0])]++;
                                dCounts[parseInt(st[1])]++;
                            }
                            if (Math.max(...dCounts) > 4) continue;
                            stats.survive_digit++;
                            
                            // 6. Lọc Cặp Chữ số Liền kề (Adjacent Digits)
                            let adj = 0;
                            for (let i=1; i<12; i++) { if (cStr[i] === cStr[i-1]) adj++; }
                            if (adj > 2) continue;
                            stats.survive_adj++;
                            
                            // 7. Lọc Điểm Ngắt Sóng (Wave Inflection)
                            let ones = [n1%10, n2%10, n3%10, n4%10, n5%10, n6%10];
                            let breaks = 0; let curDir = 0;
                            for (let i=1; i<6; i++) {
                                let dir = 0;
                                if (ones[i] > ones[i-1]) dir = 1;
                                else if (ones[i] < ones[i-1]) dir = -1;
                                if (dir !== 0) {
                                    if (curDir !== 0 && curDir !== dir) breaks++;
                                    curDir = dir;
                                }
                            }
                            if (breaks === 0) continue;
                            stats.survive_wave++;
                            
                            // 8. Lọc Chẵn/Lẻ và Cao/Thấp
                            let odd = (n1%2)+(n2%2)+(n3%2)+(n4%2)+(n5%2)+(n6%2);
                            if (odd < odd_lo || odd > odd_hi) continue;
                            let high = (n1>22?1:0)+(n2>22?1:0)+(n3>22?1:0)+(n4>22?1:0)+(n5>22?1:0)+(n6>22?1:0);
                            if (high < high_lo || high > high_hi) continue;
                            stats.survive_odd_high++;
                            
                            // 9. Lọc Tới Hạn Nhóm (Max 3/decade) & Liên tiếp (Max 3 consec)
                            let max_consec=1; let consec=1;
                            for(let i=0; i<5; i++) {
                                if (combo[i+1]-combo[i]===1) { consec++; if(consec>max_consec) max_consec=consec; }
                                else consec = 1;
                            }
                            if (max_consec > 3) continue;
                            let decs = [0,0,0,0,0];
                            for(let x of combo) decs[Math.min(Math.floor((x-1)/10), 4)]++;
                            if (Math.max(...decs) > 3) continue;
                            
                            stats.final_survivors++;
                        }
                    }
                }
            }
        }
    }
    
    console.log('\n========================================================================');
    console.log('🏆 PHỄU LỌC TỔNG HỢP: 10 LỚP KHIÊN BẢO VỆ (THE ULTIMATE SYNERGY GAUNTLET)');
    console.log('========================================================================');
    console.log(`Bắt đầu với KHÔNG GIAN TOÀN VẸN: ${stats.total.toLocaleString()} tổ hợp`);
    
    let prev = stats.total;
    let step = (name, current) => {
        let killed = prev - current;
        let pct = (killed / prev * 100).toFixed(2);
        console.log(`- Qua ${name.padEnd(20)}: Còn ${current.toLocaleString().padStart(9)} vé (Giết chết ${killed.toLocaleString().padStart(7)} vé | ${pct}%)`);
        prev = current;
    }
    
    step("Lọc Cột (Vị Trí)", stats.survive_col);
    step("Lọc Sàn Tổng", stats.survive_sum);
    step("Lọc Khối Tổng", stats.survive_block);
    step("Lọc Delta Toàn Cầu", stats.survive_delta);
    step("Lọc Tần Suất Chữ Số", stats.survive_digit);
    step("Lọc Chữ Số Kề Nhau", stats.survive_adj);
    step("Lọc Điểm Ngắt Sóng", stats.survive_wave);
    step("Lọc Chẵn Lẻ/Cao Thấp", stats.survive_odd_high);
    step("Lọc Thập kỷ/Liên tiếp", stats.final_survivors);
    
    let finalPct = ((stats.total - stats.final_survivors) / stats.total * 100).toFixed(2);
    console.log('========================================================================');
    console.log(`💥 KẾT LUẬN CUỐI CÙNG:`);
    console.log(`- Hệ thống đã HỦY DIỆT VĨNH VIỄN: ${(stats.total - stats.final_survivors).toLocaleString()} tổ hợp vô lý!`);
    console.log(`- Đạt TỶ LỆ NÉN TUYỆT ĐỐI      : ${finalPct}% KHÔNG GIAN BỊ CHÉM BAY!`);
    console.log(`- Trò chơi bây giờ chỉ còn     : MỘT CUỘC ĐẤU 1 CHỌN ${stats.final_survivors.toLocaleString()}!`);
    console.log('========================================================================');
}

main().catch(console.error);
