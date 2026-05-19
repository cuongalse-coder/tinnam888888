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
    
    // Sliding Window
    let missing_pool = new Set();
    let hot_pool = new Set();
    let window10 = data.slice(data.length - 10);
    let counts10 = Array(46).fill(0);
    for(let d of window10) for(let x of d.slice(0,6)) counts10[x]++;
    for(let n=1; n<=45; n++) {
        if(counts10[n] === 0) missing_pool.add(n);
        else if(counts10[n] >= 2) hot_pool.add(n);
    }
    
    // Markov Transitions
    let markov_transitions = [{},{},{},{},{},{}];
    for(let i=1; i<data.length; i++){
        let p_draw = data[i-1].slice(0,6);
        let c_draw = data[i].slice(0,6);
        for(let c=0; c<6; c++){
            let pv = p_draw[c], cv = c_draw[c];
            if(!markov_transitions[c][pv]) markov_transitions[c][pv] = new Set();
            markov_transitions[c][pv].add(cv);
        }
    }
    
    // Frequency Polarity
    let top_frequent = new Set();
    let window100 = data.slice(data.length - 100);
    let freqs100 = Array(46).fill(0);
    for(let d of window100) for(let x of d.slice(0,6)) freqs100[x]++;
    let arr100 = [];
    for(let n=1; n<=45; n++) arr100.push({n, c: freqs100[n]});
    arr100.sort((a,b) => b.c - a.c);
    for(let i=0; i<22; i++) top_frequent.add(arr100[i].n);
    
    // Go Board Setup
    let prev_draw_set = new Set(prevDraw);
    let go_board_liberties = new Set();
    for(let b of prevDraw) {
        let r=Math.floor((b-1)/9), c=(b-1)%9;
        if(r>0) go_board_liberties.add((r-1)*9+c+1);
        if(r<4) go_board_liberties.add((r+1)*9+c+1);
        if(c>0) go_board_liberties.add(r*9+(c-1)+1);
        if(c<8) go_board_liberties.add(r*9+(c+1)+1);
    }
    for(let b of prevDraw) go_board_liberties.delete(b);
    
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
        survive_rubik: 0,
        survive_color: 0,
        survive_go: 0,
        survive_sliding: 0,
        survive_markov: 0,
        survive_hacker: 0,
        survive_polarity: 0,
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
                            
                            // 10. Lọc Ma Trận Rubik
                            let matrix = Array(5).fill().map(()=>Array(10).fill(0));
                            for(let x of combo){ let r=Math.floor(x/10); let c=x%10; if(r<5&&c<10) matrix[r][c]=1; }
                            let has_2x2=false, has_diag3=false;
                            for(let r=0;r<4;r++)for(let c=0;c<9;c++)if(matrix[r][c]&&matrix[r][c+1]&&matrix[r+1][c]&&matrix[r+1][c+1]) has_2x2=true;
                            for(let r=0;r<3;r++){
                                for(let c=0;c<8;c++)if(matrix[r][c]&&matrix[r+1][c+1]&&matrix[r+2][c+2]) has_diag3=true;
                                for(let c=2;c<10;c++)if(matrix[r][c]&&matrix[r+1][c-1]&&matrix[r+2][c-2]) has_diag3=true;
                            }
                            if(has_2x2 || has_diag3) continue;
                            stats.survive_rubik++;
                            
                            // 11. Lọc Bảng Màu Ngũ Hành
                            let colors = [0,0,0,0,0];
                            for(let x of combo){
                                let ld=x%10;
                                if(ld===1||ld===6) colors[0]++;
                                else if(ld===2||ld===7) colors[1]++;
                                else if(ld===3||ld===8) colors[2]++;
                                else if(ld===4||ld===9) colors[3]++;
                                else if(ld===5||ld===0) colors[4]++;
                            }
                            let uColors = colors.filter(c=>c>0).length;
                            if(uColors<=2 || Math.max(...colors)>=4) continue;
                            stats.survive_color++;
                            
                            // 12. Lọc Cờ Vây (Go Board)
                            let overlap=0, contact=0;
                            for(let x of combo){
                                if(prev_draw_set.has(x)) overlap++;
                                else if(go_board_liberties.has(x)) contact++;
                            }
                            if(overlap>2 || contact>4) continue;
                            stats.survive_go++;
                            
                            // 13. Lọc Chu Kỳ 10 Kỳ (Sliding Window)
                            let missing_hit = 0;
                            let hot_hit = 0;
                            for(let x of combo) {
                                if (missing_pool.has(x)) missing_hit++;
                                else if (hot_pool.has(x)) hot_hit++;
                            }
                            if (missing_hit > 3 || hot_hit > 3) continue;
                            stats.survive_sliding++;
                            
                            // 14. Đường Rẽ Markov (Markov Transitions)
                            let markov_pass = 0;
                            for(let i=0; i<6; i++) {
                                let p_val = prevDraw[i];
                                let c_val = combo[i];
                                if (markov_transitions[i] && markov_transitions[i][p_val] && markov_transitions[i][p_val].has(c_val)) {
                                    markov_pass++;
                                }
                            }
                            if (markov_pass < 4) continue;
                            stats.survive_markov++;
                            
                            // 15. Mật Mã Hacker 12-bit (Hacker Cipher)
                            let s_bin = "";
                            for(let x of combo) {
                                let x_str = x.toString().padStart(2, '0');
                                s_bin += parseInt(x_str[0]) % 2 === 0 ? "0" : "1";
                                s_bin += parseInt(x_str[1]) % 2 === 0 ? "0" : "1";
                            }
                            let max_0 = 0, c0 = 0;
                            for(let b of s_bin) { if (b === "0") { c0++; if(c0>max_0) max_0=c0; } else c0=0; }
                            let max_1 = 0, c1 = 0;
                            for(let b of s_bin) { if (b === "1") { c1++; if(c1>max_1) max_1=c1; } else c1=0; }
                            let is_pal = s_bin === s_bin.split('').reverse().join('');
                            let is_alt = s_bin === "010101010101" || s_bin === "101010101010";
                            if (max_0 >= 7 || max_1 >= 7 || is_pal || is_alt) continue;
                            stats.survive_hacker++;
                            
                            // 16. Cân Bằng Tần Suất (Frequency Polarity)
                            let top_hit = 0;
                            for(let x of combo) {
                                if (top_frequent.has(x)) top_hit++;
                            }
                            if (top_hit < 2 || top_hit > 4) continue;
                            stats.survive_polarity++;
                            
                            stats.final_survivors++;
                        }
                    }
                }
            }
        }
    }
    
    console.log('\n========================================================================');
    console.log('🏆 PHỄU LỌC TỔNG HỢP: 16 LỚP KHIÊN BẢO VỆ (THE ULTIMATE SYNERGY GAUNTLET)');
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
    step("Lọc Thập kỷ/Liên tiếp", stats.survive_consec_dec);
    step("Lọc Ma Trận Rubik", stats.survive_rubik);
    step("Lọc Màu Ngũ Hành", stats.survive_color);
    step("Lọc Địa Bàn Cờ Vây", stats.survive_go);
    step("Lọc Chu Kỳ (Sliding)", stats.survive_sliding);
    step("Đường Rẽ (Markov)", stats.survive_markov);
    step("Mật Mã Hacker", stats.survive_hacker);
    step("Cân Bằng Tần Suất", stats.survive_polarity);
    step("LỌC TỔNG CUỐI CÙNG", stats.final_survivors);
    
    let finalPct = ((stats.total - stats.final_survivors) / stats.total * 100).toFixed(2);
    console.log('========================================================================');
    console.log(`💥 KẾT LUẬN CUỐI CÙNG:`);
    console.log(`- Hệ thống đã HỦY DIỆT VĨNH VIỄN: ${(stats.total - stats.final_survivors).toLocaleString()} tổ hợp vô lý!`);
    console.log(`- Đạt TỶ LỆ NÉN TUYỆT ĐỐI      : ${finalPct}% KHÔNG GIAN BỊ CHÉM BAY!`);
    console.log(`- Trò chơi bây giờ chỉ còn     : MỘT CUỘC ĐẤU 1 CHỌN ${stats.final_survivors.toLocaleString()}!`);
    console.log('========================================================================');
}

main().catch(console.error);
