/**
 * HEAD-TAIL TREND ANALYSIS (Phân tích Xu Hướng Đảo Chiều Đầu Đuôi)
 * Test the user's hypothesis: If Head goes up (1 -> 5), does it tend to go down next draw?
 */
const https = require('https');
function fetchData() { return new Promise((res, rej) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', r => { let d='';r.on('data',c=>d+=c);r.on('end',()=>{const dr=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)dr.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}res(dr);});r.on('error',rej);}); }); }

async function main() {
    const data = await fetchData();
    
    // Stats for HEAD (Số Đầu)
    let head_Up_then_Down = 0;
    let head_Up_then_Up = 0;
    let head_Up_then_Same = 0;
    
    let head_Down_then_Up = 0;
    let head_Down_then_Down = 0;
    let head_Down_then_Same = 0;

    // Stats for TAIL (Số Đuôi)
    let tail_Up_then_Down = 0;
    let tail_Up_then_Up = 0;
    let tail_Up_then_Same = 0;
    
    let tail_Down_then_Up = 0;
    let tail_Down_then_Down = 0;
    let tail_Down_then_Same = 0;

    for (let i = 2; i < data.length; i++) {
        // HEAD analysis
        let h_t2 = data[i-2][0];
        let h_t1 = data[i-1][0];
        let h_t  = data[i][0];
        
        if (h_t1 > h_t2) { // Upward trend previous
            if (h_t < h_t1) head_Up_then_Down++;
            else if (h_t > h_t1) head_Up_then_Up++;
            else head_Up_then_Same++;
        } else if (h_t1 < h_t2) { // Downward trend previous
            if (h_t > h_t1) head_Down_then_Up++;
            else if (h_t < h_t1) head_Down_then_Down++;
            else head_Down_then_Same++;
        }

        // TAIL analysis
        let t_t2 = data[i-2][5];
        let t_t1 = data[i-1][5];
        let t_t  = data[i][5];
        
        if (t_t1 > t_t2) { // Upward trend previous
            if (t_t < t_t1) tail_Up_then_Down++;
            else if (t_t > t_t1) tail_Up_then_Up++;
            else tail_Up_then_Same++;
        } else if (t_t1 < t_t2) { // Downward trend previous
            if (t_t > t_t1) tail_Down_then_Up++;
            else if (t_t < t_t1) tail_Down_then_Down++;
            else tail_Down_then_Same++;
        }
    }
    
    console.log('======================================================');
    console.log('📈 PHÂN TÍCH XU HƯỚNG SỐ ĐẦU (HEAD)');
    console.log('======================================================');
    let headUpTotal = head_Up_then_Down + head_Up_then_Up + head_Up_then_Same;
    console.log(`Khi Số Đầu KỲ TRƯỚC TĂNG (ví dụ 1 -> 5):`);
    console.log(`- KỲ TIẾP THEO GIẢM (Về lại < 5)   : ${head_Up_then_Down} lần (${(head_Up_then_Down/headUpTotal*100).toFixed(2)}%)`);
    console.log(`- KỲ TIẾP THEO TIẾP TỤC TĂNG (> 5) : ${head_Up_then_Up} lần (${(head_Up_then_Up/headUpTotal*100).toFixed(2)}%)`);
    console.log(`- KỲ TIẾP THEO ĐỨNG YÊN (= 5)      : ${head_Up_then_Same} lần (${(head_Up_then_Same/headUpTotal*100).toFixed(2)}%)`);
    
    console.log('');
    let headDownTotal = head_Down_then_Up + head_Down_then_Down + head_Down_then_Same;
    console.log(`Khi Số Đầu KỲ TRƯỚC GIẢM (ví dụ 8 -> 3):`);
    console.log(`- KỲ TIẾP THEO TĂNG LẠI (> 3)      : ${head_Down_then_Up} lần (${(head_Down_then_Up/headDownTotal*100).toFixed(2)}%)`);
    console.log(`- KỲ TIẾP THEO TIẾP TỤC GIẢM (< 3) : ${head_Down_then_Down} lần (${(head_Down_then_Down/headDownTotal*100).toFixed(2)}%)`);
    console.log(`- KỲ TIẾP THEO ĐỨNG YÊN (= 3)      : ${head_Down_then_Same} lần (${(head_Down_then_Same/headDownTotal*100).toFixed(2)}%)`);


    console.log('\n======================================================');
    console.log('📉 PHÂN TÍCH XU HƯỚNG SỐ ĐUÔI (TAIL)');
    console.log('======================================================');
    let tailUpTotal = tail_Up_then_Down + tail_Up_then_Up + tail_Up_then_Same;
    console.log(`Khi Số Đuôi KỲ TRƯỚC TĂNG (ví dụ 35 -> 40):`);
    console.log(`- KỲ TIẾP THEO GIẢM (Về lại < 40)  : ${tail_Up_then_Down} lần (${(tail_Up_then_Down/tailUpTotal*100).toFixed(2)}%)`);
    console.log(`- KỲ TIẾP THEO TIẾP TỤC TĂNG (> 40): ${tail_Up_then_Up} lần (${(tail_Up_then_Up/tailUpTotal*100).toFixed(2)}%)`);
    console.log(`- KỲ TIẾP THEO ĐỨNG YÊN (= 40)     : ${tail_Up_then_Same} lần (${(tail_Up_then_Same/tailUpTotal*100).toFixed(2)}%)`);
    
    console.log('');
    let tailDownTotal = tail_Down_then_Up + tail_Down_then_Down + tail_Down_then_Same;
    console.log(`Khi Số Đuôi KỲ TRƯỚC GIẢM (ví dụ 45 -> 38):`);
    console.log(`- KỲ TIẾP THEO TĂNG LẠI (> 38)     : ${tail_Down_then_Up} lần (${(tail_Down_then_Up/tailDownTotal*100).toFixed(2)}%)`);
    console.log(`- KỲ TIẾP THEO TIẾP TỤC GIẢM (< 38): ${tail_Down_then_Down} lần (${(tail_Down_then_Down/tailDownTotal*100).toFixed(2)}%)`);
    console.log(`- KỲ TIẾP THEO ĐỨNG YÊN (= 38)     : ${tail_Down_then_Same} lần (${(tail_Down_then_Same/tailDownTotal*100).toFixed(2)}%)`);
}

main().catch(console.error);
