const fs = require('fs');

const data_path = 'C:\\\\Users\\\\HQSP\\\\.gemini\\\\antigravity\\\\scratch\\\\tinnam888888_test\\\\data\\\\mega645.json';
const raw = JSON.parse(fs.readFileSync(data_path, 'utf8'));
const data = raw.map(d => d.numbers.map(Number).sort((a,b)=>a-b)).reverse(); // Cũ nhất trước

// Mô phỏng lại hàm _build_entanglement_matrix
const parasitic = new Set();
const symbiotic = {};

const single_counts = {};
const pair_counts = {};

const scan_data = data.slice(-500);

for(let draw of scan_data) {
    for(let num of draw) {
        single_counts[num] = (single_counts[num] || 0) + 1;
    }
    for(let i=0; i<draw.length; i++) {
        for(let j=i+1; j<draw.length; j++) {
            let a = draw[i], b = draw[j];
            let p = a < b ? a+"_"+b : b+"_"+a;
            pair_counts[p] = (pair_counts[p] || 0) + 1;
        }
    }
}

for(let a=1; a<=45; a++) {
    for(let b=a+1; b<=45; b++) {
        if((single_counts[a] || 0) > 10 && (single_counts[b] || 0) > 10) {
            let p = a+"_"+b;
            if(!pair_counts[p]) {
                parasitic.add(p);
            }
        }
    }
}

for(let p in pair_counts) {
    let count = pair_counts[p];
    if(count >= 4) {
        let [a, b] = p.split("_").map(Number);
        let ca = single_counts[a] || 1;
        let cb = single_counts[b] || 1;
        if(count / ca > 0.6) symbiotic[a] = b;
        if(count / cb > 0.6) symbiotic[b] = a;
    }
}

console.log("[V2500] CAC CAP TUONG KHAC (Tuyet doi cam di chung):");
let p_list = Array.from(parasitic).slice(0,15);
for(let p of p_list) {
    let [a, b] = p.split("_");
    console.log(`  X ${a} va ${b}`);
}
console.log(`Tong so cap ky nhau: ${parasitic.size}`);

console.log("\\n[V2500] CAC CAP TUONG SINH (Cong diem neu di chung):");
let s_keys = Object.keys(symbiotic).slice(0,15);
for(let k of s_keys) {
    console.log(`  + ${k} keo theo ${symbiotic[k]}`);
}
console.log(`Tong so cap tuong sinh: ${Object.keys(symbiotic).length}`);

// Giả lập loại trừ
// Tạo 10,000 vé random để test tỉ lệ bị diệt
let eliminated = 0;
let total = 100000;
for(let i=0; i<total; i++) {
    let ticket = [];
    let pool = [];
    for(let j=1; j<=45; j++) pool.push(j);
    for(let j=0; j<6; j++) {
        let idx = Math.floor(Math.random() * pool.length);
        ticket.push(pool[idx]);
        pool.splice(idx, 1);
    }
    ticket.sort((a,b)=>a-b);
    
    // Check parasitic
    let rejected = false;
    for(let a=0; a<6; a++) {
        for(let b=a+1; b<6; b++) {
            let p = ticket[a]+"_"+ticket[b];
            if(parasitic.has(p)) {
                rejected = true;
                break;
            }
        }
        if(rejected) break;
    }
    if(rejected) eliminated++;
}

console.log(`\\n[Mô Phỏng Quét 100,000 Vé]`);
console.log(`Số vé bị DIỆT TẬN GỐC bởi luật Tương Khắc: ${eliminated} vé (${(eliminated/total*100).toFixed(2)}%)`);
