/**
 * MATHEMATICAL REALITY CHECK + POOL SIZE ANALYSIS
 * What pool size do we NEED to hit 50% for 6/6?
 */

// Calculate C(n,k) = n! / (k! * (n-k)!)
function comb(n, k) {
    if (k > n) return 0;
    if (k === 0 || k === n) return 1;
    let result = 1;
    for (let i = 0; i < k; i++) {
        result = result * (n - i) / (i + 1);
    }
    return Math.round(result);
}

console.log('═'.repeat(70));
console.log('📐 PHÂN TÍCH TOÁN HỌC: XÁC SUẤT 6/6 THEO KÍCH THƯỚC POOL');
console.log('═'.repeat(70));
console.log('');
console.log('Mega 6/45: Chọn 6 từ 45 số. Tổng tổ hợp = C(45,6) =', comb(45,6).toLocaleString());
console.log('');

console.log('Pool Size | Random 6/6%  | AI cần đạt 50% | Bao nhiêu vé Bao?');
console.log('─'.repeat(65));

for (const poolSize of [6, 10, 15, 20, 25, 30, 33, 35, 37, 40]) {
    const randomProb = comb(poolSize, 6) / comb(45, 6);
    const numTickets = comb(poolSize, 6);
    const costPerTicket = 10000; // 10,000 VND per ticket
    const totalCost = numTickets * costPerTicket;
    
    console.log(
        `Pool-${String(poolSize).padEnd(3)} | ${(randomProb * 100).toFixed(2).padStart(6)}%      | ` +
        `cần ${Math.ceil(50 / (randomProb * 100 * 2)).toLocaleString().padStart(5)}x AI | ` +
        `${numTickets.toLocaleString().padStart(10)} vé (${(totalCost/1000000).toFixed(1)}M VNĐ)`
    );
}

console.log('');
console.log('═'.repeat(70));
console.log('💡 GIẢI THÍCH:');
console.log('');
console.log('• Pool-20 chọn ngẫu nhiên: 6/6 chỉ có 0.48% (1 lần / 210 kỳ)');
console.log('• AI hiện tại (V750A): 6/6 đạt 1.04% (13/1254) = GẤP 2.2x ngẫu nhiên');
console.log('• Để đạt 50% với Pool-20: AI cần dự đoán TỐT HƠN 104x so với ngẫu nhiên');
console.log('  → ĐÂY LÀ BẤT KHẢ THI vì xổ số là HOÀN TOÀN NGẪU NHIÊN');
console.log('');
console.log('📌 PHƯƠNG ÁN THỰC TẾ ĐỂ TĂNG 6/6:');
console.log('  1. TĂNG POOL SIZE lên 30-35 số → xác suất ngẫu nhiên 6/6 = 4-10%');
console.log('  2. AI boost 2-3x → có thể đạt 8-30% cho 6/6');
console.log('  3. Dùng DÀN BAO 30-35 số sẽ bao phủ rộng hơn nhiều');
console.log('');
console.log('═'.repeat(70));

// Test actual achievable rates with larger pools
console.log('');
console.log('🎯 DỰ ĐOÁN KHẢ THI (dựa trên hiệu suất AI 2.2x):');
console.log('─'.repeat(50));
for (const poolSize of [20, 25, 30, 33, 35]) {
    const randomProb = comb(poolSize, 6) / comb(45, 6);
    const aiProb = randomProb * 2.2; // AI factor from backtest
    const numTickets = comb(poolSize, 6);
    console.log(
        `Pool-${poolSize}: Random=${(randomProb*100).toFixed(1)}% → AI≈${(aiProb*100).toFixed(1)}% | ` +
        `Bao ${poolSize} = ${numTickets.toLocaleString()} vé`
    );
}
console.log('═'.repeat(70));
