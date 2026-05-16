const fs = require('fs');
const filename = 'streamlit_app.py';
const lines = fs.readFileSync(filename, 'utf-8').split('\n');

// 655:     with st.expander("Bấm để xem Phân tích Lịch sử từ kỳ đầu tiên đến nay"):
// it goes until 732. So from index 655 to 732, add 4 spaces.
for (let i = 655; i <= 732; i++) {
    lines[i] = '    ' + lines[i];
}

// 735:     with st.expander("Bấm để chạy Backtest (Kiểm thử thực tế với thuật toán Dàn Bao)"):
// it goes until 1060.
for (let i = 735; i <= 1058; i++) {
    lines[i] = '    ' + lines[i];
}

fs.writeFileSync(filename, lines.join('\n'));
console.log('Fixed indentation');
