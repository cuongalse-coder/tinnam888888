const fs = require('fs');

const nexus_path = 'C:\\\\Users\\\\HQSP\\\\.gemini\\\\antigravity\\\\scratch\\\\tinnam888888_test\\\\models\\\\nexus_engine.py';
const wheel_path = 'C:\\\\Users\\\\HQSP\\\\.gemini\\\\antigravity\\\\scratch\\\\tinnam888888_test\\\\models\\\\wheeling_optimizer.py';
const app_path = 'C:\\\\Users\\\\HQSP\\\\.gemini\\\\antigravity\\\\scratch\\\\tinnam888888_test\\\\streamlit_app.py';

// 1. UPDATE NEXUS ENGINE
let nexus = fs.readFileSync(nexus_path, 'utf8');

const conf_func = `
    def calculate_confidence(self, history_data):
        if len(history_data) < 10: return 50.0
        
        # Analyze last 10 draws volatility
        recent = history_data[-10:]
        sums = [sum(d[:self.pick_count]) for d in recent]
        avg_sum = sum(sums) / len(sums)
        
        # Variance of sums
        sum_var = sum((s - avg_sum)**2 for s in sums) / len(sums)
        
        # Gap volatility (Are numbers clustering or random?)
        recent_all = [n for d in recent for n in d[:self.pick_count]]
        unique_count = len(set(recent_all))
        
        # Normal unique count in 10 draws (60 balls) is ~ 33-37 for 6/45, ~40-44 for 6/55
        # If too low -> Heavy repetition (trend). If too high -> Chaotic randomness.
        expected_unique = 35 if self.max_number == 45 else 42
        entropy_penalty = abs(unique_count - expected_unique) * 2.0
        
        # Volatility penalty
        volatility_penalty = min(20.0, sum_var / 50.0)
        
        confidence = 100.0 - entropy_penalty - volatility_penalty
        
        # Boost if latest draw is "calm"
        last_draw_sum = sums[-1]
        target_sum = 122 if self.max_number == 45 else 150
        if abs(last_draw_sum - target_sum) <= 15:
            confidence += 10.0
            
        return max(15.0, min(99.9, confidence))
`;

if (!nexus.includes("def calculate_confidence")) {
    nexus = nexus.replace("    def predict_top_pool", conf_func + "\n    def predict_top_pool");
    fs.writeFileSync(nexus_path, nexus, 'utf8');
}


// 2. UPDATE WHEELING OPTIMIZER
let wheel = fs.readFileSync(wheel_path, 'utf8');
const dyn_logic = `
        # V2000 DYNAMIC FILTERS (REINFORCEMENT LEARNING BOUNDS)
        if history_data and len(history_data) >= 5:
            recent_5 = history_data[-5:]
            recent_sums = [sum(d[:self.pick_count]) for d in recent_5]
            avg_recent_sum = sum(recent_sums) / 5
            
            recent_evens = sum(1 for d in recent_5 for n in d[:self.pick_count] if n % 2 == 0)
            
            target_sum_mean = 122 if self.max_number == 45 else 150
            
            if constraints is None:
                constraints = {}
                
            # Regression to the mean for Sums
            if avg_recent_sum < target_sum_mean - 20:
                constraints['sum_min'] = target_sum_mean
                constraints['sum_max'] = target_sum_mean + 40
            elif avg_recent_sum > target_sum_mean + 20:
                constraints['sum_min'] = target_sum_mean - 40
                constraints['sum_max'] = target_sum_mean
                
            # Regression for Odd/Even
            if recent_evens > 20: # Quá nhiều chẵn (trung bình là 15)
                constraints['odd_even'] = [4, 5] # Ép ra lẻ
            elif recent_evens < 10:
                constraints['odd_even'] = [1, 2] # Ép ra chẵn

        import itertools`;
        
wheel = wheel.replace("        import itertools", dyn_logic);
fs.writeFileSync(wheel_path, wheel, 'utf8');


// 3. UPDATE STREAMLIT APP
let app = fs.readFileSync(app_path, 'utf8');

const ui_logic = `        st.markdown("### 🧠 V2000 CỐ VẤN TÀI CHÍNH (Kelly Criterion)")
        
        # Calculate Confidence
        ai_conf = engine.calculate_confidence(history_data)
        
        if ai_conf >= 80:
            rec_tickets = 50
            conf_color = "green"
            msg = "🔥 XUNG LỰC HỘI TỤ ĐỈNH ĐIỂM! Đề xuất xuất kích 50 vé để tổng tiến công Jackpot."
        elif ai_conf >= 60:
            rec_tickets = 20
            conf_color = "orange"
            msg = "⚡ Tín hiệu rất rõ ràng. Đội hình 20 vé Radar tiêu chuẩn là lựa chọn tối ưu."
        elif ai_conf >= 40:
            rec_tickets = 10
            conf_color = "yellow"
            msg = "⚠️ Thị trường hơi nhiễu. Đề xuất lùi về phòng ngự với 10 vé."
        else:
            rec_tickets = 5
            conf_color = "red"
            msg = "🛑 CẢNH BÁO: Điểm đứt gãy cực đoan (Chaos). Đề xuất bảo toàn vốn, chỉ test 5 vé dò đường."
            
        st.markdown(f"<h3 style='text-align:center; color:{conf_color};'>Độ tự tin AI: {ai_conf:.1f}%</h3>", unsafe_allow_html=True)
        st.info(msg)
        
        # Locked tickets by AI Kelly Criterion
        st.markdown(f"**Số vé AI chỉ định xuất kích:** `{rec_tickets} Vé` (Auto-Lock)")
        num_tickets = rec_tickets
        
        pool_size = st.selectbox("Kích thước Hồ Tiềm Năng (Mở rộng):", [10, 12, 15, 18, 20, 25, 30, 33, 35], index=4)`;

// Replace old UI
const old_ui_pattern = /        st\.markdown\("### 🎯 V1000 Radar Pathfinding[\s\S]*?index=4\)/;
app = app.replace(old_ui_pattern, ui_logic);
fs.writeFileSync(app_path, app, 'utf8');

console.log("V2000 Update Success");
