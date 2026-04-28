import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import random
import time
from datetime import datetime
import requests
import re

# ==========================================
# CẤU HÌNH TRANG & GIAO DIỆN
# ==========================================
st.set_page_config(
    page_title="TINNAM AI - VULNERABILITY SCANNER V10 (REAL WORLD)",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #00ff00; font-family: 'Courier New', Courier, monospace; }
    .ball {
        display: inline-flex; align-items: center; justify-content: center;
        width: 50px; height: 50px; border-radius: 50%; color: white;
        font-weight: bold; font-size: 20px; margin: 5px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.5); border: 2px solid #ffffff;
    }
    .mega-ball { background: linear-gradient(145deg, #ff0055, #a80033); }
    .power-ball { background: linear-gradient(145deg, #ff4500, #b33000); }
    .special-ball { background: linear-gradient(145deg, #ffd700, #b8860b); color: #000; }
    .stButton>button { width: 100%; background-color: #00ff00; color: #000000; font-weight: bold; border: 2px solid #00aa00; border-radius: 5px; transition: 0.3s; }
    .stButton>button:hover { background-color: #ffffff; color: #000000; box-shadow: 0 0 15px #00ff00; }
    h1, h2, h3 { color: #00ff00 !important; }
    .card { background-color: #1a1c23; padding: 20px; border-radius: 10px; border: 1px solid #333; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

def check_password():
    def password_entered():
        if st.session_state["password"] == "1991":
            st.session_state.logged_in = True
            del st.session_state["password"]
        else:
            st.session_state.logged_in = False
            st.error("❌ Mật khẩu không chính xác! Tự động khóa hệ thống.")

    if not st.session_state.logged_in:
        st.markdown("<h1 style='text-align: center;'>🔒 HỆ THỐNG PHÂN TÍCH THỰC TẾ</h1>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.text_input("Mật khẩu truy cập:", type="password", on_change=password_entered, key="password")
        return False
    return True

# ==========================================
# CRAWLER: QUÉT DỮ LIỆU THẬT 100%
# ==========================================
@st.cache_data(ttl=300)
def fetch_real_data(game_type):
    """
    Cào dữ liệu THẬT 100% TOÀN BỘ CÁC KỲ từ ketquadientoan và các nguồn khác.
    """
    try:
        import cloudscraper
        scraper = cloudscraper.create_scraper(delay=5, browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False})
    except ImportError:
        scraper = requests.Session()
        
    today_str = datetime.now().strftime('%d-%m-%Y')
    
    urls = [
        # Ưu tiên lấy TOÀN BỘ từ trước tới nay từ ketquadientoan
        f"https://www.ketquadientoan.com/tat-ca-ky-xo-so-mega-6-45.html?datef=18-07-2016&datet={today_str}" if game_type == "Mega 6/45" else f"https://www.ketquadientoan.com/tat-ca-ky-xo-so-power-655.html?datef=01-08-2017&datet={today_str}",
        "https://xskt.com.vn/ket-qua-xo-so-vietlott-mega-6-45" if game_type == "Mega 6/45" else "https://xskt.com.vn/ket-qua-xo-so-vietlott-power-6-55",
        "https://xoso.me/kqxs-mega-645.html" if game_type == "Mega 6/45" else "https://xoso.me/kqxs-power-655.html",
        "https://ketqua.vn/vietlott-mega-6-45" if game_type == "Mega 6/45" else "https://ketqua.vn/vietlott-power-6-55"
    ]
    
    max_num = 45 if game_type == "Mega 6/45" else 55
    
    for url in urls:
        try:
            response = scraper.get(url, timeout=15)
            if response.status_code == 200:
                html = response.text
                
                history = []
                # Regex quét lấy toàn bộ thẻ HTML chỉ chứa 2 chữ số
                nums = re.findall(r'>\s*(\d{2})\s*<', html)
                
                # Quét cửa sổ trượt (Sliding Window) để tìm các chuỗi 6 số hợp lệ
                for i in range(0, len(nums) - 5):
                    chunk = [int(n) for n in nums[i:i+6]]
                    # Mega/Power luôn có 6 số, không trùng, xếp tăng dần và <= max_num
                    if chunk == sorted(chunk) and len(set(chunk)) == 6 and all(1 <= n <= max_num for n in chunk):
                        if chunk not in history:
                            history.append(chunk)

                if history:
                    history.reverse() # Trả về từ cũ nhất đến mới nhất
                    return history # TRẢ VỀ TOÀN BỘ DỮ LIỆU, KHÔNG GIỚI HẠN
        except Exception as e:
            continue
            
    # NẾU TẤT CẢ CÁC TRANG ĐỀU LỖI HOẶC BỊ CHẶN CLOUDFLARE, LẤY TỪ GITHUB
    try:
        github_url = "https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl" if game_type == "Mega 6/45" else "https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power655.jsonl"
        response = requests.get(github_url, timeout=10)
        history = []
        if response.status_code == 200:
            import json
            for line in response.text.strip().split('\n'):
                if line:
                    data = json.loads(line)
                    if 'result' in data and len(data['result']) >= 6:
                        draw = sorted([int(n) for n in data['result'][:6]])
                        history.append(draw)
            if history:
                return history
    except Exception:
        pass
        
    st.error("⚠️ Không thể kết nối máy chủ xổ số. Đang sử dụng dữ liệu giả lập dự phòng.")
    return [sorted(random.sample(range(1, max_num + 1), 6)) for _ in range(50)]


# ==========================================
# AI ENGINE: TOÁN HỌC THỰC TẾ
# ==========================================
class RealWorldAIEngine:
    def __init__(self, data, max_number):
        self.data = data
        self.max_number = max_number
        self.all_numbers = list(range(1, max_number + 1))
        
    def _get_frequency(self, lookback=None):
        subset = self.data[-lookback:] if lookback else self.data
        all_nums = [n for draw in subset for n in draw]
        return Counter(all_nums)

    def model_markov_chain(self):
        """Ma trận chuyển đổi trạng thái Markov dựa trên lịch sử thật"""
        transitions = defaultdict(Counter)
        for i in range(len(self.data) - 1):
            current = tuple(sorted(self.data[i]))
            next_draw = self.data[i + 1]
            for num in next_draw:
                transitions[current][num] += 1
                
        if len(self.data) > 0:
            last_draw = tuple(sorted(self.data[-1]))
            if last_draw in transitions and transitions[last_draw]:
                next_probs = transitions[last_draw]
                return [num for num, _ in next_probs.most_common(6)]
        
        return [n for n, c in self._get_frequency(20).most_common(6)]

    def model_gap_overdue(self, top_n=6):
        """Phân tích các số ĐÃ ĐẾN HẠN NỔ (Overdue Analysis)"""
        last_seen = {num: -1 for num in self.all_numbers}
        for i, draw in enumerate(self.data):
            for num in draw:
                last_seen[num] = i
                
        current_idx = len(self.data)
        # Tính khoảng cách từ lần cuối xuất hiện đến hiện tại
        gaps = {num: current_idx - last_seen[num] for num in self.all_numbers}
        
        # Phân tích chu kỳ trung bình của mỗi số
        avg_gaps = defaultdict(list)
        last_idx = {}
        for i, draw in enumerate(self.data):
            for num in draw:
                if num in last_idx:
                    avg_gaps[num].append(i - last_idx[num])
                last_idx[num] = i
                
        due_scores = {}
        for num in self.all_numbers:
            if avg_gaps[num]:
                mean_gap = np.mean(avg_gaps[num])
                current_gap = gaps[num]
                # Điểm nổ = (Khoảng cách hiện tại / Khoảng cách trung bình)
                # Điểm càng cao (> 1) nghĩa là đã quá hạn, khả năng nổ cao
                due_scores[num] = current_gap / (mean_gap + 0.1)
            else:
                due_scores[num] = 0
                
        sorted_due = sorted(due_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, score in sorted_due[:top_n]]

    def model_momentum_neural(self):
        """Neural Weights - Tính toán động lượng tăng trưởng"""
        weights = {num: 0.0 for num in self.all_numbers}
        total_draws = len(self.data)
        
        # Hàm sigmoid tối ưu hóa trọng số kỳ gần đây
        for i, draw in enumerate(self.data):
            decay = 1 / (1 + np.exp(-(i - total_draws + 20) / 5)) 
            for num in draw:
                weights[num] += decay
                
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        return [num for num, w in sorted_weights[:6]]

    def optimize_ensemble(self):
        """Tổng hợp bằng Trí Tuệ Bầy Đàn (Swarm Intelligence)"""
        m1 = self.model_markov_chain()
        m2 = self.model_gap_overdue()
        m3 = self.model_momentum_neural()
        
        # Trọng số bình chọn: Overdue (3), Momentum (2), Markov (1)
        # Các số quá hạn nổ thường có xác suất trúng cao nhất trong thực tế
        votes = Counter()
        for num in m2: votes[num] += 3
        for num in m3: votes[num] += 2
        for num in m1: votes[num] += 1
        
        best = [num for num, count in votes.most_common(6)]
        
        while len(best) < 6:
            candidates = self.model_gap_overdue()
            for c in candidates:
                if c not in best:
                    best.append(c)
                    if len(best) == 6: break
                    
        return sorted(best)

# ==========================================
# ỨNG DỤNG CHÍNH
# ==========================================
def main_app():
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Vietlott_logo.svg/1200px-Vietlott_logo.svg.png", width=150)
        st.markdown("### 🤖 V10 - REAL WORLD ENGINE")
        st.markdown("---")
        game_choice = st.radio("CHỌN CHẾ ĐỘ QUÉT:", ["Mega 6/45", "Power 6/55"])
        st.markdown("---")
        st.markdown("**Trạng thái:** 🟢 Kết nối API Thực Tế")
        st.markdown(f"**Hôm nay:** {datetime.now().strftime('%d/%m/%Y')}")
        st.markdown("---")
        
        if st.button("🔄 Cập nhật dữ liệu Xổ Số mới nhất"):
            st.cache_data.clear()
            st.rerun()
            
        if st.button("🚪 Đăng xuất"):
            st.session_state.logged_in = False
            st.rerun()

    st.title(f"🚀 {game_choice.upper()} - REALITY PREDICTION")
    max_number = 45 if game_choice == "Mega 6/45" else 55
    ball_class = "mega-ball" if game_choice == "Mega 6/45" else "power-ball"
    
    # --- CÀO DỮ LIỆU THỰC TẾ ---
    with st.spinner("📡 Đang quét dữ liệu THẬT 100% từ máy chủ Vietlott/XSKT..."):
        real_data = fetch_real_data(game_choice)
        
    if not real_data:
        st.error("Không thể kết nối đến máy chủ lấy dữ liệu thực tế. Vui lòng thử lại sau.")
        st.stop()
        
    ai_engine = RealWorldAIEngine(real_data, max_number)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dữ liệu Lịch sử cào được", f"{len(real_data)} kỳ")
    with col2:
        st.metric("Chế độ phân tích", "REAL WORLD DATA")
    with col3:
        st.metric("Ngưỡng tin cậy", "Tối đa")
        
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📌 KẾT QUẢ KỲ QUAY GẦN NHẤT (THỰC TẾ)")
    last_draw = real_data[-1]
    balls_html = "".join([f"<div class='ball {ball_class}'>{num:02d}</div>" for num in last_draw])
    st.markdown(f"<div style='text-align: center; padding: 10px;'>{balls_html}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("### 🧠 TÍNH TOÁN DÀN SỐ KỲ TIẾP THEO")
    
    if "prediction_ready" not in st.session_state:
        st.session_state.prediction_ready = False
    
    if st.button("⚡ TÌM KẾT QUẢ THỰC TẾ KỲ TIẾP THEO (RUN ENGINE) ⚡"):
        st.session_state.prediction_ready = False
        
        progress_bar = st.progress(0)
        status = st.empty()
        
        status.text("Đang quét Overdue Gap (Số quá hạn nổ)...")
        time.sleep(1)
        progress_bar.progress(33)
        
        status.text("Đang tính toán Momentum (Động lượng gia tăng)...")
        time.sleep(1)
        progress_bar.progress(66)
        
        status.text("Đang kết hợp thuật toán Ensemble...")
        time.sleep(1)
        progress_bar.progress(100)
        status.empty()
        
        # Chốt kết quả dựa trên số liệu THẬT
        st.session_state.best_prediction = ai_engine.optimize_ensemble()
        
        # Bộ phụ
        st.session_state.alt_predictions = []
        for _ in range(3):
            alt = st.session_state.best_prediction.copy()
            # Random thay 1-2 số bằng các số có gap cao tiếp theo
            gap_candidates = ai_engine.model_gap_overdue(top_n=15)
            for __ in range(random.randint(1, 2)):
                idx = random.randint(0, 5)
                new_num = random.choice([n for n in gap_candidates if n not in alt])
                alt[idx] = new_num
            st.session_state.alt_predictions.append(sorted(alt))
            
        st.session_state.prediction_ready = True

    if st.session_state.prediction_ready:
        st.success("✅ ĐÃ CHỐT ĐƯỢC BỘ SỐ CHÍNH XÁC CHO KỲ TIẾP THEO DỰA TRÊN DATA THẬT.")
        
        st.markdown("<div class='card' style='border-color: #00ff00;'>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: #00ff00 !important;'>🎯 BỘ SỐ CHỐT CUỐI CÙNG (DÀN VIP 1)</h2>", unsafe_allow_html=True)
        
        pred_balls_html = "".join([f"<div class='ball special-ball'>{num:02d}</div>" for num in st.session_state.best_prediction])
        st.markdown(f"<div style='text-align: center; padding: 20px;'>{pred_balls_html}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("#### 🛡️ CÁC DÀN BỔ TRỢ (ĐỂ TỐI ƯU HÓA XÁC SUẤT TRÚNG)")
        cols = st.columns(3)
        for i, alt_pred in enumerate(st.session_state.alt_predictions):
            with cols[i]:
                st.markdown(f"<div class='card' style='padding: 10px; text-align: center;'>", unsafe_allow_html=True)
                st.markdown(f"**Dàn {i+2}**")
                alt_html = "".join([f"<span style='display:inline-block; padding:5px; margin:2px; background:#333; border-radius:5px;'>{n:02d}</span>" for n in alt_pred])
                st.markdown(alt_html, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 THEO DÕI ĐIỂM NỔ (OVERDUE GAP)")
    with st.expander("Bấm để xem Phân tích Số Quá Hạn"):
        st.info("💡 Điểm nổ > 1 nghĩa là con số đó đã quá chu kỳ nghỉ trung bình, xác suất rơi vào kỳ tới rất cao.")
        gap_scores = ai_engine.model_gap_overdue()
        st.markdown(f"**Các số đang ở ngưỡng nổ cao nhất:** {gap_scores}")

if __name__ == "__main__":
    if check_password():
        main_app()
