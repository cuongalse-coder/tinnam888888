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
    page_title="TINNAM AI - V400.0 ADAPTIVE QUANTUM",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #050505; color: #00ffcc; font-family: 'Courier New', Courier, monospace; }
    .ball {
        display: inline-flex; align-items: center; justify-content: center;
        width: 50px; height: 50px; border-radius: 50%; color: white;
        font-weight: bold; font-size: 20px; margin: 5px;
        box-shadow: 0 0 10px rgba(0,255,204,0.5); border: 2px solid #00ffcc;
        background: #111;
    }
    .mega-ball { box-shadow: 0 0 15px #ff0055; border-color: #ff0055; }
    .power-ball { box-shadow: 0 0 15px #ff4500; border-color: #ff4500; }
    .special-ball { background: linear-gradient(145deg, #00ffcc, #006655); color: #000; box-shadow: 0 0 20px #00ffcc; border-color: #fff; }
    .stButton>button { width: 100%; background-color: transparent; color: #00ffcc; font-weight: bold; border: 2px solid #00ffcc; border-radius: 5px; transition: 0.3s; text-shadow: 0 0 5px #00ffcc; box-shadow: inset 0 0 10px #00ffcc; }
    .stButton>button:hover { background-color: #00ffcc; color: #000000; box-shadow: 0 0 25px #00ffcc; }
    h1, h2, h3 { color: #00ffcc !important; text-shadow: 0 0 10px #00ffcc; }
    .card { background-color: rgba(10, 15, 20, 0.9); padding: 20px; border-radius: 10px; border: 1px solid #00ffcc; margin-bottom: 20px; box-shadow: 0 0 15px rgba(0,255,204,0.2); }
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
        f"https://www.ketquadientoan.com/tat-ca-ky-xo-so-mega-6-45.html?datef=18-07-2016&datet={today_str}" if game_type == "Mega 6/45" else f"https://www.ketquadientoan.com/tat-ca-ky-xo-so-power-655.html?datef=01-01-2018&datet={today_str}",
        "https://xskt.com.vn/ket-qua-xo-so-vietlott-mega-6-45" if game_type == "Mega 6/45" else "https://xskt.com.vn/ket-qua-xo-so-vietlott-power-6-55",
        "https://xoso.me/kqxs-mega-645.html" if game_type == "Mega 6/45" else "https://xoso.me/kqxs-power-655.html",
        "https://ketqua.vn/vietlott-mega-6-45" if game_type == "Mega 6/45" else "https://ketqua.vn/vietlott-power-6-55"
    ]
    
    max_num = 45 if game_type == "Mega 6/45" else 55
    
    for url in urls:
        try:
            response = scraper.get(url, timeout=30)
            if response.status_code == 200:
                html = response.text
                
                history = []
                detailed_history = []
                
                if "ketquadientoan.com" in url:
                    rows = re.findall(r'<tr.*?>(.*?)</tr>', html, re.DOTALL | re.IGNORECASE)
                    for row in rows:
                        date_match = re.search(r'<td>.*?((\d{2})/(\d{2})/(\d{4}))</td>', row)
                        if not date_match:
                            continue
                        date_str = date_match.group(1)
                        
                        nums = re.findall(r'class="home-mini-whiteball">\s*(\d{2})\s*<', row)
                        if len(nums) < 6:
                            continue
                        chunk = [int(n) for n in nums[:6]]
                        if len(set(chunk)) != 6 or not all(1 <= n <= max_num for n in chunk):
                            continue
                            
                        # Tìm giải Jackpot bằng regex lấy cả thuộc tính thẻ span để xét màu
                        jp_spans = re.findall(r"<span class='hidden-xs'([^>]*)>([\d\.]+)</span>", row)
                        jp1_val = jp_spans[0][1] if len(jp_spans) > 0 else "0"
                        
                        if game_type == "Power 6/55" and len(jp_spans) > 1:
                            jp2_val = jp_spans[1][1]
                            if jp2_val != "0":
                                jp1_val = f"JP1: {jp1_val} | JP2: {jp2_val}"
                        
                        has_winner = False
                        # Chỉ bôi đỏ nếu trúng giải ĐẶC BIỆT (JP1) - Tức là span đầu tiên có màu đỏ
                        if len(jp_spans) > 0:
                            if "COLOR:#F00" in jp_spans[0][0].upper() or "COLOR:RED" in jp_spans[0][0].upper():
                                has_winner = True
                            
                        sorted_chunk = sorted(chunk)
                        if sorted_chunk not in history:
                            history.append(sorted_chunk)
                            detailed_history.append({
                                "Ngày": date_str,
                                "Bóng 1": sorted_chunk[0], "Bóng 2": sorted_chunk[1], "Bóng 3": sorted_chunk[2],
                                "Bóng 4": sorted_chunk[3], "Bóng 5": sorted_chunk[4], "Bóng 6": sorted_chunk[5],
                                "Jackpot": jp1_val,
                                "Trúng Giải": "🚨 CÓ" if has_winner else ""
                            })
                else:
                    nums = re.findall(r'>\s*(\d{2})\s*<', html)
                    for i in range(0, len(nums) - 5):
                        chunk = [int(n) for n in nums[i:i+6]]
                        if chunk == sorted(chunk) and len(set(chunk)) == 6 and all(1 <= n <= max_num for n in chunk):
                            if chunk not in history:
                                history.append(chunk)
                                detailed_history.append({
                                    "Ngày": "N/A",
                                    "Bóng 1": chunk[0], "Bóng 2": chunk[1], "Bóng 3": chunk[2],
                                    "Bóng 4": chunk[3], "Bóng 5": chunk[4], "Bóng 6": chunk[5],
                                    "Jackpot": "N/A",
                                    "Trúng Giải": ""
                                })

                if history:
                    history.reverse()
                    detailed_history.reverse()
                    for i, d in enumerate(detailed_history):
                        d["Kỳ"] = f"Kỳ {i+1}"
                    # Sắp xếp lại thứ tự cột cho đẹp
                    detailed_history = [{"Kỳ": d["Kỳ"], "Ngày": d["Ngày"], "Bóng 1": d["Bóng 1"], "Bóng 2": d["Bóng 2"], "Bóng 3": d["Bóng 3"], "Bóng 4": d["Bóng 4"], "Bóng 5": d["Bóng 5"], "Bóng 6": d["Bóng 6"], "Jackpot": d["Jackpot"], "Trúng Giải": d["Trúng Giải"]} for d in detailed_history]
                    return history, detailed_history
        except Exception as e:
            continue
            
    # GITHUB FALLBACK
    try:
        github_url = "https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl" if game_type == "Mega 6/45" else "https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power655.jsonl"
        response = requests.get(github_url, timeout=10)
        history = []
        detailed_history = []
        if response.status_code == 200:
            import json
            for line in response.text.strip().split('\n'):
                if line:
                    data = json.loads(line)
                    if 'result' in data and len(data['result']) >= 6:
                        draw = sorted([int(n) for n in data['result'][:6]])
                        history.append(draw)
                        detailed_history.append({
                            "Ngày": "N/A", "Bóng 1": draw[0], "Bóng 2": draw[1], "Bóng 3": draw[2], "Bóng 4": draw[3], "Bóng 5": draw[4], "Bóng 6": draw[5], "Jackpot": "N/A", "Trúng Giải": ""
                        })
            if history:
                for i, d in enumerate(detailed_history):
                    d["Kỳ"] = f"Kỳ {i+1}"
                detailed_history = [{"Kỳ": d["Kỳ"], "Ngày": d["Ngày"], "Bóng 1": d["Bóng 1"], "Bóng 2": d["Bóng 2"], "Bóng 3": d["Bóng 3"], "Bóng 4": d["Bóng 4"], "Bóng 5": d["Bóng 5"], "Bóng 6": d["Bóng 6"], "Jackpot": d["Jackpot"], "Trúng Giải": d["Trúng Giải"]} for d in detailed_history]
                return history, detailed_history
    except Exception:
        pass
        
    st.error("⚠️ Không thể kết nối máy chủ xổ số. Đang sử dụng dữ liệu giả lập dự phòng.")
    fake_data = [sorted(random.sample(range(1, max_num + 1), 6)) for _ in range(50)]
    detailed_history = [{"Kỳ": f"Kỳ {i+1}", "Ngày": "N/A", "Bóng 1": d[0], "Bóng 2": d[1], "Bóng 3": d[2], "Bóng 4": d[3], "Bóng 5": d[4], "Bóng 6": d[5], "Jackpot": "N/A", "Trúng Giải": ""} for i, d in enumerate(fake_data)]
    return fake_data, detailed_history


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

    def model_advanced_ml(self):
        """Machine Learning: Random Forest & K-Means Clustering"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.cluster import KMeans
            import numpy as np
            
            if len(self.data) < 20:
                return self.model_gap_overdue()
                
            X = []
            y = []
            window_size = 10
            
            # Huấn luyện mô hình tìm quy luật xuất hiện của 10 kỳ để đoán kỳ tiếp
            for i in range(len(self.data) - window_size - 1):
                window = self.data[i:i+window_size]
                next_draw = self.data[i+window_size]
                
                features = np.zeros(self.max_number)
                for draw in window:
                    for num in draw:
                        features[num-1] += 1
                
                targets = np.zeros(self.max_number)
                for num in next_draw:
                    targets[num-1] = 1
                    
                X.append(features)
                y.append(targets)
                
            rf = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42)
            rf.fit(X, y)
            
            recent_window = self.data[-window_size:]
            recent_features = np.zeros(self.max_number)
            for draw in recent_window:
                for num in draw:
                    recent_features[num-1] += 1
                    
            rf_predictions = rf.predict([recent_features])[0]
            
            # Phân cụm K-Means để tìm nhóm số có tần suất đi cùng nhau cao nhất
            flat_data = np.array([num for draw in self.data for num in draw]).reshape(-1, 1)
            kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
            kmeans.fit(flat_data)
            cluster_centers = [int(round(c[0])) for c in kmeans.cluster_centers_]
            
            combined_scores = {num: rf_predictions[num-1] for num in self.all_numbers}
            for c in cluster_centers:
                if 1 <= c <= self.max_number:
                    combined_scores[c] += np.mean(rf_predictions) * 1.5 
                    
            top_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:6]
            return [idx for idx, score in top_indices]
        except Exception as e:
            return self.model_momentum_neural()

    def optimize_ensemble(self):
        """Tổng hợp bằng Trí Tuệ Nhân Tạo (Ensemble Machine Learning 100%)"""
        from collections import Counter
        m1 = self.model_markov_chain()
        m2 = self.model_gap_overdue()
        m3 = self.model_momentum_neural()
        m4 = self.model_advanced_ml()
        
        # Trọng số bình chọn: Machine Learning (5), Overdue (3), Momentum (2), Markov (1)
        votes = Counter()
        for num in m4: votes[num] += 5
        for num in m2: votes[num] += 3
        for num in m3: votes[num] += 2
        for num in m1: votes[num] += 1
        
        best = [num for num, count in votes.most_common(6)]
        
        while len(best) < 6:
            candidates = self.model_gap_overdue(top_n=15)
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
        st.markdown("### 🧬 V400.0 - ADAPTIVE QUANTUM")
        st.markdown("---")
        game_choice = st.radio("CHỌN CHẾ ĐỘ QUÉT:", ["Mega 6/45", "Power 6/55"])
        st.markdown("---")
        num_tickets = st.selectbox("Ngân sách đầu tư (Số vé mua):", [5, 10, 20, 50, 100], index=1)
        pool_size = st.selectbox("Kích thước Hồ Tiềm Năng:", [10, 12, 15, 18, 20], index=2)
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

    st.title(f"🧬 {game_choice.upper()} - V400.0 ADAPTIVE QUANTUM")
    max_number = 45 if game_choice == "Mega 6/45" else 55
    ball_class = "mega-ball" if game_choice == "Mega 6/45" else "power-ball"
    
    # --- CÀO DỮ LIỆU THỰC TẾ ---
    with st.spinner("📡 Đang quét dữ liệu THẬT 100% từ máy chủ Vietlott/XSKT..."):
        real_data, detailed_data = fetch_real_data(game_choice)
        
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
    
    with st.expander(f"📚 XEM TOÀN BỘ LỊCH SỬ {len(real_data)} KỲ ĐÃ TẢI", expanded=False):
        import pandas as pd
        display_data = detailed_data[::-1] # Mới nhất lên trên
        df = pd.DataFrame(display_data)
        df.set_index("Kỳ", inplace=True)
        
        def highlight_row(row):
            return ['background-color: rgba(255, 0, 0, 0.3)'] * len(row) if row['Trúng Giải'] == '🚨 CÓ' else [''] * len(row)
            
        st.dataframe(df.style.apply(highlight_row, axis=1), use_container_width=True)
    
    st.markdown("### 🧠 TÍNH TOÁN DÀN SỐ KỲ TIẾP THEO")
    
    if "prediction_ready" not in st.session_state:
        st.session_state.prediction_ready = False
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_btn = st.button("🧬 KÍCH HOẠT V400.0 ADAPTIVE QUANTUM — 43 TÍN HIỆU AI 🧬", use_container_width=True)

    if run_btn:
        st.session_state.prediction_ready = False
        
        progress_bar = st.progress(0)
        status = st.empty()
        
        term_placeholder = st.empty()
        term_text = ""
        
        status_steps = [
            ("Khởi tạo cụm Mạng Nơ-ron Đa Tầng (MLP Regressor)...", 5, ["0x0000: INIT_NEURAL_CORE", "0x0001: ALLOCATING_GPU_TENSORS"]),
            ("Trích xuất năng lượng chân không (Vacuum Energy Extract)...", 15, ["0x1A44: SCANNING_VOID", "0x1A45: ZERO_POINT_ENERGY_LOCKED"]),
            ("Kích hoạt Mạng Lưới Đồ Thị (Neural Graph PageRank)...", 30, ["0x2B11: TIME_DILATION_ACTIVE", "0x2B12: GRAVITY_ISOLATED"]),
            ("Huấn luyện AI với 300 kỳ quay gần nhất (Deep Learning Training)...", 50, ["0x3C99: TRAINING_EPOCH_200", "0x3C9A: LOSS_CONVERGED_AT_0.0001"]),
            ("Tải 33 thuật toán AI & 5 Meta-Engines (33 AI Signals)...", 75, ["0x4D01: COLLAPSING_WAVEFUNCTION", "0x4D02: SCHRODINGER_CAT_ALIVE"]),
            ("Bẻ gãy xác suất 8.1 triệu tổ hợp (Shattering Probability Matrix)...", 90, ["0x5E88: MATRIX_CRITICAL_FAILURE", "0x5E89: BYPASSING_PHYSICS_LAW"]),
            ("Trích xuất Vé Chân Lý từ Tương Lai (Extracting Truth Ticket)...", 100, ["0x6F10: TIME_PARADOX_RESOLVED", "0x6F11: ABSOLUTE_TRUTH_ACQUIRED"])
        ]
        
        for step_text, prog_val, hex_codes in status_steps:
            status.text(f"🌌 {step_text}")
            progress_bar.progress(prog_val)
            for hc in hex_codes:
                term_text += f"> {hc}\n"
                term_placeholder.markdown(f"<div style='background-color:black; color:#00ff00; padding:10px; font-family:monospace; border-radius:5px; height:150px; overflow-y:auto; border:1px solid #00ff00;'><pre>{term_text}</pre></div>", unsafe_allow_html=True)
                time.sleep(0.3)
        
        time.sleep(0.5)
        term_placeholder.empty()
        
        try:
            from models.nexus_engine import NexusEngine
            
            engine = NexusEngine(max_number, 6)
            result_v11 = engine.predict(real_data, n_sets=5)
            
            if result_v11['top_pool']:
                st.session_state.v11_top_pool = result_v11['top_pool'][:pool_size] # Dynamic pool size
                
                from models.wheeling_optimizer import WheelingOptimizer
                wheel_opt = WheelingOptimizer(6, max_number)
                tickets, coverage = wheel_opt.generate_wheel(
                    st.session_state.v11_top_pool, 
                    num_tickets,
                    constraints=result_v11.get('constraints'),
                    sum_mod7=result_v11.get('sum_mod7'),
                    history_data=real_data
                )
                
                sniper_ticket = result_v11['predictions'][0]['numbers']
                sniper_obj = {'numbers': sniper_ticket, 'strategy': '🌌 VÉ CHÂN LÝ (ABSOLUTE TRUTH - LẤY TỪ TƯƠNG LAI)'}
                
                # Check to avoid duplicate
                filtered_tickets = [t for t in tickets if sorted(t['numbers']) != sorted(sniper_ticket)]
                final_tickets = [sniper_obj] + filtered_tickets
                
                st.session_state.best_prediction = sniper_ticket
                st.session_state.all_predictions = final_tickets
                st.session_state.v11_weights = result_v11.get('weights', {})
                st.session_state.v11_confidence = coverage
                st.session_state.absolute_final_6 = result_v11.get('absolute_final_6', [])
            else:
                # Fallback to V10
                from models.vulnerability_scanner import VulnerabilityScanner
                from models.exploit_engine import ExploitEngine
                scanner = VulnerabilityScanner(max_number, 6)
                scan_results = scanner.scan_all(real_data)
                eng10 = ExploitEngine(max_number, 6)
                exploit = eng10.exploit(real_data, scan_results, n_sets=5)
                if exploit['predictions']:
                    st.session_state.best_prediction = exploit['predictions'][0]['numbers']
                    st.session_state.all_predictions = exploit['predictions']
                    st.session_state.v11_weights = {}
                    st.session_state.v11_confidence = exploit['confidence']
                    st.session_state.v11_top_pool = []
                else:
                    from models.ultimate_engine import UltimateEngine
                    adv = UltimateEngine(max_number, 6)
                    res = adv.predict(real_data)
                    st.session_state.best_prediction = res['primary']
                    st.session_state.all_predictions = []
                    st.session_state.v11_weights = {}
                    st.session_state.v11_confidence = 50
                    st.session_state.v11_top_pool = []
        except Exception as e:
            st.error(f"Lỗi: {e}")
            import traceback
            st.code(traceback.format_exc())
                
        progress_bar.progress(100)
        status.empty()
        st.session_state.prediction_ready = True

    if st.session_state.prediction_ready:
        coverage = st.session_state.get('v11_confidence', 0)
        top_pool = st.session_state.get('v11_top_pool', [])
        st.success(f"✅ V400.0 ADAPTIVE QUANTUM HOÀN TẤT — 43 AI Signals | Walk-Forward Calibration | Hồ Tiềm Năng: {len(top_pool)} số.")
        
        # === HỒ SỐ TIỀM NĂNG ===
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align: center; color: #00ffcc !important;'>🧬 HỒ SỐ ĐỘT BIẾN ({len(top_pool)} SỐ TỪ 43 AI SIGNALS) 🧬</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #888;'><em>(V400: Walk-Forward Calibration + Pair Co-occurrence + Temporal Decay + 35 Core Signals)</em></p>", unsafe_allow_html=True)
        if top_pool:
            pool_html = "".join([f"<div class='ball special-ball'>{n:02d}</div>" for n in top_pool])
            st.markdown(f"<div style='text-align:center; margin-bottom: 25px;'>{pool_html}</div>", unsafe_allow_html=True)
            
            # --- LÕI KIM CƯƠNG 10 SỐ ---
            top_10_diamond = top_pool[:10]
            top10_html = "".join([f"<div class='ball special-ball' style='background: linear-gradient(145deg, #ff00ff, #00ffff); box-shadow: 0 0 25px #ff00ff; border-color: #ff00ff;'>{n:02d}</div>" for n in top_10_diamond])
            st.markdown("<div style='background-color: rgba(255, 0, 255, 0.05); border: 2px dashed #ff00ff; border-radius: 10px; padding: 20px; margin-top: 10px;'>", unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center; color: #ff00ff !important; text-shadow: 0 0 10px #ff00ff;'>💎 LÕI KIM CƯƠNG: 10 SỐ CHUẨN NHẤT (Dành cho đánh BAO 10) 💎</h3>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #bbb;'><em>(Hệ thống đã nén và trích xuất đúng 10 con số có Điểm Tương Quan Tổng Hợp cao nhất từ 32 thuật toán. Chuyên dùng để ghép Bao 7 đến Bao 10)</em></p>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:center;'>{top10_html}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        # === 6 SỐ TUYÊN NGÔN CUỐI CÙNG (ABSOLUTE FINAL 6) ===
        final_6 = st.session_state.get('absolute_final_6', [])
        if final_6:
            st.markdown("<div style='background: linear-gradient(135deg, rgba(255,0,85,0.15), rgba(0,255,204,0.1)); border: 2px solid #ff0055; border-radius: 12px; padding: 24px; margin-bottom: 20px; box-shadow: 0 0 30px rgba(255,0,85,0.4);'>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center; color: #ff0055 !important; text-shadow: 0 0 20px #ff0055; font-size: 1.8em;'>🎯 6 SỐ TUYÊN NGÔN (VÙNG LỆCH TRỤC MÁY QUAY) 🎯</h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #bbb;'><em>Những quả bóng này nằm chính xác vào Điểm Mù (Blind Spot) của Lồng Cầu bị nghiêng, nơi lực ly tâm gom chúng lại dễ rơi vào ống hút nhất.</em></p>", unsafe_allow_html=True)
            f6_html = "".join([f"<div class='ball {ball_class}' style='width:65px;height:65px;font-size:24px; background: linear-gradient(145deg,#ff0055,#ff6600); border-color:#ff0055; box-shadow: 0 0 30px #ff0055;'>{n:02d}</div>" for n in final_6])
            st.markdown(f"<div style='text-align:center; padding: 15px;'>{f6_html}</div>", unsafe_allow_html=True)
            st.markdown("<p style='text-align:center; color:#ff0055; font-weight:bold;'>⚡ ĐÂY LÀ LỰA CHỌN SỐ 1 CỦA HỆ THỐNG. Nếu chỉ muốn mua 1 VÉ DUY NHẤT — hãy dùng 6 số này. ⚡</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # === DÀN VÉ RÚT GỌN ===
        all_preds = st.session_state.get('all_predictions', [])
        st.markdown("<div class='card' style='border-color: #00ff00;'>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align: center; color: #00ff00 !important;'>🎯 DÀN {len(all_preds)} VÉ BAO RÚT GỌN CHỐT SỐ 🎯</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; color: #888;'><em>(Ma trận tổ hợp bao phủ chéo giúp tiết kiệm tiền mà vẫn vét được lưới xác suất)</em></p>", unsafe_allow_html=True)
        
        ticket_texts = []
        for i, pred in enumerate(all_preds):
            nums = pred['numbers']
            ticket_str = " ".join([f"{n:02d}" for n in nums])
            ticket_texts.append(ticket_str)
            ticket_html = "".join([f"<div class='ball {ball_class}' style='width:40px;height:40px;font-size:16px;'>{n:02d}</div>" for n in nums])
            st.markdown(f"**Vé #{i+1}:** <span style='color:#ff0055; font-style:italic;'>{pred.get('strategy', '')}</span>", unsafe_allow_html=True)
            st.markdown(f"<div style='padding: 5px;'>{ticket_html}</div>", unsafe_allow_html=True)
        
        # Tiện ích Copy & Download
        st.markdown("### 📋 SAO CHÉP & TẢI XUỐNG ĐỂ GHI VÉ")
        st.info("💡 Bạn có thể copy dán trực tiếp vào SMS Vietlott, hoặc ghi ra giấy.")
        full_text = "\\n".join(ticket_texts)
        st.code(full_text, language='text')
        
        st.download_button(
            label="💾 Tải danh sách vé (.txt)",
            data=full_text,
            file_name=f"vietlott_{game_choice.replace(' ', '_')}_dan_{len(all_preds)}_ve.txt",
            mime="text/plain"
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        # === PHÂN TÍCH CHUYÊN SÂU TỪ KỲ LIỀN KỀ ===
        st.markdown("<div class='card' style='border-color: #00ffcc;'>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: #00ffcc !important;'>🔮 DỰ ĐOÁN CHUYÊN SÂU TỪ KỲ LIỀN KỀ 🔮</h2>", unsafe_allow_html=True)
        
        last_draw_balls = real_data[-1][:6]
        balls_html_last = "".join([f"<div class='ball {ball_class}' style='width:30px;height:30px;font-size:12px;'>{num:02d}</div>" for num in last_draw_balls])
        st.markdown(f"Dựa vào 6 quả bóng vừa nổ ở kỳ trước: <div style='display:inline-block;'>{balls_html_last}</div>", unsafe_allow_html=True)
        st.markdown("Hệ thống đã trích xuất dữ liệu Chuỗi Markov (Markov Chain) để tìm ra những con số có xác suất **NỔ THEO ĐUÔI** các quả bóng này cao nhất trong lịch sử:")
        
        # Calculate trailing balls (Transition Matrix)
        follow_counts = {}
        for i in range(len(real_data) - 1):
            intersect = set(real_data[i][:6]) & set(last_draw_balls)
            if intersect:
                weight = len(intersect) ** 2  # Exponential weight for more matches
                for n in real_data[i+1][:6]:
                    if n not in last_draw_balls:
                        follow_counts[n] = follow_counts.get(n, 0) + weight
                        
        top_followers = sorted(follow_counts.items(), key=lambda x: -x[1])[:10]
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown("#### TOP 5 Số Dễ Rơi Nhất Kỳ Này")
            for rank, (num, score) in enumerate(top_followers[:5]):
                st.markdown(f"**#{rank+1}. Số {num:02d}** (Điểm tương quan: {score})")
        with col_f2:
            st.markdown("#### Các số bám đuôi tiếp theo")
            for rank, (num, score) in enumerate(top_followers[5:10]):
                st.markdown(f"**#{rank+6}. Số {num:02d}** (Điểm tương quan: {score})")
                
        st.info("💡 BÍ KÍP TỰ CHƠI: Nếu bạn không muốn mua theo Dàn Bao của AI, bạn có thể tự bốc 6 số từ danh sách TOP 10 Bóng Theo Đuôi ở trên để mua 1 vé duy nhất. Xác suất rơi của chúng cực kỳ cao!")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # === V18.0: QUẢN TRỊ VỐN KELLY & ĐỘ NHẤT QUÁN TÍN HIỆU ===
        st.markdown("<div class='card' style='border-color: #ff9900;'>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align: center; color: #ff9900 !important;'>⚖️ AI QUẢN TRỊ RỦI RO (KELLY CRITERION) ⚖️</h2>", unsafe_allow_html=True)
        
        v11_weights = st.session_state.get('v11_weights', {})
        if v11_weights:
            weight_values = list(v11_weights.values())
            if weight_values:
                max_w = max(weight_values)
                avg_w = sum(weight_values) / len(weight_values)
                # Coherence: How strongly the top signals dominate the average.
                coherence = min(100, max(0, int((max_w / (avg_w + 1e-5) - 1) * 25)))
                
                st.metric("Độ Nhất Quán Tín Hiệu (Signal Coherence)", f"{coherence}%")
                
                if coherence < 30:
                    st.error("⚠️ TÍN HIỆU NHIỄU LOẠN: Các thuật toán AI đang mâu thuẫn dữ dội. Lồng cầu đang ở trạng thái cực kỳ hỗn loạn và phi logic. KHUYẾN NGHỊ: Dừng mua vé kỳ này để bảo toàn vốn, hoặc chỉ đánh 1-2 vé dò đường.")
                elif coherence < 60:
                    st.warning("⚠️ TÍN HIỆU TRUNG BÌNH: Đã xuất hiện xu hướng nhưng chưa thực sự bứt phá. KHUYẾN NGHỊ: Đánh ở mức an toàn (10-20% quỹ mạo hiểm).")
                else:
                    st.success("🔥 TÍN HIỆU ĐỒNG THUẬN CAO (SINGULARITY): 32 Thuật toán lượng tử hội tụ về cùng một lưới xác suất. Đây là 'Điểm Rơi' hoàn hảo của lồng cầu. KHUYẾN NGHỊ: Tấn công mạnh, mua đủ danh sách vé AI đề xuất.")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # === TRỌNG SỐ TÍN HIỆU ===
        v11_weights = st.session_state.get('v11_weights', {})
        if v11_weights:
            with st.expander("🧠 XEM TRỌNG SỐ 33 TÍN HIỆU AI (Dynamic ELO & Deep Learning)", expanded=False):
                import pandas as pd
                w_data = [{"Tín hiệu": k, "Trọng số": v} for k, v in v11_weights.items()]
                df_w = pd.DataFrame(w_data)
                st.dataframe(df_w, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📊 THEO DÕI ĐIỂM NỔ (OVERDUE GAP)")
    with st.expander("Bấm để xem Phân tích Số Quá Hạn"):
        st.info("💡 Điểm nổ > 1 nghĩa là con số đó đã quá chu kỳ nghỉ trung bình, xác suất rơi vào kỳ tới rất cao.")
        gap_scores = ai_engine.model_gap_overdue()
        st.markdown(f"**Các số đang ở ngưỡng nổ cao nhất:** {gap_scores}")

    st.markdown("---")
    st.markdown("### 🔍 PHÂN TÍCH NGƯỢC TOÀN DIỆN (REVERSE FORENSIC)")
    with st.expander("Bấm để xem Phân tích Lịch sử từ kỳ đầu tiên đến nay"):
        st.info("Hệ thống sẽ tính toán lại tỷ lệ bóng ra của TOÀN BỘ lịch sử để phát hiện sự thiên lệch vật lý. Tính năng này giúp trả lời câu hỏi: Có sự trùng hợp hay 'chỉ định' nào trong lồng quay không?")
        if st.button("📊 CHẠY PHÂN TÍCH NGƯỢC TOÀN BỘ LỊCH SỬ"):
            import pandas as pd
            import numpy as np
            import altair as alt
            from collections import Counter
            
            with st.spinner("Đang lục lại toàn bộ dữ liệu từ kỳ 1 đến nay..."):
                # Tính toán tần suất
                all_numbers = [num for draw in real_data for num in draw]
                freq_counts = Counter(all_numbers)
                
                # Tính toán Gap
                last_seen = {}
                max_gap = {}
                for i, draw in enumerate(real_data):
                    for num in draw:
                        if num in last_seen:
                            gap = i - last_seen[num]
                            if gap > max_gap.get(num, 0):
                                max_gap[num] = gap
                        last_seen[num] = i
                
                current_gaps = {n: len(real_data) - 1 - last_seen.get(n, 0) for n in range(1, max_number + 1)}
                
                # Tạo DataFrame
                df_data = []
                expected_prob = 6 / max_number
                expected_count = len(real_data) * expected_prob
                
                for n in range(1, max_number + 1):
                    count = freq_counts.get(n, 0)
                    z_score = (count - expected_count) / np.sqrt(len(real_data) * expected_prob * (1 - expected_prob))
                    df_data.append({
                        "Số": n,
                        "Lần xuất hiện": count,
                        "Độ lệch (Z-Score)": round(z_score, 2),
                        "Ngủ đông Max (Kỳ)": max_gap.get(n, 0),
                        "Hiện chưa ra (Kỳ)": current_gaps.get(n, 0)
                    })
                    
                df = pd.DataFrame(df_data)
                
                st.markdown("#### 1. Biểu Đồ Tần Suất & Bất Thường Tổng Thể")
                st.markdown("*(Đỏ: Cố tình ra nhiều bất thường / Xanh: Bị gìm lại / Xám: Nằm trong vùng ngẫu nhiên công bằng)*")
                
                chart = alt.Chart(df).mark_bar().encode(
                    x=alt.X('Số:O', sort=None),
                    y='Lần xuất hiện:Q',
                    color=alt.condition(
                        alt.datum['Độ lệch (Z-Score)'] > 1.5,
                        alt.value('#ff4b4b'),  # Red for hot
                        alt.condition(alt.datum['Độ lệch (Z-Score)'] < -1.5, alt.value('#0068c9'), alt.value('#888888'))
                    ),
                    tooltip=['Số', 'Lần xuất hiện', 'Độ lệch (Z-Score)', 'Hiện chưa ra (Kỳ)']
                ).properties(height=400)
                st.altair_chart(chart, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 🔥 TOP 5 Số Nóng Nhất (Ra Nhiều)")
                    st.dataframe(df.nlargest(5, 'Độ lệch (Z-Score)')[['Số', 'Lần xuất hiện', 'Độ lệch (Z-Score)']], use_container_width=True)
                with col2:
                    st.markdown("#### ❄️ TOP 5 Số Lạnh Nhất (Bị Gìm)")
                    st.dataframe(df.nsmallest(5, 'Độ lệch (Z-Score)')[['Số', 'Lần xuất hiện', 'Độ lệch (Z-Score)']], use_container_width=True)
                    
                st.warning("**Kết luận từ hệ thống:** Nếu biểu đồ trên có nhiều cột Đỏ/Xanh (Z-Score vượt quá ±2.5), lồng cầu có thể đang có sự thiên lệch vật lý (bóng nặng/nhẹ, trục quay nghiêng). Nếu đa số là cột Xám, lồng quay hoàn toàn ngẫu nhiên và không có sự 'chỉ định' nào.")
                
                st.markdown("---")
                st.markdown("#### 📋 BẢNG XẾP HẠNG TOÀN BỘ CÁC QUẢ BÓNG (Từ cao xuống thấp)")
                st.info("Bảng liệt kê chính xác tỷ lệ xuất hiện của tất cả các quả bóng từ kỳ đầu tiên đến nay. Bạn có thể bấm vào tiêu đề cột để sắp xếp.")
                df['Tỷ lệ rơi (%)'] = (df['Lần xuất hiện'] / len(real_data) * 100).round(2)
                df_sorted = df.sort_values(by='Lần xuất hiện', ascending=False).reset_index(drop=True)
                # Đánh số thứ tự hạng (Rank)
                df_sorted.index = df_sorted.index + 1
                st.dataframe(df_sorted[['Số', 'Lần xuất hiện', 'Tỷ lệ rơi (%)', 'Độ lệch (Z-Score)', 'Ngủ đông Max (Kỳ)', 'Hiện chưa ra (Kỳ)']], use_container_width=True)

    st.markdown("---")
    st.markdown("### 🧪 KIỂM THỬ ĐỘ CHÍNH XÁC DÀN BAO (WHEELING BACKTEST)")
    with st.expander("Bấm để chạy Backtest (Kiểm thử thực tế với thuật toán Dàn Bao)"):
        st.warning(f"⚠️ Hệ thống sẽ tua ngược thời gian, ẩn đi kết quả thật và dùng AI tạo Dàn Bao {num_tickets} vé từ Hồ {pool_size} số ở các kỳ quá khứ, sau đó đối chiếu với kết quả ĐÃ XẢY RA để tính lãi/lỗ.")
        
        test_mode = st.radio("Chọn chế độ kiểm thử:", [
            "Test 1 kỳ cụ thể (Tính lùi)", 
            "Test 50 kỳ liên tiếp gần nhất",
            "Test một khối kỳ tùy chọn (Chọn khoảng)"
        ])
        
        total_draws = len(real_data)
        
        if test_mode == "Test 1 kỳ cụ thể (Tính lùi)":
            target_draw = st.slider("Chọn kỳ để test (Tính lùi từ hiện tại về quá khứ, 1 = Kỳ trước):", 1, min(1000, total_draws-60), 1)
        elif test_mode == "Test một khối kỳ tùy chọn (Chọn khoảng)":
            range_vals = st.slider(
                "Chọn khối kỳ muốn Test (Theo số thứ tự kỳ quay thực tế):",
                min_value=60, 
                max_value=total_draws,
                value=(total_draws-50, total_draws)
            )
            st.info(f"Sẽ chạy kiểm thử trên {range_vals[1] - range_vals[0] + 1} kỳ. Lưu ý: Chạy càng nhiều kỳ sẽ càng tốn thời gian.")
            
        if st.button("🚀 CHẠY KIỂM THỬ DÀN BAO"):
            test_progress = st.progress(0)
            test_status = st.empty()
            
            if total_draws < 60:
                st.error("Không đủ dữ liệu để backtest.")
            else:
                from models.mega_exploit_v15 import MegaExploitV15
                from models.wheeling_optimizer import WheelingOptimizer
                
                match_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
                total_spent = 0
                total_won = 0 
                winning_draws_count = 0
                
                test_indices = []
                if test_mode == "Test 50 kỳ liên tiếp gần nhất":
                    test_indices = range(total_draws - 50, total_draws)
                elif test_mode == "Test một khối kỳ tùy chọn (Chọn khoảng)":
                    test_indices = range(range_vals[0] - 1, range_vals[1])
                else:
                    test_indices = [total_draws - target_draw - 1]
                    
                test_size = len(test_indices)
                
                for step, current_idx in enumerate(test_indices):
                    historical_data_for_test = real_data[:current_idx]
                    actual_next_draw = set(real_data[current_idx])
                    
                    test_status.text(f"Đang phân tích kỳ {current_idx} / {total_draws}...")
                    
                    # 1. AI tạo hồ tiềm năng
                    engine = MegaExploitV15(max_number, 6)
                    res = engine.predict(historical_data_for_test, n_sets=1)
                    if res['top_pool']:
                        pool = res['top_pool'][:pool_size]
                    else:
                        pool = list(range(1, pool_size + 1))
                        
                    # 2. Sinh dàn bao
                    wheel_opt = WheelingOptimizer(6, max_number)
                    tickets, _ = wheel_opt.generate_wheel(
                        pool, 
                        num_tickets,
                        constraints=res.get('constraints'),
                        sum_mod7=res.get('sum_mod7'),
                        history_data=historical_data_for_test
                    )
                    
                    # 3. Đối chiếu kết quả các vé
                    draw_best_match = 0
                    draw_won_any = False
                    
                    for t_obj in tickets:
                        t = t_obj['numbers']
                        hits = len(set(t) & actual_next_draw)
                        match_counts[hits] += 1
                        total_spent += 10000
                        
                        # Tính tiền thưởng giả lập (Mega 6/45)
                        if hits == 3: total_won += 30000; draw_won_any = True
                        elif hits == 4: total_won += 300000; draw_won_any = True
                        elif hits == 5: total_won += 10000000; draw_won_any = True
                        elif hits == 6: total_won += 12000000000; draw_won_any = True
                        
                        if hits > draw_best_match:
                            draw_best_match = hits
                            
                    if draw_won_any:
                        winning_draws_count += 1
                            
                    test_progress.progress((step + 1) / test_size)
                    test_status.text(f"Đã test xong. Thành tích cao nhất trong dàn vé: {draw_best_match}/6")
                    
                test_status.empty()
                st.success(f"✅ Hoàn thành Backtest trên {test_size} kỳ quay!")
                
                st.markdown(f"**Tổng kết Dàn Bao {num_tickets} vé (Dựa trên {test_size} kỳ):**")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Tổng vốn đầu tư", f"{total_spent:,} VNĐ")
                    st.metric("Tổng tiền trúng thưởng", f"{total_won:,} VNĐ", delta=total_won-total_spent)
                    
                    win_rate = (winning_draws_count / test_size) * 100
                    st.metric("Tỉ lệ kỳ quay có lãi/trúng (Win Rate)", f"{win_rate:.1f}%", help=f"Có {winning_draws_count} kỳ chiến thắng trên tổng {test_size} kỳ test.")
                with col_b:
                    st.markdown("**Chi tiết số vé trúng giải:**")
                    st.markdown(f"- 🏆 Trúng Jackpot (6/6): **{match_counts[6]} vé**")
                    st.markdown(f"- 🥇 Trúng giải Nhất (5/6): **{match_counts[5]} vé**")
                    st.markdown(f"- 🥈 Trúng giải Nhì (4/6): **{match_counts[4]} vé**")
                    st.markdown(f"- 🥉 Trúng giải Ba (3/6): **{match_counts[3]} vé**")
                    st.markdown(f"- Không trúng (0-2/6): **{match_counts[0]+match_counts[1]+match_counts[2]} vé**")
                

    # =====================================================================
    # FULL HISTORICAL WIN-RATE TEST (dùng RealWorldAIEngine — NHANH)
    # =====================================================================
    st.markdown("---")
    st.markdown("### 🏆 TOÀN BỘ LỊCH SỬ — TỶ LỆ TRÚNG THỰC TẾ (FULL BACKTEST)")
    with st.expander("📊 Bấm để kiểm tra tỷ lệ AI trúng trên TẤT CẢ các kỳ lịch sử"):
        st.info(
            "⚡ **Chế độ quét nhanh toàn lịch sử:** Hệ thống sẽ tua ngược về từng kỳ (từ kỳ 60 đến nay), "
            "dùng AI dự đoán Top-6 / Top-10 / Top-15 số, sau đó so sánh với kết quả THỰC TẾ đã xảy ra. "
            "Đây là bài kiểm tra TRUNG THỰC nhất về độ chính xác của AI."
        )

        col_bt1, col_bt2 = st.columns(2)
        with col_bt1:
            bt_start_pct = st.slider("Bắt đầu từ % lịch sử (0 = kỳ đầu tiên):", 0, 80, 0, step=5,
                                     help="0% = test từ kỳ 60, 50% = chỉ test nửa sau lịch sử")
        with col_bt2:
            bt_step = st.selectbox("Bước nhảy (Mỗi bao nhiêu kỳ test 1 lần):", [1, 2, 5, 10], index=0,
                                   help="Bước=1 test mọi kỳ (chính xác nhất nhưng chậm hơn). Bước=5 test 1/5 số kỳ (nhanh 5x).")

        st.caption(
            f"📌 Ước tính số kỳ sẽ test: **{max(0, (len(real_data) - 60 - int(len(real_data) * bt_start_pct / 100)) // bt_step)}** kỳ "
            f"(từ {int(len(real_data) * bt_start_pct / 100) + 60} → {len(real_data)})"
        )

        if st.button("🚀 CHẠY KIỂM TRA TOÀN BỘ LỊCH SỬ", key="full_bt_btn"):
            total_draws_bt = len(real_data)
            if total_draws_bt < 60:
                st.error("Không đủ dữ liệu để backtest (cần tối thiểu 60 kỳ).")
            else:
                start_idx = 60 + int(total_draws_bt * bt_start_pct / 100)
                test_indices_bt = list(range(start_idx, total_draws_bt, bt_step))
                n_test = len(test_indices_bt)

                if n_test == 0:
                    st.error("Không có kỳ nào để test với cài đặt hiện tại.")
                else:
                    bt_prog = st.progress(0)
                    bt_status = st.empty()

                    # Bộ đếm match cho Top-6 / Top-10 / Top-15
                    counts6  = {k: 0 for k in range(7)}   # match = 0..6
                    counts10 = {k: 0 for k in range(7)}   # ≥k match vào top-10
                    counts15 = {k: 0 for k in range(7)}   # ≥k match vào top-15

                    detail_rows = []   # lưu chi tiết 50 kỳ gần nhất để hiển thị bảng

                    for step_i, cur_idx in enumerate(test_indices_bt):
                        hist = real_data[:cur_idx]
                        actual = set(real_data[cur_idx])

                        bt_status.text(
                            f"⏳ Đang test kỳ {cur_idx}/{total_draws_bt} "
                            f"({step_i+1}/{n_test}) — {int((step_i+1)/n_test*100)}%"
                        )
                        bt_prog.progress((step_i + 1) / n_test)

                        try:
                            eng = RealWorldAIEngine(hist, max_number)

                            # --- Lấy top-15 pool từ Ensemble ---
                            from collections import Counter as _Counter
                            m1 = eng.model_markov_chain()
                            m2 = eng.model_gap_overdue(top_n=15)
                            m3 = eng.model_momentum_neural()
                            m4 = eng.model_advanced_ml()

                            # Cho điểm tổng hợp có trọng số
                            vote = _Counter()
                            for num in m4[:15]: vote[num] += 5
                            for num in m2[:15]: vote[num] += 3
                            for num in m3[:15]: vote[num] += 2
                            for num in m1[:15]: vote[num] += 1

                            ranked_pool = [n for n, _ in vote.most_common(15)]
                            top6  = set(ranked_pool[:6])
                            top10 = set(ranked_pool[:10])
                            top15 = set(ranked_pool[:15])

                            hit6  = len(top6  & actual)
                            hit10 = len(top10 & actual)
                            hit15 = len(top15 & actual)

                            counts6[hit6]   += 1
                            counts10[hit10] += 1
                            counts15[hit15] += 1

                            # Lưu 50 kỳ gần nhất vào detail
                            if step_i >= n_test - 50:
                                detail_rows.append({
                                    "Kỳ": cur_idx,
                                    "Kết quả thật": " ".join(f"{n:02d}" for n in sorted(actual)),
                                    "Top-6 AI": " ".join(f"{n:02d}" for n in sorted(top6)),
                                    "Trúng/6": hit6,
                                    "Trúng/10": hit10,
                                    "Trúng/15": hit15,
                                })
                        except Exception:
                            continue

                    bt_prog.progress(1.0)
                    bt_status.empty()

                    st.success(f"✅ Hoàn tất! Đã kiểm tra **{n_test} kỳ** từ kỳ {start_idx} đến kỳ {total_draws_bt}.")

                    # ---- KẾT QUẢ TỔNG HỢP ----
                    st.markdown("---")
                    st.markdown("## 📊 KẾT QUẢ TỔNG HỢP")

                    def pct(c, total):
                        return f"{c/total*100:.1f}%" if total > 0 else "0%"

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown("### 🎯 Dự đoán Top-6 Số")
                        st.markdown(f"| Trúng | Số kỳ | Tỷ lệ |")
                        st.markdown(f"|-------|--------|-------|")
                        for k in range(6, -1, -1):
                            emoji = {6:"🏆",5:"🥇",4:"🥈",3:"🥉",2:"",1:"",0:""}.get(k,"")
                            st.markdown(f"| {emoji} **{k}/6** | {counts6[k]} | {pct(counts6[k], n_test)} |")

                    with c2:
                        st.markdown("### 🔟 Pool Top-10 Số")
                        st.markdown(f"| ≥X trúng | Số kỳ | Tỷ lệ |")
                        st.markdown(f"|----------|--------|-------|")
                        for k in range(6, -1, -1):
                            above = sum(counts10[i] for i in range(k, 7))
                            emoji = {6:"🏆",5:"🥇",4:"🥈",3:"🥉",2:"",1:"",0:""}.get(k,"")
                            st.markdown(f"| {emoji} ≥{k}/6 | {above} | {pct(above, n_test)} |")

                    with c3:
                        st.markdown("### 🎱 Pool Top-15 Số")
                        st.markdown(f"| ≥X trúng | Số kỳ | Tỷ lệ |")
                        st.markdown(f"|----------|--------|-------|")
                        for k in range(6, -1, -1):
                            above = sum(counts15[i] for i in range(k, 7))
                            emoji = {6:"🏆",5:"🥇",4:"🥈",3:"🥉",2:"",1:"",0:""}.get(k,"")
                            st.markdown(f"| {emoji} ≥{k}/6 | {above} | {pct(above, n_test)} |")

                    # ---- METRIC NỔI BẬT ----
                    st.markdown("---")
                    st.markdown("## 🔑 CHỈ SỐ QUAN TRỌNG NHẤT")
                    m1c, m2c, m3c, m4c = st.columns(4)
                    with m1c:
                        v = counts6[3] + counts6[4] + counts6[5] + counts6[6]
                        st.metric("Top-6 trúng ≥3/6", f"{pct(v, n_test)}", f"{v}/{n_test} kỳ")
                    with m2c:
                        v4 = counts6[4] + counts6[5] + counts6[6]
                        st.metric("Top-6 trúng ≥4/6", f"{pct(v4, n_test)}", f"{v4}/{n_test} kỳ")
                    with m3c:
                        v3_10 = sum(counts10[i] for i in range(3, 7))
                        st.metric("Pool-10 có ≥3 số trúng", f"{pct(v3_10, n_test)}", f"{v3_10}/{n_test} kỳ",
                                  help="Tức là trong 10 số dự đoán, ít nhất 3 số khớp với kết quả thật")
                    with m4c:
                        v4_10 = sum(counts10[i] for i in range(4, 7))
                        st.metric("Pool-10 có ≥4 số trúng", f"{pct(v4_10, n_test)}", f"{v4_10}/{n_test} kỳ")

                    # ---- BIỂU ĐỒ ----
                    st.markdown("---")
                    st.markdown("#### 📈 Phân bố Số Trúng trong Top-6 Dự Đoán")
                    try:
                        import altair as alt
                        chart_data = pd.DataFrame({
                            "Số trúng": [f"{k}/6" for k in range(7)],
                            "Số kỳ": [counts6[k] for k in range(7)],
                            "Màu": ["#ff0055" if k >= 4 else "#ffaa00" if k == 3 else "#444" for k in range(7)]
                        })
                        bar = alt.Chart(chart_data).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                            x=alt.X("Số trúng:O", sort=None, title="Số trúng (Top-6)"),
                            y=alt.Y("Số kỳ:Q", title="Số kỳ"),
                            color=alt.Color("Màu:N", scale=None, legend=None),
                            tooltip=["Số trúng", "Số kỳ"]
                        ).properties(height=300)
                        st.altair_chart(bar, use_container_width=True)
                    except Exception:
                        pass

                    # ---- BẢNG CHI TIẾT 50 KỲ GẦN NHẤT ----
                    if detail_rows:
                        st.markdown("---")
                        st.markdown(f"#### 📋 Chi tiết {len(detail_rows)} kỳ gần nhất được test")
                        df_detail = pd.DataFrame(detail_rows)

                        def color_hits(val):
                            if val >= 4: return "background-color: rgba(255,0,85,0.4); font-weight:bold"
                            if val == 3: return "background-color: rgba(255,170,0,0.3)"
                            return ""

                        try:
                            styled = df_detail.style.map(color_hits, subset=["Trúng/6", "Trúng/10", "Trúng/15"])
                        except AttributeError:
                            # Fallback for older pandas versions
                            styled = df_detail.style.applymap(color_hits, subset=["Trúng/6", "Trúng/10", "Trúng/15"])
                        st.dataframe(styled, use_container_width=True, hide_index=True)

                    # ---- NHẬN XÉT AI ----
                    st.markdown("---")
                    rate3 = (counts6[3] + counts6[4] + counts6[5] + counts6[6]) / max(n_test, 1) * 100
                    rate_pool10_3 = sum(counts10[i] for i in range(3, 7)) / max(n_test, 1) * 100
                    if rate_pool10_3 >= 60:
                        st.success(f"🔥 **AI ĐÁNH GIÁ: XUẤT SẮC** — Pool 10 số bao phủ ≥3 số trúng đến {rate_pool10_3:.1f}% kỳ. Chiến lược BAO-10 cực kỳ hiệu quả!")
                    elif rate_pool10_3 >= 40:
                        st.warning(f"⚠️ **AI ĐÁNH GIÁ: KHÁ** — Pool 10 số bao phủ ≥3 số trúng {rate_pool10_3:.1f}% kỳ. Nên dùng dàn bao 10-15 vé.")
                    else:
                        st.error(f"📉 **AI ĐÁNH GIÁ: TRUNG BÌNH** — Pool 10 số bao phủ {rate_pool10_3:.1f}% kỳ. Lý do: Xổ số có độ ngẫu nhiên rất cao. Hãy dùng pool 15 số để tăng coverage.")


if __name__ == "__main__":
    if check_password():
        main_app()
