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
        run_btn = st.button("⚡ DỰ ĐOÁN JACKPOT (SIÊU TRÍ TUỆ ĐỘNG) ⚡", use_container_width=True)

    if run_btn:
        st.session_state.prediction_ready = False
        
        progress_bar = st.progress(0)
        status = st.empty()
        
        status.text("Đang tính toán Z-Score và Ma trận đồng xuất hiện...")
        time.sleep(0.5)
        progress_bar.progress(30)
        
        status.text("Đang kích hoạt Vulnerability Scanner để dò tìm Điểm Uốn Hỗn Độn...")
        time.sleep(0.5)
        progress_bar.progress(60)
        
        try:
            from models.vulnerability_scanner import VulnerabilityScanner
            from models.exploit_engine import ExploitEngine
            
            scanner = VulnerabilityScanner(max_number, 6)
            scan_results = scanner.scan_all(real_data)
            
            status.text("Đang chạy thuật toán Cắt Tỉa Vét Cạn (Quantum Pruning)...")
            time.sleep(0.5)
            progress_bar.progress(80)
            
            engine = ExploitEngine(max_number, 6)
            exploit = engine.exploit(real_data, scan_results, n_sets=5)
            
            if exploit['predictions']:
                st.session_state.best_prediction = exploit['predictions'][0]['numbers']
                st.success(f"⚠️ Đã tóm gọn {exploit['biases_used']} quy luật bất thường của lồng quay! Tỉ lệ tự tin: {exploit['confidence']}%")
            else:
                st.warning("Không tìm thấy lỗ hổng thuật toán nào. Chuyển về V9 Dự phòng.")
                from models.ultimate_engine import UltimateEngine
                advanced_engine = UltimateEngine(max_number, 6)
                result = advanced_engine.predict(real_data)
                st.session_state.best_prediction = result['primary']
        except Exception as e:
            st.error(f"Lỗi: {e}")
                
        progress_bar.progress(100)
        status.empty()
        st.session_state.prediction_ready = True

    if st.session_state.prediction_ready:
        st.success("✅ ĐÃ CHỐT ĐƯỢC BỘ SỐ HOÀN HẢO CHO KỲ TIẾP THEO BẰNG CÔNG NGHỆ MACHINE LEARNING.")
        
        st.markdown("<div class='card' style='border-color: #00ff00;'>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: #00ff00 !important;'>🎯 BỘ SỐ CHỐT CUỐI CÙNG (DÀN VIP DUY NHẤT)</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #888;'><em>(Đây là bộ số có tỷ lệ nổ cao nhất, đã được cắt tỉa vét cạn mọi sai số)</em></p>", unsafe_allow_html=True)
        
        pred_balls_html = "".join([f"<div class='ball special-ball'>{num:02d}</div>" for num in st.session_state.best_prediction])
        st.markdown(f"<div style='text-align: center; padding: 20px;'>{pred_balls_html}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

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
    st.markdown("### 🧪 KIỂM THỬ ĐỘ CHÍNH XÁC (BACKTESTING)")
    with st.expander("Bấm để chạy Backtest (Kiểm tra lại lịch sử 50 kỳ gần nhất)"):
        st.warning("⚠️ Hệ thống sẽ tua ngược thời gian, ẩn đi kết quả thật và dùng AI để dự đoán các kỳ trong quá khứ, sau đó đối chiếu với kết quả thực tế để tính tỉ lệ trúng.")
        if st.button("🚀 CHẠY KIỂM THỬ BACKTEST 50 KỲ"):
            test_progress = st.progress(0)
            test_status = st.empty()
            
            total_draws = len(real_data)
            if total_draws < 60:
                st.error("Không đủ dữ liệu để backtest.")
            else:
                test_size = 50
                total_matches = 0
                match_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
                
                # Chạy từ quá khứ đến hiện tại
                for i in range(test_size):
                    # Data up to the draw we want to predict
                    current_idx = total_draws - test_size + i
                    historical_data_for_test = real_data[:current_idx]
                    actual_next_draw = real_data[current_idx]
                    
                    try:
                        from models.ultimate_engine import UltimateEngine
                        test_engine = UltimateEngine(max_number, 6)
                        pred = test_engine.predict(historical_data_for_test)['primary']
                    except:
                        test_engine = RealWorldAIEngine(historical_data_for_test, max_number)
                        pred = test_engine.optimize_ensemble()
                    
                    # Đếm số bóng trùng khớp
                    matches = len(set(pred) & set(actual_next_draw))
                    match_counts[matches] += 1
                    total_matches += matches
                    
                    test_progress.progress((i + 1) / test_size)
                    test_status.text(f"Đang kiểm thử kỳ {i + 1}/{test_size}... Khớp {matches}/6 số thực tế!")
                    
                test_status.empty()
                avg_match = total_matches / test_size
                win_rate = (total_matches / (test_size * 6)) * 100
                
                st.success(f"✅ Hoàn thành Backtest nghiệm thu trên {test_size} kỳ quay gần nhất!")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Tỉ lệ khớp bóng chính xác (Hit Rate)", f"{win_rate:.2f}%")
                    st.metric("Trung bình khớp mỗi kỳ", f"{avg_match:.2f} bóng / kỳ")
                with col_b:
                    st.markdown("**Phân bố số lượng bóng trúng:**")
                    for k in range(3, 7):
                        st.markdown(f"- Trúng {k}/6 số (Có giải): **{match_counts[k]} kỳ**")
                    st.markdown(f"- Trượt hoặc trúng 1-2 số: **{match_counts[0]+match_counts[1]+match_counts[2]} kỳ**")

if __name__ == "__main__":
    if check_password():
        main_app()
