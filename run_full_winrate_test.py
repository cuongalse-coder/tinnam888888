"""
TINNAM AI — FULL HISTORICAL WIN-RATE BACKTEST
Chạy sliding-window qua TẤT CẢ kỳ lịch sử, báo cáo tỷ lệ trúng thực tế.
Dữ liệu: GitHub (vietvudanh/vietlott-data) — không cần scrape web.
"""
import sys, os, json, math, time
import numpy as np
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# 1. LẤY DỮ LIỆU THẬT TỪ GITHUB
# ============================================================
def fetch_data(game="mega"):
    import urllib.request
    urls = {
        "mega":  "https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl",
        "power": "https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power655.jsonl",
    }
    url = urls[game]
    print(f"📡 Đang tải dữ liệu {game.upper()} từ GitHub...")
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            text = r.read().decode('utf-8')
    except Exception as e:
        print(f"❌ Không kết nối được: {e}")
        sys.exit(1)

    max_num = 45 if game == "mega" else 55
    draws = []
    for line in text.strip().split('\n'):
        if not line: continue
        try:
            d = json.loads(line)
            nums = sorted([int(x) for x in d['result'][:6]])
            if len(set(nums)) == 6 and all(1 <= n <= max_num for n in nums):
                draws.append(nums)
        except Exception:
            continue
    print(f"✅ Tải xong: {len(draws)} kỳ lịch sử.")
    return draws, max_num


# ============================================================
# 2. AI ENGINE (4-model fast ensemble)
# ============================================================
class FastAIEngine:
    def __init__(self, data, max_number):
        self.data = data
        self.max_number = max_number
        self.all = list(range(1, max_number + 1))

    def _freq(self, lookback=None):
        sub = self.data[-lookback:] if lookback else self.data
        return Counter(n for d in sub for n in d)

    def model_markov(self):
        trans = defaultdict(Counter)
        for i in range(len(self.data) - 1):
            for n in self.data[i + 1]:
                trans[tuple(sorted(self.data[i]))][n] += 1
        last = tuple(sorted(self.data[-1]))
        if last in trans and trans[last]:
            return [n for n, _ in trans[last].most_common(15)]
        return [n for n, _ in self._freq(20).most_common(15)]

    def model_overdue(self, top_n=15):
        last_seen = {n: -1 for n in self.all}
        for i, d in enumerate(self.data):
            for n in d: last_seen[n] = i
        cur = len(self.data)
        gaps = {n: cur - last_seen[n] for n in self.all}
        avg = defaultdict(list)
        li = {}
        for i, d in enumerate(self.data):
            for n in d:
                if n in li: avg[n].append(i - li[n])
                li[n] = i
        due = {}
        for n in self.all:
            due[n] = gaps[n] / (np.mean(avg[n]) + 0.1) if avg[n] else 0
        return [n for n, _ in sorted(due.items(), key=lambda x: -x[1])[:top_n]]

    def model_momentum(self, top_n=15):
        w = {n: 0.0 for n in self.all}
        td = len(self.data)
        for i, d in enumerate(self.data):
            decay = 1 / (1 + np.exp(-(i - td + 20) / 5))
            for n in d: w[n] += decay
        return [n for n, _ in sorted(w.items(), key=lambda x: -x[1])[:top_n]]

    def model_ml(self, top_n=15):
        try:
            from sklearn.ensemble import RandomForestRegressor
            ws = 10
            if len(self.data) < ws + 5: return self.model_overdue(top_n)
            X, y = [], []
            for i in range(len(self.data) - ws - 1):
                feat = np.zeros(self.max_number)
                for d in self.data[i:i+ws]:
                    for n in d: feat[n-1] += 1
                tgt = np.zeros(self.max_number)
                for n in self.data[i+ws]: tgt[n-1] = 1
                X.append(feat); y.append(tgt)
            rf = RandomForestRegressor(n_estimators=80, max_depth=10, random_state=42)
            rf.fit(X, y)
            feat = np.zeros(self.max_number)
            for d in self.data[-ws:]:
                for n in d: feat[n-1] += 1
            pred = rf.predict([feat])[0]
            return [i+1 for i in np.argsort(pred)[::-1][:top_n]]
        except Exception:
            return self.model_overdue(top_n)

    def ensemble(self):
        m1 = self.model_markov()
        m2 = self.model_overdue()
        m3 = self.model_momentum()
        m4 = self.model_ml()
        vote = Counter()
        for n in m4[:15]: vote[n] += 5
        for n in m2[:15]: vote[n] += 3
        for n in m3[:15]: vote[n] += 2
        for n in m1[:15]: vote[n] += 1
        return [n for n, _ in vote.most_common(15)]


# ============================================================
# 3. CHẠY SLIDING-WINDOW BACKTEST
# ============================================================
def run_backtest(draws, max_num, step=2):
    total = len(draws)
    start = 60
    indices = list(range(start, total, step))
    n_test = len(indices)

    print(f"\n🔁 Bắt đầu backtest: {n_test} kỳ (step={step}, từ kỳ {start} → {total})\n")

    c6  = {k: 0 for k in range(7)}
    c10 = {k: 0 for k in range(7)}
    c15 = {k: 0 for k in range(7)}
    detail = []

    t0 = time.time()
    for si, idx in enumerate(indices):
        hist    = draws[:idx]
        actual  = set(draws[idx])

        eng = FastAIEngine(hist, max_num)
        pool = eng.ensemble()

        top6  = set(pool[:6])
        top10 = set(pool[:10])
        top15 = set(pool[:15])

        h6  = len(top6  & actual)
        h10 = len(top10 & actual)
        h15 = len(top15 & actual)

        c6[h6]   += 1
        c10[h10] += 1
        c15[h15] += 1

        if si >= n_test - 30:
            detail.append((idx, sorted(actual), sorted(top6), h6, h10, h15))

        # Progress every 5%
        if (si + 1) % max(1, n_test // 20) == 0 or si == n_test - 1:
            elapsed = time.time() - t0
            eta = elapsed / (si + 1) * (n_test - si - 1)
            pct_done = (si + 1) / n_test * 100
            bar = "█" * int(pct_done // 5) + "░" * (20 - int(pct_done // 5))
            print(f"  [{bar}] {pct_done:5.1f}%  ({si+1}/{n_test})  ETA: {eta:.0f}s", end="\r")

    elapsed_total = time.time() - t0
    print(f"\n\n✅ Hoàn thành trong {elapsed_total:.1f}s\n")
    return c6, c10, c15, n_test, detail


# ============================================================
# 4. IN KẾT QUẢ
# ============================================================
def print_results(c6, c10, c15, n_test, detail, game_name):
    def pct(v): return f"{v/n_test*100:5.1f}%"
    def above(cx, k): return sum(cx[i] for i in range(k, 7))

    SEP = "=" * 65

    print(SEP)
    print(f"   TINNAM AI — KẾT QUẢ BACKTEST TOÀN LỊCH SỬ ({game_name})")
    print(f"   Tổng kỳ đã test: {n_test}")
    print(SEP)

    # --- Top-6 ---
    print("\n🎯  DỰ ĐOÁN TOP-6 SỐ (đoán đúng bao nhiêu trong 6 số):")
    print(f"   {'Trúng':>6}  {'Số kỳ':>7}  {'Tỷ lệ':>7}")
    print("   " + "-" * 30)
    for k in range(6, -1, -1):
        tag = "🏆" if k==6 else ("🥇" if k==5 else ("🥈" if k==4 else ("🥉" if k==3 else "  ")))
        print(f"  {tag} {k}/6   {c6[k]:>7}   {pct(c6[k])}")

    # --- Top-10 ---
    print("\n🔟  POOL TOP-10 SỐ (≥X số rơi vào pool):")
    print(f"   {'≥X trúng':>9}  {'Số kỳ':>7}  {'Tỷ lệ':>7}")
    print("   " + "-" * 32)
    for k in range(6, -1, -1):
        ab = above(c10, k)
        tag = "🏆" if k==6 else ("🥇" if k==5 else ("🥈" if k==4 else ("🥉" if k==3 else "  ")))
        print(f"  {tag} ≥{k}/6    {ab:>7}   {pct(ab)}")

    # --- Top-15 ---
    print("\n🎱  POOL TOP-15 SỐ (≥X số rơi vào pool):")
    print(f"   {'≥X trúng':>9}  {'Số kỳ':>7}  {'Tỷ lệ':>7}")
    print("   " + "-" * 32)
    for k in range(6, -1, -1):
        ab = above(c15, k)
        tag = "🏆" if k==6 else ("🥇" if k==5 else ("🥈" if k==4 else ("🥉" if k==3 else "  ")))
        print(f"  {tag} ≥{k}/6    {ab:>7}   {pct(ab)}")

    # --- Key metrics ---
    print(f"\n{'─'*65}")
    print("🔑  CHỈ SỐ QUAN TRỌNG NHẤT:")
    v3_6   = above(c6, 3);  v4_6  = above(c6, 4)
    v3_10  = above(c10, 3); v4_10 = above(c10, 4)
    v3_15  = above(c15, 3); v4_15 = above(c15, 4)
    print(f"   Top-6  trúng ≥3/6  : {v3_6:>5} kỳ  → {pct(v3_6)}")
    print(f"   Top-6  trúng ≥4/6  : {v4_6:>5} kỳ  → {pct(v4_6)}")
    print(f"   Pool-10 có ≥3 trúng : {v3_10:>5} kỳ  → {pct(v3_10)}")
    print(f"   Pool-10 có ≥4 trúng : {v4_10:>5} kỳ  → {pct(v4_10)}")
    print(f"   Pool-15 có ≥3 trúng : {v3_15:>5} kỳ  → {pct(v3_15)}")
    print(f"   Pool-15 có ≥4 trúng : {v4_15:>5} kỳ  → {pct(v4_15)}")

    # --- Đánh giá ---
    rate_pool10_3 = v3_10 / max(n_test, 1) * 100
    print(f"\n{'─'*65}")
    print("🤖  NHẬN XÉT AI:")
    if rate_pool10_3 >= 65:
        print(f"   🔥 XUẤT SẮC — Pool 10 số bao phủ ≥3 trúng: {rate_pool10_3:.1f}%")
        print(f"   → Chiến lược BAO-10 cực kỳ hiệu quả!")
    elif rate_pool10_3 >= 45:
        print(f"   ⚠️  KHÁ — Pool 10 số bao phủ ≥3 trúng: {rate_pool10_3:.1f}%")
        print(f"   → Nên dùng Dàn Bao 10-15 vé để khai thác pool hiệu quả.")
    else:
        print(f"   📉 TRUNG BÌNH — Pool 10 số bao phủ: {rate_pool10_3:.1f}%")
        print(f"   → Xổ số có độ ngẫu nhiên rất cao. Dùng pool 15 số để tăng coverage.")

    # --- Chi tiết 30 kỳ gần nhất ---
    print(f"\n{'─'*65}")
    print("📋  CHI TIẾT 30 KỲ GẦN NHẤT ĐƯỢC TEST:")
    print(f"   {'Kỳ':>5}  {'Kết quả thật':<22}  {'Top-6 AI':<20}  T6  T10  T15")
    print("   " + "-" * 62)
    for (idx, actual, top6, h6, h10, h15) in detail:
        actual_str = " ".join(f"{n:02d}" for n in actual)
        top6_str   = " ".join(f"{n:02d}" for n in top6)
        mark6  = "🏆" if h6>=4 else ("⭐" if h6==3 else "  ")
        mark10 = "✅" if h10>=4 else ("🔸" if h10==3 else "  ")
        print(f"   {idx:>5}  {actual_str:<22}  {top6_str:<20}  {mark6}{h6}   {mark10}{h10}   {h15}")

    print(f"\n{SEP}\n")


# ============================================================
# 5. MAIN
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*65)
    print("   TINNAM AI V102.0 — FULL HISTORICAL BACKTEST RUNNER")
    print("="*65 + "\n")

    # --- MEGA 6/45 ---
    mega_draws, mega_max = fetch_data("mega")
    c6m, c10m, c15m, ntm, detm = run_backtest(mega_draws, mega_max, step=2)
    print_results(c6m, c10m, c15m, ntm, detm, "MEGA 6/45")

    # --- POWER 6/55 ---
    power_draws, power_max = fetch_data("power")
    c6p, c10p, c15p, ntp, detp = run_backtest(power_draws, power_max, step=2)
    print_results(c6p, c10p, c15p, ntp, detp, "POWER 6/55")
