"""
FULL BACKTEST: Kiểm tra tỷ lệ trúng 6/6 trên TOÀN BỘ lịch sử
Chạy 9-Model Ensemble (V750A) với Agreement Filter
"""
import json
import requests
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
import time
import sys

# ======== FETCH DATA ========
print("📡 Đang tải dữ liệu từ GitHub...")
url = "https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl"
resp = requests.get(url, timeout=30)
data = []
for line in resp.text.strip().split('\n'):
    if line:
        obj = json.loads(line)
        if obj.get('result') and len(obj['result']) >= 6:
            draw = sorted([int(n) for n in obj['result'][:6]])
            data.append(draw)

print(f"✅ Đã tải {len(data)} kỳ quay lịch sử (Mega 6/45)")

MAX_NUMBER = 45
PICK = 6
ALL_NUMBERS = list(range(1, MAX_NUMBER + 1))

# ======== 9 AI MODELS ========
def model_markov_chain(data):
    transitions = defaultdict(Counter)
    for i in range(len(data) - 1):
        current = tuple(sorted(data[i]))
        for num in data[i + 1]:
            transitions[current][num] += 1
    if data:
        last = tuple(sorted(data[-1]))
        if last in transitions and transitions[last]:
            return [num for num, _ in transitions[last].most_common(6)]
    freq = Counter(n for d in data[-20:] for n in d)
    return [n for n, _ in freq.most_common(6)]

def model_gap_overdue(data, top_n=15):
    last_seen = {num: -1 for num in ALL_NUMBERS}
    for i, draw in enumerate(data):
        for num in draw: last_seen[num] = i
    current_idx = len(data)
    gaps = {num: current_idx - last_seen[num] for num in ALL_NUMBERS}
    avg_gaps = defaultdict(list)
    last_idx = {}
    for i, draw in enumerate(data):
        for num in draw:
            if num in last_idx: avg_gaps[num].append(i - last_idx[num])
            last_idx[num] = i
    due_scores = {}
    for num in ALL_NUMBERS:
        if avg_gaps[num]:
            mean_gap = np.mean(avg_gaps[num])
            due_scores[num] = gaps[num] / (mean_gap + 0.1)
        else:
            due_scores[num] = 0
    return [num for num, _ in sorted(due_scores.items(), key=lambda x: -x[1])[:top_n]]

def model_momentum_neural(data):
    weights = {num: 0.0 for num in ALL_NUMBERS}
    total = len(data)
    for i, draw in enumerate(data):
        decay = 1 / (1 + np.exp(-(i - total + 20) / 5))
        for num in draw: weights[num] += decay
    return [num for num, _ in sorted(weights.items(), key=lambda x: -x[1])[:6]]

def model_advanced_ml(data):
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.cluster import KMeans
        if len(data) < 20: return model_momentum_neural(data)
        X, y = [], []
        ws = 10
        for i in range(len(data) - ws - 1):
            features = np.zeros(MAX_NUMBER)
            for draw in data[i:i+ws]:
                for num in draw: features[num-1] += 1
            targets = np.zeros(MAX_NUMBER)
            for num in data[i+ws]: targets[num-1] = 1
            X.append(features); y.append(targets)
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        recent = np.zeros(MAX_NUMBER)
        for draw in data[-ws:]:
            for num in draw: recent[num-1] += 1
        pred = rf.predict([recent])[0]
        return [n+1 for n in np.argsort(pred)[::-1][:6]]
    except: return model_momentum_neural(data)

def model_knn_mirror(data):
    if len(data) < 20: return model_momentum_neural(data)
    pattern = set(data[-1]) | set(data[-2]) | set(data[-3])
    if len(data) > 3: pattern |= set(data[-4])
    n = len(data)
    sims = []
    for i in range(3, n - 3):
        past = set(data[i]) | set(data[i-1]) | set(data[i-2]) | set(data[i-3])
        inter = len(pattern & past)
        recency = 1.0 + 0.5 * (i / n)
        if inter >= 5: sims.append((inter * recency, i + 1))
    sims.sort(key=lambda x: -x[0])
    votes = Counter()
    for score, nxt in sims[:30]:
        if nxt < n:
            for num in data[nxt]: votes[num] += score
    if not votes: return model_momentum_neural(data)
    return [n for n, _ in votes.most_common(20)]

def model_pair_matrix(data):
    if len(data) < 30: return model_gap_overdue(data)
    pair_scores = Counter()
    n = len(data)
    for idx, draw in enumerate(data):
        decay = 0.3 + 0.7 * (idx / n)
        for p in combinations(sorted(draw[:6]), 2): pair_scores[p] += decay
    last_draw = set(data[-1][:6])
    cand = Counter()
    for num in ALL_NUMBERS:
        if num in last_draw: continue
        for anchor in last_draw:
            key = tuple(sorted([num, anchor]))
            cand[num] += pair_scores.get(key, 0)
    triplet_bonus = Counter()
    for idx in range(max(0, n - 100), n):
        draw = data[idx]
        for trip in combinations(sorted(draw[:6]), 3):
            ts = set(trip)
            overlap = ts & last_draw
            if len(overlap) >= 2:
                for num in ts - last_draw: triplet_bonus[num] += 1.5
    for num in triplet_bonus: cand[num] += triplet_bonus[num]
    return [n for n, _ in cand.most_common(15)]

def model_delta_momentum(data):
    if len(data) < 30: return model_momentum_neural(data)
    scores = {}
    for num in ALL_NUMBERS:
        f5 = sum(1 for d in data[-5:] if num in d) / 5
        f5p = sum(1 for d in data[-10:-5] if num in d) / 5
        f15 = sum(1 for d in data[-15:] if num in d) / 15
        f15p = sum(1 for d in data[-30:-15] if num in d) / 15
        ds = f5 - f5p; dm = f15 - f15p
        m = ds * 3 + dm * 2
        if num in data[-1]: m += 0.5
        if len(data) >= 2 and num in data[-2]: m += 0.3
        scores[num] = m
    return [n for n, _ in sorted(scores.items(), key=lambda x: -x[1])[:15]]

def model_cond_prob(data):
    if len(data) < 30: return []
    last = set(data[-1])
    cc = defaultdict(lambda: defaultdict(int))
    tg = defaultdict(int)
    for i in range(len(data) - 1):
        for g in data[i]:
            tg[g] += 1
            for nn in data[i+1]: cc[g][nn] += 1
    scores = {}
    for num in ALL_NUMBERS:
        scores[num] = sum(cc[g].get(num, 0) / tg[g] for g in last if tg[g] > 0)
    return [n for n, _ in sorted(scores.items(), key=lambda x: -x[1])[:15]]

def model_freq_gap_hybrid(data):
    if len(data) < 30: return model_gap_overdue(data)
    expected = 6 / len(ALL_NUMBERS)
    scores = {}
    for num in ALL_NUMBERS:
        f5 = sum(1 for d in data[-5:] if num in d) / 5
        f15 = sum(1 for d in data[-15:] if num in d) / 15
        fs = (f5 / (expected + 0.01)) * 0.6 + (f15 / (expected + 0.01)) * 0.4
        last_seen = -1
        for i in range(len(data)-1, -1, -1):
            if num in data[i]: last_seen = i; break
        gap = len(data) - last_seen if last_seen >= 0 else len(data)
        appearances = [i for i, d in enumerate(data) if num in d]
        mg = len(ALL_NUMBERS) / 6
        if len(appearances) >= 2:
            gs = [appearances[j+1]-appearances[j] for j in range(len(appearances)-1)]
            mg = sum(gs) / len(gs)
        od = gap / (mg + 0.1)
        if fs > 0.8 and od > 0.7: scores[num] = fs * od * 3
        elif od > 1.5: scores[num] = od * 1.5
        elif fs > 1.3: scores[num] = fs * 2
        else: scores[num] = fs * 0.5 + od * 0.5
    return [n for n, _ in sorted(scores.items(), key=lambda x: -x[1])[:15]]

def run_ensemble(data, pool_size=20):
    """V750A: 9-Model Ensemble + Agreement Filter"""
    m1 = model_markov_chain(data)
    m2 = model_gap_overdue(data, top_n=15)
    m3 = model_momentum_neural(data)
    m4 = model_advanced_ml(data)
    m5 = model_knn_mirror(data)
    m6 = model_pair_matrix(data)
    m7 = model_delta_momentum(data)
    m8 = model_cond_prob(data)
    m9 = model_freq_gap_hybrid(data)
    
    votes = Counter()
    for num in m5[:15]: votes[num] += 12
    for num in m6[:15]: votes[num] += 8
    for num in m8[:15]: votes[num] += 6
    for num in m9[:15]: votes[num] += 5
    for num in m4[:15]: votes[num] += 4
    for num in m7[:15]: votes[num] += 4
    for num in m2[:15]: votes[num] += 3
    for num in m3[:6]:  votes[num] += 2
    for num in m1[:6]:  votes[num] += 1
    
    strong = [set(m5[:12]), set(m6[:12]), set(m8[:12]), set(m7[:12])]
    for num in ALL_NUMBERS:
        c = sum(1 for ml in strong if num in ml)
        if c >= 3: votes[num] += c * 5
    
    return [n for n, _ in votes.most_common(pool_size)]

# ======== BACKTEST ========
print("\n" + "="*70)
print("🧪 BẮT ĐẦU BACKTEST TOÀN BỘ LỊCH SỬ")
print("="*70)

START = 60
STEP = 1  # Test mỗi kỳ
total_draws = len(data)
test_indices = list(range(START, total_draws, STEP))
n_test = len(test_indices)

print(f"📊 Test từ kỳ {START} đến kỳ {total_draws} (Bước={STEP})")
print(f"📊 Tổng số kỳ test: {n_test}")
print(f"📊 Đang chạy... (có thể mất 5-15 phút)\n")

counts6  = {k: 0 for k in range(7)}
counts10 = {k: 0 for k in range(7)}
counts15 = {k: 0 for k in range(7)}
counts20 = {k: 0 for k in range(7)}

detail_rows = []
t0 = time.time()
errors = 0

for step_i, cur_idx in enumerate(test_indices):
    hist = data[:cur_idx]
    actual = set(data[cur_idx])
    
    if step_i % 50 == 0:
        elapsed = time.time() - t0
        eta = (elapsed / (step_i + 1)) * (n_test - step_i - 1) if step_i > 0 else 0
        print(f"  ⏳ Kỳ {cur_idx}/{total_draws} ({step_i+1}/{n_test}) — {int(step_i/n_test*100)}% — ETA: {int(eta)}s", end='\r')
    
    try:
        ranked = run_ensemble(hist, pool_size=20)
        top6  = set(ranked[:6])
        top10 = set(ranked[:10])
        top15 = set(ranked[:15])
        top20 = set(ranked[:20])
        
        hit6  = len(top6  & actual)
        hit10 = len(top10 & actual)
        hit15 = len(top15 & actual)
        hit20 = len(top20 & actual)
        
        counts6[hit6]   += 1
        counts10[hit10] += 1
        counts15[hit15] += 1
        counts20[hit20] += 1
        
        if hit6 >= 4:
            detail_rows.append({
                'draw': cur_idx,
                'actual': sorted(actual),
                'top6': sorted(top6),
                'hit6': hit6,
                'hit10': hit10,
                'hit15': hit15,
                'hit20': hit20
            })
    except Exception as e:
        errors += 1
        continue

elapsed = time.time() - t0

# ======== KẾT QUẢ ========
print("\n\n" + "="*70)
print("📊 KẾT QUẢ BACKTEST TOÀN BỘ LỊCH SỬ")
print("="*70)
print(f"Tổng kỳ test: {n_test} | Lỗi: {errors} | Thời gian: {elapsed:.0f}s\n")

def pct(c, t):
    return f"{c/t*100:.2f}%" if t > 0 else "0%"

print("┌─────────────┬────────────┬────────────┬────────────┬────────────┐")
print("│  Số trúng   │  Top-6 AI  │ Top-10 AI  │ Top-15 AI  │ Top-20 AI  │")
print("├─────────────┼────────────┼────────────┼────────────┼────────────┤")
for k in range(6, -1, -1):
    c6 = counts6[k]
    c10 = sum(counts10[i] for i in range(k, 7))
    c15 = sum(counts15[i] for i in range(k, 7))
    c20 = sum(counts20[i] for i in range(k, 7))
    emoji = {6:"🏆",5:"🥇",4:"🥈",3:"🥉",2:"  ",1:"  ",0:"  "}.get(k,"")
    label = f"{emoji} {k}/6" if k <= 6 else f"   {k}/6"
    
    if k == 6:
        print(f"│ {label:>11} │ {c6:>4} ({pct(c6,n_test):>6}) │ {c10:>4} ({pct(c10,n_test):>6}) │ {c15:>4} ({pct(c15,n_test):>6}) │ {c20:>4} ({pct(c20,n_test):>6}) │")
    else:
        print(f"│ {label:>11} │ {c6:>4} ({pct(c6,n_test):>6}) │ ≥{k}: {c10:>3} ({pct(c10,n_test):>6}) │ ≥{k}: {c15:>3} ({pct(c15,n_test):>6}) │ ≥{k}: {c20:>3} ({pct(c20,n_test):>6}) │")
print("└─────────────┴────────────┴────────────┴────────────┴────────────┘")

print(f"\n🔑 CHỈ SỐ QUAN TRỌNG:")
v6_66 = counts6[6]
v6_56 = counts6[5] + counts6[6]
v6_46 = counts6[4] + v6_56
v6_36 = counts6[3] + v6_46
print(f"  Top-6  trúng 6/6: {v6_66:>4} kỳ ({pct(v6_66, n_test)})")
print(f"  Top-6  trúng ≥5/6: {v6_56:>4} kỳ ({pct(v6_56, n_test)})")
print(f"  Top-6  trúng ≥4/6: {v6_46:>4} kỳ ({pct(v6_46, n_test)})")
print(f"  Top-6  trúng ≥3/6: {v6_36:>4} kỳ ({pct(v6_36, n_test)})")

v10_6 = sum(counts10[i] for i in range(6, 7))
v10_5 = sum(counts10[i] for i in range(5, 7))
v10_4 = sum(counts10[i] for i in range(4, 7))
v10_3 = sum(counts10[i] for i in range(3, 7))
print(f"\n  Top-10 chứa 6/6: {v10_6:>4} kỳ ({pct(v10_6, n_test)})")
print(f"  Top-10 chứa ≥5/6: {v10_5:>4} kỳ ({pct(v10_5, n_test)})")
print(f"  Top-10 chứa ≥4/6: {v10_4:>4} kỳ ({pct(v10_4, n_test)})")
print(f"  Top-10 chứa ≥3/6: {v10_3:>4} kỳ ({pct(v10_3, n_test)})")

v20_6 = sum(counts20[i] for i in range(6, 7))
v20_5 = sum(counts20[i] for i in range(5, 7))
v20_4 = sum(counts20[i] for i in range(4, 7))
print(f"\n  Top-20 chứa 6/6: {v20_6:>4} kỳ ({pct(v20_6, n_test)})")
print(f"  Top-20 chứa ≥5/6: {v20_5:>4} kỳ ({pct(v20_5, n_test)})")
print(f"  Top-20 chứa ≥4/6: {v20_4:>4} kỳ ({pct(v20_4, n_test)})")

# Xác suất ngẫu nhiên để so sánh
from math import comb
total_combos = comb(45, 6)  # 8,145,060
random_6_6 = 1 / total_combos * 100
random_5_6 = comb(6,5) * comb(39,1) / total_combos * 100
random_4_6 = comb(6,4) * comb(39,2) / total_combos * 100
random_3_6 = comb(6,3) * comb(39,3) / total_combos * 100

print(f"\n📈 SO SÁNH VỚI XÁC SUẤT NGẪU NHIÊN:")
print(f"  Ngẫu nhiên 6/6:  {random_6_6:.6f}% (1 trong {total_combos:,})")
print(f"  Ngẫu nhiên 5/6:  {random_5_6:.4f}%")
print(f"  Ngẫu nhiên 4/6:  {random_4_6:.3f}%")
print(f"  Ngẫu nhiên 3/6:  {random_3_6:.2f}%")

if detail_rows:
    print(f"\n🏆 CÁC KỲ TRÚNG LỚN (≥4/6 trong Top-6):")
    for d in detail_rows:
        print(f"  Kỳ {d['draw']:>4}: Thật={d['actual']} | AI Top-6={d['top6']} | Trúng: {d['hit6']}/6 (Top-10: {d['hit10']}, Top-15: {d['hit15']}, Top-20: {d['hit20']})")

print("\n" + "="*70)
print("✅ BACKTEST HOÀN TẤT")
print("="*70)
