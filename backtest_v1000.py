import sys
import os
import time
import json
from collections import Counter
from math import comb

# Cần load data và chạy wheeling
sys.path.append(r"C:\Users\HQSP\.gemini\antigravity\scratch\tinnam888888_test")
from models.nexus_engine import MegaExploitV15
from models.wheeling_optimizer import WheelingOptimizer

data_path = r"C:\Users\HQSP\.gemini\antigravity\scratch\tinnam888888_test\data\mega645.json"
with open(data_path, "r", encoding="utf-8") as f:
    raw = json.load(f)
    data = [sorted([int(x) for x in d["numbers"]]) for d in raw]

data = data[::-1] # Cũ nhất trước

print("Khoi tao Nexus Engine...")
engine = MegaExploitV15(max_number=45)
wheel = WheelingOptimizer(max_number=45)

total_draws = len(data)
START = max(100, total_draws - 50) # Test 50 kỳ gần nhất cho nhanh
STEP = 1

print(f"Bat dau backtest V1000 Radar tu ky {START} den {total_draws} (Tong: {total_draws - START} ky)")

hits_6 = 0
hits_5 = 0
hits_4 = 0
hits_3 = 0
total_tested = 0

t0 = time.time()

for cur_idx in range(START, total_draws, STEP):
    hist = data[:cur_idx]
    actual = set(data[cur_idx])
    
    # 1. AI đoán Top 22
    try:
        top_pool = engine.predict_top_pool(hist, pool_size=22)
    except Exception as e:
        print(f"Loi predict: {e}")
        continue
        
    # 2. Sinh 20 vé V1000 Radar
    try:
        tickets, cov, stats, total_gen = wheel.generate_wheel(
            pool=top_pool,
            num_tickets=20,
            constraints=None,
            history_data=hist,
            ai_top_core=top_pool[:6], # Hard core top 6
            hard_core_lock=4 # Khóa cứng 4 số lõi
        )
    except Exception as e:
        print(f"Loi wheel: {e}")
        continue
        
    total_tested += 1
    
    # 3. Kiểm tra trúng giải
    best_hit = 0
    for t in tickets:
        t_set = set(t['numbers'])
        hit = len(t_set & actual)
        if hit > best_hit:
            best_hit = hit
            
    if best_hit == 6: hits_6 += 1
    if best_hit >= 5: hits_5 += 1
    if best_hit >= 4: hits_4 += 1
    if best_hit >= 3: hits_3 += 1
    
    print(f"Ky {cur_idx} | Tot nhat: {best_hit}/6 | Mat {time.time()-t0:.1f}s", end='\r')

print("\n" + "="*50)
print(f"KET QUA TEST V1000 (20 VE BAN TIA) - {total_tested} KY")
print(f"Trung 6/6: {hits_6} ky ({hits_6/total_tested*100:.2f}%)")
print(f"Trung 5/6: {hits_5} ky ({hits_5/total_tested*100:.2f}%)")
print(f"Trung 4/6: {hits_4} ky ({hits_4/total_tested*100:.2f}%)")
print(f"Trung 3/6: {hits_3} ky ({hits_3/total_tested*100:.2f}%)")
print("="*50)
