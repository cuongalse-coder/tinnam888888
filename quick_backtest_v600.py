"""Quick backtest: Compare V500 (old) vs V600 (stacking) hit rates."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from collections import Counter

# Simulate lottery data fetch
def get_test_data():
    """Load from local HTML or generate realistic test data."""
    try:
        import re
        with open(os.path.join(os.path.dirname(__file__), '..', 'all_mega.html'), 'r', encoding='utf-8') as f:
            html = f.read()
        history = []
        rows = re.findall(r'<tr.*?>(.*?)</tr>', html, re.DOTALL | re.IGNORECASE)
        for row in rows:
            nums = re.findall(r'class="home-mini-whiteball">\s*(\d{2})\s*<', row)
            if len(nums) >= 6:
                chunk = sorted([int(n) for n in nums[:6]])
                if len(set(chunk)) == 6 and all(1 <= n <= 45 for n in chunk):
                    if chunk not in history:
                        history.append(chunk)
        if history:
            history.reverse()
            return history
    except:
        pass
    
    # Fallback: use GitHub data
    try:
        import requests, json
        r = requests.get("https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl", timeout=15)
        history = []
        for line in r.text.strip().split('\n'):
            if line:
                data = json.loads(line)
                if 'result' in data and len(data['result']) >= 6:
                    draw = sorted([int(n) for n in data['result'][:6]])
                    history.append(draw)
        return history
    except:
        pass
    
    # Last fallback: random data
    import random
    return [sorted(random.sample(range(1, 46), 6)) for _ in range(500)]


def run_backtest():
    data = get_test_data()
    print(f"Loaded {len(data)} draws")
    
    max_number = 45
    pick_count = 6
    
    # Test last 100 draws
    test_start = max(80, len(data) - 100)
    test_indices = list(range(test_start, len(data)))
    n_test = len(test_indices)
    
    print(f"Testing {n_test} draws (from {test_start} to {len(data)})")
    print("="*60)
    
    # V600 Stacking Engine
    from models.stacking_engine import StackingEngine
    
    counts_stack = {k: 0 for k in range(7)}  # top-6 hits
    counts_pool10 = {k: 0 for k in range(7)}  # pool-10 hits  
    counts_pool15 = {k: 0 for k in range(7)}  # pool-15 hits
    
    for step, idx in enumerate(test_indices):
        hist = data[:idx]
        actual = set(data[idx])
        
        stacker = StackingEngine(max_number, pick_count)
        result = stacker.predict_top_pool(hist, pool_size=15)
        
        pool = result['pool']
        top6 = set(pool[:6])
        top10 = set(pool[:10])
        top15 = set(pool[:15])
        
        hit6 = len(top6 & actual)
        hit10 = len(top10 & actual)
        hit15 = len(top15 & actual)
        
        counts_stack[hit6] += 1
        counts_pool10[hit10] += 1
        counts_pool15[hit15] += 1
        
        if (step + 1) % 10 == 0 or step == 0:
            print(f"  Step {step+1}/{n_test}: Draw {idx} | Hit6={hit6} Hit10={hit10} Hit15={hit15}")
    
    print("\n" + "="*60)
    print("V600 STACKING ENGINE RESULTS")
    print("="*60)
    
    def pct(c): return f"{c/n_test*100:.1f}%"
    
    print("\nTop-6 predictions:")
    for k in range(6, -1, -1):
        print(f"  {k}/6: {counts_stack[k]} draws ({pct(counts_stack[k])})")
    
    print(f"\n  >= 3/6: {sum(counts_stack[k] for k in range(3,7))} ({pct(sum(counts_stack[k] for k in range(3,7)))})")
    print(f"  >= 4/6: {sum(counts_stack[k] for k in range(4,7))} ({pct(sum(counts_stack[k] for k in range(4,7)))})")
    
    print("\nPool-10 predictions:")
    for k in range(6, -1, -1):
        above = sum(counts_pool10[i] for i in range(k, 7))
        print(f"  >= {k}/6: {above} ({pct(above)})")
    
    print("\nPool-15 predictions:")
    for k in range(6, -1, -1):
        above = sum(counts_pool15[i] for i in range(k, 7))
        print(f"  >= {k}/6: {above} ({pct(above)})")


if __name__ == "__main__":
    run_backtest()
