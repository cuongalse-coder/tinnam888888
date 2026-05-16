"""
NEXUS V200 FULL HISTORY BACKTEST
Tests prediction accuracy across ALL historical draws.
"""
import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
import requests, re

def fetch_data(game_type="Mega 6/45"):
    """Fetch all historical data."""
    today_str = datetime.now().strftime('%d-%m-%Y')
    max_num = 45 if game_type == "Mega 6/45" else 55
    
    url = f"https://www.ketquadientoan.com/tat-ca-ky-xo-so-mega-6-45.html?datef=18-07-2016&datet={today_str}" if game_type == "Mega 6/45" else f"https://www.ketquadientoan.com/tat-ca-ky-xo-so-power-655.html?datef=01-01-2018&datet={today_str}"
    
    try:
        import cloudscraper
        scraper = cloudscraper.create_scraper(delay=5, browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False})
    except:
        scraper = requests.Session()
    
    response = scraper.get(url, timeout=30)
    html = response.text
    history = []
    
    rows = re.findall(r'<tr.*?>(.*?)</tr>', html, re.DOTALL | re.IGNORECASE)
    for row in rows:
        nums = re.findall(r'class="home-mini-whiteball">\s*(\d{2})\s*<', row)
        if len(nums) < 6:
            continue
        chunk = sorted([int(n) for n in nums[:6]])
        if len(set(chunk)) == 6 and all(1 <= n <= max_num for n in chunk):
            if chunk not in history:
                history.append(chunk)
    
    if history:
        history.reverse()
    
    if not history:
        # GitHub fallback
        gh_url = "https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl"
        resp = requests.get(gh_url, timeout=10)
        if resp.status_code == 200:
            for line in resp.text.strip().split('\n'):
                if line:
                    data = json.loads(line)
                    if 'result' in data and len(data['result']) >= 6:
                        draw = sorted([int(n) for n in data['result'][:6]])
                        history.append(draw)
    
    return history

def run_backtest():
    print("=" * 70)
    print("  TINNAM AI V200.0 QUANTUM NEXUS — FULL HISTORY BACKTEST")
    print("=" * 70)
    
    print("\n[1/3] Fetching ALL historical data...")
    data = fetch_data("Mega 6/45")
    total = len(data)
    print(f"  => Got {total} draws")
    
    if total < 70:
        print("ERROR: Not enough data for backtest")
        return
    
    from models.nexus_engine import NexusEngine
    
    # Test from draw 60 onwards (need 60 for training)
    start_idx = 60
    step = 1  # Test every draw
    test_indices = list(range(start_idx, total, step))
    n_test = len(test_indices)
    
    print(f"\n[2/3] Running backtest on {n_test} draws (from draw {start_idx} to {total})...")
    print(f"  Testing EVERY draw, step={step}")
    
    # Counters for Top-6, Top-10, Top-15
    counts6 = {k: 0 for k in range(7)}
    counts10 = {k: 0 for k in range(7)}
    counts15 = {k: 0 for k in range(7)}
    
    max_number = 45
    t0 = time.time()
    
    for step_i, cur_idx in enumerate(test_indices):
        hist = data[:cur_idx]
        actual = set(data[cur_idx])
        
        if (step_i + 1) % 50 == 0 or step_i == 0:
            elapsed = time.time() - t0
            eta = (elapsed / (step_i + 1)) * (n_test - step_i - 1)
            print(f"  Progress: {step_i+1}/{n_test} ({(step_i+1)/n_test*100:.1f}%) | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")
        
        try:
            engine = NexusEngine(max_number, 6)
            result = engine.predict(hist, n_sets=1)
            
            top_pool = result.get('top_pool', [])
            if not top_pool:
                continue
            
            top6 = set(top_pool[:6])
            top10 = set(top_pool[:10])
            top15 = set(top_pool[:15])
            
            hit6 = len(top6 & actual)
            hit10 = len(top10 & actual)
            hit15 = len(top15 & actual)
            
            counts6[hit6] += 1
            counts10[hit10] += 1
            counts15[hit15] += 1
            
        except Exception as e:
            print(f"  Error at draw {cur_idx}: {e}")
            continue
    
    elapsed_total = time.time() - t0
    
    # ===== RESULTS =====
    print(f"\n{'=' * 70}")
    print(f"  RESULTS — Tested {n_test} draws in {elapsed_total:.1f}s")
    print(f"{'=' * 70}")
    
    def pct(c, t):
        return f"{c/t*100:.1f}%" if t > 0 else "0%"
    
    print(f"\n--- TOP-6 PREDICTION (6 so chinh xac nhat) ---")
    for k in range(6, -1, -1):
        emoji = {6:"JACKPOT", 5:"GIAI 1", 4:"GIAI 2", 3:"GIAI 3", 2:"", 1:"", 0:""}.get(k, "")
        print(f"  Trung {k}/6: {counts6[k]:>5} ky  ({pct(counts6[k], n_test)})  {emoji}")
    
    ge3_6 = sum(counts6[k] for k in range(3, 7))
    ge4_6 = sum(counts6[k] for k in range(4, 7))
    print(f"\n  => Top-6 trung >=3/6: {pct(ge3_6, n_test)} ({ge3_6}/{n_test})")
    print(f"  => Top-6 trung >=4/6: {pct(ge4_6, n_test)} ({ge4_6}/{n_test})")
    
    print(f"\n--- TOP-10 POOL (Ho 10 so) ---")
    for k in range(6, -1, -1):
        above = sum(counts10[i] for i in range(k, 7))
        print(f"  >=  {k}/6 trung: {above:>5} ky  ({pct(above, n_test)})")
    
    print(f"\n--- TOP-15 POOL (Ho 15 so) ---")
    for k in range(6, -1, -1):
        above = sum(counts15[i] for i in range(k, 7))
        print(f"  >=  {k}/6 trung: {above:>5} ky  ({pct(above, n_test)})")
    
    # Save results
    results = {
        "version": "V200.0 QUANTUM NEXUS",
        "total_draws": total,
        "tested_draws": n_test,
        "elapsed_seconds": round(elapsed_total, 1),
        "top6": {str(k): counts6[k] for k in range(7)},
        "top10": {str(k): counts10[k] for k in range(7)},
        "top15": {str(k): counts15[k] for k in range(7)},
        "top6_ge3_pct": round(ge3_6/n_test*100, 2),
        "top6_ge4_pct": round(ge4_6/n_test*100, 2),
    }
    
    with open("nexus_backtest_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n=> Results saved to nexus_backtest_results.json")
    print("=" * 70)

if __name__ == "__main__":
    run_backtest()
