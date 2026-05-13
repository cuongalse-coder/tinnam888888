"""
TINNAM AI V500 — MACHINE LEARNING ENSEMBLE BACKTEST
Uses Gradient Boosting / Random Forest to learn non-linear relationships 
between the 25 signals to maximize prediction accuracy.
"""
import sys, os, json, time, math, warnings
warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime
import requests, re

# ML Tools
try:
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
except ImportError:
    pass

MAX = 45; PICK = 6

def fetch_data():
    today = datetime.now().strftime('%d-%m-%Y')
    url = f"https://www.ketquadientoan.com/tat-ca-ky-xo-so-mega-6-45.html?datef=18-07-2016&datet={today}"
    try:
        import cloudscraper
        s = cloudscraper.create_scraper(delay=5, browser={'browser':'chrome','platform':'windows','mobile':False})
    except: s = requests.Session()
    resp = s.get(url, timeout=30); html = resp.text; history = []
    for row in re.findall(r'<tr.*?>(.*?)</tr>', html, re.DOTALL|re.IGNORECASE):
        nums = re.findall(r'class="home-mini-whiteball">\s*(\d{2})\s*<', row)
        if len(nums)<6: continue
        chunk = sorted([int(n) for n in nums[:6]])
        if len(set(chunk))==6 and all(1<=n<=MAX for n in chunk) and chunk not in history:
            history.append(chunk)
    if history: history.reverse()
    if not history:
        gh = requests.get("https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl", timeout=10)
        for line in gh.text.strip().split('\n'):
            obj = json.loads(line)
            if 'result' in obj and len(obj['result'])>=6:
                draw = sorted([int(n) for n in obj['result'][:6]])
                if len(set(draw))==6: history.append(draw)
    return history

# ===== SIGNAL FUNCTIONS (Lightweight) =====
def s_freq(d):
    f=Counter(n for x in d for n in x); ep=PICK/MAX; ec=len(d)*ep; st=math.sqrt(len(d)*ep*(1-ep)) if len(d)>0 else 1
    return {n:((f.get(n,0)-ec)/st) for n in range(1,MAX+1)}

def s_trans(d):
    fw=defaultdict(Counter); pc=Counter()
    for i in range(len(d)-1):
        for p in d[i]: pc[p]+=1
        for p in d[i]:
            for nx in d[i+1]: fw[p][nx]+=1
    last=set(d[-1]); base=PICK/MAX
    return {n:(sum(fw[p].get(n,0) for p in last)/max(sum(pc[p] for p in last),1)/base-1) for n in range(1,MAX+1)}

def s_gap(d):
    nd=len(d); sc={}
    for n in range(1,MAX+1):
        ap=[i for i,x in enumerate(d) if n in x]
        if len(ap)<3: sc[n]=0; continue
        gs=[ap[j+1]-ap[j] for j in range(len(ap)-1)]; mg=np.mean(gs); sg=np.std(gs)
        cur=nd-ap[-1]; z=(cur-mg)/sg if sg>0 else 0
        pa=sum(1 for g in gs if g<=cur)/len(gs)
        sc[n]=z+pa if z>0 else (-1 if z<-1 else 0)
    return sc

def s_mom(d):
    n=len(d)
    return {num:(sum(1 for x in d[-5:] if num in x)/5-sum(1 for x in d[-10:] if num in x)/10) for num in range(1,MAX+1)}

def s_streak(d):
    sc={}; eg=MAX/PICK
    for n in range(1,MAX+1):
        c=0
        for x in reversed(d):
            if n not in x: c+=1
            else: break
        sc[n]=1/(1+math.exp(-3*(c/eg-0.8))) if c>0 else 0
    return sc

def s_knn(d):
    last=set(d[-1]); ks=Counter()
    for i in range(len(d)-2):
        sim=len(set(d[i])&last)
        if sim>=3:
            for n in d[i+1]: ks[n]+=sim**2
    mx=max(ks.values()) if ks else 1
    return {n:ks.get(n,0)/mx for n in range(1,MAX+1)}

def s_regime(d):
    if len(d)<60: return {n:0 for n in range(1,MAX+1)}
    fr=Counter(n for x in d[-30:] for n in x); fp=Counter(n for x in d[-60:-30] for n in x); ep=30*PICK/MAX
    return {n:(fr.get(n,0)/ep-fp.get(n,0)/ep) for n in range(1,MAX+1)}

def s_sliding(d):
    sc={n:0 for n in range(1,MAX+1)}
    for w in [5,10,20]:
        if len(d)<w: continue
        f=Counter(n for x in d[-w:] for n in x); ep=PICK/MAX
        for n in range(1,MAX+1): sc[n]+=((f.get(n,0)/w-ep)/(ep+.001))
    return sc

def s_delta(d):
    sc={n:0 for n in range(1,MAX+1)}
    if len(d)<30: return sc
    for n in range(1,MAX+1):
        f1=sum(1 for x in d[-10:] if n in x)/10; f2=sum(1 for x in d[-20:-10] if n in x)/10; f3=sum(1 for x in d[-30:-20] if n in x)/10
        v1,v2=f1-f2,f2-f3; a=v1-v2
        if a>0 and v1>0: sc[n]=a*2+v1
        elif a>0: sc[n]=a
    return sc

def s_hotcold(d):
    sc={n:0 for n in range(1,MAX+1)}
    if len(d)<50: return sc
    ep=PICK/MAX
    for n in range(1,MAX+1):
        s=sum(1 for x in d[-10:] if n in x)/10; m=sum(1 for x in d[-30:-10] if n in x)/20
        if s>ep*1.2 and m<ep*0.8: sc[n]=(s-m)
    return sc

def s_decay(d):
    sc={n:0 for n in range(1,MAX+1)}; nd=len(d); lam=0.05
    for i,draw in enumerate(d):
        w=math.exp(-lam*(nd-1-i))
        for n in draw: sc[n]+=w
    mx=max(sc.values()) if sc else 1
    if mx>0:
        for n in sc: sc[n]=(sc[n]/mx)
    return sc

def s_antirepeat(d):
    last=set(d[-1]); prev=set(d[-2]) if len(d)>1 else set()
    f=Counter(n for x in d for n in x); sc={}
    for n in range(1,MAX+1):
        b=f.get(n,0)/len(d)
        if n in last and n in prev: sc[n]=b*-2
        elif n in last: sc[n]=b*-0.5
        elif n in prev: sc[n]=b*1.5
        else: sc[n]=b*0.5
    return sc

def s_pair(d):
    sc={n:0 for n in range(1,MAX+1)}; last=set(d[-1]); pf=Counter()
    for x in d[-100:]:
        for p in combinations(sorted(x),2): pf[p]+=1
    for n in range(1,MAX+1):
        for p in last:
            key=tuple(sorted([p,n]))
            cnt=pf.get(key,0)
            if cnt>2: sc[n]+=cnt*0.05
    return sc

ALL_SIGS = [s_freq,s_trans,s_gap,s_mom,s_streak,s_knn,s_regime,
            s_sliding,s_delta,s_hotcold,s_decay,s_antirepeat,s_pair]

# --- ML MODELING ---
def extract_features(data, idx_target):
    """Generate X (features) and y (target) up to idx_target."""
    X = []; y = []
    
    # We need enough history to calculate features. Start from idx_target - window
    # Window size: learn from the past `train_window` draws
    train_window = 100
    start_idx = max(50, idx_target - train_window)
    
    for i in range(start_idx, idx_target):
        hist = data[:i]
        actual = set(data[i])
        
        # Calculate all 13 signals for this history
        sig_results = [fn(hist) for fn in ALL_SIGS]
        
        # For each number 1 to 45, create a feature row
        for n in range(1, MAX+1):
            row = [sig_res.get(n, 0.0) for sig_res in sig_results]
            X.append(row)
            y.append(1.0 if n in actual else 0.0)
            
    return np.array(X), np.array(y)

def predict_ml(data, model):
    """Predict next draw using the trained model."""
    sig_results = [fn(data) for fn in ALL_SIGS]
    X_pred = []
    for n in range(1, MAX+1):
        row = [sig_res.get(n, 0.0) for sig_res in sig_results]
        X_pred.append(row)
    
    preds = model.predict(X_pred)
    # Rank numbers based on prediction score
    ranked = sorted([(n+1, preds[n]) for n in range(MAX)], key=lambda x: -x[1])
    return [n for n, _ in ranked]

def main():
    print("="*70, flush=True)
    print("  TINNAM AI V500 — DEEP MACHINE LEARNING BACKTEST", flush=True)
    print("  Using HistGradientBoostingRegressor to learn non-linear patterns", flush=True)
    print("="*70, flush=True)
    
    data = fetch_data()
    print(f"  => {len(data)} draws fetched", flush=True)
    
    start = 150; total = len(data); ntest = total - start
    c6={k:0 for k in range(7)}; c10={k:0 for k in range(7)}; c15={k:0 for k in range(7)}
    
    print(f"\n[2] Backtesting {ntest} draws...", flush=True)
    t0 = time.time()
    
    model = HistGradientBoostingRegressor(max_iter=50, max_depth=5, learning_rate=0.05, random_state=42)
    
    last_train = -999
    
    for idx in range(start, total):
        # Retrain every 20 draws to save time but keep it adaptive
        if idx - last_train >= 20:
            X_train, y_train = extract_features(data, idx)
            model.fit(X_train, y_train)
            last_train = idx
            
        hist = data[:idx]
        actual = set(data[idx])
        pool = predict_ml(hist, model)
        
        h6 = len(set(pool[:6]) & actual)
        h10 = len(set(pool[:10]) & actual)
        h15 = len(set(pool[:15]) & actual)
        c6[h6]+=1; c10[h10]+=1; c15[h15]+=1
        
        done = idx - start + 1
        if done % 50 == 0 or done == ntest:
            el = time.time()-t0; eta = (el/done)*(ntest-done)
            g3 = sum(c10[k] for k in range(3,7))
            print(f"  {done}/{ntest} ({done/ntest*100:.1f}%) | {el:.0f}s | ETA: {eta:.0f}s | Top10≥3: {g3/done*100:.1f}%", flush=True)
    
    el = time.time()-t0
    pct = lambda c,t: f"{c/t*100:.1f}%" if t>0 else "0%"
    
    print(f"\n{'='*70}", flush=True)
    print(f"  RESULTS — {ntest} draws in {el:.1f}s", flush=True)
    print(f"{'='*70}", flush=True)
    
    for label,cc,top_k in [("TOP-6",c6,6),("TOP-10",c10,10),("TOP-15",c15,15)]:
        print(f"\n--- {label} ---", flush=True)
        for k in range(6,-1,-1):
            ab = sum(cc[j] for j in range(k,7))
            print(f"  ≥{k}/6: {ab:>5} ({pct(ab,ntest):>6})")

if __name__=="__main__": main()
