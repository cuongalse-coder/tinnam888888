"""
TINNAM AI V300 — ULTRA-FAST BACKTEST (ALL DRAWS)
=================================================
25 signals with FIXED weights (no slow calibration per draw).
Optimized for speed: ~1 second per draw.
"""
import sys, os, json, time, math, warnings
warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime
import requests, re

MAX = 45
PICK = 6

def fetch_data():
    today = datetime.now().strftime('%d-%m-%Y')
    url = f"https://www.ketquadientoan.com/tat-ca-ky-xo-so-mega-6-45.html?datef=18-07-2016&datet={today}"
    try:
        import cloudscraper
        s = cloudscraper.create_scraper(delay=5, browser={'browser':'chrome','platform':'windows','mobile':False})
    except:
        s = requests.Session()
    resp = s.get(url, timeout=30)
    html = resp.text
    history = []
    rows = re.findall(r'<tr.*?>(.*?)</tr>', html, re.DOTALL|re.IGNORECASE)
    for row in rows:
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

# ===== ALL SIGNALS (FAST — no heavy computation) =====

def sig_frequency(data):
    """Basic frequency Z-score."""
    all_n = [n for d in data for n in d]
    freq = Counter(all_n)
    ep = PICK/MAX; ec = len(data)*ep; std = math.sqrt(len(data)*ep*(1-ep)) if len(data)>0 else 1
    return {n: ((freq.get(n,0)-ec)/std*2) if std>0 else 0 for n in range(1,MAX+1)}

def sig_transition(data):
    """Markov transition probability from last draw."""
    follow = defaultdict(Counter); pc = Counter()
    for i in range(len(data)-1):
        for p in data[i]: pc[p]+=1
        for p in data[i]:
            for nx in data[i+1]: follow[p][nx]+=1
    last = set(data[-1]); base = PICK/MAX; scores = {}
    for num in range(1,MAX+1):
        tf = sum(follow[p].get(num,0) for p in last)
        tp = max(sum(pc[p] for p in last), 1)
        scores[num] = (tf/tp/base-1)*3
    return scores

def sig_gap(data):
    """Overdue gap analysis."""
    n = len(data); scores = {}
    for num in range(1,MAX+1):
        apps = [i for i,d in enumerate(data) if num in d]
        if len(apps)<3: scores[num]=0; continue
        gaps = [apps[j+1]-apps[j] for j in range(len(apps)-1)]
        mg = np.mean(gaps); sg = np.std(gaps)
        cur = n-apps[-1]
        z = (cur-mg)/sg if sg>0 else 0
        pa = sum(1 for g in gaps if g<=cur)/len(gaps)
        scores[num] = z*1.5+pa*2 if z>0.5 else (-1 if z<-1 else 0)
    return scores

def sig_momentum(data):
    """Multi-window momentum."""
    scores = {}; n = len(data)
    for num in range(1,MAX+1):
        f5 = sum(1 for d in data[-5:] if num in d)/5
        f10 = sum(1 for d in data[-10:] if num in d)/10
        f20 = sum(1 for d in data[-20:] if num in d)/20
        f50 = sum(1 for d in data[-50:] if num in d)/50 if n>=50 else f20
        scores[num] = (f5-f10)*15+(f10-f20)*8+(f20-f50)*4
    return scores

def sig_streak(data):
    """Cold streak sigmoid."""
    scores = {}; eg = MAX/PICK
    for num in range(1,MAX+1):
        cold = 0
        for d in reversed(data):
            if num not in d: cold+=1
            else: break
        scores[num] = 1/(1+math.exp(-3*(cold/eg-0.8)))*2 if cold>0 else 0
    return scores

def sig_cooccurrence(data):
    """Pair co-occurrence with last draw."""
    last = set(data[-1]); pf = Counter()
    for draw in data[-200:]:
        for pair in combinations(sorted(draw),2): pf[pair]+=1
    return {num: sum(pf.get(tuple(sorted([p,num])),0) for p in last)*0.1 for num in range(1,MAX+1)}

def sig_knn(data):
    """K-nearest-neighbor pattern matching."""
    last = set(data[-1]); ks = Counter()
    for i in range(len(data)-2):
        sim = len(set(data[i])&last)
        if sim>=3:
            for num in data[i+1]: ks[num]+=sim**2
    mx = max(ks.values()) if ks else 1
    return {num: ks.get(num,0)/mx*3 for num in range(1,MAX+1)}

def sig_fft(data):
    """FFT cycle detection (lightweight)."""
    scores = {}; w = min(150,len(data))
    for num in range(1,MAX+1):
        seq = np.array([1.0 if num in x else 0.0 for x in data[-w:]])
        if len(seq)<30: scores[num]=0; continue
        sc = seq-np.mean(seq); ft = np.fft.rfft(sc); pw = np.abs(ft)**2
        if len(pw)<3: scores[num]=0; continue
        pi = np.argmax(pw[2:])+2; fr = np.fft.rfftfreq(len(sc))
        pf_ = fr[pi] if pi<len(fr) else 0; pp = pw[pi] if pi<len(pw) else 0
        sr = pp/(np.sum(pw[1:])+1e-10)
        if sr>0.15 and pf_>0:
            phase = math.cos(2*math.pi*((len(seq)%(1/pf_))/(1/pf_)))
            scores[num] = sr*max(0,phase)*3
        else: scores[num]=0
    return scores

def sig_regime(data):
    """Regime change detection."""
    if len(data)<100: return {n:0 for n in range(1,MAX+1)}
    fr = Counter(n for d in data[-30:] for n in d)
    fp = Counter(n for d in data[-60:-30] for n in d)
    exp = 30*PICK/MAX
    return {n: (fr.get(n,0)/exp-fp.get(n,0)/exp)*2 for n in range(1,MAX+1)}

def sig_lag_repeat(data):
    """Lag-specific repeat probability."""
    n = len(data); lag_stats = defaultdict(lambda:defaultdict(int)); last_seen = {}
    for i,draw in enumerate(data):
        for num in draw:
            if num in last_seen: lag_stats[num][i-last_seen[num]]+=1
            last_seen[num]=i
    scores = {}
    for num in range(1,MAX+1):
        cur = n-last_seen.get(num,0)
        if num not in lag_stats: scores[num]=0; continue
        stats = lag_stats[num]; total = sum(stats.values())
        p = stats.get(cur,0)/total if total>0 else 0
        gaps = []; [gaps.extend([l]*c) for l,c in stats.items()]
        med = np.median(gaps) if gaps else MAX/PICK
        od = cur/med if med>0 else 1
        scores[num] = p*5+max(0,od-1)*2
    return scores

def sig_ngram(data):
    """Bigram transition probability."""
    bg = defaultdict(Counter)
    for i in range(1,len(data)):
        for pn in data[i-1]:
            for cn in data[i]: bg[pn][cn]+=1
    sc = Counter()
    for pn in data[-1]:
        t = sum(bg[pn].values())
        if t>0:
            for nn,cnt in bg[pn].most_common(10): sc[nn]+=cnt/t
    return {n: sc.get(n,0) for n in range(1,MAX+1)}

def sig_entropy(data):
    """Markov entropy-based transition."""
    if len(data)<60: return {n:0 for n in range(1,MAX+1)}
    scores = {}
    for num in range(1,MAX+1):
        seq = [1 if num in x else 0 for x in data[-60:]]
        tr = {0:[0,0],1:[0,0]}
        for i in range(1,len(seq)): tr[seq[i-1]][seq[i]]+=1
        cs = seq[-1]; t = sum(tr[cs])
        pa = tr[cs][1]/t if t>0 else PICK/MAX
        ent = 0
        for st in [0,1]:
            tt = sum(tr[st])
            if tt==0: continue
            for c in tr[st]:
                if c>0: p = c/tt; ent -= p*math.log2(p)*(tt/len(seq))
        scores[num] = pa*max(0,1-ent)
    return scores

def sig_ma_cross(data):
    """Moving average crossover."""
    return {n: (sum(1 for d in data[-10:] if n in d)/10-sum(1 for d in data[-30:] if n in d)/30)*8 for n in range(1,MAX+1)}

def sig_anti_repeat(data):
    """Anti-consecutive-repeat."""
    last = set(data[-1]); prev = set(data[-2]) if len(data)>1 else set()
    freq = Counter(n for d in data for n in d)
    scores = {}
    for n in range(1,MAX+1):
        b = freq.get(n,0)/len(data)
        if n in last and n in prev: scores[n]=b*-2
        elif n in last: scores[n]=b*-0.5
        elif n in prev: scores[n]=b*1.5
        else: scores[n]=b*0.5
    return scores

def sig_oddeven(data):
    lo = sum(1 for x in data[-1] if x%2==1)
    return {n: 0.3 if (lo>3 and n%2==0) or (lo<=3 and n%2==1) else 0 for n in range(1,MAX+1)}

def sig_highlow(data):
    mid = MAX//2; lh = sum(1 for x in data[-1] if x>mid)
    return {n: 0.3 if (lh>3 and n<=mid) or (lh<=3 and n>mid) else 0 for n in range(1,MAX+1)}

def sig_sliding_window(data):
    scores = {n:0 for n in range(1,MAX+1)}
    for w,ww in [(5,5),(10,3),(20,2),(40,1),(80,0.5)]:
        if len(data)<w: continue
        freq = Counter(n for d in data[-w:] for n in d)
        exp = PICK/MAX
        for n in range(1,MAX+1):
            obs = freq.get(n,0)/w
            scores[n] += ((obs-exp)/(exp+0.001))*ww
    return scores

def sig_cond_prob(data):
    scores = {n:0 for n in range(1,MAX+1)}
    if len(data)<30: return scores
    last = data[-1]; cond = defaultdict(Counter); tot = Counter()
    for i in range(len(data)-1):
        for g in data[i]:
            tot[g]+=1
            for nx in data[i+1]: cond[g][nx]+=1
    for n in range(1,MAX+1):
        ps = sum(cond[g].get(n,0)/tot[g] for g in last if tot[g]>0)
        scores[n] = ps*3
    return scores

def sig_gap_accel(data):
    scores = {n:0 for n in range(1,MAX+1)}; nd = len(data)
    for num in range(1,MAX+1):
        apps = [i for i,d in enumerate(data) if num in d]
        if len(apps)<4: continue
        gaps = [apps[j+1]-apps[j] for j in range(len(apps)-1)]
        if len(gaps)<3: continue
        rg = gaps[-5:]
        if len(rg)<2: continue
        diffs = [rg[i]-rg[i-1] for i in range(1,len(rg))]
        aa = sum(diffs)/len(diffs)
        cg = nd-apps[-1]; mg = sum(gaps)/len(gaps); od = cg/(mg+0.1)
        if aa<0 and od>0.8: scores[num]=abs(aa)*od*2
        elif od>1.5: scores[num]=od*1.5
    return scores

def sig_delta_mom(data):
    scores = {n:0 for n in range(1,MAX+1)}
    if len(data)<30: return scores
    for n in range(1,MAX+1):
        f1=sum(1 for d in data[-10:] if n in d)/10
        f2=sum(1 for d in data[-20:-10] if n in d)/10
        f3=sum(1 for d in data[-30:-20] if n in d)/10
        v1,v2=f1-f2,f2-f3; a=v1-v2
        if a>0 and v1>0: scores[n]=a*15+v1*5
        elif a>0: scores[n]=a*8
    return scores

def sig_hot_cold(data):
    scores = {n:0 for n in range(1,MAX+1)}
    if len(data)<50: return scores
    exp = PICK/MAX
    for n in range(1,MAX+1):
        s=sum(1 for d in data[-10:] if n in d)/10
        m=sum(1 for d in data[-30:-10] if n in d)/20
        l=sum(1 for d in data[-80:] if n in d)/80
        if s>exp*1.3 and m<exp*0.7: scores[n]=(s-m)*8
        elif s<exp*0.5 and l>exp*1.2: scores[n]=(l-s)*3
    return scores

def sig_sector_rot(data):
    scores = {n:0 for n in range(1,MAX+1)}
    if len(data)<40: return scores
    ns = (MAX+9)//10; rs=[0]*ns; ps=[0]*ns
    for d in data[-15:]:
        for n in d: rs[(n-1)//10]+=1
    for d in data[-30:-15]:
        for n in d: ps[(n-1)//10]+=1
    for n in range(1,MAX+1):
        sec=(n-1)//10; r=rs[sec]; p=ps[sec]
        if r>p*1.2: scores[n]=(r-p)*0.3
        elif r<p*0.8: scores[n]=-0.5
    return scores

def sig_temporal_decay(data):
    scores = {n:0 for n in range(1,MAX+1)}
    nd = len(data); lam = 0.05
    for i,draw in enumerate(data):
        age = nd-1-i; w = math.exp(-lam*age)
        for n in draw: scores[n]+=w
    mx = max(scores.values()) if scores else 1
    if mx>0:
        for n in scores: scores[n]=(scores[n]/mx)*4
    return scores

def sig_markov_steady(data):
    """Markov steady-state (lightweight)."""
    scores = {n:0 for n in range(1,MAX+1)}
    trans = np.zeros((MAX,MAX))
    for i in range(1,min(200, len(data))):
        for p in data[i-1]:
            for n in data[i]: trans[p-1,n-1]+=1
    rs = trans.sum(axis=1,keepdims=True)
    trans = np.divide(trans,rs,out=np.zeros_like(trans),where=rs!=0)
    curr = np.zeros(MAX)
    for p in data[-1]: curr[p-1]=1.0/PICK
    for _ in range(3): curr = curr.dot(trans)
    for i,v in enumerate(curr): scores[i+1]=v*10
    return scores

ALL_SIGNALS = [
    (sig_frequency, 1.2),
    (sig_transition, 1.5),
    (sig_gap, 1.3),
    (sig_momentum, 1.8),
    (sig_streak, 1.0),
    (sig_cooccurrence, 0.8),
    (sig_knn, 1.4),
    (sig_fft, 0.7),
    (sig_regime, 0.9),
    (sig_lag_repeat, 1.1),
    (sig_ngram, 1.3),
    (sig_entropy, 0.6),
    (sig_ma_cross, 1.0),
    (sig_anti_repeat, 1.2),
    (sig_oddeven, 0.4),
    (sig_highlow, 0.4),
    (sig_sliding_window, 1.5),
    (sig_cond_prob, 1.6),
    (sig_gap_accel, 1.2),
    (sig_delta_mom, 1.4),
    (sig_hot_cold, 1.3),
    (sig_sector_rot, 0.7),
    (sig_temporal_decay, 1.0),
    (sig_markov_steady, 0.9),
]

def predict_pool(data):
    """Score all 45 numbers using 24 signals with fixed weights."""
    scores = {n:0.0 for n in range(1,MAX+1)}
    for fn, weight in ALL_SIGNALS:
        try:
            sig = fn(data)
        except:
            continue
        vals = list(sig.values())
        mx = max(abs(v) for v in vals) if vals else 1
        if mx<0.001: continue
        for n, sc in sig.items():
            scores[n] += (sc/mx)*weight
    return [n for n,_ in sorted(scores.items(), key=lambda x:-x[1])]

def main():
    print("="*70, flush=True)
    print("  TINNAM AI V300 — ULTRA-FAST BACKTEST (24 Signals, Fixed Weights)", flush=True)
    print("="*70, flush=True)
    
    print("\n[1] Fetching data...", flush=True)
    data = fetch_data()
    print(f"  => {len(data)} draws", flush=True)
    
    start = 80; total = len(data); ntest = total-start
    
    # Track hit counts for TOP-6, TOP-10, TOP-15
    c6 = {k:0 for k in range(7)}
    c10 = {k:0 for k in range(7)}
    c15 = {k:0 for k in range(7)}
    
    print(f"\n[2] Backtesting {ntest} draws (kỳ {start+1} → {total})...", flush=True)
    t0 = time.time()
    
    for idx in range(start, total):
        hist = data[:idx]
        actual = set(data[idx])
        pool = predict_pool(hist)
        
        h6 = len(set(pool[:6]) & actual)
        h10 = len(set(pool[:10]) & actual)
        h15 = len(set(pool[:15]) & actual)
        c6[h6]+=1; c10[h10]+=1; c15[h15]+=1
        
        done = idx-start+1
        if done % 100 == 0 or done == ntest:
            el = time.time()-t0
            eta = (el/done)*(ntest-done) if done<ntest else 0
            # Running stats for top-10 >= 3
            run_ge3 = sum(c10[k] for k in range(3,7))
            print(f"  {done}/{ntest} ({done/ntest*100:.1f}%) | {el:.0f}s | ETA: {eta:.0f}s | Top10≥3: {run_ge3/done*100:.1f}%", flush=True)
    
    el = time.time()-t0
    pct = lambda c,t: f"{c/t*100:.1f}%" if t>0 else "0%"
    
    print(f"\n{'='*70}", flush=True)
    print(f"  RESULTS — {ntest} draws in {el:.1f}s", flush=True)
    print(f"{'='*70}", flush=True)
    
    print(f"\n--- TOP-6 (Mua 1 vé 6 số) ---", flush=True)
    for k in range(6,-1,-1):
        tag = {6:'🏆 JACKPOT!',5:'GIẢI 1',4:'GIẢI 2',3:'GIẢI 3'}.get(k,'')
        cum = sum(c6[j] for j in range(k,7))
        print(f"  ≥{k}/6: {cum:>5} ({pct(cum,ntest):>6})  {tag}", flush=True)
    
    print(f"\n--- TOP-10 (Bao 10 số) ---", flush=True)
    for k in range(6,-1,-1):
        ab = sum(c10[j] for j in range(k,7))
        tag = {6:'🏆 JACKPOT!',5:'GIẢI 1',4:'GIẢI 2',3:'GIẢI 3'}.get(k,'')
        print(f"  ≥{k}/6: {ab:>5} ({pct(ab,ntest):>6})  {tag}", flush=True)
    
    print(f"\n--- TOP-15 (Bao 15 số) ---", flush=True)
    for k in range(6,-1,-1):
        ab = sum(c15[j] for j in range(k,7))
        tag = {6:'🏆 JACKPOT!',5:'GIẢI 1',4:'GIẢI 2',3:'GIẢI 3'}.get(k,'')
        print(f"  ≥{k}/6: {ab:>5} ({pct(ab,ntest):>6})  {tag}", flush=True)
    
    # === RANDOM BASELINE ===
    print(f"\n--- RANDOM BASELINE (So sánh) ---", flush=True)
    from scipy.special import comb as _comb
    for top_k in [6, 10, 15]:
        print(f"  Random Top-{top_k}:", flush=True)
        for k in range(6, -1, -1):
            # Hypergeometric probability
            try:
                p = _comb(6,k)*_comb(MAX-6,top_k-k)/_comb(MAX,top_k)
                cum_p = sum(_comb(6,j)*_comb(MAX-6,top_k-j)/_comb(MAX,top_k) for j in range(k,7))
            except:
                p = 0; cum_p = 0
            print(f"    ≥{k}/6: {cum_p*100:.2f}%", flush=True)
    
    res = {"v":"V300_24sig_fixed","n":ntest,
           "c6":{str(k):v for k,v in c6.items()},
           "c10":{str(k):v for k,v in c10.items()},
           "c15":{str(k):v for k,v in c15.items()},
           "time_sec": round(el,1)}
    with open("nexus_backtest_v3.json","w") as f:
        json.dump(res, f, indent=2)
    print(f"\n=> Saved to nexus_backtest_v3.json", flush=True)

if __name__=="__main__": main()
