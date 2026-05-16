"""
TINNAM AI V200 — OPTIMIZED PYTHON BACKTEST (1508 kỳ)
Uses 30+ fast signals WITHOUT MLP (too slow for backtest loop).
Includes walk-forward calibration.
"""
import sys, os, json, time, math, warnings
warnings.filterwarnings('ignore')
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

# ===== FAST SIGNAL IMPLEMENTATIONS =====

def sig_physical_weight(data):
    all_n = [n for d in data for n in d]
    freq = Counter(all_n)
    ep = PICK/MAX; ec = len(data)*ep; std = math.sqrt(len(data)*ep*(1-ep))
    return {n: ((freq.get(n,0)-ec)/std*2) if std>0 else 0 for n in range(1,MAX+1)}

def sig_transition(data):
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

def sig_gap_timing(data):
    n = len(data); scores = {}
    for num in range(1,MAX+1):
        apps = [i for i,d in enumerate(data) if num in d]
        if len(apps)<5: scores[num]=0; continue
        gaps = [apps[j+1]-apps[j] for j in range(len(apps)-1)]
        mg = np.mean(gaps); sg = np.std(gaps)
        cur = n-apps[-1]
        z = (cur-mg)/sg if sg>0 else 0
        pa = sum(1 for g in gaps if g<=cur)/len(gaps)
        scores[num] = z*1.5+pa*2 if z>0.5 else (-1 if z<-1 else 0)
    return scores

def sig_momentum(data):
    scores = {}; n = len(data)
    for num in range(1,MAX+1):
        f5 = sum(1 for d in data[-5:] if num in d)/5
        f10 = sum(1 for d in data[-10:] if num in d)/10
        f20 = sum(1 for d in data[-20:] if num in d)/20
        f50 = sum(1 for d in data[-50:] if num in d)/50 if n>=50 else f20
        scores[num] = (f5-f10)*15+(f10-f20)*8+(f20-f50)*4
    return scores

def sig_streak(data):
    scores = {}; eg = MAX/PICK
    for num in range(1,MAX+1):
        cold = 0
        for d in reversed(data):
            if num not in d: cold+=1
            else: break
        scores[num] = 1/(1+math.exp(-3*(cold/eg-0.8)))*2 if cold>0 else 0
    return scores

def sig_cooccurrence(data):
    last = set(data[-1]); pf = Counter()
    for draw in data[-200:]:
        for pair in combinations(sorted(draw),2): pf[pair]+=1
    return {num: sum(pf.get(tuple(sorted([p,num])),0) for p in last)*0.1 for num in range(1,MAX+1)}

def sig_knn(data):
    last = set(data[-1]); ks = Counter()
    for i in range(len(data)-2):
        sim = len(set(data[i])&last)
        if sim>=3:
            for num in data[i+1]: ks[num]+=sim**2
    mx = max(ks.values()) if ks else 1
    return {num: ks.get(num,0)/mx*3 for num in range(1,MAX+1)}

def sig_fft_cycle(data):
    scores = {}; w = min(200,len(data))
    for num in range(1,MAX+1):
        seq = np.array([1.0 if num in x else 0.0 for x in data[-w:]])
        if len(seq)<30: scores[num]=0; continue
        sc = seq-np.mean(seq); ft = np.fft.rfft(sc); pw = np.abs(ft)**2
        if len(pw)<3: scores[num]=0; continue
        fr = np.fft.rfftfreq(len(sc)); pi = np.argmax(pw[2:])+2
        pf_ = fr[pi] if pi<len(fr) else 0; pp = pw[pi] if pi<len(pw) else 0
        sr = pp/(np.sum(pw[1:])+1e-10)
        if sr>0.15 and pf_>0:
            phase = math.cos(2*math.pi*((len(seq)%(1/pf_))/(1/pf_)))
            scores[num] = sr*max(0,phase)*3
        else: scores[num]=0
    return scores

def sig_regime(data):
    if len(data)<100: return {n:0 for n in range(1,MAX+1)}
    fr = Counter(n for d in data[-30:] for n in d)
    fp = Counter(n for d in data[-60:-30] for n in d)
    exp = 30*PICK/MAX
    return {n: (fr.get(n,0)/exp-fp.get(n,0)/exp)*2 for n in range(1,MAX+1)}

def sig_lag_repeat(data):
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

def sig_ma_crossover(data):
    return {n: (sum(1 for d in data[-10:] if n in d)/10-sum(1 for d in data[-30:] if n in d)/30)*8 for n in range(1,MAX+1)}

def sig_anti_repeat(data):
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

def sig_pair_boost(data):
    last = set(data[-1]); pf = Counter()
    for x in data[-100:]:
        for p in combinations(sorted(x),2): pf[p]+=1
    return {n: sum(pf.get(tuple(sorted([p,n])),0) for p in last if pf.get(tuple(sorted([p,n])),0)>3)*0.05 for n in range(1,MAX+1)}

def sig_oddeven(data):
    lo = sum(1 for x in data[-1] if x%2==1)
    return {n: 0.3 if (lo>3 and n%2==0) or (lo<=3 and n%2==1) else 0 for n in range(1,MAX+1)}

def sig_highlow(data):
    mid = MAX//2; lh = sum(1 for x in data[-1] if x>mid)
    return {n: 0.3 if (lh>3 and n<=mid) or (lh<=3 and n>mid) else 0 for n in range(1,MAX+1)}

# NEW NEXUS SIGNALS
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

def sig_delta_momentum(data):
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

def sig_sector_rotation(data):
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
    scores = {n:0 for n in range(1,MAX+1)}
    trans = np.zeros((MAX,MAX))
    for i in range(1,len(data)):
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
    sig_physical_weight, sig_transition, sig_gap_timing, sig_momentum,
    sig_streak, sig_cooccurrence, sig_knn, sig_fft_cycle, sig_regime,
    sig_lag_repeat, sig_ngram, sig_entropy, sig_ma_crossover, sig_anti_repeat,
    sig_pair_boost, sig_oddeven, sig_highlow, sig_sliding_window, sig_cond_prob,
    sig_gap_accel, sig_delta_momentum, sig_hot_cold, sig_sector_rotation,
    sig_temporal_decay, sig_markov_steady
]

def calibrate(data):
    ts = min(30, len(data)-70)
    if ts<5: return [1.0]*len(ALL_SIGNALS)
    hits = [0.0]*len(ALL_SIGNALS); tw = 0
    for idx in range(len(data)-ts, len(data)):
        hist = data[:idx]; actual = set(data[idx])
        rec = math.exp((idx-(len(data)-ts))/6); tw+=rec
        for si,fn in enumerate(ALL_SIGNALS):
            sig = fn(hist)
            top = set(n for n,_ in sorted(sig.items(),key=lambda x:-x[1])[:10])
            hits[si] += len(top&actual)*rec
    base = 10*(PICK/MAX); exp = tw*base
    return [max(h/exp,0.1) if exp>0 and h>0 else 0.1 for h in hits]

def predict_pool(data):
    weights = calibrate(data)
    scores = {n:0 for n in range(1,MAX+1)}
    for si,fn in enumerate(ALL_SIGNALS):
        sig = fn(data)
        vals = list(sig.values())
        mx = max(abs(v) for v in vals) if vals else 1
        if mx<0.001: continue
        for n,sc in sig.items():
            scores[n]+=(sc/mx)*weights[si]
    return [n for n,_ in sorted(scores.items(),key=lambda x:-x[1])]

def main():
    print("="*70)
    print("  TINNAM AI V200 — PYTHON FULL BACKTEST (25 Signals + Calibration)")
    print("="*70)
    
    print("\n[1] Fetching data...")
    data = fetch_data()
    print(f"  => {len(data)} draws")
    
    start = 80; total = len(data); ntest = total-start
    c6={k:0 for k in range(7)}
    c10={k:0 for k in range(7)}
    c15={k:0 for k in range(7)}
    
    print(f"\n[2] Backtesting {ntest} draws...")
    t0 = time.time()
    
    for idx in range(start, total):
        hist = data[:idx]; actual = set(data[idx])
        pool = predict_pool(hist)
        
        h6 = len(set(pool[:6])&actual)
        h10 = len(set(pool[:10])&actual)
        h15 = len(set(pool[:15])&actual)
        c6[h6]+=1; c10[h10]+=1; c15[h15]+=1
        
        done = idx-start+1
        if done%50==0:
            el = time.time()-t0
            eta = (el/done)*(ntest-done)
            print(f"  {done}/{ntest} ({done/ntest*100:.1f}%) | {el:.0f}s | ETA: {eta:.0f}s")
    
    el = time.time()-t0
    pct = lambda c,t: f"{c/t*100:.1f}%" if t>0 else "0%"
    
    print(f"\n{'='*70}")
    print(f"  RESULTS — {ntest} draws in {el:.1f}s")
    print(f"{'='*70}")
    
    print(f"\n--- TOP-6 ---")
    for k in range(6,-1,-1):
        tag = {6:'JACKPOT',5:'GIAI 1',4:'GIAI 2',3:'GIAI 3'}.get(k,'')
        print(f"  {k}/6: {c6[k]:>5} ({pct(c6[k],ntest):>6})  {tag}")
    
    ge3 = sum(c6[k] for k in range(3,7))
    ge4 = sum(c6[k] for k in range(4,7))
    print(f"\n  Top-6 >=3: {pct(ge3,ntest)} ({ge3}/{ntest})")
    print(f"  Top-6 >=4: {pct(ge4,ntest)} ({ge4}/{ntest})")
    
    print(f"\n--- TOP-10 ---")
    for k in range(6,-1,-1):
        ab = sum(c10[i] for i in range(k,7))
        print(f"  >={k}: {ab:>5} ({pct(ab,ntest):>6})")
    
    print(f"\n--- TOP-15 ---")
    for k in range(6,-1,-1):
        ab = sum(c15[i] for i in range(k,7))
        print(f"  >={k}: {ab:>5} ({pct(ab,ntest):>6})")
    
    res = {"v":"V200_Python_25sig","n":ntest,"c6":c6,"c10":c10,"c15":c15,
           "ge3":round(ge3/ntest*100,2),"ge4":round(ge4/ntest*100,2)}
    json.dump(res, open("nexus_backtest_python.json","w"), indent=2)
    print(f"\n=> Saved to nexus_backtest_python.json")

if __name__=="__main__": main()
