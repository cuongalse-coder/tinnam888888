"""
TINNAM AI V400 — ADAPTIVE BACKTEST
Key improvements over V300:
1. Walk-forward calibration (learn signal weights from recent history)
2. Pair co-occurrence signals (analyze number pairs, not just singles)
3. Sum/Range structural constraints
4. Ensemble diversification with multiple ranking methods
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

# ===== SIGNAL FUNCTIONS =====
def s_freq(d):
    f=Counter(n for x in d for n in x); ep=PICK/MAX; ec=len(d)*ep; st=math.sqrt(len(d)*ep*(1-ep)) if len(d)>0 else 1
    return {n:((f.get(n,0)-ec)/st*2) if st>0 else 0 for n in range(1,MAX+1)}

def s_trans(d):
    fw=defaultdict(Counter); pc=Counter()
    for i in range(len(d)-1):
        for p in d[i]: pc[p]+=1
        for p in d[i]:
            for nx in d[i+1]: fw[p][nx]+=1
    last=set(d[-1]); base=PICK/MAX
    return {n:(sum(fw[p].get(n,0) for p in last)/max(sum(pc[p] for p in last),1)/base-1)*3 for n in range(1,MAX+1)}

def s_gap(d):
    nd=len(d); sc={}
    for n in range(1,MAX+1):
        ap=[i for i,x in enumerate(d) if n in x]
        if len(ap)<3: sc[n]=0; continue
        gs=[ap[j+1]-ap[j] for j in range(len(ap)-1)]; mg=np.mean(gs); sg=np.std(gs)
        cur=nd-ap[-1]; z=(cur-mg)/sg if sg>0 else 0
        pa=sum(1 for g in gs if g<=cur)/len(gs)
        sc[n]=z*1.5+pa*2 if z>0.5 else (-1 if z<-1 else 0)
    return sc

def s_mom(d):
    n=len(d)
    return {num:(sum(1 for x in d[-5:] if num in x)/5-sum(1 for x in d[-10:] if num in x)/10)*15+(sum(1 for x in d[-10:] if num in x)/10-sum(1 for x in d[-20:] if num in x)/20)*8 for num in range(1,MAX+1)}

def s_streak(d):
    sc={}; eg=MAX/PICK
    for n in range(1,MAX+1):
        c=0
        for x in reversed(d):
            if n not in x: c+=1
            else: break
        sc[n]=1/(1+math.exp(-3*(c/eg-0.8)))*2 if c>0 else 0
    return sc

def s_knn(d):
    last=set(d[-1]); ks=Counter()
    for i in range(len(d)-2):
        sim=len(set(d[i])&last)
        if sim>=3:
            for n in d[i+1]: ks[n]+=sim**2
    mx=max(ks.values()) if ks else 1
    return {n:ks.get(n,0)/mx*3 for n in range(1,MAX+1)}

def s_ngram(d):
    bg=defaultdict(Counter)
    for i in range(1,len(d)):
        for p in d[i-1]:
            for c in d[i]: bg[p][c]+=1
    sc=Counter()
    for p in d[-1]:
        t=sum(bg[p].values())
        if t>0:
            for nn,cnt in bg[p].most_common(10): sc[nn]+=cnt/t
    return {n:sc.get(n,0) for n in range(1,MAX+1)}

def s_regime(d):
    if len(d)<100: return {n:0 for n in range(1,MAX+1)}
    fr=Counter(n for x in d[-30:] for n in x); fp=Counter(n for x in d[-60:-30] for n in x); ep=30*PICK/MAX
    return {n:(fr.get(n,0)/ep-fp.get(n,0)/ep)*2 for n in range(1,MAX+1)}

def s_sliding(d):
    sc={n:0 for n in range(1,MAX+1)}
    for w,ww in [(5,5),(10,3),(20,2),(40,1)]:
        if len(d)<w: continue
        f=Counter(n for x in d[-w:] for n in x); ep=PICK/MAX
        for n in range(1,MAX+1): sc[n]+=((f.get(n,0)/w-ep)/(ep+.001))*ww
    return sc

def s_cond(d):
    sc={n:0 for n in range(1,MAX+1)}
    if len(d)<30: return sc
    last=d[-1]; cd=defaultdict(Counter); tot=Counter()
    for i in range(len(d)-1):
        for g in d[i]: tot[g]+=1
        for g in d[i]:
            for nx in d[i+1]: cd[g][nx]+=1
    for n in range(1,MAX+1): sc[n]=sum(cd[g].get(n,0)/tot[g] for g in last if tot[g]>0)*3
    return sc

def s_delta(d):
    sc={n:0 for n in range(1,MAX+1)}
    if len(d)<30: return sc
    for n in range(1,MAX+1):
        f1=sum(1 for x in d[-10:] if n in x)/10; f2=sum(1 for x in d[-20:-10] if n in x)/10; f3=sum(1 for x in d[-30:-20] if n in x)/10
        v1,v2=f1-f2,f2-f3; a=v1-v2
        if a>0 and v1>0: sc[n]=a*15+v1*5
        elif a>0: sc[n]=a*8
    return sc

def s_hotcold(d):
    sc={n:0 for n in range(1,MAX+1)}
    if len(d)<50: return sc
    ep=PICK/MAX
    for n in range(1,MAX+1):
        s=sum(1 for x in d[-10:] if n in x)/10; m=sum(1 for x in d[-30:-10] if n in x)/20
        l=sum(1 for x in d[-80:] if n in x)/80
        if s>ep*1.3 and m<ep*0.7: sc[n]=(s-m)*8
        elif s<ep*0.5 and l>ep*1.2: sc[n]=(l-s)*3
    return sc

def s_decay(d):
    sc={n:0 for n in range(1,MAX+1)}; nd=len(d); lam=0.05
    for i,draw in enumerate(d):
        w=math.exp(-lam*(nd-1-i))
        for n in draw: sc[n]+=w
    mx=max(sc.values()) if sc else 1
    if mx>0:
        for n in sc: sc[n]=(sc[n]/mx)*4
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
    """Pair co-occurrence boost — new V400 signal."""
    sc={n:0 for n in range(1,MAX+1)}
    last=set(d[-1]); pf=Counter()
    for x in d[-150:]:
        for p in combinations(sorted(x),2): pf[p]+=1
    # For each candidate, sum pair freq with last draw numbers
    for n in range(1,MAX+1):
        for p in last:
            key=tuple(sorted([p,n]))
            cnt=pf.get(key,0)
            if cnt>3: sc[n]+=cnt*0.08
    return sc

def s_sum_zone(d):
    """Numbers in historically likely sum zones — new V400 signal."""
    sc={n:0 for n in range(1,MAX+1)}
    if len(d)<50: return sc
    sums=[sum(x) for x in d[-100:]]
    avg_s=np.mean(sums); std_s=np.std(sums)
    # Favor numbers that keep total sum near historical average
    last_sum=sum(d[-1])
    target=avg_s  # Next draw sum likely near mean
    for n in range(1,MAX+1):
        # How much does adding this number contribute to reaching target?
        contrib=n/target if target>0 else 0
        if 0.08<contrib<0.25: sc[n]=0.5  # Reasonable contribution
        if 0.12<contrib<0.18: sc[n]=1.0  # Sweet spot
    return sc

ALL_SIGS = [s_freq,s_trans,s_gap,s_mom,s_streak,s_knn,s_ngram,s_regime,
            s_sliding,s_cond,s_delta,s_hotcold,s_decay,s_antirepeat,s_pair,s_sum_zone]

def calibrate_weights(data, calib_size=25):
    """Walk-forward: learn which signals performed best on recent draws."""
    n=len(data)
    if n<calib_size+50: return [1.0]*len(ALL_SIGS)
    hits=[0.0]*len(ALL_SIGS); tw=0
    for idx in range(n-calib_size, n):
        hist=data[:idx]; actual=set(data[idx])
        rec=math.exp((idx-(n-calib_size))/5); tw+=rec
        for si,fn in enumerate(ALL_SIGS):
            try:
                sig=fn(hist)
                top=set(k for k,_ in sorted(sig.items(),key=lambda x:-x[1])[:10])
                hits[si]+=len(top&actual)*rec
            except: pass
    base=10*(PICK/MAX); exp=tw*base
    return [max(h/exp,0.1) if exp>0 else 1.0 for h in hits]

def predict_pool(data, weights=None):
    if weights is None: weights=[1.0]*len(ALL_SIGS)
    scores={n:0.0 for n in range(1,MAX+1)}
    for si,(fn,w) in enumerate(zip(ALL_SIGS, weights)):
        try: sig=fn(data)
        except: continue
        vals=list(sig.values()); mx=max(abs(v) for v in vals) if vals else 1
        if mx<0.001: continue
        for n,sc in sig.items(): scores[n]+=(sc/mx)*w
    return [n for n,_ in sorted(scores.items(),key=lambda x:-x[1])]

def main():
    print("="*70, flush=True)
    print("  TINNAM AI V400 — ADAPTIVE BACKTEST (16 Signals + Walk-Forward)", flush=True)
    print("="*70, flush=True)
    
    print("\n[1] Fetching data...", flush=True)
    data=fetch_data()
    print(f"  => {len(data)} draws", flush=True)
    
    start=80; total=len(data); ntest=total-start
    c6={k:0 for k in range(7)}; c10={k:0 for k in range(7)}; c15={k:0 for k in range(7)}
    
    print(f"\n[2] Backtesting {ntest} draws...", flush=True)
    t0=time.time()
    
    # Calibrate every 50 draws to save time
    cached_weights=None; last_calib=-999
    
    for idx in range(start, total):
        # Re-calibrate every 50 draws
        if idx-last_calib>=50:
            hist_for_calib=data[:idx]
            if len(hist_for_calib)>100:
                cached_weights=calibrate_weights(hist_for_calib, calib_size=20)
            else:
                cached_weights=[1.0]*len(ALL_SIGS)
            last_calib=idx
        
        hist=data[:idx]; actual=set(data[idx])
        pool=predict_pool(hist, cached_weights)
        
        h6=len(set(pool[:6])&actual); h10=len(set(pool[:10])&actual); h15=len(set(pool[:15])&actual)
        c6[h6]+=1; c10[h10]+=1; c15[h15]+=1
        
        done=idx-start+1
        if done%100==0 or done==ntest:
            el=time.time()-t0; eta=(el/done)*(ntest-done) if done<ntest else 0
            g3=sum(c10[k] for k in range(3,7))
            print(f"  {done}/{ntest} ({done/ntest*100:.1f}%) | {el:.0f}s | ETA: {eta:.0f}s | Top10≥3: {g3/done*100:.1f}%", flush=True)
    
    el=time.time()-t0
    pct=lambda c,t: f"{c/t*100:.1f}%" if t>0 else "0%"
    
    print(f"\n{'='*70}", flush=True)
    print(f"  RESULTS — {ntest} draws in {el:.1f}s", flush=True)
    print(f"{'='*70}", flush=True)
    
    for label,cc,top_k in [("TOP-6",c6,6),("TOP-10",c10,10),("TOP-15",c15,15)]:
        print(f"\n--- {label} ---", flush=True)
        for k in range(6,-1,-1):
            ab=sum(cc[j] for j in range(k,7))
            tag={6:'🏆 JACKPOT!',5:'GIẢI 1',4:'GIẢI 2',3:'GIẢI 3'}.get(k,'')
            print(f"  ≥{k}/6: {ab:>5} ({pct(ab,ntest):>6})  {tag}", flush=True)
    
    # Random baseline
    print(f"\n--- RANDOM BASELINE ---", flush=True)
    from scipy.special import comb as _c
    for tk in [6,10,15]:
        print(f"  Random Top-{tk}:", flush=True)
        for k in range(6,-1,-1):
            cp=sum(_c(6,j)*_c(MAX-6,tk-j)/_c(MAX,tk) for j in range(k,7))
            print(f"    ≥{k}/6: {cp*100:.2f}%", flush=True)
    
    res={"v":"V400_adaptive","n":ntest,"c6":{str(k):v for k,v in c6.items()},"c10":{str(k):v for k,v in c10.items()},"c15":{str(k):v for k,v in c15.items()},"time":round(el,1)}
    with open("nexus_backtest_v4.json","w") as f: json.dump(res,f,indent=2)
    print(f"\n=> Saved to nexus_backtest_v4.json", flush=True)

if __name__=="__main__": main()
