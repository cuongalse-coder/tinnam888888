"""
Ultimate Engine V9 — Block Puzzle Engine.

KEY FINDINGS from block-puzzle analysis:
1. Position 1 stays within ±3: 43% (3x random)
2. Position 6 stays within ±3: 44%
3. Bridge numbers (repeats) tend to stay at same position (10% pos6→pos6)
4. Numbers have ROLES: 1-11=HEAD, 12-32=BRIDGE, 33-45=TAIL
5. Block shapes are predictable (19% conditional P for top shapes)
6. Internal gaps: median=5 between consecutive positions

V9 STRATEGY:
  - Use POSITIONAL CONSTRAINTS: filter combos where each position
    is within historical range of that position
  - BRIDGE-AWARE generation: include 0-2 bridging numbers from last draw,
    preferring numbers that stay at their SAME position
  - SHAPE-CONSTRAINED: only generate combos matching top-30 block shapes
  - Combine with multi-pool exhaustive from V8
"""
import sys, os, time, math
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations

sys.stdout.reconfigure(encoding='utf-8')
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class UltimateEngine:
    """V9: Block Puzzle — positional + bridge + shape constraints."""

    def __init__(self, max_number, pick_count):
        self.max_number = max_number
        self.pick_count = pick_count

    def predict(self, data, dates=None, n_portfolio=500):
        n = len(data)
        last = sorted(data[-1][:6])
        last_set = set(last)

        # PHASE 1: Learn positional ranges from history
        sorted_data = [sorted(d[:6]) for d in data]
        pos_ranges = self._learn_positional_ranges(sorted_data)
        
        # PHASE 2: Learn bridge patterns
        bridge_probs = self._learn_bridge_probs(sorted_data)
        
        # PHASE 3: Learn allowed shapes
        allowed_shapes = self._learn_allowed_shapes(sorted_data)
        
        # PHASE 4: Score (20 signals + ensemble)
        signals = self._compute_all_signals(data, dates)
        weights = self._walk_forward_weights(data, signals)
        vote_counts = Counter()
        for sig_name, sig_scores in signals.items():
            if not sig_scores: continue
            w = weights.get(sig_name, 1.0)
            for rank, (num, _) in enumerate(sorted(sig_scores.items(), key=lambda x: -x[1])[:12]):
                vote_counts[num] += w * (12 - rank) / 12
        all_scores = {num: max(vote_counts.get(num, 0), 0.001) for num in range(1, self.max_number + 1)}
        
        # Standard constraints
        constraints = self._tight_constraints(data)
        sum_mod7 = self._get_sum_mod7_targets(data)

        # PHASE 5: Build diverse pools
        ranked = sorted(all_scores.items(), key=lambda x: -x[1])
        ensemble_pool = [num for num, _ in ranked]
        
        # Overdue pool
        last_seen = {}
        for i, d in enumerate(data):
            for num in d: last_seen[num] = i
        overdue_pool = sorted(range(1, self.max_number + 1), key=lambda x: -(n - last_seen.get(x, 0)))
        
        # Frequency pool
        freq_50 = Counter(num for d in data[-50:] for num in d)
        freq_pool = [num for num, _ in freq_50.most_common()]
        
        # Transition pool
        follow_scores = Counter()
        for i in range(n - 1):
            for p in data[i]:
                if p in last_set:
                    for nx in data[i + 1]: follow_scores[nx] += 1
        transition_pool = [num for num, _ in follow_scores.most_common()]

        pools = {
            'ensemble': ensemble_pool[:15],
            'overdue': overdue_pool[:15],
            'frequency': freq_pool[:15],
            'transition': transition_pool[:15],
        }

        # PHASE 6: Generate with BLOCK-PUZZLE constraints
        portfolio = []
        used = set()

        for pool_name, pool in pools.items():
            for combo in combinations(pool, self.pick_count):
                combo = sorted(combo)
                t = tuple(combo)
                if t in used: continue
                # Standard validation
                if not self._validate(combo, constraints): continue
                if not self._check_sum_mod7(combo, sum_mod7): continue
                if not self._check_decade_balance(combo): continue
                # BLOCK PUZZLE validation: positional range check
                if not self._check_positional(combo, pos_ranges, last): continue
                # Shape check
                if not self._check_shape(combo, allowed_shapes): continue
                
                portfolio.append({
                    'numbers': combo, 'strategy': f'block_{pool_name}',
                    'score': round(sum(all_scores.get(n, 0) for n in combo), 2),
                })
                used.add(t)

        # Bridge-aware sets (include 1-2 numbers from last draw at same position)
        for n_bridge in [1, 2]:
            for _ in range(100):
                combo = self._bridge_sample(n_bridge, last, ensemble_pool, all_scores,
                                           constraints, used, sum_mod7, pos_ranges, allowed_shapes)
                if combo:
                    portfolio.append({
                        'numbers': combo, 'strategy': f'bridge_{n_bridge}',
                        'score': round(sum(all_scores.get(n, 0) for n in combo), 2),
                    })
                    used.add(tuple(combo))

        # Broad repeat-aware (200 sets — no positional constraint, just standard)
        non_last = [num for num in ensemble_pool if num not in last_set]
        for _ in range(100):
            combo = self._smart_sample(non_last[:25], all_scores, constraints, used, sum_mod7)
            if combo:
                portfolio.append({'numbers': combo, 'strategy': 'broad_zero',
                                'score': round(sum(all_scores.get(n,0) for n in combo), 2)})
                used.add(tuple(combo))
        for _ in range(100):
            combo = self._repeat_sample(1, last_set, ensemble_pool, all_scores, constraints, used, sum_mod7)
            if combo:
                portfolio.append({'numbers': combo, 'strategy': 'broad_one',
                                'score': round(sum(all_scores.get(n,0) for n in combo), 2)})
                used.add(tuple(combo))

        # KNN (20)
        for combo in self._knn_replay(data)[:20]:
            t = tuple(sorted(combo))
            if self._validate(combo, constraints) and t not in used:
                portfolio.append({'numbers': sorted(combo), 'strategy': 'knn',
                                'score': round(sum(all_scores.get(n,0) for n in combo), 2)})
                used.add(t)

        portfolio.sort(key=lambda x: -x['score'])
        all_nums = set()
        for p in portfolio: all_nums.update(p['numbers'])

        return {
            'primary': portfolio[0]['numbers'] if portfolio else sorted(ensemble_pool[:6]),
            'portfolio': portfolio,
            'total_sets': len(portfolio),
            'top_30': ensemble_pool[:30],
            'scores': {num: round(s, 3) for num, s in ranked[:30]},
            'weights': weights,
            'n_signals': len(signals),
            'coverage': len(all_nums),
            'constraints': constraints,
        }

    # ================================================================
    # BLOCK PUZZLE CONSTRAINTS
    # ================================================================
    def _learn_positional_ranges(self, sorted_data):
        """Learn P10-P90 range for each position from history."""
        ranges = {}
        for pos in range(6):
            values = [sd[pos] for sd in sorted_data[-200:]]
            ranges[pos] = {
                'lo': int(np.percentile(values, 5)),
                'hi': int(np.percentile(values, 95)),
                'mean': np.mean(values),
            }
        return ranges

    def _check_positional(self, combo, pos_ranges, last_block):
        """Check if combo's positions fall within learned ranges."""
        for pos in range(6):
            v = combo[pos]
            r = pos_ranges[pos]
            if v < r['lo'] or v > r['hi']:
                return False
            # Position-to-position continuity: within ±15 of last block
            if abs(v - last_block[pos]) > 15:
                return False
        return True

    def _learn_bridge_probs(self, sorted_data):
        """Learn which numbers bridge between consecutive blocks."""
        bridge_counts = Counter()
        for i in range(len(sorted_data) - 1):
            bridges = set(sorted_data[i]) & set(sorted_data[i + 1])
            for b in bridges:
                bridge_counts[b] += 1
        return bridge_counts

    def _learn_allowed_shapes(self, sorted_data):
        """Learn top-50 block signatures (decade patterns)."""
        def get_decade(n):
            if n <= 9: return 0
            elif n <= 19: return 1
            elif n <= 29: return 2
            elif n <= 39: return 3
            else: return 4
        
        sig_count = Counter()
        for sd in sorted_data:
            sig = tuple(get_decade(n) for n in sd)
            sig_count[sig] += 1
        
        # Allow top shapes covering 80%+ of draws
        allowed = set()
        cumulative = 0
        for sig, cnt in sig_count.most_common():
            allowed.add(sig)
            cumulative += cnt
            if cumulative / len(sorted_data) > 0.85:
                break
        return allowed

    def _check_shape(self, combo, allowed_shapes):
        """Check if combo's decade shape is in allowed set."""
        def get_decade(n):
            if n <= 9: return 0
            elif n <= 19: return 1
            elif n <= 29: return 2
            elif n <= 39: return 3
            else: return 4
        sig = tuple(get_decade(n) for n in sorted(combo))
        return sig in allowed_shapes

    def _bridge_sample(self, n_bridge, last, pool, scores, constraints, used, sum_mod7, pos_ranges, allowed_shapes):
        """Generate combo with n_bridge numbers from last draw, preferring same position."""
        for _ in range(80):
            try:
                # Pick bridge positions
                bridge_positions = np.random.choice(6, n_bridge, replace=False)
                bridges = [last[p] for p in bridge_positions]
                # Fill remaining from pool (excluding bridges)
                remaining = [n for n in pool if n not in bridges]
                if len(remaining) < self.pick_count - n_bridge: continue
                w = np.array([max(scores.get(n,0.01),0.01) for n in remaining]); w = w/w.sum()
                fi = np.random.choice(len(remaining), self.pick_count - n_bridge, replace=False, p=w)
                fills = [remaining[i] for i in fi]
                combo = sorted(bridges + fills); t = tuple(combo)
                if t in used: continue
                if not self._validate(combo, constraints): continue
                if not self._check_sum_mod7(combo, sum_mod7): continue
                if not self._check_decade_balance(combo): continue
                if not self._check_positional(combo, pos_ranges, last): continue
                if not self._check_shape(combo, allowed_shapes): continue
                return combo
            except: continue
        return None

    # ================================================================
    # STANDARD HELPERS (same as V8)
    # ================================================================
    def _smart_sample(self, pool, scores, constraints, used, sum_targets):
        if len(pool) < self.pick_count: return None
        w = np.array([max(scores.get(n,0.01),0.01) for n in pool]); w = w/w.sum()
        for _ in range(80):
            try:
                idx = np.random.choice(len(pool), self.pick_count, replace=False, p=w)
                combo = sorted([pool[i] for i in idx]); t = tuple(combo)
                if t in used: continue
                if not self._validate(combo, constraints): continue
                if not self._check_sum_mod7(combo, sum_targets): continue
                if not self._check_decade_balance(combo): continue
                return combo
            except: continue
        return None

    def _repeat_sample(self, n_rep, last, pool, scores, constraints, used, sum_targets):
        last_list = sorted(last)
        non_last = [n for n in pool if n not in last]
        for _ in range(80):
            try:
                ri = np.random.choice(len(last_list), n_rep, replace=False)
                reps = [last_list[i] for i in ri]
                rem = self.pick_count - n_rep
                w = np.array([max(scores.get(n,0.01),0.01) for n in non_last]); w = w/w.sum()
                fi = np.random.choice(len(non_last), rem, replace=False, p=w)
                combo = sorted(reps + [non_last[i] for i in fi]); t = tuple(combo)
                if t in used: continue
                if not self._validate(combo, constraints): continue
                if not self._check_sum_mod7(combo, sum_targets): continue
                if not self._check_decade_balance(combo): continue
                return combo
            except: continue
        return None

    def _get_sum_mod7_targets(self, data):
        transitions = defaultdict(Counter)
        for i in range(1, len(data)):
            transitions[sum(data[i-1])%7][sum(data[i])%7] += 1
        counts = transitions[sum(data[-1])%7]
        return set(m for m,_ in counts.most_common(5)) if sum(counts.values()) > 0 else set(range(7))

    def _check_sum_mod7(self, combo, targets):
        return not targets or sum(combo)%7 in targets

    def _check_decade_balance(self, combo):
        dec = [0]*5
        for n in combo:
            if n <= 9: dec[0] += 1
            elif n <= 19: dec[1] += 1
            elif n <= 29: dec[2] += 1
            elif n <= 39: dec[3] += 1
            else: dec[4] += 1
        if max(dec[:4]) > 3: return False
        if dec[4] > 2: return False
        if sum(1 for d in dec if d > 0) < 3: return False
        return True

    def _validate(self, combo, c):
        if not c: return True
        s = sum(combo)
        if s < c.get('sum_lo',0) or s > c.get('sum_hi',999): return False
        odd = sum(1 for x in combo if x%2==1)
        if odd < c.get('odd_lo',0) or odd > c.get('odd_hi',6): return False
        mid = self.max_number//2; high = sum(1 for x in combo if x > mid)
        if high < c.get('high_lo',0) or high > c.get('high_hi',6): return False
        rng = max(combo)-min(combo)
        if rng < c.get('range_lo',0) or rng > c.get('range_hi',99): return False
        return True

    def _tight_constraints(self, data):
        r = data[-50:]
        sums = [sum(d[:self.pick_count]) for d in r]
        odds = [sum(1 for x in d[:self.pick_count] if x%2==1) for d in r]
        mid = self.max_number//2
        highs = [sum(1 for x in d[:self.pick_count] if x > mid) for d in r]
        ranges = [max(d[:self.pick_count])-min(d[:self.pick_count]) for d in r]
        return {
            'sum_lo': int(np.percentile(sums,10)), 'sum_hi': int(np.percentile(sums,90)),
            'odd_lo': max(0,int(np.percentile(odds,10))), 'odd_hi': min(self.pick_count,int(np.percentile(odds,90))),
            'high_lo': max(0,int(np.percentile(highs,10))), 'high_hi': min(self.pick_count,int(np.percentile(highs,90))),
            'range_lo': int(np.percentile(ranges,10)), 'range_hi': int(np.percentile(ranges,90)),
        }

    def _knn_replay(self, data):
        n, last = len(data), set(data[-1])
        m = [(len(set(data[i])&last), data[i+1]) for i in range(n-2) if len(set(data[i])&last) >= 3]
        m.sort(key=lambda x:-x[0])
        return [sorted(x[1][:self.pick_count]) for x in m[:20]]

    # ================================================================
    # 20 SIGNALS (compact)
    # ================================================================
    def _compute_all_signals(self, data, dates=None):
        s = {}
        s['transition'] = self._sig_transition(data)
        s['momentum'] = self._sig_momentum(data)
        s['gap_timing'] = self._sig_gap_timing(data)
        s['lag_repeat'] = self._sig_lag_repeat(data)
        s['cooccurrence'] = self._sig_cooccurrence(data)
        s['position'] = self._sig_position(data)
        s['streak'] = self._sig_streak(data)
        s['knn'] = self._sig_knn(data)
        s['fft_cycle'] = self._sig_fft_cycle(data)
        s['ngram'] = self._sig_ngram(data)
        s['context3'] = self._sig_context3(data)
        s['entropy'] = self._sig_entropy(data)
        s['triplet'] = self._sig_triplet(data)
        s['seq_pattern'] = self._sig_seq_pattern(data)
        s['runlength'] = self._sig_runlength(data)
        if dates: s['day_profile'] = self._sig_day_profile(data, dates)
        s['pair_boost'] = self._sig_pair_boost(data)
        s['consecutive'] = self._sig_consecutive(data)
        s['oddeven'] = self._sig_oddeven(data)
        s['highlow'] = self._sig_highlow(data)
        return s

    def _sig_transition(self, d):
        n,l=len(d),set(d[-1]);f,pc=defaultdict(Counter),Counter()
        for i in range(n-1):
            for p in d[i]:pc[p]+=1;[f[p].__setitem__(x,f[p].get(x,0)+1) for x in d[i+1]]
        b=self.pick_count/self.max_number
        return{num:((sum(f[p].get(num,0) for p in l)/max(sum(pc[p] for p in l),1))/b-1)*3 for num in range(1,self.max_number+1)}
    def _sig_momentum(self, d):
        n=len(d);return{num:(sum(1 for x in d[-5:] if num in x)/5-sum(1 for x in d[-20:] if num in x)/20)*10+(sum(1 for x in d[-20:] if num in x)/20-sum(1 for x in d[-50:] if num in x)/50)*5 if n>=50 else 0 for num in range(1,self.max_number+1)}
    def _sig_gap_timing(self, d):
        n,s=len(d),{};
        for num in range(1,self.max_number+1):
            a=[i for i,x in enumerate(d) if num in x]
            if len(a)<5:s[num]=0;continue
            g=[a[j+1]-a[j] for j in range(len(a)-1)];mg,sg=np.mean(g),np.std(g)
            z=(n-a[-1]-mg)/sg if sg>0 else 0;s[num]=z*1.5+(sum(1 for x in g if x<=n-a[-1])/len(g))*2 if z>0.5 else(-1 if z<-1 else 0)
        return s
    def _sig_lag_repeat(self, d):
        n,ls=len(d),{};lg=defaultdict(lambda:defaultdict(int))
        for i,x in enumerate(d):
            for num in x:
                if num in ls:lg[num][i-ls[num]]+=1
                ls[num]=i
        s={}
        for num in range(1,self.max_number+1):
            cl=n-ls.get(num,0)
            if num not in lg:s[num]=0;continue
            t=sum(lg[num].values());gl=[];[gl.extend([l]*c) for l,c in lg[num].items()]
            med=np.median(gl) if gl else self.max_number/self.pick_count
            s[num]=lg[num].get(cl,0)/t*5+max(0,cl/med-1)*2 if t>0 else 0
        return s
    def _sig_cooccurrence(self, d):
        l,pf=set(d[-1]),Counter()
        for x in d[-200:]:
            for p in combinations(sorted(x[:self.pick_count]),2):pf[p]+=1
        s={num:sum(pf.get(tuple(sorted([p,num])),0) for p in l)*0.1 for num in range(1,self.max_number+1)}
        tf=Counter()
        for x in d[-150:]:
            for t in combinations(sorted(x[:self.pick_count]),3):tf[t]+=1
        for t,c in tf.most_common(500):
            if c<2:break
            ts=set(t);ov=ts&l
            if len(ov)==2:m=(ts-l).pop();s[m]=s.get(m,0)+c*0.5
        return s
    def _sig_position(self, d):
        n,pf=len(d),[Counter() for _ in range(self.pick_count)]
        for x in d:
            sd=sorted(x[:self.pick_count])
            for p,num in enumerate(sd):pf[p][num]+=1
        return{num:sum(pf[p].get(num,0) for p in range(self.pick_count))/n for num in range(1,self.max_number+1)}
    def _sig_streak(self, d):
        s={}
        for num in range(1,self.max_number+1):
            c=0
            for x in reversed(d):
                if num not in x:c+=1
                else:break
            s[num]=c*0.1 if c>=10 else 0
        return s
    def _sig_knn(self, d):
        n,l,ks=len(d),set(d[-1]),Counter()
        for i in range(n-2):
            sim=len(set(d[i])&l)
            if sim>=3:
                for num in d[i+1]:ks[num]+=sim**2
        mx=max(ks.values()) if ks else 1;return{num:ks.get(num,0)/mx*3 for num in range(1,self.max_number+1)}
    def _sig_fft_cycle(self, d):
        s,w={},min(200,len(d))
        for num in range(1,self.max_number+1):
            seq=np.array([1.0 if num in x else 0.0 for x in d[-w:]])
            if len(seq)<30:s[num]=0;continue
            sc=seq-np.mean(seq);ft=np.fft.rfft(sc);pw=np.abs(ft)**2
            if len(pw)<3:s[num]=0;continue
            fr=np.fft.rfftfreq(len(sc));pi=np.argmax(pw[2:])+2
            pf_=fr[pi] if pi<len(fr) else 0;pp=pw[pi] if pi<len(pw) else 0;sr=pp/(np.sum(pw[1:])+1e-10)
            s[num]=sr*max(0,math.cos(2*math.pi*((len(seq)%(1/pf_))/(1/pf_))))*3 if sr>0.15 and pf_>0 else 0
        return s
    def _sig_ngram(self, d):
        bg=defaultdict(Counter)
        for i in range(1,len(d)):
            for pn in d[i-1]:
                for cn in d[i]:bg[pn][cn]+=1
        sc=Counter()
        for pn in d[-1]:
            t=sum(bg[pn].values())
            if t>0:
                for nn,cnt in bg[pn].most_common(10):sc[nn]+=cnt/t
        return{num:sc.get(num,0) for num in range(1,self.max_number+1)}
    def _sig_context3(self, d):
        n,sc=len(d),Counter()
        if n<20:return{num:0 for num in range(1,self.max_number+1)}
        l3=[set(x) for x in d[-3:]]
        for i in range(3,n-1):
            h3=[set(x) for x in d[i-3:i]];sim=sum(len(h3[j]&l3[j]) for j in range(3))
            if sim>=4:
                for num in d[i]:sc[num]+=sim**2
        mx=max(sc.values()) if sc else 1;return{num:sc.get(num,0)/max(1,mx)*3 for num in range(1,self.max_number+1)}
    def _sig_entropy(self, d):
        n=len(d)
        if n<60:return{num:0 for num in range(1,self.max_number+1)}
        s={}
        for num in range(1,self.max_number+1):
            seq=[1 if num in x else 0 for x in d[-60:]];tr={0:[0,0],1:[0,0]}
            for i in range(1,len(seq)):tr[seq[i-1]][seq[i]]+=1
            cs=seq[-1];t=sum(tr[cs]);pa=tr[cs][1]/t if t>0 else self.pick_count/self.max_number
            ent=0
            for st in [0,1]:
                tt=sum(tr[st])
                if tt==0:continue
                for c in tr[st]:
                    if c>0:p=c/tt;ent-=p*math.log2(p)*(tt/len(seq))
            s[num]=pa*max(0,1-ent)
        return s
    def _sig_triplet(self, d):
        n,l,sc=len(d),set(d[-1]),Counter()
        if n<50:return{num:0 for num in range(1,self.max_number+1)}
        tf=Counter()
        for x in(d[-100:] if n>100 else d):
            for t in combinations(sorted(x[:self.pick_count]),3):tf[t]+=1
        for t,c in tf.most_common(300):
            if c<2:break
            ts=set(t);ov=ts&l
            if len(ov)>=2:
                for num in ts-l:sc[num]+=c*len(ov)
        mx=max(sc.values()) if sc else 1;return{num:sc.get(num,0)/max(1,mx)*2.5 for num in range(1,self.max_number+1)}
    def _sig_seq_pattern(self, d):
        n,l,sc=len(d),set(d[-1]),Counter()
        if n<30:return{num:0 for num in range(1,self.max_number+1)}
        for cl in[3,4]:
            cc=[frozenset(x) for x in d[-cl:]]
            for i in range(cl,min(n-cl,n-1)):
                hc=[frozenset(x) for x in d[i-cl:i]];sim=sum(len(cc[j]&hc[j]) for j in range(cl))
                if sim>=cl*2 and i<n-1:
                    for num in d[i]:
                        if num not in l:sc[num]+=sim*cl*0.1
        mx=max(sc.values()) if sc else 1;return{num:sc.get(num,0)/max(1,mx)*3 for num in range(1,self.max_number+1)}
    def _sig_runlength(self, d):
        n,eg=len(d),self.max_number/self.pick_count;s={}
        for num in range(1,self.max_number+1):
            ca=0
            for x in reversed(d):
                if num not in x:ca+=1
                else:break
            if ca>0:
                sa=[1 if num in x else 0 for x in d];ar,run=[],0
                for sv in sa:
                    if sv==0:run+=1
                    else:
                        if run>0:ar.append(run);run=0
                avg=np.mean(ar) if ar else eg;s[num]=1/(1+math.exp(-3*(ca/avg-0.8)))*2 if avg>0 else 0
            else:s[num]=0
        return s
    def _sig_day_profile(self, d, dates):
        from datetime import datetime;dd=defaultdict(list)
        for dt,x in zip(dates,d):
            try:dd[datetime.strptime(dt,'%Y-%m-%d').weekday()].append(x)
            except:continue
        try:ndow=(datetime.strptime(dates[-1],'%Y-%m-%d').weekday()+2)%7
        except:return{num:0 for num in range(1,self.max_number+1)}
        return{num:(sum(1 for x in dd.get(ndow,[]) if num in x)/max(len(dd.get(ndow,[])),1))/(sum(1 for x in d if num in x)/len(d)+1e-10)*2-2 for num in range(1,self.max_number+1)}
    def _sig_pair_boost(self, d):
        l,pf=set(d[-1]),Counter()
        for x in d[-100:]:
            for p in combinations(sorted(x[:self.pick_count]),2):pf[p]+=1
        return{num:sum(pf.get(tuple(sorted([p,num])),0) for p in l if pf.get(tuple(sorted([p,num])),0)>3)*0.05 for num in range(1,self.max_number+1)}
    def _sig_consecutive(self, d):
        s={num:0.0 for num in range(1,self.max_number+1)}
        for x in d[-50:]:
            sd=sorted(x[:self.pick_count])
            for i in range(len(sd)-1):
                if sd[i+1]-sd[i]==1:s[sd[i]]+=0.05;s[sd[i+1]]+=0.05
        return s
    def _sig_oddeven(self, d):
        lo=sum(1 for x in d[-1] if x%2==1);return{num:0.3 if(lo>3 and num%2==0)or(lo<=3 and num%2==1) else 0 for num in range(1,self.max_number+1)}
    def _sig_highlow(self, d):
        mid,lh=self.max_number//2,sum(1 for x in d[-1] if x>self.max_number//2);return{num:0.3 if(lh>3 and num<=mid)or(lh<=3 and num>mid) else 0 for num in range(1,self.max_number+1)}
    def _walk_forward_weights(self, data, signals):
        n,cs=len(data),min(30,len(data)-70)
        if cs<10:return{name:1.0 for name in signals}
        sh={name:0 for name in signals};tc=0
        for idx in range(n-cs-1,n-1):
            actual=set(data[idx+1]);tc+=1
            for sn,ss in signals.items():
                if not ss:continue
                pred=set(num for num,_ in sorted(ss.items(),key=lambda x:-x[1])[:self.pick_count])
                sh[sn]+=len(pred&actual)
        base=self.pick_count/self.max_number;return{name:max(sh[name]/tc/(base*self.pick_count),0.1) if tc>0 and sh[name]>0 else 0.5 for name in signals}
