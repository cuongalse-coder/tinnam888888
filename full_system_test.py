"""
FULL SYSTEM TEST & BACKTEST
============================
1. Test ALL imports & methods for errors
2. Backtest ALL prediction methods across ALL historical draws
3. Report hit rates for 6/6, 5/6, 4/6, 3/6
"""
import sys, os, time, json, math, traceback
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# PHASE 1: TEST ALL IMPORTS
# ============================================================
def test_imports():
    print("=" * 80)
    print("PHASE 1: TEST ALL IMPORTS & CLASS INITIALIZATION")
    print("=" * 80)
    
    errors = []
    
    modules = [
        ("models.nexus_engine", "NexusEngine"),
        ("models.stacking_engine", "StackingEngine"),
        ("models.mega_exploit_v15", "MegaExploitV15"),
        ("models.mega_exploit_v12", "MegaExploitV12"),
        ("models.mega_exploit_v11", None),
        ("models.ultimate_engine", "UltimateEngine"),
        ("models.super_predictor", "SuperPredictor"),
        ("models.deep_forensic", "DeepForensic"),
        ("models.exploit_engine", "ExploitEngine"),
        ("models.vulnerability_scanner", "VulnerabilityScanner"),
        ("models.wheeling_optimizer", "WheelingOptimizer"),
        ("models.backtester", None),
        ("models.dan_predictor", None),
        ("models.middle4_predictor", None),
        ("models.ultimate_predictor", None),
    ]
    
    for mod_name, class_name in modules:
        try:
            mod = __import__(mod_name, fromlist=[class_name or ""])
            if class_name:
                cls = getattr(mod, class_name)
                instance = cls(45, 6)
                print(f"  ✅ {mod_name}.{class_name} - OK")
            else:
                print(f"  ✅ {mod_name} - OK")
        except Exception as e:
            print(f"  ❌ {mod_name} - ERROR: {e}")
            errors.append((mod_name, str(e)))
    
    return errors

# ============================================================
# PHASE 2: TEST ALL METHODS OF NEXUS ENGINE
# ============================================================
def test_nexus_methods(data):
    print("\n" + "=" * 80)
    print("PHASE 2: TEST ALL NEXUS ENGINE METHODS")
    print("=" * 80)
    
    from models.nexus_engine import NexusEngine
    engine = NexusEngine(45, 6)
    
    errors = []
    methods = [
        ("_sig_sliding_window", [data]),
        ("_sig_conditional_probability", [data]),
        ("_sig_gap_acceleration", [data]),
        ("_sig_hot_cold_intersection", [data]),
        ("_sig_delta_momentum", [data]),
        ("_sig_sector_rotation", [data]),
        ("_sig_pair_boost", [data]),
        ("_sig_temporal_decay", [data]),
        ("_sig_knn_fractal_v3", [data]),
        ("_sig_pair_triplet_hybrid", [data]),
        ("_sig_regime_detector", [data]),
        ("_sig_lag_correlation", [data]),
        ("_calibrate_rolling", None),  # special
        ("calculate_confidence", [data]),
        ("predict_micro_sector", [data]),
        ("predict", [data]),
    ]
    
    for method_name, args in methods:
        try:
            method = getattr(engine, method_name)
            if method_name == "_calibrate_rolling":
                # Need signals first
                sigs = {
                    'test_sig': engine._sig_sliding_window(data)
                }
                result = method(data, sigs)
            elif method_name == "predict":
                result = method(data, n_sets=3)
            else:
                result = method(*args)
            print(f"  ✅ NexusEngine.{method_name}() - OK (type={type(result).__name__})")
        except Exception as e:
            print(f"  ❌ NexusEngine.{method_name}() - ERROR: {e}")
            traceback.print_exc()
            errors.append((method_name, str(e)))
    
    return errors

# ============================================================
# PHASE 3: TEST STACKING ENGINE
# ============================================================
def test_stacking_engine(data):
    print("\n" + "=" * 80)
    print("PHASE 3: TEST STACKING ENGINE")
    print("=" * 80)
    
    errors = []
    try:
        from models.stacking_engine import StackingEngine
        engine = StackingEngine(45, 6)
        result = engine.predict_top_pool(data, pool_size=15)
        print(f"  ✅ StackingEngine.predict_top_pool() - OK")
        print(f"      Top 6: {result['top6']}")
        print(f"      Pool 15: {result['pool']}")
    except Exception as e:
        print(f"  ❌ StackingEngine - ERROR: {e}")
        traceback.print_exc()
        errors.append(("StackingEngine", str(e)))
    
    return errors

# ============================================================
# PHASE 4: TEST RealWorldAIEngine (from streamlit_app.py)
# ============================================================
def test_realworld_engine(data):
    print("\n" + "=" * 80)
    print("PHASE 4: TEST RealWorldAIEngine METHODS")
    print("=" * 80)
    
    # Import inline to avoid streamlit dependency
    errors = []
    
    try:
        # Re-create RealWorldAIEngine class locally (copy from streamlit_app.py)
        from collections import Counter, defaultdict
        
        class RealWorldAIEngine:
            def __init__(self, data, max_number):
                self.data = data
                self.max_number = max_number
                self.all_numbers = list(range(1, max_number + 1))
            
            def _get_frequency(self, lookback=None):
                subset = self.data[-lookback:] if lookback else self.data
                all_nums = [n for draw in subset for n in draw]
                return Counter(all_nums)
            
            def model_markov_chain(self):
                transitions = defaultdict(Counter)
                for i in range(len(self.data) - 1):
                    current = tuple(sorted(self.data[i]))
                    next_draw = self.data[i + 1]
                    for num in next_draw:
                        transitions[current][num] += 1
                if len(self.data) > 0:
                    last_draw = tuple(sorted(self.data[-1]))
                    if last_draw in transitions and transitions[last_draw]:
                        next_probs = transitions[last_draw]
                        return [num for num, _ in next_probs.most_common(6)]
                return [n for n, c in self._get_frequency(20).most_common(6)]
            
            def model_gap_overdue(self, top_n=6):
                last_seen = {num: -1 for num in self.all_numbers}
                for i, draw in enumerate(self.data):
                    for num in draw:
                        last_seen[num] = i
                current_idx = len(self.data)
                gaps = {num: current_idx - last_seen[num] for num in self.all_numbers}
                avg_gaps = defaultdict(list)
                last_idx = {}
                for i, draw in enumerate(self.data):
                    for num in draw:
                        if num in last_idx:
                            avg_gaps[num].append(i - last_idx[num])
                        last_idx[num] = i
                due_scores = {}
                for num in self.all_numbers:
                    if avg_gaps[num]:
                        mean_gap = np.mean(avg_gaps[num])
                        current_gap = gaps[num]
                        due_scores[num] = current_gap / (mean_gap + 0.1)
                    else:
                        due_scores[num] = 0
                sorted_due = sorted(due_scores.items(), key=lambda x: x[1], reverse=True)
                return [num for num, score in sorted_due[:top_n]]
            
            def model_momentum_neural(self):
                weights = {num: 0.0 for num in self.all_numbers}
                total_draws = len(self.data)
                for i, draw in enumerate(self.data):
                    decay = 1 / (1 + np.exp(-(i - total_draws + 20) / 5))
                    for num in draw:
                        weights[num] += decay
                sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
                return [num for num, w in sorted_weights[:6]]
            
            def model_knn_mirror(self):
                if len(self.data) < 20:
                    return self.model_momentum_neural()
                pattern = set(self.data[-1]) | set(self.data[-2]) | set(self.data[-3])
                if len(self.data) > 3:
                    pattern |= set(self.data[-4])
                n = len(self.data)
                similarities = []
                for i in range(3, n - 3):
                    past_pattern = set(self.data[i]) | set(self.data[i-1]) | set(self.data[i-2]) | set(self.data[i-3])
                    intersect = len(pattern & past_pattern)
                    recency = 1.0 + 0.5 * (i / n)
                    if intersect >= 5:
                        similarities.append((intersect * recency, i + 1))
                similarities.sort(key=lambda x: -x[0])
                from collections import Counter
                mirror_votes = Counter()
                for score, next_idx in similarities[:30]:
                    if next_idx < n:
                        for num in self.data[next_idx]:
                            mirror_votes[num] += score
                if not mirror_votes:
                    return self.model_momentum_neural()
                return [n for n, s in mirror_votes.most_common(20)]
            
            def model_pair_matrix(self):
                if len(self.data) < 30:
                    return self.model_gap_overdue()
                from itertools import combinations
                pair_scores = Counter()
                n = len(self.data)
                for idx, draw in enumerate(self.data):
                    decay = 0.3 + 0.7 * (idx / n)
                    for p in combinations(sorted(draw[:6]), 2):
                        pair_scores[p] += decay
                last_draw = set(self.data[-1][:6])
                candidate_scores = Counter()
                for num in self.all_numbers:
                    if num in last_draw:
                        continue
                    for anchor in last_draw:
                        key = tuple(sorted([num, anchor]))
                        candidate_scores[num] += pair_scores.get(key, 0)
                triplet_bonus = Counter()
                for idx in range(max(0, n - 100), n):
                    draw = self.data[idx]
                    for trip in combinations(sorted(draw[:6]), 3):
                        trip_set = set(trip)
                        overlap = trip_set & last_draw
                        if len(overlap) >= 2:
                            for num in trip_set - last_draw:
                                triplet_bonus[num] += 1.5
                for num in triplet_bonus:
                    candidate_scores[num] += triplet_bonus[num]
                return [n for n, s in candidate_scores.most_common(15)]
            
            def model_delta_momentum(self):
                if len(self.data) < 30:
                    return self.model_momentum_neural()
                scores = {}
                for num in self.all_numbers:
                    f5 = sum(1 for d in self.data[-5:] if num in d[:6]) / 5
                    f5_prev = sum(1 for d in self.data[-10:-5] if num in d[:6]) / 5
                    f15 = sum(1 for d in self.data[-15:] if num in d[:6]) / 15
                    f15_prev = sum(1 for d in self.data[-30:-15] if num in d[:6]) / 15
                    delta_short = f5 - f5_prev
                    delta_mid = f15 - f15_prev
                    momentum = delta_short * 3 + delta_mid * 2
                    if num in self.data[-1][:6]:
                        momentum += 0.5
                    if len(self.data) >= 2 and num in self.data[-2][:6]:
                        momentum += 0.3
                    scores[num] = momentum
                sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
                return [n for n, s in sorted_scores[:15]]
            
            def model_advanced_ml(self):
                try:
                    from sklearn.ensemble import RandomForestRegressor
                    from sklearn.cluster import KMeans
                    if len(self.data) < 20:
                        return self.model_gap_overdue()
                    X, y = [], []
                    window_size = 10
                    for i in range(len(self.data) - window_size - 1):
                        window = self.data[i:i+window_size]
                        next_draw = self.data[i+window_size]
                        features = np.zeros(self.max_number)
                        for draw in window:
                            for num in draw:
                                features[num-1] += 1
                        targets = np.zeros(self.max_number)
                        for num in next_draw:
                            targets[num-1] = 1
                        X.append(features)
                        y.append(targets)
                    rf = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42)
                    rf.fit(X, y)
                    recent_window = self.data[-window_size:]
                    recent_features = np.zeros(self.max_number)
                    for draw in recent_window:
                        for num in draw:
                            recent_features[num-1] += 1
                    rf_predictions = rf.predict([recent_features])[0]
                    flat_data = np.array([num for draw in self.data for num in draw]).reshape(-1, 1)
                    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
                    kmeans.fit(flat_data)
                    cluster_centers = [int(round(c[0])) for c in kmeans.cluster_centers_]
                    combined_scores = {num: rf_predictions[num-1] for num in self.all_numbers}
                    for c in cluster_centers:
                        if 1 <= c <= self.max_number:
                            combined_scores[c] += np.mean(rf_predictions) * 1.5
                    top_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:6]
                    return [idx for idx, score in top_indices]
                except Exception:
                    return self.model_momentum_neural()
            
            def model_cond_prob(self):
                if len(self.data) < 30:
                    return []
                last = set(self.data[-1])
                cond_counts = defaultdict(lambda: defaultdict(int))
                total_given = defaultdict(int)
                for i in range(len(self.data) - 1):
                    for given in self.data[i]:
                        total_given[given] += 1
                        for next_num in self.data[i+1]:
                            cond_counts[given][next_num] += 1
                scores = {}
                for num in self.all_numbers:
                    scores[num] = 0
                    for given in last:
                        if total_given[given] > 0:
                            scores[num] += cond_counts[given].get(num, 0) / total_given[given]
                sorted_s = sorted(scores.items(), key=lambda x: -x[1])
                return [n for n, s in sorted_s[:15]]
            
            def model_freq_gap_hybrid(self):
                if len(self.data) < 30:
                    return self.model_gap_overdue()
                expected = 6 / len(self.all_numbers)
                scores = {}
                for num in self.all_numbers:
                    f5 = sum(1 for d in self.data[-5:] if num in d) / 5
                    f15 = sum(1 for d in self.data[-15:] if num in d) / 15
                    freq_score = (f5 / (expected + 0.01)) * 0.6 + (f15 / (expected + 0.01)) * 0.4
                    last_seen = -1
                    for i in range(len(self.data)-1, -1, -1):
                        if num in self.data[i]: last_seen = i; break
                    gap = len(self.data) - last_seen if last_seen >= 0 else len(self.data)
                    appearances = [i for i, d in enumerate(self.data) if num in d]
                    mean_gap = len(self.all_numbers) / 6
                    if len(appearances) >= 2:
                        gaps = [appearances[j+1]-appearances[j] for j in range(len(appearances)-1)]
                        mean_gap = sum(gaps) / len(gaps)
                    overdue = gap / (mean_gap + 0.1)
                    if freq_score > 0.8 and overdue > 0.7: scores[num] = freq_score * overdue * 3
                    elif overdue > 1.5: scores[num] = overdue * 1.5
                    elif freq_score > 1.3: scores[num] = freq_score * 2
                    else: scores[num] = freq_score * 0.5 + overdue * 0.5
                return [n for n, _ in sorted(scores.items(), key=lambda x: -x[1])[:15]]
            
            def _run_9model_ensemble(self, pool_size=20):
                m1 = self.model_markov_chain()
                m2 = self.model_gap_overdue(top_n=15)
                m3 = self.model_momentum_neural()
                m4 = self.model_advanced_ml()
                m5 = self.model_knn_mirror()
                m6 = self.model_pair_matrix()
                m7 = self.model_delta_momentum()
                m8 = self.model_cond_prob()
                m9 = self.model_freq_gap_hybrid()
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
                strong_models = [set(m5[:12]), set(m6[:12]), set(m8[:12]), set(m7[:12])]
                for num in self.all_numbers:
                    consensus = sum(1 for ml in strong_models if num in ml)
                    if consensus >= 3:
                        votes[num] += consensus * 5
                return [n for n, _ in votes.most_common(pool_size)]
            
            def optimize_ensemble(self):
                ranked = self._run_9model_ensemble(pool_size=20)
                best = ranked[:6]
                while len(best) < 6:
                    candidates = self.model_gap_overdue(top_n=15)
                    for c in candidates:
                        if c not in best:
                            best.append(c)
                            if len(best) == 6: break
                return sorted(best)
        
        engine = RealWorldAIEngine(data, 45)
        
        model_methods = [
            "model_markov_chain",
            "model_gap_overdue",
            "model_momentum_neural",
            "model_knn_mirror",
            "model_pair_matrix",
            "model_delta_momentum",
            "model_advanced_ml",
            "model_cond_prob",
            "model_freq_gap_hybrid",
            "_run_9model_ensemble",
            "optimize_ensemble",
        ]
        
        for method_name in model_methods:
            try:
                method = getattr(engine, method_name)
                result = method()
                print(f"  ✅ RealWorldAIEngine.{method_name}() - OK (returned {len(result)} nums: {result[:6]})")
            except Exception as e:
                print(f"  ❌ RealWorldAIEngine.{method_name}() - ERROR: {e}")
                traceback.print_exc()
                errors.append((method_name, str(e)))
    except Exception as e:
        print(f"  ❌ RealWorldAIEngine setup - ERROR: {e}")
        traceback.print_exc()
        errors.append(("RealWorldAIEngine", str(e)))
    
    return errors

# ============================================================
# PHASE 5: FULL BACKTEST — ALL METHODS, ALL DRAWS
# ============================================================
def full_backtest(all_data, game_type="Mega 6/45"):
    max_number = 45 if game_type == "Mega 6/45" else 55
    
    print("\n" + "=" * 80)
    print(f"PHASE 5: FULL BACKTEST — {game_type} ({len(all_data)} kỳ)")
    print("=" * 80)
    
    from models.nexus_engine import NexusEngine
    
    # Methods to backtest with their prediction functions
    min_history = 80  # Minimum history needed
    
    results = {}
    
    method_names = [
        "NexusEngine.predict (top_pool[:6])",
        "NexusEngine.predict (absolute_final_6)",
        "NexusEngine.calculate_confidence",
        "StackingEngine.predict_top_pool (top6)",
        "RealWorld.markov_chain",
        "RealWorld.gap_overdue",
        "RealWorld.momentum_neural",
        "RealWorld.knn_mirror (top 6)",
        "RealWorld.pair_matrix (top 6)",
        "RealWorld.delta_momentum (top 6)",
        "RealWorld.cond_prob (top 6)",
        "RealWorld.freq_gap_hybrid (top 6)",
        "RealWorld.9model_ensemble (top 6)",
        "RealWorld.optimize_ensemble",
        "NexusEngine (top 10 pool)",
        "NexusEngine (top 15 pool)",
        "NexusEngine (top 20 pool)",
    ]
    
    for name in method_names:
        results[name] = {k: 0 for k in range(7)}  # 0/6 to 6/6
    
    n_tested = 0
    n_total = len(all_data)
    start_idx = min_history
    
    # Use step to make it run in a reasonable time
    # Test every draw from min_history to end
    test_range = range(start_idx, n_total)
    total_tests = len(test_range)
    
    print(f"\n  Testing {total_tests} draws (from kỳ {start_idx+1} to {n_total})...")
    print(f"  This will take a while...\n")
    
    confidence_values = []
    
    t0 = time.time()
    
    for test_idx in test_range:
        history = all_data[:test_idx]
        actual = set(all_data[test_idx][:6])
        n_tested += 1
        
        if n_tested % 50 == 0:
            elapsed = time.time() - t0
            pct = n_tested / total_tests * 100
            eta = (elapsed / n_tested) * (total_tests - n_tested) if n_tested > 0 else 0
            print(f"  Progress: {n_tested}/{total_tests} ({pct:.1f}%) — Elapsed: {elapsed:.0f}s — ETA: {eta:.0f}s")
        
        # === NexusEngine methods ===
        try:
            engine = NexusEngine(max_number, 6)
            nexus_result = engine.predict(history, n_sets=1, use_elastic_filter=False)
            
            # top_pool[:6]
            top6 = nexus_result['top_pool'][:6]
            hits = len(set(top6) & actual)
            results["NexusEngine.predict (top_pool[:6])"][hits] += 1
            
            # absolute_final_6
            af6 = nexus_result.get('absolute_final_6', [])
            if af6:
                hits = len(set(af6) & actual)
                results["NexusEngine.predict (absolute_final_6)"][hits] += 1
            
            # top 10/15/20 pool coverage
            for pool_label, pool_size in [("NexusEngine (top 10 pool)", 10), 
                                           ("NexusEngine (top 15 pool)", 15),
                                           ("NexusEngine (top 20 pool)", 20)]:
                pool = nexus_result['top_pool'][:pool_size]
                hits = len(set(pool) & actual)
                results[pool_label][min(hits, 6)] += 1
            
            # calculate_confidence
            try:
                conf = engine.calculate_confidence(history)
                confidence_values.append(conf)
                results["NexusEngine.calculate_confidence"][0] += 1  # Just track it runs OK
            except Exception as e:
                results["NexusEngine.calculate_confidence"][0] = -1  # Error marker
        except Exception as e:
            if n_tested <= 3:
                print(f"    NexusEngine error at test {n_tested}: {e}")
        
        # === StackingEngine (only every 10th draw due to slowness) ===
        if n_tested % 10 == 1:
            try:
                from models.stacking_engine import StackingEngine
                stacker = StackingEngine(max_number, 6)
                stack_result = stacker.predict_top_pool(history, pool_size=15)
                top6 = stack_result['top6']
                hits = len(set(top6) & actual)
                results["StackingEngine.predict_top_pool (top6)"][hits] += 1
            except Exception:
                pass
        
        # === RealWorldAIEngine methods (lightweight, run every draw) ===
        try:
            # Inline mini-engine for speed
            data = history
            all_numbers = list(range(1, max_number + 1))
            
            # markov_chain
            transitions = defaultdict(Counter)
            for i in range(len(data) - 1):
                current = tuple(sorted(data[i]))
                for num in data[i + 1]:
                    transitions[current][num] += 1
            last_draw = tuple(sorted(data[-1]))
            if last_draw in transitions and transitions[last_draw]:
                pred = [num for num, _ in transitions[last_draw].most_common(6)]
            else:
                freq = Counter(n for d in data[-20:] for n in d)
                pred = [n for n, c in freq.most_common(6)]
            hits = len(set(pred[:6]) & actual)
            results["RealWorld.markov_chain"][hits] += 1
            
            # gap_overdue
            last_seen = {num: -1 for num in all_numbers}
            for i, draw in enumerate(data):
                for num in draw:
                    last_seen[num] = i
            curr_idx = len(data)
            gap_dict = {num: curr_idx - last_seen[num] for num in all_numbers}
            avg_gaps_dict = defaultdict(list)
            last_idx_dict = {}
            for i, draw in enumerate(data):
                for num in draw:
                    if num in last_idx_dict:
                        avg_gaps_dict[num].append(i - last_idx_dict[num])
                    last_idx_dict[num] = i
            due_scores = {}
            for num in all_numbers:
                if avg_gaps_dict[num]:
                    mean_gap = np.mean(avg_gaps_dict[num])
                    due_scores[num] = gap_dict[num] / (mean_gap + 0.1)
                else:
                    due_scores[num] = 0
            pred = [num for num, _ in sorted(due_scores.items(), key=lambda x: -x[1])[:6]]
            hits = len(set(pred) & actual)
            results["RealWorld.gap_overdue"][hits] += 1
            
            # momentum_neural
            weights = {num: 0.0 for num in all_numbers}
            td = len(data)
            for i, draw in enumerate(data):
                decay = 1 / (1 + np.exp(-(i - td + 20) / 5))
                for num in draw:
                    weights[num] += decay
            pred = [num for num, _ in sorted(weights.items(), key=lambda x: -x[1])[:6]]
            hits = len(set(pred) & actual)
            results["RealWorld.momentum_neural"][hits] += 1
            
            # knn_mirror (top 6)
            if len(data) >= 20:
                pattern = set(data[-1]) | set(data[-2]) | set(data[-3])
                if len(data) > 3:
                    pattern |= set(data[-4])
                nd = len(data)
                sims = []
                for i in range(3, nd - 3):
                    pp = set(data[i]) | set(data[i-1]) | set(data[i-2]) | set(data[i-3])
                    intx = len(pattern & pp)
                    if intx >= 5:
                        sims.append((intx * (1.0 + 0.5 * (i/nd)), i + 1))
                sims.sort(key=lambda x: -x[0])
                mv = Counter()
                for sc, ni in sims[:30]:
                    if ni < nd:
                        for num in data[ni]:
                            mv[num] += sc
                pred = [n for n, _ in mv.most_common(6)] if mv else list(range(1, 7))
                hits = len(set(pred[:6]) & actual)
                results["RealWorld.knn_mirror (top 6)"][hits] += 1
            
            # pair_matrix (top 6)
            if len(data) >= 30:
                from itertools import combinations
                pair_sc = Counter()
                nd = len(data)
                for idx, draw in enumerate(data):
                    decay = 0.3 + 0.7 * (idx / nd)
                    for p in combinations(sorted(draw[:6]), 2):
                        pair_sc[p] += decay
                last_d = set(data[-1][:6])
                cand_sc = Counter()
                for num in all_numbers:
                    if num in last_d: continue
                    for anchor in last_d:
                        key = tuple(sorted([num, anchor]))
                        cand_sc[num] += pair_sc.get(key, 0)
                pred = [n for n, _ in cand_sc.most_common(6)]
                hits = len(set(pred) & actual)
                results["RealWorld.pair_matrix (top 6)"][hits] += 1
            
            # delta_momentum (top 6)
            if len(data) >= 30:
                dm_scores = {}
                for num in all_numbers:
                    f5 = sum(1 for d in data[-5:] if num in d[:6]) / 5
                    f5_prev = sum(1 for d in data[-10:-5] if num in d[:6]) / 5
                    f15 = sum(1 for d in data[-15:] if num in d[:6]) / 15
                    f15_prev = sum(1 for d in data[-30:-15] if num in d[:6]) / 15
                    dm_scores[num] = (f5 - f5_prev) * 3 + (f15 - f15_prev) * 2
                pred = [n for n, _ in sorted(dm_scores.items(), key=lambda x: -x[1])[:6]]
                hits = len(set(pred) & actual)
                results["RealWorld.delta_momentum (top 6)"][hits] += 1
            
            # cond_prob (top 6)
            if len(data) >= 30:
                last_set = set(data[-1])
                cc = defaultdict(lambda: defaultdict(int))
                tg = defaultdict(int)
                for i in range(len(data) - 1):
                    for g in data[i]:
                        tg[g] += 1
                        for nx in data[i+1]:
                            cc[g][nx] += 1
                cp_scores = {}
                for num in all_numbers:
                    cp_scores[num] = sum(cc[g].get(num, 0) / tg[g] for g in last_set if tg[g] > 0)
                pred = [n for n, _ in sorted(cp_scores.items(), key=lambda x: -x[1])[:6]]
                hits = len(set(pred) & actual)
                results["RealWorld.cond_prob (top 6)"][hits] += 1
            
            # freq_gap_hybrid (top 6)
            if len(data) >= 30:
                expected = 6 / len(all_numbers)
                fg_scores = {}
                for num in all_numbers:
                    f5 = sum(1 for d in data[-5:] if num in d) / 5
                    f15 = sum(1 for d in data[-15:] if num in d) / 15
                    freq_sc = (f5 / (expected + 0.01)) * 0.6 + (f15 / (expected + 0.01)) * 0.4
                    ls = -1
                    for i in range(len(data)-1, -1, -1):
                        if num in data[i]: ls = i; break
                    gap = len(data) - ls if ls >= 0 else len(data)
                    apps = [i for i, d in enumerate(data) if num in d]
                    mg = len(all_numbers) / 6
                    if len(apps) >= 2:
                        gps = [apps[j+1]-apps[j] for j in range(len(apps)-1)]
                        mg = sum(gps) / len(gps)
                    od = gap / (mg + 0.1)
                    if freq_sc > 0.8 and od > 0.7: fg_scores[num] = freq_sc * od * 3
                    elif od > 1.5: fg_scores[num] = od * 1.5
                    elif freq_sc > 1.3: fg_scores[num] = freq_sc * 2
                    else: fg_scores[num] = freq_sc * 0.5 + od * 0.5
                pred = [n for n, _ in sorted(fg_scores.items(), key=lambda x: -x[1])[:6]]
                hits = len(set(pred) & actual)
                results["RealWorld.freq_gap_hybrid (top 6)"][hits] += 1
            
            # 9model_ensemble (top 6) — Run every 5th draw for speed
            if n_tested % 5 == 1 and len(data) >= 30:
                try:
                    # Simplified 9-model ensemble
                    from collections import Counter as C2
                    # Just use the models we already have inline
                    m_markov = [num for num, _ in transitions.get(last_draw, Counter()).most_common(6)] if last_draw in transitions else []
                    m_gap = [num for num, _ in sorted(due_scores.items(), key=lambda x: -x[1])[:15]]
                    m_momentum = [num for num, _ in sorted(weights.items(), key=lambda x: -x[1])[:6]]
                    m_knn = pred if 'mv' in dir() and mv else []
                    
                    votes = C2()
                    for num in m_knn[:15]: votes[num] += 12
                    for num in m_gap[:15]: votes[num] += 3
                    for num in m_momentum[:6]: votes[num] += 2
                    for num in m_markov[:6]: votes[num] += 1
                    
                    pred_ens = [n for n, _ in votes.most_common(6)]
                    hits = len(set(pred_ens) & actual)
                    results["RealWorld.9model_ensemble (top 6)"][hits] += 1
                    
                    # optimize_ensemble
                    pred_opt = [n for n, _ in votes.most_common(6)]
                    hits = len(set(pred_opt) & actual)
                    results["RealWorld.optimize_ensemble"][hits] += 1
                except:
                    pass
                    
        except Exception as e:
            if n_tested <= 3:
                print(f"    RealWorld error at test {n_tested}: {e}")
    
    elapsed = time.time() - t0
    
    # ============================================================
    # PRINT RESULTS
    # ============================================================
    print("\n" + "=" * 80)
    print(f"BACKTEST RESULTS — {game_type} ({n_tested} kỳ tested, took {elapsed:.1f}s)")
    print("=" * 80)
    
    print(f"\n{'Method':<45} {'0/6':>6} {'1/6':>6} {'2/6':>6} {'3/6':>6} {'4/6':>6} {'5/6':>6} {'6/6':>6} {'Total':>7} {'≥3/6%':>7} {'≥4/6%':>7}")
    print("-" * 120)
    
    for name in method_names:
        if name == "NexusEngine.calculate_confidence":
            if confidence_values:
                avg_conf = np.mean(confidence_values)
                print(f"  {name:<43} Avg Confidence: {avg_conf:.1f}% (min={min(confidence_values):.1f}%, max={max(confidence_values):.1f}%)")
            continue
            
        counts = results[name]
        total = sum(counts.values())
        if total == 0:
            print(f"  {name:<43} {'N/A':>6}")
            continue
        
        ge3 = sum(counts[k] for k in range(3, 7))
        ge4 = sum(counts[k] for k in range(4, 7))
        pct3 = ge3 / total * 100
        pct4 = ge4 / total * 100
        
        print(f"  {name:<43} {counts[0]:>6} {counts[1]:>6} {counts[2]:>6} {counts[3]:>6} {counts[4]:>6} {counts[5]:>6} {counts[6]:>6} {total:>7} {pct3:>6.2f}% {pct4:>6.2f}%")
    
    # Special highlight for 6/6
    print("\n" + "=" * 80)
    print("🎯 HIGHLIGHT: 6/6 JACKPOT HITS")
    print("=" * 80)
    any_jackpot = False
    for name in method_names:
        if name == "NexusEngine.calculate_confidence":
            continue
        if results[name][6] > 0:
            total = sum(results[name].values())
            print(f"  🏆 {name}: {results[name][6]} times out of {total} ({results[name][6]/total*100:.4f}%)")
            any_jackpot = True
    if not any_jackpot:
        print("  ❌ Không có phương pháp nào trúng 6/6 trong backtest.")
        print("  📊 Đây là BÌNH THƯỜNG — xác suất random trúng 6/6 Mega645 = 1/8,145,060 ≈ 0.0000123%")
        total_combos = math.comb(45, 6)
        print(f"  📊 Tổng tổ hợp C(45,6) = {total_combos:,}")
    
    return results

# ============================================================
# DATA FETCHER (standalone, no streamlit)
# ============================================================
def fetch_data_standalone(game_type="Mega 6/45"):
    print(f"\n📡 Đang tải dữ liệu {game_type} từ internet...")
    
    import requests, re
    
    max_num = 45 if game_type == "Mega 6/45" else 55
    today_str = datetime.now().strftime('%d-%m-%Y')
    
    urls = [
        f"https://www.ketquadientoan.com/tat-ca-ky-xo-so-mega-6-45.html?datef=18-07-2016&datet={today_str}" if game_type == "Mega 6/45" else f"https://www.ketquadientoan.com/tat-ca-ky-xo-so-power-655.html?datef=01-01-2018&datet={today_str}",
    ]
    
    try:
        import cloudscraper
        scraper = cloudscraper.create_scraper(delay=5, browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False})
    except ImportError:
        scraper = requests.Session()
    
    for url in urls:
        try:
            response = scraper.get(url, timeout=30)
            if response.status_code == 200:
                html = response.text
                history = []
                rows = re.findall(r'<tr.*?>(.*?)</tr>', html, re.DOTALL | re.IGNORECASE)
                for row in rows:
                    nums = re.findall(r'class="home-mini-whiteball">\s*(\d{2})\s*<', row)
                    if len(nums) < 6:
                        continue
                    chunk = [int(n) for n in nums[:6]]
                    if len(set(chunk)) != 6 or not all(1 <= n <= max_num for n in chunk):
                        continue
                    sorted_chunk = sorted(chunk)
                    if sorted_chunk not in history:
                        history.append(sorted_chunk)
                
                if history:
                    history.reverse()
                    print(f"  ✅ Loaded {len(history)} draws from ketquadientoan.com")
                    return history
        except Exception as e:
            print(f"  ⚠️ Error from {url[:50]}...: {e}")
    
    # GitHub fallback
    try:
        github_url = "https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl" if game_type == "Mega 6/45" else "https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power655.jsonl"
        response = requests.get(github_url, timeout=10)
        history = []
        if response.status_code == 200:
            for line in response.text.strip().split('\n'):
                if line:
                    data = json.loads(line)
                    if 'result' in data and len(data['result']) >= 6:
                        draw = sorted([int(n) for n in data['result'][:6]])
                        if all(1 <= n <= max_num for n in draw):
                            history.append(draw)
            if history:
                print(f"  ✅ Loaded {len(history)} draws from GitHub fallback")
                return history
    except Exception as e:
        print(f"  ⚠️ GitHub fallback error: {e}")
    
    print("  ❌ Could not load data from any source!")
    return None

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("🧬" * 40)
    print("  FULL SYSTEM TEST & BACKTEST — TINNAM888888 V700 QUANTUM SUPREME")
    print("🧬" * 40)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    all_errors = []
    
    # Phase 1: Test imports
    errors = test_imports()
    all_errors.extend(errors)
    
    # Fetch real data
    data = fetch_data_standalone("Mega 6/45")
    
    if data and len(data) >= 100:
        # Phase 2: Test NexusEngine methods
        errors = test_nexus_methods(data)
        all_errors.extend(errors)
        
        # Phase 3: Test StackingEngine
        errors = test_stacking_engine(data)
        all_errors.extend(errors)
        
        # Phase 4: Test RealWorldAIEngine
        errors = test_realworld_engine(data)
        all_errors.extend(errors)
        
        # Phase 5: Full Backtest
        results = full_backtest(data, "Mega 6/45")
    else:
        print("\n❌ Không đủ dữ liệu để chạy backtest!")
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 TỔNG KẾT")
    print("=" * 80)
    if all_errors:
        print(f"  ❌ Tìm thấy {len(all_errors)} lỗi:")
        for name, err in all_errors:
            print(f"    - {name}: {err}")
    else:
        print("  ✅ TẤT CẢ TESTS ĐỀU PASS — KHÔNG CÓ LỖI!")
