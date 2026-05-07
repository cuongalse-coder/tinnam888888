import os

with open('models/mega_exploit_v12.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace header
content = content.replace('MEGA EXPLOIT ENGINE V12.0 — QUANTUM SUPREMACY EDITION', 'MEGA EXPLOIT ENGINE V15.0 — NEURAL QUANTUM (30 SIGNALS + ELO RATING)')
content = content.replace('24 AI SIGNALS', '30 AI SIGNALS')
content = content.replace('MegaExploitV12', 'MegaExploitV15')
content = content.replace('V12.0: Quantum Supremacy — 24 signals', 'V15.0: Neural Quantum — 30 signals')

# Add new signals in Phase 1
phase_1_add = """        sd['poisson_spike'] = self._sig_poisson_spike(data)
        sd['markov_steady'] = self._sig_markov_steady(data)
        sd['golden_ratio'] = self._sig_golden_ratio(data)
        sd['std_reversion'] = self._sig_std_reversion(data)
        sd['acf_peaks'] = self._sig_acf_peaks(data)
        sd['bayesian_post'] = self._sig_bayesian_posterior(data)
"""
content = content.replace("sd['kmeans'] = self._sig_kmeans_closeness(data)", "sd['kmeans'] = self._sig_kmeans_closeness(data)\n" + phase_1_add)

# Replace strategy name
content = content.replace("'strategy': 'v12_quantum_optimal'", "'strategy': 'v15_neural_optimal'")
content = content.replace("'strategy': 'v12_quantum_diversified'", "'strategy': 'v15_neural_diversified'")
content = content.replace("'strategy': 'MegaExploitV12.0'", "'strategy': 'MegaExploitV15.0'")

# Replace _calibrate_walkforward with Elo Rating
elo_replacement = """    def _calibrate_walkforward(self, data, signal_details):
        \"\"\"Dynamic Elo-like Rating: exponential weighting over the last 50 draws.\"\"\"
        n = len(data)
        test_size = min(50, n - 70)
        if test_size < 10:
            return {name: 1.0 for name in signal_details}

        hits = {name: 0.0 for name in signal_details}
        tc = 0

        for idx in range(n - test_size - 1, n - 1):
            actual = set(data[idx + 1][:self.pick_count])
            tc += 1
            decay = math.exp((tc - test_size) / 10.0) # More weight to recent draws
            for sig_name, sig_scores in signal_details.items():
                if not sig_scores:
                    continue
                top = set(num for num, _ in sorted(sig_scores.items(), key=lambda x: -x[1])[:self.pick_count])
                match_cnt = len(top & actual)
                hits[sig_name] += match_cnt * decay

        base_match = self.pick_count * (self.pick_count / self.max_number)
        expected_total = sum(math.exp((i + 1 - test_size) / 10.0) for i in range(tc)) * base_match
        
        weights = {}
        for name in signal_details:
            if expected_total > 0 and hits[name] > 0:
                lift = hits[name] / expected_total
                weights[name] = max(lift, 0.1)
            else:
                weights[name] = 0.1
        return weights"""
        
import re
content = re.sub(r'    def _calibrate_walkforward.*?return weights', elo_replacement, content, flags=re.DOTALL)

# Add new signal definitions
new_sigs = """
    # === NEW SIGNALS FOR V15.0 NEURAL QUANTUM ===
    
    def _sig_poisson_spike(self, data):
        scores = {}
        lam = len(data) * self.pick_count / self.max_number
        import math
        for num in range(1, self.max_number + 1):
            k = sum(1 for d in data if num in d[:self.pick_count])
            try:
                p = (math.exp(-lam) * (lam**k)) / math.factorial(k)
            except:
                p = 0
            scores[num] = -math.log10(p + 1e-10) if p < 0.05 else 0
        return scores

    def _sig_markov_steady(self, data):
        scores = {n: 0 for n in range(1, self.max_number + 1)}
        try:
            import numpy as np
            trans = np.zeros((self.max_number, self.max_number))
            for i in range(1, len(data)):
                for p in data[i-1][:self.pick_count]:
                    for n in data[i][:self.pick_count]:
                        trans[p-1, n-1] += 1
            row_sums = trans.sum(axis=1, keepdims=True)
            trans = np.divide(trans, row_sums, out=np.zeros_like(trans), where=row_sums!=0)
            curr = np.zeros(self.max_number)
            for p in data[-1][:self.pick_count]: curr[p-1] = 1.0/self.pick_count
            for _ in range(3): curr = curr.dot(trans)
            for i, val in enumerate(curr): scores[i+1] = val * 10.0
        except: pass
        return scores

    def _sig_golden_ratio(self, data):
        golden = 1.61803398875
        scores = {n: 0 for n in range(1, self.max_number + 1)}
        targets = [self.max_number / golden, self.max_number - (self.max_number / golden)]
        for n in range(1, self.max_number + 1):
            if any(abs(n - t) < 1.5 for t in targets):
                scores[n] = 1.5 if n not in data[-1][:self.pick_count] else 0.0
        return scores

    def _sig_std_reversion(self, data):
        scores = {}
        all_nums = [n for d in data for n in d[:self.pick_count]]
        from collections import Counter
        freq = Counter(all_nums)
        vals = list(freq.values())
        mean = sum(vals)/max(len(vals), 1)
        std = (sum((v - mean)**2 for v in vals)/max(len(vals), 1))**0.5
        for num in range(1, self.max_number + 1):
            c = freq.get(num, 0)
            z = (c - mean) / (std + 1e-5)
            scores[num] = -z * 1.5 if abs(z) > 1.5 else 0
        return scores

    def _sig_acf_peaks(self, data):
        scores = {n: 0 for n in range(1, self.max_number + 1)}
        try:
            import numpy as np
            for num in range(1, self.max_number + 1):
                seq = np.array([1 if num in d[:self.pick_count] else 0 for d in data[-100:]])
                if seq.sum() < 5: continue
                acfs = []
                for lag in range(1, 11):
                    a = seq[:-lag]
                    b = seq[lag:]
                    acfs.append(np.corrcoef(a, b)[0, 1] if np.std(a)>0 and np.std(b)>0 else 0)
                best_lag = np.argmax(acfs) + 1
                last_seen = len(data) - 1 - max([i for i, d in enumerate(data) if num in d[:self.pick_count]])
                if last_seen == best_lag:
                    scores[num] = 2.0
        except: pass
        return scores

    def _sig_bayesian_posterior(self, data):
        scores = {}
        lam = self.pick_count / self.max_number
        for num in range(1, self.max_number + 1):
            prior = lam
            likelihood = sum(1 for d in data[-20:] if num in d[:self.pick_count]) / 20.0
            posterior = (likelihood * prior) / (likelihood * prior + (1-likelihood)*(1-prior) + 1e-10)
            scores[num] = posterior * 5.0
        return scores
"""
content = content.replace('    # ================================================================', new_sigs + '\n    # ================================================================', 1)

with open('models/mega_exploit_v15.py', 'w', encoding='utf-8') as f:
    f.write(content)
