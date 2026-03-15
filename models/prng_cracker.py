"""
PRNG Cracker & Scientific Pattern Detector
============================================
If TinNam uses a PRNG (Pseudo-Random Number Generator) instead of true randomness,
these scientific methods can detect and potentially crack it.

Methods:
1. LCG Detection - Crack Linear Congruential Generator parameters (a, b, m)
2. FFT Spectral Analysis - Detect hidden periodicities via Fourier Transform
3. Autocorrelation - Detect repeating cycles in the number sequence
4. Runs Test - Statistical test for non-randomness
5. Serial Correlation - Test if consecutive numbers are correlated
6. Entropy Analysis - Measure information content (PRNG has lower entropy)
7. Kolmogorov-Smirnov Test - Compare distribution to uniform
8. Modular Pattern Search - Find modular arithmetic relationships
9. Neural Sequence Cracker - Train neural net to learn hidden algorithm
10. Difference Analysis - Analyze differences between consecutive draws
"""
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
import math
import warnings
warnings.filterwarnings('ignore')


class PRNGCracker:
    """Scientific analysis to detect if lottery numbers come from a PRNG."""
    
    def __init__(self, max_number, pick_count):
        self.max_number = max_number
        self.pick_count = pick_count
        self.results = {}
    
    def analyze(self, data):
        """Run ALL scientific analyses on the data."""
        self.data = data
        
        # Flatten all numbers into a single sequence for some tests
        self.flat_sequence = []
        for draw in data:
            self.flat_sequence.extend(sorted(draw))
        
        # Run all analyses
        print("[PRNG Cracker] Running 10 scientific analyses...")
        
        self.results['lcg'] = self._detect_lcg()
        print("  [1/10] LCG Detection complete")
        
        self.results['fft'] = self._fft_spectral()
        print("  [2/10] FFT Spectral Analysis complete")
        
        self.results['autocorrelation'] = self._autocorrelation()
        print("  [3/10] Autocorrelation complete")
        
        self.results['runs_test'] = self._runs_test()
        print("  [4/10] Runs Test complete")
        
        self.results['serial_correlation'] = self._serial_correlation()
        print("  [5/10] Serial Correlation complete")
        
        self.results['entropy'] = self._entropy_analysis()
        print("  [6/10] Entropy Analysis complete")
        
        self.results['ks_test'] = self._ks_test()
        print("  [7/10] KS Test complete")
        
        self.results['modular'] = self._modular_patterns()
        print("  [8/10] Modular Pattern Search complete")
        
        self.results['difference'] = self._difference_analysis()
        print("  [9/10] Difference Analysis complete")
        
        self.results['neural'] = self._neural_cracker()
        print("  [10/10] Neural Sequence Cracker complete")
        
        # Generate verdict
        self.results['verdict'] = self._generate_verdict()
        
        print("[PRNG Cracker] Analysis complete!")
        return self.results
    
    # ===== 1. LCG DETECTION =====
    def _detect_lcg(self):
        """
        Try to crack Linear Congruential Generator: X(n+1) = (a*X(n) + b) mod m
        If numbers come from LCG, we can find a, b, m.
        """
        seq = self.flat_sequence[:500]  # Use first 500 numbers
        
        # Try different moduli
        best_match = None
        best_score = 0
        
        for m in [self.max_number, self.max_number + 1, 
                  2**16, 2**31 - 1, 2**32, 127, 256]:
            # Try to find 'a' and 'b' from pairs of consecutive numbers
            # X1 = (a*X0 + b) mod m
            # X2 = (a*X1 + b) mod m
            # X2 - a*X1 = X1 - a*X0 (mod m)
            
            found_params = []
            
            for i in range(min(50, len(seq) - 2)):
                x0, x1, x2 = seq[i], seq[i+1], seq[i+2]
                
                # Try to solve for a: (x2 - x1) / (x1 - x0) mod m
                denom = (x1 - x0) % m
                numer = (x2 - x1) % m
                
                if denom != 0:
                    # Try all possible 'a' values
                    for a_try in range(1, min(m, 100)):
                        if (a_try * denom) % m == numer:
                            b_try = (x1 - a_try * x0) % m
                            found_params.append((a_try, b_try, m))
                            break
            
            if found_params:
                # Validate: check if these params predict subsequent numbers
                param_counts = Counter(found_params)
                most_common = param_counts.most_common(1)[0]
                a, b, m_val = most_common[0]
                count = most_common[1]
                
                # Verify on more data
                correct = 0
                total = 0
                for i in range(min(200, len(seq) - 1)):
                    predicted = (a * seq[i] + b) % m_val
                    if predicted == seq[i + 1]:
                        correct += 1
                    total += 1
                
                accuracy = correct / total if total > 0 else 0
                
                if accuracy > best_score:
                    best_score = accuracy
                    best_match = {
                        'a': int(a), 'b': int(b), 'm': int(m_val),
                        'accuracy': round(accuracy * 100, 2),
                        'formula': f'X(n+1) = ({a} * X(n) + {b}) mod {m_val}',
                        'consistent_pairs': int(count)
                    }
        
        is_lcg = best_score > 0.5  # More than 50% match = likely LCG
        
        return {
            'name': 'LCG Detection',
            'is_prng': is_lcg,
            'best_params': best_match,
            'confidence': round(best_score * 100, 2),
            'conclusion': 'PRNG DETECTED! LCG pattern found.' if is_lcg 
                         else 'No LCG pattern found. Numbers appear non-deterministic.'
        }
    
    # ===== 2. FFT SPECTRAL ANALYSIS =====
    def _fft_spectral(self):
        """Use Fast Fourier Transform to detect hidden periodicities."""
        seq = np.array(self.flat_sequence[:3000], dtype=float)
        
        # Normalize
        seq = (seq - np.mean(seq)) / (np.std(seq) + 1e-10)
        
        # FFT
        fft_vals = np.fft.fft(seq)
        magnitudes = np.abs(fft_vals[:len(seq)//2])
        frequencies = np.fft.fftfreq(len(seq))[:len(seq)//2]
        
        # Find dominant frequencies (peaks)
        mean_mag = np.mean(magnitudes[1:])  # Exclude DC component
        std_mag = np.std(magnitudes[1:])
        threshold = mean_mag + 3 * std_mag  # 3 sigma threshold
        
        peaks = []
        for i in range(1, len(magnitudes)):
            if magnitudes[i] > threshold:
                period = 1.0 / frequencies[i] if frequencies[i] != 0 else float('inf')
                peaks.append({
                    'frequency': round(float(frequencies[i]), 6),
                    'magnitude': round(float(magnitudes[i]), 2),
                    'period': round(float(period), 1),
                    'sigma': round(float((magnitudes[i] - mean_mag) / std_mag), 2)
                })
        
        peaks.sort(key=lambda x: -x['magnitude'])
        peaks = peaks[:10]  # Top 10 peaks
        
        has_periodicity = len(peaks) > 0
        max_sigma = max([p['sigma'] for p in peaks]) if peaks else 0
        
        return {
            'name': 'FFT Spectral Analysis',
            'is_prng': has_periodicity and max_sigma > 5,
            'peaks': peaks,
            'num_significant_peaks': len(peaks),
            'max_sigma': round(max_sigma, 2),
            'mean_magnitude': round(float(mean_mag), 2),
            'conclusion': f'Found {len(peaks)} significant frequency peaks (max {max_sigma:.1f}σ). '
                         + ('POSSIBLE hidden periodicity!' if max_sigma > 5 
                            else 'No strong periodicity detected.')
        }
    
    # ===== 3. AUTOCORRELATION =====
    def _autocorrelation(self):
        """Test autocorrelation at various lags to detect repeating patterns."""
        seq = np.array(self.flat_sequence[:2000], dtype=float)
        seq = seq - np.mean(seq)
        n = len(seq)
        
        # Test lags from 1 to 200
        max_lag = min(200, n // 4)
        correlations = []
        
        var = np.sum(seq ** 2)
        
        for lag in range(1, max_lag + 1):
            corr = np.sum(seq[:n-lag] * seq[lag:]) / var
            correlations.append({
                'lag': lag,
                'correlation': round(float(corr), 6)
            })
        
        # Find significant correlations (> 2/sqrt(n) threshold for 95% CI)
        threshold = 2.0 / np.sqrt(n)
        significant = [c for c in correlations if abs(c['correlation']) > threshold]
        
        # Check for periodic pattern
        strong_corr = [c for c in correlations if abs(c['correlation']) > threshold * 2]
        
        return {
            'name': 'Autocorrelation Analysis',
            'is_prng': len(strong_corr) > 5,
            'threshold_95pct': round(float(threshold), 6),
            'significant_lags': len(significant),
            'strong_correlations': strong_corr[:15],
            'top_correlations': sorted(correlations, key=lambda x: -abs(x['correlation']))[:10],
            'conclusion': f'{len(significant)} significant autocorrelations found. '
                         + ('PATTERN DETECTED in sequence!' if len(strong_corr) > 5
                            else 'Consistent with random data.')
        }
    
    # ===== 4. RUNS TEST =====
    def _runs_test(self):
        """
        Wald-Wolfowitz runs test for randomness.
        A 'run' is a maximal sequence of consecutive similar elements.
        """
        # Test each number position independently
        results_per_pos = []
        
        for pos in range(self.pick_count):
            values = [d[pos] if pos < len(d) else 0 for d in self.data]
            median = np.median(values)
            
            # Convert to binary: above/below median
            binary = [1 if v > median else 0 for v in values]
            
            # Count runs
            runs = 1
            for i in range(1, len(binary)):
                if binary[i] != binary[i-1]:
                    runs += 1
            
            n1 = sum(binary)
            n0 = len(binary) - n1
            n = len(binary)
            
            # Expected runs and std
            if n1 == 0 or n0 == 0:
                continue
            
            expected_runs = (2 * n1 * n0) / n + 1
            std_runs = math.sqrt((2 * n1 * n0 * (2 * n1 * n0 - n)) / (n * n * (n - 1)))
            
            if std_runs > 0:
                z_score = (runs - expected_runs) / std_runs
            else:
                z_score = 0
            
            results_per_pos.append({
                'position': pos + 1,
                'runs': runs,
                'expected': round(expected_runs, 1),
                'z_score': round(z_score, 3),
                'is_random': abs(z_score) < 1.96  # 95% CI
            })
        
        all_random = all(r['is_random'] for r in results_per_pos)
        avg_z = np.mean([abs(r['z_score']) for r in results_per_pos])
        
        return {
            'name': 'Runs Test (Wald-Wolfowitz)',
            'is_prng': not all_random,
            'results': results_per_pos,
            'avg_abs_z_score': round(float(avg_z), 3),
            'all_pass': all_random,
            'conclusion': ('ALL positions pass randomness test.' if all_random
                          else 'SOME positions FAIL randomness test - possible pattern!')
        }
    
    # ===== 5. SERIAL CORRELATION =====
    def _serial_correlation(self):
        """Test if consecutive draws are correlated."""
        correlations = {}
        
        for pos in range(self.pick_count):
            values = [d[pos] if pos < len(d) else 0 for d in self.data]
            
            if len(values) < 3:
                continue
            
            # Lag-1 correlation
            x = np.array(values[:-1], dtype=float)
            y = np.array(values[1:], dtype=float)
            
            if np.std(x) > 0 and np.std(y) > 0:
                corr = np.corrcoef(x, y)[0, 1]
            else:
                corr = 0
            
            correlations[f'pos_{pos+1}'] = round(float(corr), 6)
        
        # Cross-position correlation (does number in pos 1 predict pos 2?)
        cross_corr = {}
        for p1, p2 in combinations(range(self.pick_count), 2):
            v1 = [d[p1] if p1 < len(d) else 0 for d in self.data]
            v2 = [d[p2] if p2 < len(d) else 0 for d in self.data]
            
            if len(v1) > 2 and np.std(v1) > 0 and np.std(v2) > 0:
                c = np.corrcoef(v1, v2)[0, 1]
                cross_corr[f'pos{p1+1}_pos{p2+1}'] = round(float(c), 4)
        
        max_corr = max(abs(v) for v in correlations.values()) if correlations else 0
        threshold = 2.0 / np.sqrt(len(self.data))
        
        return {
            'name': 'Serial Correlation',
            'is_prng': max_corr > 0.1,
            'lag1_correlations': correlations,
            'cross_correlations': cross_corr,
            'max_correlation': round(max_corr, 6),
            'significance_threshold': round(float(threshold), 6),
            'conclusion': (f'Max serial correlation: {max_corr:.4f}. '
                          + ('SIGNIFICANT correlation found!' if max_corr > 0.1
                             else 'No significant serial correlation.'))
        }
    
    # ===== 6. ENTROPY ANALYSIS =====
    def _entropy_analysis(self):
        """
        Shannon entropy measurement. True random has maximum entropy.
        PRNG typically has slightly lower entropy.
        """
        # Entropy of individual numbers
        freq = Counter(self.flat_sequence)
        total = len(self.flat_sequence)
        
        probs = [count / total for count in freq.values()]
        shannon = -sum(p * math.log2(p) for p in probs if p > 0)
        max_entropy = math.log2(self.max_number)
        entropy_ratio = shannon / max_entropy
        
        # Entropy of number pairs (bigrams)
        bigrams = []
        for i in range(len(self.flat_sequence) - 1):
            bigrams.append((self.flat_sequence[i], self.flat_sequence[i+1]))
        
        bigram_freq = Counter(bigrams)
        bigram_total = len(bigrams)
        bigram_probs = [c / bigram_total for c in bigram_freq.values()]
        bigram_entropy = -sum(p * math.log2(p) for p in bigram_probs if p > 0)
        max_bigram_entropy = 2 * max_entropy  # Max for independent pairs
        bigram_ratio = bigram_entropy / max_bigram_entropy if max_bigram_entropy > 0 else 0
        
        # Entropy of draw sums
        sums = [sum(d) for d in self.data]
        sum_freq = Counter(sums)
        sum_total = len(sums)
        sum_probs = [c / sum_total for c in sum_freq.values()]
        sum_entropy = -sum(p * math.log2(p) for p in sum_probs if p > 0)
        
        return {
            'name': 'Entropy Analysis',
            'is_prng': entropy_ratio < 0.95,
            'shannon_entropy': round(shannon, 4),
            'max_entropy': round(max_entropy, 4),
            'entropy_ratio': round(entropy_ratio, 4),
            'bigram_entropy': round(bigram_entropy, 4),
            'bigram_ratio': round(bigram_ratio, 4),
            'sum_entropy': round(sum_entropy, 4),
            'conclusion': (f'Entropy ratio: {entropy_ratio:.4f} (1.0 = perfectly uniform). '
                          + ('LOW entropy - possible PRNG!' if entropy_ratio < 0.95
                             else 'High entropy consistent with true randomness.'))
        }
    
    # ===== 7. KOLMOGOROV-SMIRNOV TEST =====
    def _ks_test(self):
        """Compare number distribution to uniform distribution."""
        seq = np.array(self.flat_sequence, dtype=float)
        n = len(seq)
        
        # Empirical CDF
        sorted_seq = np.sort(seq)
        ecdf = np.arange(1, n + 1) / n
        
        # Theoretical CDF (uniform on [1, max_number])
        tcdf = (sorted_seq - 0.5) / self.max_number
        
        # KS statistic
        d_plus = np.max(ecdf - tcdf)
        d_minus = np.max(tcdf - ecdf + 1/n)
        ks_stat = max(d_plus, d_minus)
        
        # Critical value at 95% confidence
        critical_95 = 1.36 / math.sqrt(n)
        critical_99 = 1.63 / math.sqrt(n)
        
        passes = ks_stat < critical_95
        
        return {
            'name': 'Kolmogorov-Smirnov Test',
            'is_prng': not passes,
            'ks_statistic': round(float(ks_stat), 6),
            'critical_95': round(float(critical_95), 6),
            'critical_99': round(float(critical_99), 6),
            'passes_95': passes,
            'passes_99': ks_stat < critical_99,
            'conclusion': (f'KS = {ks_stat:.6f}, Critical(95%) = {critical_95:.6f}. '
                          + ('FAILS uniformity test!' if not passes
                             else 'Distribution consistent with uniform random.'))
        }
    
    # ===== 8. MODULAR PATTERN SEARCH =====
    def _modular_patterns(self):
        """Search for modular arithmetic patterns: a*x + b ≡ y (mod m)."""
        seq = self.flat_sequence[:1000]
        patterns_found = []
        
        # Test various moduli
        for mod in [2, 3, 5, 7, 9, 10, 11, 13, 45, 55]:
            residues = [x % mod for x in seq]
            
            # Check if residues follow a pattern
            res_freq = Counter(residues)
            expected = len(seq) / mod
            
            # Chi-square for uniformity of residues
            chi_sq = sum((count - expected) ** 2 / expected 
                        for count in res_freq.values())
            df = mod - 1
            
            # Rough p-value check (chi-square critical values)
            # For df=1: 3.84, df=2: 5.99, df=4: 9.49, df=6: 12.59
            critical = df * 2  # Rough approximation
            
            is_significant = chi_sq > critical * 1.5
            
            if is_significant:
                patterns_found.append({
                    'modulus': mod,
                    'chi_square': round(chi_sq, 2),
                    'critical': round(critical, 2),
                    'distribution': {str(k): v for k, v in sorted(res_freq.items())},
                    'significant': True
                })
        
        # Check for differences following mod pattern
        diffs = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
        diff_mods = {}
        for mod in [3, 5, 7, 10]:
            diff_residues = [d % mod for d in diffs]
            diff_mods[str(mod)] = dict(Counter(diff_residues))
        
        return {
            'name': 'Modular Pattern Search',
            'is_prng': len(patterns_found) > 2,
            'patterns': patterns_found,
            'diff_mod_patterns': diff_mods,
            'conclusion': (f'Found {len(patterns_found)} significant modular patterns. '
                          + ('MODULAR PATTERNS DETECTED!' if len(patterns_found) > 2
                             else 'No significant modular patterns.'))
        }
    
    # ===== 9. DIFFERENCE ANALYSIS =====
    def _difference_analysis(self):
        """Analyze differences and ratios between consecutive draws."""
        # Sum of each draw
        sums = [sum(d) for d in self.data]
        
        # First differences
        diff1 = [sums[i+1] - sums[i] for i in range(len(sums)-1)]
        # Second differences
        diff2 = [diff1[i+1] - diff1[i] for i in range(len(diff1)-1)]
        
        # Analyze difference distribution
        diff1_mean = float(np.mean(diff1))
        diff1_std = float(np.std(diff1))
        diff2_mean = float(np.mean(diff2))
        diff2_std = float(np.std(diff2))
        
        # Check if differences follow a pattern
        # Autocorrelation of differences
        d1 = np.array(diff1, dtype=float)
        d1_norm = d1 - np.mean(d1)
        var_d1 = np.sum(d1_norm ** 2)
        
        diff_autocorr = []
        for lag in range(1, min(30, len(d1) // 2)):
            if var_d1 > 0:
                corr = np.sum(d1_norm[:len(d1)-lag] * d1_norm[lag:]) / var_d1
                diff_autocorr.append({'lag': lag, 'correlation': round(float(corr), 4)})
        
        max_diff_corr = max(abs(c['correlation']) for c in diff_autocorr) if diff_autocorr else 0
        
        # Ratio analysis
        ratios = []
        for i in range(len(sums) - 1):
            if sums[i] > 0:
                ratios.append(sums[i+1] / sums[i])
        
        ratio_std = float(np.std(ratios)) if ratios else 0
        
        return {
            'name': 'Difference Analysis',
            'is_prng': max_diff_corr > 0.15,
            'diff1_stats': {'mean': round(diff1_mean, 2), 'std': round(diff1_std, 2)},
            'diff2_stats': {'mean': round(diff2_mean, 2), 'std': round(diff2_std, 2)},
            'diff_autocorrelations': diff_autocorr[:10],
            'max_diff_autocorr': round(max_diff_corr, 4),
            'ratio_std': round(ratio_std, 4),
            'conclusion': (f'Max difference autocorrelation: {max_diff_corr:.4f}. '
                          + ('PATTERN in differences detected!' if max_diff_corr > 0.15
                             else 'Differences appear random.'))
        }
    
    # ===== 10. NEURAL SEQUENCE CRACKER =====
    def _neural_cracker(self):
        """
        Simple neural network trying to learn the hidden algorithm.
        Uses numpy only (no TensorFlow needed).
        A 2-layer perceptron that tries to predict next draw from previous draws.
        """
        # Prepare data: input = last 3 draws flattened, output = next draw
        window = 3
        X = []
        Y = []
        
        for i in range(window, len(self.data)):
            features = []
            for j in range(window):
                features.extend(self.data[i - window + j])
            X.append(features)
            Y.append(self.data[i])
        
        if len(X) < 100:
            return {
                'name': 'Neural Sequence Cracker',
                'is_prng': False,
                'conclusion': 'Not enough data for neural analysis.'
            }
        
        X = np.array(X, dtype=float) / self.max_number  # Normalize
        Y = np.array(Y, dtype=float) / self.max_number
        
        # Split
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        Y_train, Y_test = Y[:split], Y[split:]
        
        # Simple 2-layer neural network
        np.random.seed(42)
        input_size = window * self.pick_count
        hidden_size = 32
        output_size = self.pick_count
        
        W1 = np.random.randn(input_size, hidden_size) * 0.1
        b1 = np.zeros(hidden_size)
        W2 = np.random.randn(hidden_size, output_size) * 0.1
        b2 = np.zeros(output_size)
        
        lr = 0.01
        
        # Train
        train_losses = []
        for epoch in range(200):
            # Forward
            z1 = X_train @ W1 + b1
            a1 = np.maximum(z1, 0)  # ReLU
            z2 = a1 @ W2 + b2
            pred = z2  # Linear output
            
            # Loss
            loss = np.mean((pred - Y_train) ** 2)
            train_losses.append(float(loss))
            
            # Backward
            d_loss = 2 * (pred - Y_train) / len(X_train)
            dW2 = a1.T @ d_loss
            db2 = np.sum(d_loss, axis=0)
            d_a1 = d_loss @ W2.T
            d_z1 = d_a1 * (z1 > 0)
            dW1 = X_train.T @ d_z1
            db1 = np.sum(d_z1, axis=0)
            
            W2 -= lr * dW2
            b2 -= lr * db2
            W1 -= lr * dW1
            b1 -= lr * db1
        
        # Test
        z1 = X_test @ W1 + b1
        a1 = np.maximum(z1, 0)
        test_pred = (a1 @ W2 + b2) * self.max_number
        test_actual = Y_test * self.max_number
        
        # Calculate accuracy: average number of correct predictions
        matches = []
        for i in range(len(test_pred)):
            pred_nums = set(np.round(test_pred[i]).astype(int).clip(1, self.max_number))
            actual_nums = set(np.round(test_actual[i]).astype(int))
            matches.append(len(pred_nums & actual_nums))
        
        avg_matches = float(np.mean(matches))
        max_matches = int(np.max(matches))
        
        # Random baseline for comparison
        random_expected = self.pick_count * self.pick_count / self.max_number
        
        return {
            'name': 'Neural Sequence Cracker',
            'is_prng': avg_matches > random_expected * 1.5,
            'avg_matches': round(avg_matches, 3),
            'max_matches': max_matches,
            'random_expected': round(random_expected, 3),
            'improvement_over_random': round((avg_matches / random_expected - 1) * 100, 1) if random_expected > 0 else 0,
            'final_train_loss': round(train_losses[-1], 6),
            'initial_train_loss': round(train_losses[0], 6),
            'loss_reduction': round((1 - train_losses[-1] / train_losses[0]) * 100, 1) if train_losses[0] > 0 else 0,
            'test_size': len(X_test),
            'conclusion': (f'Neural net avg matches: {avg_matches:.3f}/6 '
                          f'(random: {random_expected:.3f}). '
                          + ('NEURAL NET LEARNED PATTERN!' if avg_matches > random_expected * 1.5
                             else 'Neural net cannot learn pattern beyond random.'))
        }
    
    # ===== VERDICT =====
    def _generate_verdict(self):
        """Generate final verdict based on all analyses."""
        prng_indicators = 0
        total_tests = 0
        evidence = []
        
        for name, result in self.results.items():
            if name == 'verdict':
                continue
            if isinstance(result, dict) and 'is_prng' in result:
                total_tests += 1
                if result['is_prng']:
                    prng_indicators += 1
                    evidence.append(f"+ {result['name']}: {result.get('conclusion', 'Pattern detected')}")
                else:
                    evidence.append(f"- {result['name']}: {result.get('conclusion', 'No pattern')}")
        
        prng_score = round(prng_indicators / total_tests * 100, 1) if total_tests > 0 else 0
        
        if prng_score >= 60:
            verdict = "STRONG EVIDENCE of non-random generation (possible PRNG)"
        elif prng_score >= 30:
            verdict = "WEAK EVIDENCE of patterns (inconclusive)"
        else:
            verdict = "NO EVIDENCE of PRNG - data appears truly random"
        
        return {
            'prng_score': prng_score,
            'prng_indicators': prng_indicators,
            'total_tests': total_tests,
            'verdict': verdict,
            'evidence': evidence
        }
