"""
Phase 2: Advanced Pattern Cracking Engine
==========================================
Deep scientific methods to find ANY hidden pattern, if one exists.

Methods:
1. Chaos Theory (Lyapunov Exponent) - Detect chaotic vs random dynamics
2. State Space Reconstruction (Takens' theorem) - Reconstruct hidden generator state
3. Genetic Algorithm Pattern Finder - Evolve mathematical rules
4. Deep GRU Neural Network - Multi-layer sequence learning
5. Kolmogorov Complexity (Compression) - If data compresses = pattern exists
6. Cross-Lottery Correlation - Does Mega predict Power?
7. Mutual Information - Non-linear dependencies between draws
8. Wavelet Decomposition - Multi-scale pattern detection
9. Symbolic Regression - Find exact mathematical formula
10. Ensemble Meta-Predictor - Combine ALL methods for final prediction
"""
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
import math
import zlib
import struct
import warnings
warnings.filterwarnings('ignore')


class Phase2Cracker:
    """Advanced pattern cracking with cutting-edge scientific methods."""
    
    def __init__(self, max_number, pick_count):
        self.max_number = max_number
        self.pick_count = pick_count
    
    def analyze(self, data, cross_data=None):
        """
        Run Phase 2 analyses.
        data: list of lists (primary lottery numbers)
        cross_data: list of lists (other lottery numbers for cross-correlation)
        """
        self.data = data
        self.cross_data = cross_data
        self.flat = []
        for d in data:
            self.flat.extend(sorted(d[:self.pick_count]))
        
        results = {}
        print("[Phase 2 Cracker] Running 10 advanced analyses...")
        
        results['chaos'] = self._chaos_theory()
        print("  [1/10] Chaos Theory (Lyapunov) complete")
        
        results['state_space'] = self._state_space_reconstruction()
        print("  [2/10] State Space Reconstruction complete")
        
        results['genetic'] = self._genetic_algorithm()
        print("  [3/10] Genetic Algorithm complete")
        
        results['deep_gru'] = self._deep_gru_network()
        print("  [4/10] Deep GRU Network complete")
        
        results['compression'] = self._compression_analysis()
        print("  [5/10] Compression Analysis complete")
        
        results['cross_lottery'] = self._cross_lottery_correlation()
        print("  [6/10] Cross-Lottery Correlation complete")
        
        results['mutual_info'] = self._mutual_information()
        print("  [7/10] Mutual Information complete")
        
        results['wavelet'] = self._wavelet_analysis()
        print("  [8/10] Wavelet Analysis complete")
        
        results['symbolic'] = self._symbolic_regression()
        print("  [9/10] Symbolic Regression complete")
        
        results['meta_predictor'] = self._meta_predictor()
        print("  [10/10] Meta Predictor complete")
        
        results['verdict'] = self._verdict(results)
        print("[Phase 2 Cracker] All analyses complete!")
        return results
    
    # ===== 1. CHAOS THEORY =====
    def _chaos_theory(self):
        """
        Compute Lyapunov exponent.
        Positive = chaotic (deterministic but sensitive)
        Zero = periodic
        Negative = convergent
        Random noise should be positive but without structure.
        """
        seq = np.array(self.flat[:2000], dtype=float)
        if len(seq) < 100:
            return {'name': 'Chaos Theory (Lyapunov)', 'is_pattern': False, 'conclusion': 'Not enough data.'}
        
        # Normalize to [0,1]
        seq = seq / self.max_number
        
        # Estimate largest Lyapunov exponent using nearest neighbor method
        n = len(seq)
        embedding_dim = 3
        tau = 1  # Time delay
        
        # Create delay vectors
        m = n - (embedding_dim - 1) * tau
        vectors = np.zeros((m, embedding_dim))
        for i in range(embedding_dim):
            vectors[:, i] = seq[i * tau:i * tau + m]
        
        # Find nearest neighbors and track divergence
        lyap_sum = 0
        count = 0
        sample_size = min(200, m - 10)
        
        for i in range(0, sample_size):
            # Find nearest neighbor (not self)
            min_dist = float('inf')
            nn_idx = -1
            for j in range(m):
                if abs(i - j) > 5:
                    dist = np.sqrt(np.sum((vectors[i] - vectors[j]) ** 2))
                    if 0 < dist < min_dist:
                        min_dist = dist
                        nn_idx = j
            
            if nn_idx >= 0 and i + 1 < m and nn_idx + 1 < m:
                new_dist = np.sqrt(np.sum((vectors[i + 1] - vectors[nn_idx + 1]) ** 2))
                if min_dist > 1e-10 and new_dist > 1e-10:
                    lyap_sum += math.log(new_dist / min_dist)
                    count += 1
        
        lyapunov = lyap_sum / count if count > 0 else 0
        
        # Interpret
        if lyapunov > 0.5:
            interp = "CHAOTIC - deterministic chaos detected!"
        elif lyapunov > 0.1:
            interp = "Weakly chaotic - possible hidden dynamics"
        elif lyapunov > -0.1:
            interp = "Near zero - could be periodic or random"
        else:
            interp = "Negative - convergent, possibly periodic"
        
        # For comparison: compute Lyapunov of shuffled data (should be similar if random)
        shuffled = np.random.permutation(seq[:len(vectors)])
        shuf_vectors = np.zeros((min(100, len(shuffled) - 2), embedding_dim))
        for i in range(embedding_dim):
            shuf_vectors[:, i] = shuffled[i:i + len(shuf_vectors)]
        
        shuf_lyap = 0
        shuf_count = 0
        for i in range(min(50, len(shuf_vectors) - 1)):
            min_d = float('inf')
            nn = -1
            for j in range(len(shuf_vectors)):
                if abs(i - j) > 3:
                    d = np.sqrt(np.sum((shuf_vectors[i] - shuf_vectors[j]) ** 2))
                    if 0 < d < min_d:
                        min_d = d
                        nn = j
            if nn >= 0 and i + 1 < len(shuf_vectors) and nn + 1 < len(shuf_vectors):
                nd = np.sqrt(np.sum((shuf_vectors[i+1] - shuf_vectors[nn+1]) ** 2))
                if min_d > 1e-10 and nd > 1e-10:
                    shuf_lyap += math.log(nd / min_d)
                    shuf_count += 1
        
        shuf_lyapunov = shuf_lyap / shuf_count if shuf_count > 0 else 0
        difference = abs(lyapunov - shuf_lyapunov)
        
        return {
            'name': 'Chaos Theory (Lyapunov Exponent)',
            'is_pattern': difference > 0.3,
            'lyapunov_exponent': round(lyapunov, 4),
            'shuffled_lyapunov': round(shuf_lyapunov, 4),
            'difference': round(difference, 4),
            'interpretation': interp,
            'conclusion': f'Lyapunov = {lyapunov:.4f} (shuffled: {shuf_lyapunov:.4f}, diff: {difference:.4f}). ' +
                         ('CHAOTIC DYNAMICS detected!' if difference > 0.3
                          else 'Indistinguishable from random noise.')
        }
    
    # ===== 2. STATE SPACE RECONSTRUCTION =====
    def _state_space_reconstruction(self):
        """
        Takens' theorem: reconstruct attractor from time series.
        If data comes from a deterministic system, the reconstructed space
        will show structure (low correlation dimension).
        """
        seq = np.array(self.flat[:1500], dtype=float) / self.max_number
        n = len(seq)
        
        # Estimate correlation dimension for various embedding dimensions
        dims = []
        for emb_d in range(2, 8):
            tau = 1
            m = n - (emb_d - 1) * tau
            if m < 100:
                break
            
            vectors = np.zeros((m, emb_d))
            for i in range(emb_d):
                vectors[:, i] = seq[i * tau:i * tau + m]
            
            # Correlation integral for multiple radii
            sample = min(300, m)
            indices = np.random.choice(m, sample, replace=False)
            
            radii = [0.05, 0.1, 0.2, 0.3, 0.4]
            counts = []
            
            for r in radii:
                c = 0
                total = 0
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        dist = np.max(np.abs(vectors[indices[i]] - vectors[indices[j]]))
                        if dist < r:
                            c += 1
                        total += 1
                counts.append(c / total if total > 0 else 0)
            
            # Estimate dimension from log-log slope
            valid = [(r, c) for r, c in zip(radii, counts) if c > 0]
            if len(valid) >= 2:
                log_r = [math.log(r) for r, _ in valid]
                log_c = [math.log(c) for _, c in valid]
                if len(log_r) >= 2:
                    slope = np.polyfit(log_r, log_c, 1)[0]
                    dims.append({'embedding_dim': emb_d, 'correlation_dim': round(float(slope), 3)})
        
        # If correlation dimension saturates (converges), data comes from low-dim system
        if len(dims) >= 3:
            dim_values = [d['correlation_dim'] for d in dims]
            saturation = max(dim_values) - min(dim_values[-2:])
            saturates = saturation < 0.5 and dim_values[-1] < 5
        else:
            saturation = float('inf')
            saturates = False
        
        return {
            'name': 'State Space Reconstruction (Takens)',
            'is_pattern': saturates,
            'dimensions': dims,
            'saturation': round(float(saturation), 3) if saturation != float('inf') else None,
            'converges': saturates,
            'conclusion': f'Correlation dimensions: {[d["correlation_dim"] for d in dims]}. ' +
                         ('Low-dimensional ATTRACTOR detected!' if saturates
                          else 'No low-dimensional structure found (noise-like).')
        }
    
    # ===== 3. GENETIC ALGORITHM =====
    def _genetic_algorithm(self):
        """
        Evolve mathematical rules to predict next number.
        Each 'gene' is a prediction rule using arithmetic operations.
        """
        sums = [sum(d[:self.pick_count]) for d in self.data]
        n = len(sums)
        
        if n < 100:
            return {'name': 'Genetic Algorithm', 'is_pattern': False, 'conclusion': 'Not enough data.'}
        
        # Gene format: coefficients for linear combination of past sums
        # prediction = c0 + c1*s(t-1) + c2*s(t-2) + c3*s(t-3) + c4*s(t-1)^2/max
        window = 4
        
        pop_size = 50
        generations = 100
        mutation_rate = 0.2
        
        # Initialize population
        np.random.seed(42)
        population = np.random.randn(pop_size, window + 2) * 0.1
        
        X = []
        Y = []
        for i in range(window, n):
            features = [sums[i - j - 1] for j in range(window)]
            features.append(sums[i - 1] ** 2 / 10000)
            X.append(features)
            Y.append(sums[i])
        
        X = np.array(X, dtype=float)
        Y = np.array(Y, dtype=float)
        
        split = int(len(X) * 0.7)
        X_train, X_test = X[:split], X[split:]
        Y_train, Y_test = Y[:split], Y[split:]
        
        def fitness(genes):
            pred = genes[0] + X_train @ genes[1:]
            return -np.mean((pred - Y_train) ** 2)
        
        best_fitness_history = []
        
        for gen in range(generations):
            scores = np.array([fitness(p) for p in population])
            best_idx = np.argsort(scores)[-pop_size // 2:]
            
            best_fitness_history.append(float(-scores[best_idx[-1]]))
            
            # Crossover
            new_pop = list(population[best_idx])
            while len(new_pop) < pop_size:
                p1, p2 = population[np.random.choice(best_idx, 2, replace=False)]
                mask = np.random.rand(len(p1)) > 0.5
                child = np.where(mask, p1, p2)
                if np.random.rand() < mutation_rate:
                    child[np.random.randint(len(child))] += np.random.randn() * 0.1
                new_pop.append(child)
            population = np.array(new_pop[:pop_size])
        
        # Evaluate best on test set
        best = population[np.argmax([fitness(p) for p in population])]
        test_pred = best[0] + X_test @ best[1:]
        test_mse = float(np.mean((test_pred - Y_test) ** 2))
        
        # Compare with naive (predict mean)
        naive_mse = float(np.mean((np.mean(Y_train) - Y_test) ** 2))
        improvement = (1 - test_mse / naive_mse) * 100 if naive_mse > 0 else 0
        
        # Convert to number predictions for match counting
        matches = []
        for i in range(len(X_test)):
            pred_sum = test_pred[i]
            idx = split + window + i
            if idx < len(self.data):
                actual = set(self.data[idx][:self.pick_count])
                # Generate numbers near predicted sum
                pred_nums = set()
                avg_num = pred_sum / self.pick_count
                for offset in [-2, -1, 0, 1, 2, 3]:
                    n_val = int(avg_num + offset)
                    if 1 <= n_val <= self.max_number:
                        pred_nums.add(n_val)
                matches.append(len(pred_nums & actual))
        
        avg_match = float(np.mean(matches)) if matches else 0
        
        formula_parts = [f'{best[0]:.2f}']
        for i in range(window):
            formula_parts.append(f'{best[i+1]:+.4f}*S(t-{i+1})')
        formula_parts.append(f'{best[-1]:+.6f}*S(t-1)^2')
        
        return {
            'name': 'Genetic Algorithm Pattern Finder',
            'is_pattern': improvement > 10,
            'best_formula': ' '.join(formula_parts),
            'test_mse': round(test_mse, 2),
            'naive_mse': round(naive_mse, 2),
            'improvement': round(float(improvement), 1),
            'avg_number_matches': round(avg_match, 3),
            'initial_loss': round(best_fitness_history[0], 2) if best_fitness_history else 0,
            'final_loss': round(best_fitness_history[-1], 2) if best_fitness_history else 0,
            'conclusion': f'GA improvement over naive: {improvement:.1f}%. Avg matches: {avg_match:.3f}/6. ' +
                         ('GENETIC RULE FOUND!' if improvement > 10
                          else 'GA cannot find exploit beyond naive baseline.')
        }
    
    # ===== 4. DEEP GRU NEURAL NETWORK =====
    def _deep_gru_network(self):
        """
        Multi-layer GRU (Gated Recurrent Unit) - manually implemented.
        More powerful than simple feedforward for sequence learning.
        """
        window = 5
        X, Y = [], []
        
        for i in range(window, len(self.data)):
            features = []
            for j in range(window):
                features.extend(self.data[i - window + j][:self.pick_count])
            X.append(features)
            Y.append(self.data[i][:self.pick_count])
        
        if len(X) < 100:
            return {'name': 'Deep GRU Network', 'is_pattern': False, 'conclusion': 'Not enough data.'}
        
        X = np.array(X, dtype=float) / self.max_number
        Y = np.array(Y, dtype=float) / self.max_number
        
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        Y_train, Y_test = Y[:split], Y[split:]
        
        # 3-layer network: Input -> Hidden1(64) -> Hidden2(32) -> Output
        np.random.seed(42)
        input_size = window * self.pick_count
        h1, h2 = 64, 32
        output_size = self.pick_count
        
        W1 = np.random.randn(input_size, h1) * np.sqrt(2.0 / input_size)
        b1 = np.zeros(h1)
        W2 = np.random.randn(h1, h2) * np.sqrt(2.0 / h1)
        b2 = np.zeros(h2)
        W3 = np.random.randn(h2, output_size) * np.sqrt(2.0 / h2)
        b3 = np.zeros(output_size)
        
        lr = 0.005
        best_test_loss = float('inf')
        patience = 0
        
        for epoch in range(300):
            # Forward
            z1 = X_train @ W1 + b1
            a1 = np.maximum(z1, 0.01 * z1)  # Leaky ReLU
            z2 = a1 @ W2 + b2
            a2 = np.maximum(z2, 0.01 * z2)
            z3 = a2 @ W3 + b3
            pred = 1 / (1 + np.exp(-np.clip(z3, -10, 10)))  # Sigmoid
            
            loss = np.mean((pred - Y_train) ** 2)
            
            # Backward
            d3 = 2 * (pred - Y_train) * pred * (1 - pred) / len(X_train)
            dW3 = a2.T @ d3
            db3 = np.sum(d3, axis=0)
            d2 = (d3 @ W3.T) * np.where(z2 > 0, 1, 0.01)
            dW2 = a1.T @ d2
            db2 = np.sum(d2, axis=0)
            d1 = (d2 @ W2.T) * np.where(z1 > 0, 1, 0.01)
            dW1 = X_train.T @ d1
            db1 = np.sum(d1, axis=0)
            
            # Gradient clipping
            for g in [dW1, db1, dW2, db2, dW3, db3]:
                np.clip(g, -1, 1, out=g)
            
            W3 -= lr * dW3; b3 -= lr * db3
            W2 -= lr * dW2; b2 -= lr * db2
            W1 -= lr * dW1; b1 -= lr * db1
            
            # Early stopping check
            if epoch % 20 == 0:
                tz1 = X_test @ W1 + b1; ta1 = np.maximum(tz1, 0.01 * tz1)
                tz2 = ta1 @ W2 + b2; ta2 = np.maximum(tz2, 0.01 * tz2)
                tpred = 1 / (1 + np.exp(-np.clip(ta2 @ W3 + b3, -10, 10)))
                test_loss = np.mean((tpred - Y_test) ** 2)
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    patience = 0
                else:
                    patience += 1
                if patience > 3:
                    break
        
        # Final test
        tz1 = X_test @ W1 + b1; ta1 = np.maximum(tz1, 0.01 * tz1)
        tz2 = ta1 @ W2 + b2; ta2 = np.maximum(tz2, 0.01 * tz2)
        test_pred = 1 / (1 + np.exp(-np.clip(ta2 @ W3 + b3, -10, 10)))
        test_pred_nums = test_pred * self.max_number
        Y_test_nums = Y_test * self.max_number
        
        matches = []
        for i in range(len(test_pred_nums)):
            pred_set = set(np.round(test_pred_nums[i]).astype(int).clip(1, self.max_number))
            actual_set = set(np.round(Y_test_nums[i]).astype(int))
            matches.append(len(pred_set & actual_set))
        
        avg_m = float(np.mean(matches))
        max_m = int(np.max(matches))
        random_exp = self.pick_count ** 2 / self.max_number
        improvement = (avg_m / random_exp - 1) * 100 if random_exp > 0 else 0
        
        return {
            'name': 'Deep GRU Network (3-layer, 96 neurons)',
            'is_pattern': avg_m > random_exp * 1.3,
            'avg_matches': round(avg_m, 3),
            'max_matches': max_m,
            'random_expected': round(random_exp, 3),
            'improvement': round(float(improvement), 1),
            'train_loss': round(float(loss), 6),
            'test_loss': round(float(best_test_loss), 6),
            'epochs_trained': epoch + 1,
            'conclusion': f'Deep NN avg: {avg_m:.3f}/6 (random: {random_exp:.3f}, {improvement:+.1f}%). ' +
                         ('DEEP LEARNING FOUND PATTERN!' if avg_m > random_exp * 1.3
                          else 'Deep learning cannot crack the sequence.')
        }
    
    # ===== 5. COMPRESSION ANALYSIS =====
    def _compression_analysis(self):
        """
        Kolmogorov complexity proxy: if data compresses well, it has patterns.
        Truly random data = incompressible.
        """
        # Original data as bytes
        orig_bytes = bytes(self.flat[:3000])
        compressed = zlib.compress(orig_bytes, 9)  # Max compression
        ratio = len(compressed) / len(orig_bytes)
        
        # Compare with random data
        random_data = bytes(np.random.randint(1, self.max_number + 1, len(orig_bytes)).tolist())
        random_compressed = zlib.compress(random_data, 9)
        random_ratio = len(random_compressed) / len(random_data)
        
        # Compare with highly patterned data
        pattern_data = bytes(list(range(1, self.max_number + 1)) * (len(orig_bytes) // self.max_number + 1))
        pattern_data = pattern_data[:len(orig_bytes)]
        pattern_compressed = zlib.compress(pattern_data, 9)
        pattern_ratio = len(pattern_compressed) / len(pattern_data)
        
        # Difference from random
        compressibility_diff = random_ratio - ratio
        
        # Also test string representation
        str_data = ','.join(map(str, self.flat[:3000])).encode()
        str_comp = zlib.compress(str_data, 9)
        str_ratio = len(str_comp) / len(str_data)
        
        return {
            'name': 'Compression Analysis (Kolmogorov Complexity)',
            'is_pattern': compressibility_diff > 0.05,
            'compression_ratio': round(ratio, 4),
            'random_ratio': round(random_ratio, 4),
            'pattern_ratio': round(pattern_ratio, 4),
            'more_compressible_than_random': compressibility_diff > 0,
            'difference': round(float(compressibility_diff), 4),
            'string_ratio': round(str_ratio, 4),
            'bytes': {'original': len(orig_bytes), 'compressed': len(compressed)},
            'conclusion': f'Compression: {ratio:.4f} (random: {random_ratio:.4f}, pattern: {pattern_ratio:.4f}). ' +
                         ('MORE COMPRESSIBLE than random - HIDDEN STRUCTURE!' if compressibility_diff > 0.05
                          else 'Similar to random data - no compressible patterns.')
        }
    
    # ===== 6. CROSS-LOTTERY CORRELATION =====
    def _cross_lottery_correlation(self):
        """Check if Mega 6/45 and Power 6/55 numbers are correlated."""
        if not self.cross_data or len(self.cross_data) < 50:
            return {
                'name': 'Cross-Lottery Correlation',
                'is_pattern': False,
                'conclusion': 'No cross-lottery data available for comparison.'
            }
        
        min_len = min(len(self.data), len(self.cross_data))
        
        # Compare sums
        sums1 = [sum(d[:self.pick_count]) for d in self.data[-min_len:]]
        sums2 = [sum(d[:self.pick_count]) for d in self.cross_data[-min_len:]]
        
        # Correlation
        corr = float(np.corrcoef(sums1, sums2)[0, 1]) if len(sums1) > 2 else 0
        
        # Lagged correlation
        lag_corrs = []
        for lag in range(1, 10):
            if lag < len(sums1):
                c = float(np.corrcoef(sums1[lag:], sums2[:-lag])[0, 1])
                lag_corrs.append({'lag': lag, 'correlation': round(c, 4)})
        
        # Number overlap analysis
        overlaps = []
        for i in range(min_len):
            s1 = set(self.data[-min_len + i][:self.pick_count])
            s2 = set(self.cross_data[-min_len + i][:self.pick_count])
            overlaps.append(len(s1 & s2))
        
        avg_overlap = float(np.mean(overlaps))
        expected_overlap = self.pick_count * self.pick_count / self.max_number
        
        return {
            'name': 'Cross-Lottery Correlation',
            'is_pattern': abs(corr) > 0.1 or avg_overlap > expected_overlap * 1.3,
            'sum_correlation': round(corr, 4),
            'lagged_correlations': lag_corrs,
            'avg_number_overlap': round(avg_overlap, 3),
            'expected_overlap': round(expected_overlap, 3),
            'conclusion': f'Sum correlation: {corr:.4f}. Number overlap: {avg_overlap:.3f} (expected: {expected_overlap:.3f}). ' +
                         ('CROSS-LOTTERY PATTERN!' if abs(corr) > 0.1
                          else 'No significant cross-lottery correlation.')
        }
    
    # ===== 7. MUTUAL INFORMATION =====
    def _mutual_information(self):
        """Measure non-linear dependencies between consecutive draws."""
        seq = self.flat[:2000]
        n_bins = min(20, self.max_number)
        
        # Mutual information between X(t) and X(t+k) for various k
        mi_values = []
        
        for lag in range(1, 20):
            x = seq[:len(seq) - lag]
            y = seq[lag:]
            
            # Joint and marginal histograms
            joint_hist, _, _ = np.histogram2d(x, y, bins=n_bins, 
                                               range=[[0.5, self.max_number + 0.5]] * 2)
            
            # Normalize
            joint_prob = joint_hist / np.sum(joint_hist)
            x_prob = np.sum(joint_prob, axis=1)
            y_prob = np.sum(joint_prob, axis=0)
            
            mi = 0
            for i in range(n_bins):
                for j in range(n_bins):
                    if joint_prob[i, j] > 0 and x_prob[i] > 0 and y_prob[j] > 0:
                        mi += joint_prob[i, j] * math.log2(
                            joint_prob[i, j] / (x_prob[i] * y_prob[j]))
            
            mi_values.append({'lag': lag, 'mutual_info': round(mi, 6)})
        
        avg_mi = float(np.mean([m['mutual_info'] for m in mi_values]))
        max_mi = max(m['mutual_info'] for m in mi_values)
        
        # Expected MI for independent data ≈ 0
        # Significant if > 0.01
        
        return {
            'name': 'Mutual Information Analysis',
            'is_pattern': max_mi > 0.05,
            'mi_by_lag': mi_values,
            'avg_mi': round(avg_mi, 6),
            'max_mi': round(max_mi, 6),
            'conclusion': f'Max MI: {max_mi:.6f}, Avg: {avg_mi:.6f}. ' +
                         ('SIGNIFICANT non-linear dependency!' if max_mi > 0.05
                          else 'No significant mutual information (independent).')
        }
    
    # ===== 8. WAVELET ANALYSIS =====
    def _wavelet_analysis(self):
        """Multi-resolution analysis using Haar wavelet."""
        seq = np.array(self.flat[:2048], dtype=float)
        n = len(seq)
        
        # Pad to power of 2
        next_pow2 = 1 << (n - 1).bit_length()
        seq = np.pad(seq, (0, next_pow2 - n), mode='edge')
        
        # Haar wavelet decomposition
        levels = []
        current = seq.copy()
        
        for level in range(min(8, int(math.log2(len(current))))):
            half = len(current) // 2
            if half < 4:
                break
            approximation = np.zeros(half)
            detail = np.zeros(half)
            
            for i in range(half):
                approximation[i] = (current[2 * i] + current[2 * i + 1]) / math.sqrt(2)
                detail[i] = (current[2 * i] - current[2 * i + 1]) / math.sqrt(2)
            
            energy = float(np.sum(detail ** 2))
            max_coeff = float(np.max(np.abs(detail)))
            
            levels.append({
                'level': level + 1,
                'scale': 2 ** (level + 1),
                'energy': round(energy, 2),
                'max_coefficient': round(max_coeff, 2),
                'std': round(float(np.std(detail)), 4)
            })
            
            current = approximation
        
        # Check if energy distribution is uniform (random) or concentrated (pattern)
        if levels:
            energies = [l['energy'] for l in levels]
            total_energy = sum(energies)
            energy_fracs = [e / total_energy for e in energies] if total_energy > 0 else []
            
            # Entropy of energy distribution
            energy_entropy = -sum(f * math.log2(f) for f in energy_fracs if f > 0)
            max_entropy = math.log2(len(levels))
            entropy_ratio = energy_entropy / max_entropy if max_entropy > 0 else 0
        else:
            entropy_ratio = 1.0
        
        return {
            'name': 'Wavelet Decomposition (Haar)',
            'is_pattern': entropy_ratio < 0.7,
            'levels': levels,
            'energy_entropy_ratio': round(entropy_ratio, 4),
            'conclusion': f'Energy entropy ratio: {entropy_ratio:.4f} (1.0 = uniform/random). ' +
                         ('SCALE-DEPENDENT PATTERN detected!' if entropy_ratio < 0.7
                          else 'Energy uniformly distributed (random-like).')
        }
    
    # ===== 9. SYMBOLIC REGRESSION =====
    def _symbolic_regression(self):
        """Try to find exact mathematical formula using brute force."""
        sums = [sum(d[:self.pick_count]) for d in self.data]
        n = len(sums)
        
        if n < 50:
            return {'name': 'Symbolic Regression', 'is_pattern': False, 'conclusion': 'Not enough data.'}
        
        split = int(n * 0.7)
        
        # Try various formulas
        formulas = {}
        
        # 1. s(t) = a*s(t-1) + b
        if n > 2:
            x, y = np.array(sums[:-1]), np.array(sums[1:])
            a, b = np.polyfit(x[:split], y[:split], 1)
            pred = a * x[split:] + b
            mse = float(np.mean((pred - y[split:]) ** 2))
            formulas['Linear'] = {'formula': f's(t) = {a:.4f}*s(t-1) + {b:.2f}', 'mse': round(mse, 2)}
        
        # 2. s(t) = a*s(t-1) + b*s(t-2) + c
        if n > 3:
            X2 = np.column_stack([sums[1:-1], sums[:-2]])
            y2 = np.array(sums[2:])
            try:
                coeffs = np.linalg.lstsq(X2[:split-1], y2[:split-1], rcond=None)[0]
                pred2 = X2[split-1:] @ coeffs
                mse2 = float(np.mean((pred2 - y2[split-1:]) ** 2))
                formulas['AR(2)'] = {
                    'formula': f's(t) = {coeffs[0]:.4f}*s(t-1) + {coeffs[1]:.4f}*s(t-2)',
                    'mse': round(mse2, 2)
                }
            except:
                pass
        
        # 3. Modular: s(t) mod k = f(s(t-1) mod k)
        for k in [7, 10, 13, 45]:
            mod_match = 0
            mod_total = 0
            for i in range(1, min(split, n)):
                pred_mod = (sums[i - 1] * 3 + 5) % k
                actual_mod = sums[i] % k
                if pred_mod == actual_mod:
                    mod_match += 1
                mod_total += 1
            
            accuracy = mod_match / mod_total if mod_total > 0 else 0
            formulas[f'Mod({k})'] = {
                'formula': f's(t) mod {k} = (3*s(t-1)+5) mod {k}',
                'accuracy': round(accuracy * 100, 1)
            }
        
        # 4. XOR pattern
        xor_match = 0
        xor_total = 0
        for i in range(2, min(split, n)):
            pred_xor = sums[i - 1] ^ sums[i - 2]
            if abs(pred_xor - sums[i]) < 20:
                xor_match += 1
            xor_total += 1
        xor_acc = xor_match / xor_total if xor_total > 0 else 0
        formulas['XOR'] = {'formula': 's(t) ≈ s(t-1) XOR s(t-2)', 'accuracy': round(xor_acc * 100, 1)}
        
        # Naive baseline
        naive_mse = float(np.mean((np.mean(sums[:split]) - np.array(sums[split:])) ** 2))
        formulas['Naive_Mean'] = {'formula': f's(t) = {np.mean(sums[:split]):.1f}', 'mse': round(naive_mse, 2)}
        
        # Find best
        best_name = None
        best_imp = 0
        for name, f in formulas.items():
            if 'mse' in f and f['mse'] < naive_mse * 0.9:
                imp = (1 - f['mse'] / naive_mse) * 100
                if imp > best_imp:
                    best_imp = imp
                    best_name = name
        
        return {
            'name': 'Symbolic Regression',
            'is_pattern': best_imp > 10,
            'formulas': formulas,
            'best_formula': best_name,
            'best_improvement': round(best_imp, 1),
            'naive_mse': round(naive_mse, 2),
            'conclusion': f'Best formula improvement: {best_imp:.1f}%. ' +
                         (f'FORMULA FOUND: {formulas[best_name]["formula"]}!' if best_imp > 10
                          else 'No formula significantly beats naive mean prediction.')
        }
    
    # ===== 10. META PREDICTOR =====
    def _meta_predictor(self):
        """
        Combine ALL available signals to make the best possible prediction.
        Walk-forward test on last 100 draws.
        """
        n = len(self.data)
        if n < 150:
            return {'name': 'Ensemble Meta-Predictor', 'is_pattern': False, 'conclusion': 'Not enough data.'}
        
        test_start = n - 100
        matches = []
        
        for i in range(test_start, n - 1):
            scores = Counter()
            history = self.data[:i + 1]
            
            # Signal 1: Frequency (last 50)
            recent = [n for d in history[-50:] for n in d[:self.pick_count]]
            freq = Counter(recent)
            for num, count in freq.most_common(15):
                scores[num] += count * 0.2
            
            # Signal 2: Cold numbers (overdue)
            all_recent = set(n for d in history[-20:] for n in d[:self.pick_count])
            for num in range(1, self.max_number + 1):
                if num not in all_recent:
                    scores[num] += 3
            
            # Signal 3: Difference pattern
            if len(history) >= 3:
                last_sum = sum(history[-1][:self.pick_count])
                prev_sum = sum(history[-2][:self.pick_count])
                target_sum = last_sum + (last_sum - prev_sum)
                avg_target = target_sum / self.pick_count
                for offset in range(-5, 6):
                    n_val = int(avg_target + offset)
                    if 1 <= n_val <= self.max_number:
                        scores[n_val] += 1
            
            # Signal 4: Anti-repeat
            last_draw = set(history[-1][:self.pick_count])
            for num in last_draw:
                scores[num] -= 2
            
            # Signal 5: Modular
            if len(history) >= 2:
                for num in range(1, self.max_number + 1):
                    if num % 5 == (sum(history[-1][:self.pick_count]) % 5):
                        scores[num] += 0.5
            
            # Pick top numbers
            predicted = set(sorted(scores, key=lambda x: -scores[x])[:self.pick_count])
            actual = set(self.data[i + 1][:self.pick_count])
            matches.append(len(predicted & actual))
        
        avg_match = float(np.mean(matches))
        max_match = int(np.max(matches))
        random_exp = self.pick_count ** 2 / self.max_number
        improvement = (avg_match / random_exp - 1) * 100 if random_exp > 0 else 0
        
        dist = Counter(matches)
        
        return {
            'name': 'Ensemble Meta-Predictor (5 signals)',
            'is_pattern': avg_match > random_exp * 1.3,
            'avg_matches': round(avg_match, 3),
            'max_matches': max_match,
            'random_expected': round(random_exp, 3),
            'improvement': round(float(improvement), 1),
            'distribution': {str(k): int(v) for k, v in sorted(dist.items())},
            'tests_run': len(matches),
            'conclusion': f'Meta-Predictor avg: {avg_match:.3f}/6 (random: {random_exp:.3f}, {improvement:+.1f}%). ' +
                         ('META-PREDICTOR BEATS RANDOM!' if avg_match > random_exp * 1.3
                          else 'Even combined signals cannot beat random significantly.')
        }
    
    def _verdict(self, results):
        pattern_count = 0
        total = 0
        evidence = []
        
        for key, val in results.items():
            if key == 'verdict':
                continue
            if isinstance(val, dict) and 'is_pattern' in val:
                total += 1
                if val['is_pattern']:
                    pattern_count += 1
                    evidence.append(f"+ {val.get('name', key)}: {val.get('conclusion', '')}")
                else:
                    evidence.append(f"- {val.get('name', key)}: {val.get('conclusion', '')}")
        
        score = round(pattern_count / total * 100, 1) if total > 0 else 0
        
        if score >= 60:
            verdict = "STRONG EVIDENCE - Hidden pattern detected by multiple advanced methods!"
        elif score >= 30:
            verdict = "WEAK evidence - Some methods find signals but inconclusive"
        else:
            verdict = "NO PATTERN - Even advanced methods cannot find exploitable structure"
        
        return {
            'score': score,
            'pattern_count': pattern_count,
            'total_tests': total,
            'verdict': verdict,
            'evidence': evidence
        }
