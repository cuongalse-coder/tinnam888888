"""
STACKING ENGINE V600.0 — NEURAL APEX
=====================================
True ML Stacking Ensemble with proper walk-forward validation.
Fixes critical data leakage bugs from V500.

Key improvements:
1. 20+ engineered features per number (not just 6)
2. Proper expanding-window walk-forward (no future data leak)
3. 3-model stacking: HistGBR + Ridge + RandomForest
4. Meta-learner blending layer
"""
import math
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations

try:
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class StackingEngine:
    def __init__(self, max_number, pick_count):
        self.max_number = max_number
        self.pick_count = pick_count

    # ================================================================
    # FEATURE ENGINEERING: 20 features per number
    # ================================================================

    def _build_features(self, data, num):
        """Build 20-dim feature vector for a single number given historical data."""
        n = len(data)
        if n < 30:
            return [0.0] * 20

        # 1-5: Multi-window frequency ratios
        windows = [5, 10, 20, 40, 80]
        freq_feats = []
        expected = self.pick_count / self.max_number
        for w in windows:
            if n >= w:
                f = sum(1 for d in data[-w:] if num in d[:self.pick_count]) / w
                freq_feats.append(f / (expected + 1e-6))
            else:
                freq_feats.append(1.0)

        # 6: Current gap (draws since last appearance)
        last_idx = -1
        for i in range(n - 1, -1, -1):
            if num in data[i][:self.pick_count]:
                last_idx = i
                break
        current_gap = (n - last_idx) if last_idx >= 0 else n

        # 7: Mean gap
        appearances = [i for i, d in enumerate(data) if num in d[:self.pick_count]]
        if len(appearances) >= 2:
            gaps = [appearances[j+1] - appearances[j] for j in range(len(appearances)-1)]
            mean_gap = np.mean(gaps)
            std_gap = np.std(gaps) + 1e-6
        else:
            mean_gap = n / max(len(appearances), 1)
            std_gap = mean_gap
            gaps = []

        # 8: Overdue ratio
        overdue = current_gap / (mean_gap + 0.1)

        # 9: Gap z-score
        gap_z = (current_gap - mean_gap) / std_gap if std_gap > 0 else 0

        # 10-11: Momentum (velocity and acceleration)
        f5 = sum(1 for d in data[-5:] if num in d[:self.pick_count]) / 5 if n >= 5 else 0
        f10 = sum(1 for d in data[-10:] if num in d[:self.pick_count]) / 10 if n >= 10 else 0
        f20 = sum(1 for d in data[-20:] if num in d[:self.pick_count]) / 20 if n >= 20 else 0
        velocity = f5 - f10
        accel = (f5 - f10) - (f10 - f20)

        # 12: Conditional probability given last draw
        last_draw = set(data[-1][:self.pick_count])
        cond_score = 0.0
        for given in last_draw:
            count_given = sum(1 for d in data[:-1] if given in d[:self.pick_count])
            count_follow = 0
            for i in range(len(data) - 1):
                if given in data[i][:self.pick_count] and num in data[i+1][:self.pick_count]:
                    count_follow += 1
            if count_given > 0:
                cond_score += count_follow / count_given
        cond_score /= max(len(last_draw), 1)

        # 13: Pair co-occurrence with last draw
        pair_score = 0.0
        pair_counts = Counter()
        for d in data[-150:]:
            for p in combinations(sorted(d[:self.pick_count]), 2):
                pair_counts[p] += 1
        for p in last_draw:
            key = tuple(sorted([p, num]))
            pair_score += pair_counts.get(key, 0)

        # 14: Temporal decay weighted frequency
        decay_score = 0.0
        lam = 0.05
        for i, d in enumerate(data):
            if num in d[:self.pick_count]:
                decay_score += math.exp(-lam * (n - 1 - i))

        # 15: Sector heat (which decade is hot)
        sector = (num - 1) // 10
        sector_heat = sum(1 for d in data[-20:] for x in d[:self.pick_count] if (x-1)//10 == sector)
        avg_sector = 20 * self.pick_count / ((self.max_number + 9) // 10)
        sector_ratio = sector_heat / (avg_sector + 1e-6)

        # 16: Is in last draw (binary)
        in_last = 1.0 if num in last_draw else 0.0

        # 17: In second-to-last draw
        in_prev = 1.0 if len(data) > 1 and num in set(data[-2][:self.pick_count]) else 0.0

        # 18: Streak length (consecutive appearances or absences)
        streak = 0
        for d in reversed(data):
            if num in d[:self.pick_count]:
                streak += 1
            else:
                break

        # 19: Overall frequency z-score
        total_freq = sum(1 for d in data if num in d[:self.pick_count])
        exp_freq = n * expected
        std_freq = math.sqrt(n * expected * (1 - expected)) + 1e-6
        freq_z = (total_freq - exp_freq) / std_freq

        # 20: Gap acceleration (are gaps getting shorter or longer?)
        gap_accel = 0.0
        if len(gaps) >= 4:
            recent = gaps[-3:]
            diffs = [recent[i] - recent[i-1] for i in range(1, len(recent))]
            gap_accel = -np.mean(diffs)  # negative diffs = gaps shrinking = more likely

        return [
            *freq_feats,        # 1-5
            current_gap,        # 6
            mean_gap,           # 7
            overdue,            # 8
            gap_z,              # 9
            velocity,           # 10
            accel,              # 11
            cond_score,         # 12
            pair_score,         # 13
            decay_score,        # 14
            sector_ratio,       # 15
            in_last,            # 16
            in_prev,            # 17
            float(streak),      # 18
            freq_z,             # 19
            gap_accel,          # 20
        ]

    def _build_all_features(self, data):
        """Build feature matrix for all numbers."""
        X = []
        for num in range(1, self.max_number + 1):
            X.append(self._build_features(data, num))
        return np.array(X)

    # ================================================================
    # PROPER WALK-FORWARD STACKING
    # ================================================================

    def predict_top_pool(self, data, pool_size=15):
        """
        Main prediction method. Returns ranked pool of numbers.
        Uses proper expanding-window walk-forward to avoid data leakage.
        """
        if not HAS_SKLEARN or len(data) < 80:
            return self._fallback_predict(data, pool_size)

        n = len(data)
        # Training: use draws 60..n-1 as expanding window
        # Each step i: train on data[:i], predict data[i], collect OOF predictions
        train_start = max(60, n - 120)  # Last 120 draws for training speed

        # Collect training examples with proper walk-forward
        X_train = []
        y_train = []

        for i in range(train_start, n):
            hist = data[:i]
            actual = set(data[i][:self.pick_count])

            # Build features using ONLY data[:i] — no future leak
            for num in range(1, self.max_number + 1):
                feats = self._build_features(hist, num)
                X_train.append(feats)
                y_train.append(1.0 if num in actual else 0.0)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Train 3 base models on all training data
        models = []

        m1 = HistGradientBoostingRegressor(
            loss='log_loss', max_iter=250, max_depth=6, learning_rate=0.03,
            min_samples_leaf=15, l2_regularization=0.5, random_state=42
        )
        m1.fit(X_train, y_train)
        models.append(('hgbr', m1))

        m2 = RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            max_features='sqrt', random_state=42, n_jobs=-1
        )
        m2.fit(X_train, y_train)
        models.append(('rf', m2))

        m3 = Ridge(alpha=1.0)
        m3.fit(X_train, y_train)
        models.append(('ridge', m3))

        # Generate OOF predictions for meta-learner calibration
        # Use last 30 training steps
        oof_start = max(0, len(X_train) - 30 * self.max_number)
        X_oof = X_train[oof_start:]
        y_oof = y_train[oof_start:]

        meta_X = []
        for name, model in models:
            preds = model.predict(X_oof)
            meta_X.append(preds)
        meta_X = np.column_stack(meta_X)

        # Meta-learner: simple Ridge on stacked predictions
        meta = Ridge(alpha=0.5)
        meta.fit(meta_X, y_oof)

        # Final prediction on current state
        X_pred = self._build_all_features(data)

        base_preds = []
        for name, model in models:
            base_preds.append(model.predict(X_pred))
        meta_input = np.column_stack(base_preds)

        final_scores = meta.predict(meta_input)

        # Rank numbers by score
        ranked = sorted(
            [(num + 1, final_scores[num]) for num in range(self.max_number)],
            key=lambda x: -x[1]
        )

        return {
            'pool': [n for n, _ in ranked[:pool_size]],
            'scores': {n: round(s, 5) for n, s in ranked},
            'top6': sorted([n for n, _ in ranked[:6]]),
            'model_weights': {name: round(meta.coef_[i], 3) for i, (name, _) in enumerate(models)},
        }

    def _fallback_predict(self, data, pool_size):
        """Simple frequency-based fallback when sklearn unavailable."""
        n = len(data)
        scores = {}
        for num in range(1, self.max_number + 1):
            f10 = sum(1 for d in data[-10:] if num in d[:self.pick_count]) / 10
            f30 = sum(1 for d in data[-30:] if num in d[:self.pick_count]) / 30

            last_seen = -1
            for i in range(n-1, -1, -1):
                if num in data[i][:self.pick_count]:
                    last_seen = i
                    break
            gap = n - last_seen if last_seen >= 0 else n
            exp_gap = self.max_number / self.pick_count

            scores[num] = f10 * 3 + (gap / exp_gap) * 2 + f30

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return {
            'pool': [n for n, _ in ranked[:pool_size]],
            'scores': {n: round(s, 5) for n, s in ranked},
            'top6': sorted([n for n, _ in ranked[:6]]),
            'model_weights': {},
        }
