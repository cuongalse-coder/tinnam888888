"""
Dàn Predictor - Progressive Block Elimination System
=====================================================
V1: Full dàn (block + direction + S/L filters)
V2: Optimized dàn (V1 + gap constraints + mid4 sum filter)
"""
import numpy as np
from itertools import product
from collections import Counter

def _extract_pos(data, pick):
    pos = [[] for _ in range(pick)]
    for d in data:
        sd = sorted(d[:pick])
        for p in range(pick):
            pos[p].append(sd[p])
    return pos

def _get_dir(a, b):
    if b > a: return 'U'
    elif b < a: return 'D'
    return 'S'

def _to_sl(n):
    r = n % 10
    return 'S' if (1 <= r <= 5) or r == 0 else 'L'

def _to_block(n, is_mega):
    if n <= 9: return 'A'
    elif n <= 19: return 'B'
    elif n <= 29: return 'C'
    elif n <= 39: return 'D'
    elif is_mega: return 'E'
    elif n <= 49: return 'E'
    else: return 'F'

MEGA_BLOCKS = {'A': (1, 9), 'B': (10, 19), 'C': (20, 29), 'D': (30, 39), 'E': (40, 45)}
POWER_BLOCKS = {'A': (1, 9), 'B': (10, 19), 'C': (20, 29), 'D': (30, 39), 'E': (40, 49), 'F': (50, 55)}

PATTERN_3D = {
    ('D', 'D', 'D'): ('U', 0.85), ('U', 'U', 'U'): ('D', 0.80),
    ('D', 'U', 'U'): ('D', 0.72), ('U', 'D', 'D'): ('U', 0.69),
    ('U', 'D', 'U'): ('D', 0.63), ('D', 'U', 'D'): ('U', 0.62),
}


def predict_dan(data, max_num, pick, is_mega=True, version="v1"):
    """
    Generate dàn combos.
    version="v1": full dàn (block + direction + S/L)
    version="v2": optimized (V1 + gap + mid4 sum constraints)
    Returns: (candidates_per_col, valid_combos, info_dict)
    """
    pos_data = _extract_pos(data, pick)
    n = len(data)
    blocks_def = MEGA_BLOCKS if is_mega else POWER_BLOCKS

    # === Block prediction per column ===
    predicted_blocks = []
    for pos in range(pick):
        h = pos_data[pos]
        bseq = [_to_block(v, is_mega) for v in h]
        pred = None
        if len(bseq) >= 3:
            p3 = (bseq[-3], bseq[-2], bseq[-1])
            p3n = Counter()
            for i in range(len(bseq) - 3):
                if (bseq[i], bseq[i + 1], bseq[i + 2]) == p3:
                    p3n[bseq[i + 3]] += 1
            if sum(p3n.values()) >= 3:
                pred = [b for b, _ in p3n.most_common(2)]
        if not pred:
            bc = Counter(bseq)
            pred = [b for b, _ in bc.most_common(3)]
        predicted_blocks.append(pred)

    # === Hot numbers within predicted blocks ===
    candidates = []
    for pos in range(pick):
        h = pos_data[pos]
        nh = len(h)
        freq = Counter(h)
        valid = set()
        for b in predicted_blocks[pos]:
            blo, bhi = blocks_def[b]
            for num in range(blo, bhi + 1):
                if freq.get(num, 0) > 0:
                    valid.add(num)
        ranked = sorted(valid, key=lambda x: -freq.get(x, 0))
        total = 0
        hot = []
        for num in ranked:
            hot.append(num)
            total += freq[num] / nh * 100
            if total >= 70 or len(hot) >= 8:
                break
        candidates.append(sorted(hot) if hot else [int(np.median(h[-10:]))])

    # === Direction filter ===
    for pos in range(pick):
        h = pos_data[pos]
        nh = len(h)
        if nh < 4:
            continue
        dirs = [_get_dir(h[i], h[i + 1]) for i in range(nh - 1)]
        if len(dirs) >= 3:
            p3 = (dirs[-3], dirs[-2], dirs[-1])
            if p3 in PATTERN_3D:
                ed, conf = PATTERN_3D[p3]
                if conf >= 0.62:
                    lv = h[-1]
                    mg = int(np.std(h[-10:]) * 0.5)
                    if ed == 'U':
                        f = {x for x in candidates[pos] if x >= lv - mg}
                    else:
                        f = {x for x in candidates[pos] if x <= lv + mg}
                    if len(f) >= 2:
                        candidates[pos] = sorted(f)

    # === S/L filter ===
    for pos in range(pick):
        h = pos_data[pos]
        sl = [_to_sl(v) for v in h]
        if len(sl) >= 3:
            sp = sl[-3] + sl[-2] + sl[-1]
            sn = Counter()
            for i in range(len(sl) - 3):
                if sl[i] + sl[i + 1] + sl[i + 2] == sp:
                    sn[sl[i + 3]] += 1
            ts = sum(sn.values())
            if ts >= 5:
                bs = sn.most_common(1)[0][0]
                if sn.most_common(1)[0][1] / ts >= 0.62:
                    f = [x for x in candidates[pos] if _to_sl(x) == bs]
                    if len(f) >= 2:
                        candidates[pos] = sorted(f)

    # === Generate combos (strictly increasing) ===
    all_combos = []
    for combo in product(*candidates):
        if all(combo[i] < combo[i + 1] for i in range(len(combo) - 1)):
            all_combos.append(combo)

    before = len(all_combos)

    # === V2: Gap + Mid4 sum constraints ===
    if version == "v2" and len(all_combos) > 0:
        # Gap constraints
        gap_constraints = []
        for i in range(pick - 1):
            gaps = [pos_data[i + 1][t] - pos_data[i][t] for t in range(n)]
            lo = max(1, int(np.percentile(gaps, 15)))
            hi = int(np.percentile(gaps, 85))
            gap_constraints.append((lo, hi))

        # Mid4 sum constraint
        mid4_sums = [sum(pos_data[j][t] for j in range(1, 5)) for t in range(n)]
        sum_lo = int(np.percentile(mid4_sums, 10))
        sum_hi = int(np.percentile(mid4_sums, 90))

        filtered = []
        for combo in all_combos:
            ok = True
            for i in range(5):
                gap = combo[i + 1] - combo[i]
                lo, hi = gap_constraints[i]
                if gap < lo or gap > hi:
                    ok = False
                    break
            if not ok:
                continue
            mid4 = sum(combo[1:5])
            if mid4 < sum_lo or mid4 > sum_hi:
                continue
            filtered.append(combo)
        all_combos = filtered

    after = len(all_combos)

    info = {
        "candidates": candidates,
        "total_before_filter": before,
        "total_after_filter": after,
        "version": version,
        "predicted_blocks": predicted_blocks,
    }

    return candidates, all_combos, info
