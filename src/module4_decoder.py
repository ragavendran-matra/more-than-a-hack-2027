"""
MODULE 4 (Varsha): Period-Aware Decoder & XML-Guided Spell Stitcher
Balances flicker removal (sandwich_win=48) with Set-Piece Dead-Ball preservation.
"""
import numpy as np
import pandas as pd

STATES = np.array(["H", "A", "D"])


def _clean_half_spells(
    raw_labels: np.ndarray,
    setpiece_mask: np.ndarray,
    sandwich_win: int = 48,
    step_win: int = 18
) -> np.ndarray:
    out = np.array(raw_labels, copy=True)
    n = len(out)

    # Pass 1: Sandwich stitch (merges brief H<->A flickers < sandwich_win,
    #         while protecting XML-confirmed Set-Piece 'D' stoppages >= 15 frames)
    for w in [sandwich_win // 2, sandwich_win]:
        if w <= 0:
            continue
        ch = np.flatnonzero(out[1:] != out[:-1]) + 1
        starts, ends = np.r_[0, ch], np.r_[ch, n]
        for i in range(1, len(starts) - 1):
            seg_len = ends[i] - starts[i]
            cur_lbl = out[starts[i]]
            if cur_lbl == "X":
                continue
            # Protect XML-confirmed Set-Piece Dead Balls >= 0.6s (15 frames)
            if cur_lbl == "D" and seg_len >= 15 and np.mean(setpiece_mask[starts[i] : ends[i]]) > 0.2:
                continue
            if seg_len < w:
                if out[starts[i - 1]] == out[starts[i + 1]] and out[starts[i - 1]] != "X":
                    out[starts[i] : ends[i]] = out[starts[i - 1]]

    # Pass 2: Step stitch (absorbs tiny transition blips < step_win)
    if step_win > 0:
        ch = np.flatnonzero(out[1:] != out[:-1]) + 1
        starts, ends = np.r_[0, ch], np.r_[ch, n]
        lengths = ends - starts
        for i in range(1, len(starts) - 1):
            if lengths[i] < step_win and out[starts[i]] != "X":
                # Do not step-stitch away an XML-confirmed Dead Ball >= 15 frames
                if out[starts[i]] == "D" and lengths[i] >= 15 and np.mean(setpiece_mask[starts[i] : ends[i]]) > 0.2:
                    continue
                prev_l, next_l = out[starts[i - 1]], out[starts[i + 1]]
                if prev_l != "X" and next_l != "X":
                    out[starts[i] : ends[i]] = prev_l if lengths[i - 1] >= lengths[i + 1] else next_l

    return out


def decode_possession_spells(
    df_probs: pd.DataFrame,
    h1_bounds: tuple = (11185, 90651),
    h2_bounds: tuple = (113742, 190197)
) -> np.ndarray:
    probs = df_probs[["p_H", "p_A", "p_D"]].to_numpy(dtype=np.float64)
    init_labels = STATES[np.argmax(probs, axis=1)]
    sp_mask = (
        df_probs["is_setpiece_window"].to_numpy()
        if "is_setpiece_window" in df_probs.columns
        else np.zeros(len(df_probs))
    )

    out = np.full(len(df_probs), "X", dtype=object)
    for (s_f, e_f) in [h1_bounds, h2_bounds]:
        out[s_f : e_f + 1] = _clean_half_spells(
            init_labels[s_f : e_f + 1],
            sp_mask[s_f : e_f + 1]
        )
    return out