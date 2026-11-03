#!/usr/bin/env python3
"""
Master Integration Pipeline — More than a Hack 2027
Team Lead: Ragavendran
Executes Modules 1 -> 2 -> 3 -> 4, enforces official graded frame boundaries,
writes team submission CSV, and runs official evaluation/validate_submission.py.
"""

import argparse
import os
import sys
import subprocess
import numpy as np
import pandas as pd

from src.module1_tracking import build_tracking_features
from src.module2_deadball_tags import build_deadball_and_tag_features
from src.module3_possession import compute_frame_probabilities
from src.module4_decoder import decode_possession_spells

N_FRAMES = 199319
H1_START, H1_END = 11185, 90651          # 1st half graded inclusive (79,467 frames)[cite: 1, 7]
H2_START, H2_END = 110808, 190197        # 2nd half graded inclusive (79,390 frames)[cite: 1, 7]
H2_KICKOFF = 113742                      # 2nd half actual kickoff (110808..113741 are 'D')[cite: 1]


def enforce_hard_boundary_rules(labels: np.ndarray) -> np.ndarray:
    """
    Guarantees 100% compliance with DATA.md and FAQ.md frame boundaries:
      - Outside graded halves -> 'X'
      - Graded 2nd-half pre-kickoff window (110,808 to 113,741) -> 'D'[cite: 1]
    """
    out = np.full(N_FRAMES, "X", dtype=object)

    # Copy predictions strictly inside the two graded halves[cite: 1, 7]
    out[H1_START : H1_END + 1] = labels[H1_START : H1_END + 1]
    out[H2_START : H2_END + 1] = labels[H2_START : H2_END + 1]

    # FAQ.md trap fix: frames 110,808..113,741 are graded before 2H kickoff and are ALL 'D'[cite: 1]
    out[H2_START : H2_KICKOFF] = "D"
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate full-match possession submission.")
    parser.add_argument("--team_name", type=str, default="our_team",
                        help="Team name for output CSV in evaluation/results/<team_name>.csv")
    args = parser.parse_args()

    print("=" * 70)
    print(f" RUNNING FULL-MATCH PIPELINE FOR TEAM: {args.team_name} (Lead: Ragavendran)")
    print("=" * 70)

    print("[1/4] Running Module 1 (Suriya): Tactical ATD & Missing Ball Recovery...")
    df_track = build_tracking_features(n_frames=N_FRAMES)
    assert len(df_track) == N_FRAMES, f"Module 1 must return {N_FRAMES} rows!"

    print("[2/4] Running Module 2 (Abhishek): Smart Tagging & Dead-Ball Priors...")
    df_tags = build_deadball_and_tag_features(df_track, n_frames=N_FRAMES)
    assert len(df_tags) == N_FRAMES, f"Module 2 must return {N_FRAMES} rows!"

    print("[3/4] Running Module 3 (Ranganathan): Possession Control Physics & LGBM...")
    df_probs = compute_frame_probabilities(df_track, df_tags, n_frames=N_FRAMES)
    assert len(df_probs) == N_FRAMES, f"Module 3 must return {N_FRAMES} rows!"

    print("[4/4] Running Module 4 (Varsha): Period-Aware Viterbi & Spell Merger...")
    raw_pred_labels = decode_possession_spells(
        df_probs,
        h1_bounds=(H1_START, H1_END),
        h2_bounds=(H2_KICKOFF, H2_END)
    )

    final_labels = enforce_hard_boundary_rules(raw_pred_labels)

    out_path = os.path.join("evaluation", "results", f"{args.team_name}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sub_df = pd.DataFrame({
        "frame": np.arange(N_FRAMES, dtype=np.int32),
        "label": final_labels
    })
    sub_df.to_csv(out_path, index=False)
    print(f"\n[Done] Wrote {len(sub_df):,} rows to {out_path}")

    print("\n--- Validating Submission (validate_submission.py) ---")
    subprocess.run([sys.executable, "evaluation/validate_submission.py", out_path], check=False)

    print("\n--- Running Public 5-Min Pipeline Check (run_evaluation.py) ---")
    subprocess.run([sys.executable, "evaluation/run_evaluation.py"], check=False)


if __name__ == "__main__":
    main()