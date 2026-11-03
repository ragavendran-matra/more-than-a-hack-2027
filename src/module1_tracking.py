"""
MODULE 1 (Suriya): Pure Raw ATD Parser, Missing-Ball Imputation & Top-K Player Distances
Reads data/tactical/BAR-ATH_tactical_atd.xml and BAR-ATH_tactical_atd.txt directly
without Kloppy or baseline CSV dependencies. Caches parsed arrays for 0.2s reloads.
"""
import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

from games import GAME, GAMES, require

FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING = "N"
HOME, AWAY = 0, 1


def _team_of_each_track(meta_path: str) -> np.ndarray:
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    home_id = team_ids[0]
    slots = [p.get("teamId") for p in root.iter("Player")]
    return np.array([HOME if tid == home_id else AWAY for tid in slots], dtype=np.int8)


def _parse_raw_atd_to_arrays(meta_path: str, raw_path: str, n_frames: int, pitch: tuple) -> dict:
    length, width = float(pitch[0]), float(pitch[1])
    team_of_track = _team_of_each_track(meta_path)

    ball_x = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y = np.full(n_frames, np.nan, dtype=np.float32)
    dist_h1 = np.full(n_frames, 99.0, dtype=np.float32)
    dist_a1 = np.full(n_frames, 99.0, dtype=np.float32)
    dist_h2 = np.full(n_frames, 99.0, dtype=np.float32)
    dist_a2 = np.full(n_frames, 99.0, dtype=np.float32)
    n_players_vis = np.zeros(n_frames, dtype=np.int16)

    print(f"  -> [Module 1] Streaming raw ATD ({os.path.basename(raw_path)}, {n_frames:,} frames)...")
    with open(raw_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            f_idx = int(parts[0]) - 1
            if f_idx < 0 or f_idx >= n_frames:
                continue

            ball = parts[2]
            has_ball = ball[0] != MISSING
            if has_ball:
                c1 = ball.index(",")
                c2 = ball.index(",", c1 + 1)
                bx = float(ball[:c1]) * length
                by = float(ball[c1 + 1 : c2]) * width
                ball_x[f_idx] = bx
                ball_y[f_idx] = by

            h_best1, h_best2 = 9999.0, 9999.0
            a_best1, a_best2 = 9999.0, 9999.0
            p_count = 0

            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if group[0] == MISSING:
                    continue
                p_count += 1
                if has_ball:
                    c1 = group.index(",")
                    c2 = group.index(",", c1 + 1)
                    dx = float(group[:c1]) * length - bx
                    dy = float(group[c1 + 1 : c2]) * width - by
                    d2 = dx * dx + dy * dy
                    if team_of_track[slot] == HOME:
                        if d2 < h_best1:
                            h_best2 = h_best1
                            h_best1 = d2
                        elif d2 < h_best2:
                            h_best2 = d2
                    else:
                        if d2 < a_best1:
                            a_best2 = a_best1
                            a_best1 = d2
                        elif d2 < a_best2:
                            a_best2 = d2

            n_players_vis[f_idx] = p_count
            if has_ball:
                if h_best1 < 9000.0:
                    dist_h1[f_idx] = np.sqrt(h_best1)
                if h_best2 < 9000.0:
                    dist_h2[f_idx] = np.sqrt(h_best2)
                if a_best1 < 9000.0:
                    dist_a1[f_idx] = np.sqrt(a_best1)
                if a_best2 < 9000.0:
                    dist_a2[f_idx] = np.sqrt(a_best2)

    return {
        "ball_x": ball_x,
        "ball_y": ball_y,
        "dist_h1": dist_h1,
        "dist_a1": dist_a1,
        "dist_h2": dist_h2,
        "dist_a2": dist_a2,
        "n_players_vis": n_players_vis,
    }


def _fast_cumulative_votes(codes: np.ndarray, window: int) -> tuple:
    """Exact O(N) cumulative-sum window voter from starter/baseline.py."""
    h_cum = np.concatenate(([0], np.cumsum(codes == HOME)))
    a_cum = np.concatenate(([0], np.cumsum(codes == AWAY)))
    idx = np.arange(len(codes))
    low = np.maximum(0, idx - window)
    high = np.minimum(len(codes), idx + window + 1)
    h_cnt = (h_cum[high] - h_cum[low]).astype(np.float32)
    a_cnt = (a_cum[high] - a_cum[low]).astype(np.float32)
    tot = (high - low).astype(np.float32)
    seen = (h_cnt + a_cnt) > 0
    s_H = np.where(seen & (h_cnt >= a_cnt), 1.0, 0.0).astype(np.float32)
    s_A = np.where(seen & (a_cnt > h_cnt), 1.0, 0.0).astype(np.float32)
    s_D = np.where(~seen, 1.0, 0.0).astype(np.float32)
    # Also return continuous vote ratios
    r_H = h_cnt / np.maximum(1.0, h_cnt + a_cnt)
    r_D = 1.0 - ((h_cnt + a_cnt) / tot)
    return s_H, s_A, s_D, r_H, r_D


def build_tracking_features(n_frames: int = 199319) -> pd.DataFrame:
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, None)
    cache_path = os.path.join(os.path.dirname(raw_path), "atd_extracted_cache.npz")

    if os.path.exists(cache_path):
        print(f"  -> [Module 1] Loading cached raw ATD features from {cache_path}...")
        data = dict(np.load(cache_path))
    else:
        data = _parse_raw_atd_to_arrays(meta_path, raw_path, n_frames, GAMES[GAME]["pitch"])
        np.savez_compressed(cache_path, **data)
        print(f"  -> [Module 1] Cached raw ATD arrays to {cache_path} for fast reloads!")

    ball_x = data["ball_x"]
    ball_y = data["ball_y"]
    ball_vis = (~np.isnan(ball_x)).astype(np.float32)

    df = pd.DataFrame({"frame": np.arange(n_frames, dtype=np.int32)})
    df["ball_visible"] = ball_vis
    df["ball_x"] = pd.Series(ball_x).interpolate(method="linear", limit=62).ffill().bfill().to_numpy()
    df["ball_y"] = pd.Series(ball_y).interpolate(method="linear", limit=62).ffill().bfill().to_numpy()

    # True ball velocity & acceleration (25 fps)
    dx = np.gradient(df["ball_x"].to_numpy())
    dy = np.gradient(df["ball_y"].to_numpy())
    speed = np.sqrt(dx * dx + dy * dy) * 25.0
    df["ball_speed"] = pd.Series(speed).rolling(5, center=True, min_periods=1).mean().to_numpy()
    df["ball_accel"] = np.gradient(df["ball_speed"].to_numpy()) * 25.0

    # Out of bounds flag (105.3m x 68.0m pitch)
    df["out_of_bounds"] = (
        (df["ball_x"] < -0.2) | (df["ball_x"] > 105.5) |
        (df["ball_y"] < -0.2) | (df["ball_y"] > 68.2)
    ).astype(np.float32)

    # Interpolate player distances across short ball occlusions (< 50 frames)
    d_h1_raw = np.where(ball_vis > 0.5, data["dist_h1"], np.nan)
    d_a1_raw = np.where(ball_vis > 0.5, data["dist_a1"], np.nan)
    d_h2_raw = np.where(ball_vis > 0.5, data["dist_h2"], np.nan)
    d_a2_raw = np.where(ball_vis > 0.5, data["dist_a2"], np.nan)

    df["dist_h1"] = pd.Series(d_h1_raw).interpolate(limit=50).fillna(10.0).to_numpy(dtype=np.float32)
    df["dist_a1"] = pd.Series(d_a1_raw).interpolate(limit=50).fillna(10.0).to_numpy(dtype=np.float32)
    df["dist_h2"] = pd.Series(d_h2_raw).interpolate(limit=50).fillna(15.0).to_numpy(dtype=np.float32)
    df["dist_a2"] = pd.Series(d_a2_raw).interpolate(limit=50).fillna(15.0).to_numpy(dtype=np.float32)

    # Exact nearest-player codes & multi-scale cumulative voters directly from raw ATD!
    codes = np.where(
        ball_vis < 0.5,
        -1,
        np.where(data["dist_h1"] <= data["dist_a1"], HOME, AWAY)
    ).astype(np.int8)

    df["raw_H"] = (codes == HOME).astype(np.float32)
    df["raw_A"] = (codes == AWAY).astype(np.float32)
    df["raw_D"] = (codes == -1).astype(np.float32)

    for w, name in [(25, "s25"), (50, "s50"), (85, "s85"), (125, "s125")]:
        sH, sA, sD, rH, rD = _fast_cumulative_votes(codes, w)
        df[f"{name}_H"] = sH
        df[f"{name}_A"] = sA
        df[f"{name}_D"] = sD
        df[f"{name}_ratio_H"] = rH
        df[f"{name}_missing_rate"] = rD

    return df