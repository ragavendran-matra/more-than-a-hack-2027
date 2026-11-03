"""
MODULE 2 (Abhishek): Smart Tagging XML Parser, Dead-Ball Engine & Throw-In Resolver
Parses all 638 tactical phases from BAR-ATH_tactical_smart_tagging.xml and resolves
the 36 Team=N/A events (including all 26 Throw-ins) using Module 1's real player distances.
"""
import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd


def _find_smart_tagging_xml() -> str:
    candidates = [
        os.path.join("data", "tactical", "BAR-ATH_tactical_smart_tagging.xml"),
        os.path.join("data", "BAR-ATH_tactical_smart_tagging.xml"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    if os.path.exists("data"):
        for root, _, files in os.walk("data"):
            for f in files:
                if f.endswith("smart_tagging.xml"):
                    return os.path.join(root, f)
    return ""


def build_deadball_and_tag_features(df_track: pd.DataFrame, n_frames: int = 199319) -> pd.DataFrame:
    prob_dead = (
        0.45 * df_track["raw_D"].to_numpy()
        + 0.30 * df_track["s25_D"].to_numpy()
        + 0.25 * df_track["s50_D"].to_numpy()
    )
    tag_H = np.zeros(n_frames, dtype=np.float32)
    tag_A = np.zeros(n_frames, dtype=np.float32)
    is_setpiece = np.zeros(n_frames, dtype=np.float32)
    restart_flag = np.zeros(n_frames, dtype=np.float32)
    restart_team = np.full(n_frames, "NA", dtype=object)

    dist_h1 = df_track["dist_h1"].to_numpy()
    dist_a1 = df_track["dist_a1"].to_numpy()
    ball_vis = df_track["ball_visible"].to_numpy()

    xml_path = _find_smart_tagging_xml()
    if xml_path:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        instances = root.findall(".//instance")
        n_loaded, n_na_resolved = 0, 0

        for inst in instances:
            start_el = inst.find("start")
            end_el = inst.find("end")
            code_el = inst.find("code")
            if start_el is None or end_el is None or code_el is None:
                continue

            s_f = max(0, min(n_frames - 1, int(round(float(start_el.text) * 25.0))))
            e_f = max(0, min(n_frames - 1, int(round(float(end_el.text) * 25.0))))
            code = code_el.text.strip().upper()

            team = "NA"
            for label in inst.findall("label"):
                grp = label.find("group")
                txt = label.find("text")
                if grp is not None and txt is not None and grp.text.strip().lower() == "team":
                    t_val = txt.text.strip().lower()
                    if "barcelona" in t_val:
                        team = "H"
                    elif "atletic" in t_val or "athletic" in t_val:
                        team = "A"

            # Resolve Team=N/A events (especially the 26 Throw-ins!) using post-restart player proximity
            if team == "NA" and code == "SET PIECES":
                look_s = min(n_frames - 1, s_f)
                look_e = min(n_frames, e_f + 50)
                vis_idx = np.flatnonzero(ball_vis[look_s:look_e] > 0.5) + look_s
                if len(vis_idx) > 0:
                    h_mean = np.mean(dist_h1[vis_idx[:25]])
                    a_mean = np.mean(dist_a1[vis_idx[:25]])
                    team = "H" if h_mean <= a_mean else "A"
                    n_na_resolved += 1

            if code in ("SET PIECES", "GOALS"):
                sp_start = max(0, s_f - 100)
                is_setpiece[sp_start : e_f + 1] = 1.0
                restart_flag[s_f] = 1.0
                if team in ("H", "A"):
                    restart_team[s_f : min(n_frames, e_f + 35)] = team
            elif code in ("BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION",
                          "LONG BALL", "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX"):
                if team == "H":
                    tag_H[s_f : e_f + 1] = 1.0
                elif team == "A":
                    tag_A[s_f : e_f + 1] = 1.0
            elif "DEFENDING" in code or "DEFENSIVE TRANSITION" in code:
                if team == "H":
                    tag_A[s_f : e_f + 1] = 1.0
                elif team == "A":
                    tag_H[s_f : e_f + 1] = 1.0
            n_loaded += 1
        print(f"  -> [Module 2] Loaded {n_loaded} XML tags & resolved {n_na_resolved} N/A set-pieces!")

    # FAQ.md trap fix: frames 110,808 to 113,741 are 100% Dead Ball before 2H kickoff
    prob_dead[110808:113742] = 0.999
    is_setpiece[110808:113742] = 1.0

    return pd.DataFrame({
        "frame": np.arange(n_frames, dtype=np.int32),
        "prob_dead_prior": np.clip(prob_dead, 0.01, 0.999),
        "tag_active_H": tag_H,
        "tag_active_A": tag_A,
        "is_setpiece_window": is_setpiece,
        "restart_frame_flag": restart_flag,
        "restart_team": restart_team,
    })