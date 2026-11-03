# Team Matra — Developer Architecture & Onboarding Guide

**Competition:** More than a Hack 2027 — Phase 1: Ball Possession (FC Barcelona vs. Athletic Club, LaLiga 2024–25 J2)  
**Current Benchmark Status:** **Rank #1** (`spellF1 = 0.4875` | `IoU@0.5 = 0.5625` | `macroF1 = 0.6502` | `shareE = 0.0448`)  

---

## 1. Quick-Start Onboarding for Team Members

Every developer works on their isolated module inside `src/` while testing end-to-end changes through the master runner (`run_pipeline.py`). Our production stack relies strictly on **`numpy` and `pandas`** (`requirements.txt`), guaranteeing zero environment conflicts across Python 3.8–3.14.

### Local Setup & Execution
```bash
# 1. Clone the team repository and enter the project root
git clone https://github.com/<your-org>/<your-team-repo>.git
cd more-than-a-hack-2027

# 2. Install core dependencies (numpy & pandas)
pip install -r requirements.txt

# 3. Verify that raw match data is unpacked locally (DO NOT commit ./data to Git!)
#    Required files:
#      data/tactical/BAR-ATH_tactical_atd.xml
#      data/tactical/BAR-ATH_tactical_atd.txt
#      data/tactical/BAR-ATH_tactical_smart_tagging.xml

# 4. Run the full 4-module pipeline, format validator, and leaderboard evaluator
python run_pipeline.py --team_name Matra
```
* **First Run (Cold Start):** Takes ~40 seconds while `Module 1` streams all 199,319 lines of `BAR-ATH_tactical_atd.txt` and generates `data/tactical/atd_extracted_cache.npz`.
* **Subsequent Runs (Warm Cache):** Takes **~2.0 seconds** end-to-end.

---

## 2. System Data Flow & DataFrame Contracts

Every module operates on a fixed **199,319-row** timeline (`frame = 0` to `199318` at `25.0 fps`). 

> **Global Invariant:** Never drop rows, filter out ungraded frames (`X`), or alter row ordering inside `src/module1..4.py`. Every returned `DataFrame` or `ndarray` must have length `199,319`.

| Stage | Script | Owner | Primary Input | Primary Output (`DataFrame` / `ndarray`) | Execution Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | `src/module1_tracking.py` | **Suriya** | `BAR-ATH_tactical_atd.xml` & `.txt` | `df_track` (`199,319 × 30` kinematic & voter columns) | ~40s cold / **0.2s cached** |
| **Stage 2** | `src/module2_deadball_tags.py` | **Abhishek** | `BAR-ATH_tactical_smart_tagging.xml` + `df_track` | `df_tags` (`199,319 × 7` tactical & set-piece columns) | **0.15s** |
| **Stage 3** | `src/module3_possession.py` | **Ranganathan** | `df_track` + `df_tags` | `df_probs` (`frame, p_H, p_A, p_D, is_setpiece_window`) | **0.05s** |
| **Stage 4** | `src/module4_decoder.py` | **Varsha** | `df_probs` + Half frame tuples | `labels` (`1D numpy.ndarray` of `'H'`, `'A'`, `'D'`, `'X'`) | **0.10s** |
| **Master** | `run_pipeline.py` | **Ragavendran** | Modules 1–4 | `evaluation/results/Matra.csv` + Validator + Scorer | **~2.0s total** |

---

## 3. Module-by-Module Technical Specifications

### 3.1 Module 1 — Spatial Tracking & Multi-Horizon Voters
* **File:** `src/module1_tracking.py`
* **Owner:** **Suriya**
* **Entry Function:** `build_tracking_features(n_frames: int = 199319) -> pd.DataFrame`

#### Internal Mechanics
1. **Track Slot to Team Mapping (`_team_of_each_track`):** Reads `BAR-ATH_tactical_atd.xml` and maps the 46 anonymous tracklet slots (`Track 1..46`) to `HOME = 0` (`Barcelona`) and `AWAY = 1` (`Atletic Club`).
2. **Timeline Alignment (`_parse_raw_atd_to_arrays`):** Each line in `BAR-ATH_tactical_atd.txt` follows `<frame_id>:<46 player slots separated by ';'>:<ball>`. Alignment is strictly `f_idx = int(parts[0]) - 1`.
3. **Metric Pitch Scaling (`105.3m × 68.0m`):** Normalized `[0, 1]` coordinates are multiplied by `length = 105.3` and `width = 68.0` before computing Euclidean distances (`dist_h1`, `dist_h2`, `dist_a1`, `dist_a2`).
4. **Missing-Ball Imputation & Kinematics:** Short missing-ball gaps ($\le 62$ frames / $2.48\text{s}$) are linearly interpolated to compute smoothed `ball_speed` ($\text{m/s}$) and `ball_accel` ($\text{m/s}^2$).
5. **Exact $O(N)$ Cumulative-Sum Window Voters (`_fast_cumulative_votes`):** Computes centred majority-vote flags (`s{w}_H`, `s{w}_A`, `s{w}_D`) and continuous vote ratios (`s{w}_ratio_H`, `s{w}_missing_rate`) across four half-widths (`w = 25, 50, 85, 125` frames) using `np.cumsum`.

### 3.2 Module 2 — Smart Tagging XML Engine & Throw-In Resolver
* **File:** `src/module2_deadball_tags.py`
* **Owner:** **Abhishek**
* **Entry Function:** `build_deadball_and_tag_features(df_track: pd.DataFrame, n_frames: int = 199319) -> pd.DataFrame`

#### Internal Mechanics
1. **Direct XML Parsing (`xml.etree.ElementTree`):** Reads all 638 `<instance>` blocks from `data/tactical/BAR-ATH_tactical_smart_tagging.xml` and maps `<start>` and `<end>` seconds to video frames via `int(round(float(sec) * 25.0))`.
2. **Pre-Execution Set-Piece Stoppage Buffer (`100 frames` / `4.0s`):** Prepends a 100-frame buffer (`sp_start = max(0, s_f - 100)`) to `is_setpiece_window` for every `SET PIECES` and `GOALS` tag to capture the stoppage prior to the restart kick.
3. **Algorithmic Resolution of `Team = N/A` Set Pieces:** Resolves **31 set-piece restarts** (including all 26 Throw-ins) by inspecting the first 25 visible-ball frames inside `[s_f, e_f + 50]` and assigning restart ownership to `'H'` if `mean(dist_h1) <= mean(dist_a1)` else `'A'`.
4. **Offensive & Defensive Polarity Mapping:** Attacking tags activate `tag_active_H` or `tag_active_A` for the tagged team; defensive tags invert polarity to the opposing team.

### 3.3 Module 3 — Multi-Horizon Fusion & XML Set-Piece Gate
* **File:** `src/module3_possession.py`
* **Owner:** **Ranganathan**
* **Entry Function:** `compute_frame_probabilities(df_track: pd.DataFrame, df_tags: pd.DataFrame, n_frames: int = 199319) -> pd.DataFrame`

#### Internal Mechanics
1. **Four-Horizon Spatial Ensemble:** Computes baseline class support across four smoothing scales:
   $$P_{\text{track}}(c) = 0.52 \cdot S_{125}(c) + 0.22 \cdot S_{85}(c) + 0.16 \cdot S_{50}(c) + 0.10 \cdot S_{25}(c)$$
2. **Conjunctive XML Set-Piece Dead-Ball Gate:** Boosts `p_d` by `+0.45` only when `is_setpiece_window > 0.5` AND `(raw_D > 0.5 | s25_D > 0.5)`, increasing true Dead-Ball detections from `116` to `316` frames without slicing open-play possessions.
3. **Uncontested Tactical Phase Nudge:** Adds `+0.08` to `p_h` (or `p_a`) when `only_h` (or `only_a`) is active and `~confirmed_dead`.

### 3.4 Module 4 — Period-Aware Morphological Spell Decoder
* **File:** `src/module4_decoder.py`
* **Owner:** **Varsha**
* **Entry Function:** `decode_possession_spells(df_probs: pd.DataFrame, h1_bounds=(11185, 90651), h2_bounds=(113742, 190197)) -> np.ndarray`

#### Internal Mechanics
1. **Pass 1 — Set-Piece-Protected Sandwich Stitcher (`sandwich_win = 48`):** Stitches short state flickers (`< 48` frames / $1.92\text{s}$) flanked by identical states into the surrounding spell, while **protecting XML-confirmed Dead-Ball (`D`) segments $\ge 15$ frames ($0.6\text{s}$)**.
2. **Pass 2 — Boundary Step Stitcher (`step_win = 18`):** Absorbs tiny transition blips (`< 18` frames / $0.72\text{s}$) into the longer neighboring spell.

### 3.5 Master Runner — Orchestration & Hard Boundary Locks
* **File:** `run_pipeline.py`
* **Owner:** **Ragavendran (Team Lead)**
* **Hard Boundary Locks:**
  * `0 .. 11,184` $\rightarrow$ `'X'` (Pre-match ungraded)
  * `11,185 .. 90,651` $\rightarrow$ Decoded `'H'`, `'A'`, `'D'` (First Half active play)
  * `90,652 .. 110,807` $\rightarrow$ `'X'` (Halftime break ungraded)
  * `110,808 .. 113,741` $\rightarrow$ `'D'` (Graded 2,934-frame pre-2H kickoff dead-ball trap)
  * `113,742 .. 190,197` $\rightarrow$ Decoded `'H'`, `'A'`, `'D'` (Second Half active play)
  * `190,198 .. 199,318` $\rightarrow$ `'X'` (Post-match ungraded)