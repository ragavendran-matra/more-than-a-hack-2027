# Team Matra — Technical Report: Period-Aware Spatio-Temporal Possession Decoding
**Competition:** More than a Hack 2027 — Phase 1: Ball Possession (FC Barcelona vs. Athletic Club, LaLiga 2024–25 J2)  
**Team Name:** Matra  
**Team Members:**  
* **Ragavendran** (Team Lead — Pipeline Integration, Boundary Enforcement & Full-Match QA)  
* **Suriya** (Spatial Tracking, Missing-Ball Imputation & Top-K Player Kinematics)  
* **Abhishek** (Tactical Smart Tagging XML Engine, Dead-Ball Priors & Throw-In Resolution)  
* **Ranganathan** (Multi-Scale Horizon Fusion & Set-Piece Gate Calibration)  
* **Varsha** (Period-Aware Sequence Decoding & Morphological Spell Stitching)  

---

## 1. Executive Summary & System Overview

Determining continuous ball possession in football from optical tracking data requires bridging the gap between noisy frame-level spatial proximity and high-level tactical continuity. Raw nearest-player heuristics suffer from severe state flickering during passes in flight, contested duels, and the 41% of graded frames where the ball track is missing (`NaN`). Conversely, naive wide-window majority smoothing (such as a $\pm 125$-frame rolling voter) erases genuine 2-to-4 second dead-ball stoppages (`D`), merging separate consecutive possessions by the same team into artificial mega-spells.

**Team Matra** solves this trade-off through a **4-stage, pure NumPy/Pandas modular architecture** that requires zero fragile third-party tracking libraries or pre-computed baseline files:
1. **Direct FIFA/EPTS Streamer & Kinematic Imputer (`src/module1_tracking.py`):** Streams all 199,319 frames and 46 anonymous player track slots directly from `BAR-ATH_tactical_atd.txt` and `.xml`, scaling coordinates to metres (`105.3m × 68.0m`), interpolating short ball occlusions ($\le 62$ frames), and computing exact $O(N)$ cumulative-sum possession voters across four temporal horizons ($\pm 1.0\text{s}$, $\pm 2.0\text{s}$, $\pm 3.4\text{s}$, and $\pm 5.0\text{s}$).
2. **Smart Tagging XML Engine & `Team=N/A` Restart Resolver (`src/module2_deadball_tags.py`):** Parses all 638 tactical annotations from `BAR-ATH_tactical_smart_tagging.xml`, introduces a 4.0-second (`100-frame`) pre-execution stoppage buffer prior to tagged set pieces, and algorithmically resolves 31 `Team = N/A` set-piece events (including all 26 Throw-ins) by measuring post-restart player proximity.
3. **Multi-Scale Probability Fusion & Set-Piece Gate (`src/module3_possession.py`):** Fuses multi-horizon tracking ratios with a strict **XML Set-Piece Dead-Ball Gate** that boosts Dead Ball (`D`) probability only when a tracking stoppage coincides with a verified tactical set-piece/goal window, preventing false dead-ball splits during live play.
4. **Period-Aware Morphological Spell Decoder (`src/module4_decoder.py`):** Decodes the 1st and 2nd halves independently, applying a two-pass morphological spell stitcher (sandwich stitching and step stitching) that eliminates sub-2-second mid-play flickers while strictly protecting XML-confirmed dead-ball stoppages ($\ge 15$ frames).

On the public evaluation benchmark, **Team Matra ranks #1 overall** on both the primary ranking metric (`spellF1 = 0.4875` vs. `0.4242` for the strongest baseline; `IoU@0.5 = 0.5625` with 9/17 matched spells) and the tie-breaker metric (`macroF1 = 0.6502`), while achieving the lowest possession share error (`0.0448`) on the leaderboard.

---

## 2. Dataset Assets Used, Excluded & Justification

| Asset File | Status | Role in Team Matra Pipeline / Reason for Exclusion |
| :--- | :---: | :--- |
| `data/tactical/BAR-ATH_tactical_atd.xml` | **Used** | Parsed via `xml.etree.ElementTree` to map the 46 anonymous tracklet slots (`Track 1..46`) to Home (`Barcelona`, `H`) and Away (`Atletic Club`, `A`) teams and retrieve pitch dimensions (`105.3m × 68.0m`). |
| `data/tactical/BAR-ATH_tactical_atd.txt` | **Used** | Streamed line-by-line (`199,319` records; `video_frame = frame_id - 1`) to extract 2D ball coordinates, player-to-ball Euclidean distances (`dist_h1, dist_h2, dist_a1, dist_a2`), and multi-horizon cumulative votes. |
| `data/tactical/BAR-ATH_tactical_smart_tagging.xml` | **Used** | Parsed all 638 coach-tagged tactical intervals (`frame = round(seconds * 25)`), utilizing the 59 `SET PIECES` and 4 `GOALS` tags to anchor dead-ball stoppages and offensive/defensive phase tags to disambiguate contested sequences. |
| `data/tactical/BAR-ATH_tactical.mp4` | **Used (QA)** | Used strictly for visual inspection of edge cases and verifying the 2nd-half pre-kickoff dead-ball window (`110,808–113,741`). No manual frame labeling was performed. |
| `data/tactical/BAR-ATH_tactical_homography.csv` | **Excluded** | Excluded because the Tactical ATD already provides pitch-plane $(x, y)$ coordinates across 100% of frames, making pixel-to-pitch projection redundant. |
| `data/broadcast/*` (All 4 Broadcast Assets) | **Excluded** | Intentionally excluded. The broadcast feed has a separate timeline (`164,488` frames), contains TV camera cuts and replays, and suffers from `34.2%` invalid homography matrices. Mixing unsynchronized broadcast tracks degrades temporal boundary precision. |

---

## 3. Methodology & Four-Module Pipeline Architecture

### 3.1 Module 1: Spatial Tracking & Missing-Ball Imputation (`src/module1_tracking.py`)
* **Coordinate Scaling & Alignment:** Each record `frame_id` (`1..199319`) is mapped to submission index `frame = frame_id - 1` (`0..199318`). Normalized `[0, 1]` coordinates are scaled to metres (`105.3m × 68.0m`) prior to distance calculation so touchline distances are not distorted by the pitch aspect ratio.
* **Missing-Ball Imputation:** The ball is `NaN` on 41% of graded frames. For short occlusions ($\le 62$ frames / $2.5\text{s}$), ball coordinates $(x, y)$ and nearest-player distances (`dist_h1..2`, `dist_a1..2`) are linearly interpolated to compute continuous ball speed ($\text{m/s}$) and acceleration ($\text{m/s}^2$).
* **Exact $O(N)$ Multi-Horizon Cumulative Voters:** Using cumulative sums over the per-frame nearest-team array (`HOME = 0`, `AWAY = 1`, `NOBODY = -1`), Module 1 computes centred majority-vote indicators and continuous team ratios across four complementary half-widths: `w = 25` ($\pm 1.0\text{s}$), `w = 50` ($\pm 2.0\text{s}$), `w = 85` ($\pm 3.4\text{s}$), and `w = 125` ($\pm 5.0\text{s}$).

### 3.2 Module 2: Smart Tagging XML Engine & Throw-In Resolution (`src/module2_deadball_tags.py`)
* **Pre-Execution Set-Piece Stoppage Buffer:** In `BAR-ATH_tactical_smart_tagging.xml`, `SET PIECES` timestamps mark the *execution* of the restart, whereas the dead-ball stoppage begins several seconds earlier when the ball exits play or a foul occurs. Module 2 prepends a `100-frame` ($4.0\text{s}$) lead-in window (`[max(0, s_f - 100), e_f]`) to every `SET PIECES` and `GOALS` tag.
* **Algorithmic Resolution of `Team = N/A` Throw-Ins:** Exactly 36 Smart Tagging events—including all 26 Throw-ins—have `Team = N/A`. Module 2 resolves 31 set-piece restarts by scanning the first 25 visible-ball frames inside `[s_f, e_f + 50]` and assigning restart ownership to the team with the lower mean nearest-player distance (`mean(dist_h1)` vs. `mean(dist_a1)`).
* **Offensive & Defensive Polarity Mapping:** Attacking tags (`BUILD UP`, `PROGRESSION`, `CREATING CHANCES`, `ATTACKING TRANSITION`, `BALL IN FINAL THIRD`, `BALL IN THE BOX`) assign positive prior support to the tagged team, while defensive tags (`DEFENDING IN ...`, `DEFENSIVE TRANSITION`) assign possession support to the **opposing** team.

### 3.3 Module 3: Multi-Scale Probability Fusion & XML Set-Piece Gate (`src/module3_possession.py`)
* **Multi-Horizon Blending:** Rather than relying on a single smoothing window (which either over-fragments at `w=50` or over-smooths at `w=125`), Module 3 blends four temporal scales:
  $$P_{\text{track}}(c) = 0.52 \cdot S_{125}(c) + 0.22 \cdot S_{85}(c) + 0.16 \cdot S_{50}(c) + 0.10 \cdot S_{25}(c), \quad c \in \{H, A, D\}$$
* **XML Set-Piece Dead-Ball Gate:** A critical flaw in `baseline_smooth125` is that if the ball is visible on even a single frame within a 10-second ($\pm 125$-frame) window, the entire window is forced to `H` or `A`, destroying Dead Ball (`D`) recall (`7.9%`). Conversely, trusting raw missing-ball frames (`raw_D`) everywhere injects false `D` splits during mid-pitch aerial duels. Module 3 solves this via a conjunctive gate:
  $$\text{confirmed\_dead}(t) = \mathbb{I}\big(\text{is\_setpiece\_window}(t) = 1\big) \land \mathbb{I}\big(\text{raw\_D}(t) = 1 \lor S_{25}(D, t) = 1\big)$$
  When `confirmed_dead(t)` is true, $P(D)$ receives a `+0.45` boost, recovering genuine stoppages before throw-ins, corners, and free kicks while keeping false dead-ball splits during open play near zero.

### 3.4 Module 4 & Master Runner: Period-Aware Spell Stitching & Boundary Rules (`src/module4_decoder.py` & `run_pipeline.py`)
* **Two-Pass Morphological Spell Stitcher:** Decodes Half 1 (`11,185–90,651`) and Half 2 (`113,742–190,197`) separately:
  1. *Pass 1 (Set-Piece-Protected Sandwich Stitcher):* Eliminates short state flickers (`A` inside `H..H` or `H` inside `A..A`) shorter than `sandwich_win = 48` frames ($1.92\text{s}$), **except** when the segment is a Dead Ball (`D`) of length $\ge 15$ frames ($0.6\text{s}$) overlapping an XML set-piece window (`mean(setpiece_mask) > 0.2`).
  2. *Pass 2 (Step Stitcher):* Absorbs ultra-brief boundary transition blips (`< 18` frames / $0.72\text{s}$) between live play and stoppages into the longer adjacent spell.
* **Hard Boundary Enforcement (`run_pipeline.py`):** Forces `X` on all ungraded intervals (`0–11,184`, `90,652–110,807`, `190,198–199,318`) and enforces `D` across the entire 2,934-frame graded pre-kickoff 2nd-half window (`110,808–113,741`).

---

## 4. Quantitative Evaluation & Ablation Results

Evaluated on the official public ground truth (`frames 11,185–18,684`, 7,500 graded frames, 17 possessions of mean duration $14.2\text{s}$), **Team Matra** outperforms all four official baselines across both primary and secondary metrics:

| Rank | Submission | `spellF1` (Mean) | `IoU@0.3` | `IoU@0.5` | `IoU@0.7` | `macroF1` | `F1 (H)` | `F1 (A)` | `F1 (D)` | Poss (`GT=17`) | Dur (`GT=14.2s`) | Share Err | Frag (`GT=1.0`) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **`Matra` (Ours)** | **`0.4875`** | **`0.6875`** | **`0.5625`** | **`0.2500`** | **`0.6502`** | **`0.7868`** | **`0.8217`** | **`0.3422`** | **`15`** | **`19.0s`** | **`0.0448`** | **`0.81`** |
| 2 | `baseline_smooth125` | `0.4242` | `0.6061` | `0.4242` | `0.2424` | `0.5812` | `0.7864` | `0.8108` | `0.1464` | `16` | `18.5s` | `0.0520` | `0.81` |
| 3 | `baseline_smooth50` | `0.3256` | `0.5116` | `0.3256` | `0.0930` | `0.6207` | `0.7781` | `0.7615` | `0.3225` | `26` | `10.8s` | `0.0485` | `1.43` |
| 4 | `baseline_smooth25` | `0.2868` | `0.4528` | `0.3019` | `0.0755` | `0.6200` | `0.7620` | `0.7123` | `0.3857` | `36` | `7.6s` | `0.0556` | `1.90` |
| 5 | `baseline_raw` | `0.0378` | `0.0946` | `0.0270` | `0.0000` | `0.5972` | `0.7290` | `0.6430` | `0.4197` | `131` | `2.0s` | `0.0457` | `6.52` |

### Key Diagnostic Highlights
* **+13.83 Point Gain at `IoU@0.5`:** `Matra` matches **9 out of 17** true possessions at `IoU >= 0.5` (`F1 = 0.5625`, Precision `0.6000`, Recall `0.5294`) with only 6 spurious spells, compared to 7 matched and 9 spurious for `baseline_smooth125` (`F1 = 0.4242`).
* **2.3x Higher Dead-Ball F1 (`0.3422` vs. `0.1464`):** The XML Set-Piece Gate increases true positive Dead Ball frames from `116` to **`316`** (`83.6%` precision), separating consecutive same-team possessions that `baseline_smooth125` erroneously merges.
* **Full-Match Validation (`199,319` frames):** Across the entire 90-minute match (`158,857` graded frames), `Matra.csv` produces realistic LaLiga macro-statistics: `30.7%` Home (`61,179` frames), `21.9%` Away (`43,689` frames), `27.1%` Dead Ball (`53,989` frames), and `20.3%` ungraded `X` (`40,462` frames), with a full-match mean spell duration of ~`15.0s`.

---

## 5. Known Limitations & Future Extensions

1. **Un-tagged Mid-Half Stoppages (Drop-Balls & Minor Fouls Without Set-Piece Tags):** Because the Dead-Ball Gate requires confirmation from `SET PIECES` or `GOALS` tags to override the long-horizon smoother, brief stoppages that lack an XML set-piece tag can still be smoothed over if the ball disappears for less than 3 seconds. Incorporating collective player deceleration (`mean_player_speed < 1.2 m/s`) would capture untagged stoppages.
2. **Exact Frame-Level Transition Latency on Quick One-Touch Turnovers:** Rapid midfield interceptions followed by immediate counter-passes exhibit a median boundary lag of `-3.0` frames ($0.12\text{s}$). Detecting sharp changes in the ball's velocity vector ($\Delta \vec{v}_{\text{ball}}$) at player contact could snap turnover boundaries to the exact touch frame.
3. **Anonymous Tracklet Swaps During Crowded Set Pieces:** Because the 46 ATD track slots do not preserve player identities across occlusions, goalkeeper vs. outfielder roles inside the penalty box are not explicitly distinguished. Adding a penalty-box spatial prior for goalkeeper catches would further sharpen set-piece end frames.

---

## 6. Exact Reproducibility Instructions

Our pipeline requires **only Python 3.8+ with `numpy` and `pandas`** (the exact dependencies in `requirements.txt`) and runs end-to-end from the raw `./data` directory in under 45 seconds on a standard CPU (and ~2 seconds on subsequent runs using the auto-generated `.npz` tracking cache).

```bash
# 1. Install standard dependencies
pip install -r requirements.txt

# 2. Ensure raw match files are unpacked under ./data/tactical/
#    - data/tactical/BAR-ATH_tactical_atd.xml
#    - data/tactical/BAR-ATH_tactical_atd.txt
#    - data/tactical/BAR-ATH_tactical_smart_tagging.xml

# 3. Reproduce evaluation/results/Matra.csv, validate format, and score on leaderboard
python run_pipeline.py --team_name Matra

# 4. (Optional) Run 1,000-draw bootstrap confidence interval evaluation
python evaluation/run_evaluation.py --bootstrap 1000
```