# Team Matra — Ball Possession Prediction Pipeline
**More than a Hack 2027 — Phase 1: Ball Possession (FC Barcelona vs. Athletic Club, LaLiga 2024–25 J2)**

---

## Leaderboard Status & Performance
* **Status:** **Rank #1** across both primary and secondary evaluation metrics.
* **Primary Metric (`spellF1`):** **`0.4875`** (vs. `0.4242` for the strongest official baseline; **`0.5625`** at `IoU@0.5` matching 9 of 17 true possessions).
* **Tie-Breaker Metric (`macroF1`):** **`0.6502`** (Highest frame-level accuracy on the board: `H = 0.7868`, `A = 0.8217`, `D = 0.3422`).
* **Possession Share Error (`shareE`):** **`0.0448`** (Lowest possession share error on the leaderboard).

---

## Team Matra Roster
* **Ragavendran** (Team Lead — Pipeline Orchestration, Boundary Locks & Full-Match QA)
* **Suriya** (Spatial Tracking, Missing-Ball Imputation & Top-K Player Kinematics)
* **Abhishek** (Tactical Smart Tagging XML Engine, Dead-Ball Priors & Throw-In Resolution)
* **Ranganathan** (Multi-Scale Horizon Fusion & Set-Piece Gate Calibration)
* **Varsha** (Period-Aware Sequence Decoding & Morphological Spell Stitching)

---

## Single-Command Reproduction Instructions

Our production pipeline is implemented purely in **Python using standard NumPy and Pandas** (`requirements.txt`), requiring no third-party tracking libraries (`kloppy`) or pre-computed baseline files.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Dataset Placement
Ensure the official match dataset is unpacked under `./data/tactical/`:
* `data/tactical/BAR-ATH_tactical_atd.xml`
* `data/tactical/BAR-ATH_tactical_atd.txt`
* `data/tactical/BAR-ATH_tactical_smart_tagging.xml`

### 3. Run the Full Pipeline & Evaluators
Execute the master orchestration script from the repository root:
```bash
python run_pipeline.py --team_name Matra
```

This single command automatically executes all five stages:
1. **Module 1 (`src/module1_tracking.py` — Suriya):** Streams all 199,319 frames and 46 track slots from `BAR-ATH_tactical_atd.txt`, scales coordinates to metric metres (`105.3m × 68.0m`), imputes short missing-ball occlusions, and computes $O(N)$ cumulative-sum voters (`w = 25, 50, 85, 125`).
2. **Module 2 (`src/module2_deadball_tags.py` — Abhishek):** Parses 638 tactical intervals from `BAR-ATH_tactical_smart_tagging.xml`, prepends a 100-frame ($4.0\text{s}$) pre-execution stoppage buffer to set pieces, and resolves 31 `Team = N/A` restarts via player proximity.
3. **Module 3 (`src/module3_possession.py` — Ranganathan):** Blends multi-scale tracking signals with the conjunctive XML Set-Piece Dead-Ball Gate.
4. **Module 4 (`src/module4_decoder.py` — Varsha):** Period-aware spell decoding (sandwich and step stitching) applied separately to Half 1 (`11,185–90,651`) and Half 2 (`113,742–190,197`), locking ungraded frames to `X` and pre-2H kickoff frames to `D` (`110,808–113,741`).
5. **Validation & Scoring:** Writes `evaluation/results/Matra.csv`, validates its format via `evaluation/validate_submission.py`, and prints the ranked scoreboard via `evaluation/run_evaluation.py`.

---

## Repository Directory Structure

```text
more-than-a-hack-2027/
├── data/                        # Local match files (Excluded via .gitignore)
├── evaluation/
│   ├── gt/                      # Public ground truth CSVs
│   ├── results/                 # Submission CSVs (Matra.csv, baselines)
│   ├── reports/                 # Per-submission diagnostic reports
│   ├── run_evaluation.py        # Official scoreboard & metrics evaluator
│   └── validate_submission.py   # Format & boundary validator
├── starter/                     # Organizer baseline scripts & reference loaders
├── src/
│   ├── __init__.py
│   ├── module1_tracking.py      # Suriya: ATD tracking & multi-horizon voters
│   ├── module2_deadball_tags.py # Abhishek: Smart Tagging XML engine & throw-ins
│   ├── module3_possession.py    # Ranganathan: Horizon fusion & Set-Piece Gate
│   └── module4_decoder.py       # Varsha: Period-aware Viterbi & spell stitcher
├── run_pipeline.py              # Ragavendran: Master orchestration runner
├── DEVELOPER_GUIDE.md           # Detailed technical architecture handbook
├── Matra.md                     # Complete 6-section Technical Report Markdown
├── Matra.pdf                    # Standalone PDF Technical Report for submission
├── requirements.txt             # Core dependencies (numpy, pandas)
├── .gitignore                   # Blocks data/ and *.npz from being uploaded
└── README.md                    # Single-command reproduction instructions
```