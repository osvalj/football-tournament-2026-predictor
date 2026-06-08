# ⚽ Football Tournament 2026 Predictor

> Multiclass classification model to predict football match results (home win / draw / away win), applied to simulate the 2026 international football tournament using Monte Carlo simulation.

**Live Dashboard → [football-tournament-2026.streamlit.app](https://share.streamlit.io)**

---

## Why this project

Predicting football results is a classic machine learning challenge — three possible outcomes, class imbalance, high inherent randomness, and limited public data. The 2026 tournament, with its expanded 48-team format across 12 groups, offered a concrete and timely application.

The goal was not to build a perfect predictor (the sport's randomness makes that impossible), but to build a rigorous, well-documented pipeline that demonstrates the full ML workflow: from raw data to a deployed interactive dashboard.

---

## What we built

A complete ML pipeline that:

1. **Cleans and merges** four public datasets (FIFA rankings, match results, EA FC 26 player ratings, Transfermarkt squad data)
2. **Engineers 26 features** capturing team quality, recent form, head-to-head history, and scoring patterns
3. **Trains and compares** 8 models (Decision Tree, Logistic Regression, Random Forest, XGBoost, LightGBM, Extra Trees, Stacking, Soft Voting)
4. **Tunes** the best model via RandomizedSearchCV with 150 iterations
5. **Evaluates** on a held-out temporal test set (June 2025 → 2026) — never touched during training or tuning
6. **Simulates** the full tournament 100,000 times using Monte Carlo
7. **Visualizes** results in an interactive Streamlit dashboard

---

## Results

| Metric | Value |
|--------|-------|
| Model | Random Forest |
| Test Accuracy | **65.1%** |
| F1 Macro | **56.1%** |
| Test Period | Jun 2025 → Jun 2026 (502 matches) |
| Simulations | 100,000 Monte Carlo iterations |

**Top champion probabilities:**

| Team | Champion | Final | Semifinals |
|------|----------|-------|------------|
| France | 10.7% | 17.0% | 26.8% |
| Argentina | 9.7% | 15.7% | 25.1% |
| Brazil | 8.5% | 14.5% | 23.8% |
| England | 7.9% | 13.7% | 22.8% |
| Belgium | 7.4% | 13.3% | 22.6% |
| Spain | 6.8% | 12.5% | 21.9% |

---


## What is Monte Carlo simulation?

The model can predict the *probability* of each match result — for example, France vs Argentina might give France a 52% chance of winning, 24% draw, 24% Argentina win.

But a tournament is not one match. It's a chain of 6–7 matches where the winner of each depends on who survived the previous round. Calculating the exact probability of each team winning the whole tournament analytically would require evaluating millions of possible paths through the bracket.

Monte Carlo solves this by brute force: **simulate the entire tournament at random, respecting the probabilities, and repeat 100,000 times**.

Here is what one simulation looks like:

```
Simulation #1:
  Group stage  → France wins Group I, Senegal qualifies 2nd
  Round of 32  → France beats Norway
  Round of 16  → France beats Belgium
  Quarterfinals → France beats England
  Semifinals   → France beats Brazil
  Final        → France beats Argentina  ← France wins this simulation
```

After 100,000 simulations:

```
France won the tournament in 10,700 simulations  → 10.7% probability
Argentina won in 9,700 simulations               →  9.7% probability
...
```

The key insight is that **a strong team can still lose** — and does, often. A team with 60% win probability per match still loses 40% of the time. In a 6-match tournament, even the heavy favorite fails to win most simulations. That is why no team exceeds ~11% in our results: the tournament is genuinely unpredictable.


## How it works

### Data sources

| Dataset | Source | Records |
|---------|--------|---------|
| Match results (1872–2026) | [martj42/international-football-results](https://github.com/martj42/international-football-results) | 49,287 |
| FIFA Rankings | Kaggle | 67,472 |
| EA FC 26 Player Ratings | Kaggle | 16,228 |
| Transfermarkt Squads | Kaggle | 47,689 |

### Feature engineering

| Feature | Description |
|---------|-------------|
| `ranking_diff` | FIFA ranking difference (home − away) |
| `points_diff` | FIFA points difference |
| `overall_diff` | EA FC avg overall rating difference |
| `top3_overall_diff` | EA FC top 3 players rating difference |
| `home_form_w` / `away_form_w` | Weighted recent form (last 5 competitive matches, weights [5,4,3,2,1], normalized [0,1]) |
| `h2h_home_rate` | Historical home win rate in head-to-head matchups |
| `goals_scored_diff` | Average goals scored difference |
| `goals_conceded_diff` | Average goals conceded difference |
| `neutral` | Whether the match is played on neutral ground |
| `tournament_weight` | Match importance (3=finals, 2=qualifiers, 1=friendlies) |

### Model selection

All models evaluated with `TimeSeriesSplit(n_splits=5)` to respect temporal order and avoid leakage.

```
Baseline Decision Tree      F1: 0.433
Decision Tree (depth=5)     F1: 0.489
Logistic Regression         F1: 0.514
Random Forest v1            F1: 0.520
+ form features (v2)        F1: 0.533
+ weighted form (v3)        F1: 0.536
+ RandomizedSearchCV        F1: 0.538  ← final model
Extra Trees (tuned)         F1: 0.538  (tie, RF preferred)
Soft Voting ensemble        F1: 0.534  (worse)
Calibrated RF               F1: 0.444  (rejected)
```

### Final model hyperparameters

```python
RandomForestClassifier(
    bootstrap             = False,
    max_depth             = 4,
    max_features          = 0.5,
    min_impurity_decrease = 0.00139,
    min_samples_leaf      = 9,
    min_samples_split     = 18,
    n_estimators          = 649,
    class_weight          = "balanced",
)
```

### Monte Carlo simulation

The model produces probabilities for each match, not point predictions. The simulation:

1. Pre-computes probabilities for all 1,128 team pairs (model called once per pair)
2. Simulates the full tournament 100,000 times
3. In group stage: draws can occur; standings by points + goal difference proxy
4. In knockouts: draw probability is split equally between the two teams
5. Reports champion / finalist / semifinalist frequency as probability

Pre-computing probabilities reduces model calls from ~6,000,000 to 1,128 — a 5,000× speedup.

---

## Project structure

```
football-tournament-2026-predictor/
│
├── notebook/
│   └── predictor.ipynb          # Full pipeline (EDA → model → simulation)
│
├── dashboard/
│   ├── app.py                   # Streamlit dashboard
│   ├── requirements.txt
│   ├── champion_probs.json      # Champion probabilities (100k simulations)
│   ├── match_probs.json         # Match-level probabilities (group stage)
│   └── group_probs.json         # Group qualification probabilities
│
├── models/
│   ├── rf_ft2026_final.pkl      # Serialized pipeline (BoolToInt + RF)
│   ├── features_v3.pkl          # Feature list
│   └── classes.pkl              # Class labels
│
├── data/                        # Raw datasets (not tracked by git)
│   ├── dataset_international_football/
│   ├── dataset_ranking_FIFA/
│   ├── dataset_EA_playerrating/
│   └── dataset_transfermarkt/
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## Limitations

1. **EA FC 26 ratings applied retroactively** — player ratings are static snapshots used as a proxy for historical squad quality. Teams that have improved significantly (Morocco, Japan) may be undervalued in earlier matches.

2. **Minor leakage in goal/form imputation** — global mean imputation for missing values was computed before the train/test split. Affects < 2% of rows; negligible impact.

3. **Simplified bracket** — the knockout bracket uses random pairing rather than the exact official draw rules. This may slightly affect probabilities for teams in the same group.

4. **Draw class is hard** — F1 of 0.22 for draws. Draws are the most random outcome in football and systematically underperform in structural models. This is a known limitation across the academic literature.

5. **Public data ceiling** — without access to lineups, injuries, or tracking data, the realistic F1 ceiling with public data is ~0.56–0.62. Our model reaches 0.561.

---

## Run locally

```bash
# Clone the repo
git clone https://github.com/your-username/football-tournament-2026-predictor
cd football-tournament-2026-predictor

# Create environment
conda create -n ft2026 python=3.11
conda activate ft2026

# Install dependencies
pip install -r dashboard/requirements.txt
pip install jupyter scikit-learn xgboost lightgbm joblib

# Launch dashboard
streamlit run dashboard/app.py

# Or open the notebook
jupyter notebook notebook/predictor.ipynb
```

---

## Tech stack

`Python 3.11` · `scikit-learn` · `pandas` · `numpy` · `XGBoost` · `LightGBM` · `Streamlit` · `Plotly` · `joblib`

---

## License

© 2026 Osvaldo Hernández. All rights reserved.

This project is made public for educational and portfolio purposes. Reproduction, distribution, or commercial use without explicit written permission from the author is prohibited.

---

*Predictions are probabilistic estimates based on historical data and model assumptions. They do not constitute betting advice.*
