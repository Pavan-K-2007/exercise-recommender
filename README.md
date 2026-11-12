# Personalised Exercise Recommendation (academic prototype)

Hybrid, explainable recommender: simulated population → K-means personas →
hard safety/equipment gate → content score → LinUCB adaptation → explanations.
Spec: `Personalised_Exercise_Recommendation_PRD.docx` (authoritative).

> **Not medical advice.** Academic prototype; rules are not clinically validated.

## Status (PRD Section 13)
| Phase | State |
|---|---|
| 0 Specification | **Open** – team sign-off needed (see below) |
| 1 Data foundation | Code written: schema validation, seeded synthetic users, **draft** catalogue |
| 2 Core recommendation | Partial: safety filter, content scorer. TODO: preprocessing, K-means/PCA, explanations, profile UI |
| 3 Adaptive learning | Partial: LinUCB + persistence. TODO: feedback logging |
| 4 Evaluation | Partial: metrics. TODO: baselines, ablations, runner |
| 5 Streamlit | Not started |

**Tests have not yet been executed** (no working Python on the dev machine at
time of writing). Run them after setup.

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.synthetic_users      # writes data/simulated_users.csv
pytest
```

## Safety design
* `data/exercises.csv` is a **draft**: every row is `draft_unreviewed` with empty
  `contraindication_tags`, so the safety gate excludes all of it. No clinical
  rules were invented. The team must author/review tags, then set
  `review_status=reviewed` and use `none` for "reviewed, no restrictions".
* Missing/unreviewed metadata ⇒ excluded. `prefer_not_to_say` ⇒ excludes anything
  carrying a tag. `bodyweight` is always implicitly available.

## Open decisions (PRD Section 14)
Vocabularies in `src/config.py` are provisional defaults; injury taxonomy and
contraindication tags, catalogue source, reward mapping, relevance-label
function and persistence mode need team approval.
