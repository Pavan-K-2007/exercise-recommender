# Personalised Exercise Recommendation Using Machine Learning
### A Safe, Adaptive and Explainable Hybrid Approach

Academic research prototype developed for the Department of Electronics and Communication, RVCE.

> **⚠️ Non-Medical Disclaimer:** This system is an academic machine-learning demonstration. It does not provide medical diagnosis, clinical treatment, or injury rehabilitation. Exercise recommendations apply conservative rule-based exclusion filters but cannot replace a certified healthcare or fitness professional.

---

## 🌟 Architecture & Features

The system implements the full hybrid pipeline defined in the Project Requirements Document (PRD):
1. **User Profile & Validation**: Strict validation over age, BMI, goal, fitness level, available equipment, and injury restrictions (`src/data_loader.py`).
2. **K-Means Persona Clustering & PCA**: Groups simulated user populations into interpretable cohorts with automated Silhouette-based optimal $K$ selection and 2D PCA projection (`src/clustering.py`).
3. **Hard Rule-Based Safety Gate**: Hard eligibility filter running strictly *before* scoring and re-ranking. Excludes exercises requiring unavailable equipment or conflicting with stated injury restrictions (`src/safety_filter.py`).
4. **Content-Based Scoring**: Transparent baseline scoring combining goal alignment and fitness level calibration (`src/content_ranker.py`).
5. **Adaptive Contextual LinUCB Bandit**: Re-ranks candidates from feedback events (Completed $+1.0$, Like $+0.5$, Skip $-0.25$) without ever violating hard safety constraints (`src/linucb.py`).
6. **Evidence-Grounded Explanations**: Plain-language explanations citing specific profile features, safety clearances, and adaptation signals (`src/explanations.py`).
7. **Evaluation & Ablation Benchmarks**: Evaluation against Random and Popularity baselines with Precision@K, NDCG@K, Constraint-Violation Rate, and Cumulative Reward trajectories (`src/evaluation.py`).

---

## 🚀 Quick Start Guide

### 1. Environment Setup
```powershell
# Navigate to the project directory
cd "exercise-recommender"

# Create a Python virtual environment (Python 3.10+)
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Users
```powershell
python -m src.synthetic_users
```

### 3. Run the Test Suite (26 Unit & Integration Tests)
```powershell
python -m pytest -v
```

### 4. Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📁 Repository Structure

```
exercise-recommender/
├── app.py                     # Interactive Streamlit application
├── data/
│   ├── exercises.csv          # Curated exercise catalogue with safety tags
│   ├── simulated_users.csv    # Seeded synthetic user population
│   └── simulated_users.config.json
├── src/
│   ├── __init__.py
│   ├── config.py              # Vocabularies, bounds, weights, and rewards
│   ├── data_loader.py         # Schemas, dataclasses, and CSV validators
│   ├── preprocessing.py       # One-hot encoding & standard scaling pipeline
│   ├── clustering.py          # Silhouette search, K-means personas, and PCA
│   ├── safety_filter.py       # Hard equipment & contraindication filters
│   ├── content_ranker.py      # Transparent content-based scoring
│   ├── features.py            # Feature vector representation for LinUCB
│   ├── linucb.py              # Contextual bandit with persistence and reset
│   ├── explanations.py        # Plain-language explanation templates
│   ├── metrics.py             # Precision@K, NDCG@K, violation rate, reward
│   ├── evaluation.py          # Benchmark runner (Random, Popularity, Ablations)
│   └── synthetic_users.py     # Reproducible synthetic generator
├── tests/
│   └── test_core.py           # 26 unit and integration test assertions
├── requirements.txt
└── README.md
```

---

## 👥 Authors
- **P Sujith Kumar** (1RV25EC126)
- **Pallavi M C** (1RV25EC127)
- **Pavan K** (1RV25EC128)  
*Department of Electronics and Communication, RV College of Engineering (RVCE)*
