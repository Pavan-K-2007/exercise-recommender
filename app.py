"""Personalised Exercise Recommendation Web Application (Streamlit).

An academic prototype combining:
- Hard rule-based safety & feasibility filters
- K-Means user persona clustering & PCA projection
- Content-based scoring with explainable evidence
- Adaptive contextual LinUCB bandit learning from feedback
"""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import config as C
from src.clustering import PersonaClusterer
from src.content_ranker import content_score, rank_by_content
from src.data_loader import Exercise, UserProfile, load_catalogue, make_profile, validate_profile
from src.evaluation import Evaluator
from src.explanations import generate_explanation
from src.features import FEATURE_DIM, feature_vector
from src.linucb import LinUCB
from src.safety_filter import filter_exercises, violations

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Personalised Exercise Recommender",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------- CUSTOM STYLING -----------------
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 0.85rem 1.1rem;
        border-radius: 6px;
        color: #92400E;
        font-size: 0.92rem;
        margin-bottom: 1.5rem;
    }
    .rec-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.3rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }
    .rec-badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        font-size: 0.78rem;
        font-weight: 600;
        border-radius: 9999px;
        margin-right: 0.4rem;
    }
    .badge-diff-beginner { background-color: #DEF7EC; color: #03543F; }
    .badge-diff-intermediate { background-color: #FEF08A; color: #854D0E; }
    .badge-diff-advanced { background-color: #FEE2E2; color: #991B1B; }
    .badge-type { background-color: #E0E7FF; color: #3730A3; }
    .badge-equipment { background-color: #F1F5F9; color: #334155; }
    .expl-box {
        background: #F8FAFC;
        border-left: 3px solid #3B82F6;
        padding: 0.65rem 0.9rem;
        border-radius: 4px;
        font-size: 0.88rem;
        color: #1E3A8A;
        margin-top: 0.75rem;
    }
    .metric-pill {
        background-color: #EEF2F6;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- SESSION STATE INIT -----------------
if "catalogue" not in st.session_state:
    st.session_state.catalogue = load_catalogue()

if "clusterer" not in st.session_state:
    sim_path = C.USERS_PATH
    if not sim_path.exists():
        from src.synthetic_users import write
        write()
    df_users = pd.read_csv(sim_path)
    cl = PersonaClusterer(seed=C.SEED)
    cl.fit(df_users)
    st.session_state.clusterer = cl
    st.session_state.df_users = df_users

if "bandit" not in st.session_state:
    st.session_state.bandit = LinUCB(FEATURE_DIM, alpha=C.LINUCB_ALPHA)

if "feedback_history" not in st.session_state:
    st.session_state.feedback_history = []

if "user_profile" not in st.session_state:
    st.session_state.user_profile = None

if "eval_cache" not in st.session_state:
    st.session_state.eval_cache = None


# ----------------- SIDEBAR NAVIGATION -----------------
st.sidebar.image("https://img.icons8.com/fluency/96/dumbbell.png", width=64)
st.sidebar.title("FitGuide ML")
st.sidebar.caption("Personalised & Explainable Exercise Recommender")

nav_choice = st.sidebar.radio(
    "Navigation",
    ["🏋️ Recommendation Hub", "📊 Evaluation & Ablations", "ℹ️ System Architecture & Ethics"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Bandit Learning State")
st.sidebar.markdown(f"**Total Feedback Updates:** `{st.session_state.bandit.n_updates}`")
if st.session_state.feedback_history:
    st.sidebar.caption(f"Last Action: {st.session_state.feedback_history[-1]['action']} ({st.session_state.feedback_history[-1]['exercise_name']})")

if st.sidebar.button("🔄 Reset Bandit & Feedback", help="Restores bandit parameters and clears session feedback."):
    st.session_state.bandit.reset()
    st.session_state.feedback_history = []
    st.sidebar.success("Bandit reset to initial state!")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Project: AEC EL • RVCE\nVersion 1.0 (Nov 2026)")


# ----------------- VIEW 1: RECOMMENDATION HUB -----------------
if nav_choice == "🏋️ Recommendation Hub":
    st.markdown('<div class="main-header">Personalised Exercise Recommendation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Safe, transparent, and adaptive recommendations grounded in your goals and physical constraints.</div>', unsafe_allow_html=True)

    # Prominent Disclaimer (PRD 10, 11 AC-15)
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚠️ Academic Prototype & Safety Notice:</strong> This system is an academic research prototype demonstrating 
            transparent recommendation pipelines. It does <strong>not</strong> provide medical diagnosis, clinical treatment, or injury rehabilitation. 
            Safety filters apply conservative exclusion rules based on stated inputs, but cannot replace a qualified physician or certified fitness professional. 
            If you experience pain or active injury, discontinue exercise and seek medical advice.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. PROFILE SECTION
    with st.expander("👤 Step 1: Set Your Profile & Constraints", expanded=(st.session_state.user_profile is None)):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=14, max_value=90, value=25, step=1)
            use_calc = st.checkbox("Calculate BMI from Height & Weight", value=False)
            if use_calc:
                weight_kg = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=68.0, step=0.5)
                height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=230.0, value=172.0, step=0.5)
                bmi = round(weight_kg / ((height_cm / 100.0) ** 2), 1)
                st.caption(f"Computed BMI: **{bmi}**")
            else:
                bmi = st.number_input("BMI", min_value=12.0, max_value=60.0, value=23.0, step=0.1)

        with col2:
            goal_options = list(C.GOALS)
            goal_labels = [g.replace("_", " ").title() for g in goal_options]
            selected_goal_label = st.selectbox("Primary Fitness Goal", goal_labels, index=0)
            goal = goal_options[goal_labels.index(selected_goal_label)]

            fitness_level_options = list(C.FITNESS_LEVELS)
            fitness_labels = [f.title() for f in fitness_level_options]
            selected_level_label = st.selectbox("Fitness Level", fitness_labels, index=0)
            fitness_level = fitness_level_options[fitness_labels.index(selected_level_label)]

        with col3:
            equipment_options = list(C.EQUIPMENT)
            equip_labels = [e.replace("_", " ").title() for e in equipment_options]
            selected_equip_labels = st.multiselect(
                "Available Equipment",
                equip_labels,
                default=["Bodyweight", "Dumbbells"],
            )
            equipment = [equipment_options[equip_labels.index(l)] for l in selected_equip_labels]
            if "bodyweight" not in equipment:
                equipment.append("bodyweight")

            restr_options = ["none", "prefer_not_to_say"] + list(C.RESTRICTION_TAGS)
            restr_labels = [r.replace("_", " ").title() for r in restr_options]
            selected_restr_labels = st.multiselect(
                "Joint / Body Areas to Avoid (Injury Restrictions)",
                restr_labels,
                default=["None"],
            )
            restrictions = [restr_options[restr_labels.index(l)] for l in selected_restr_labels]
            if "None" in selected_restr_labels and len(selected_restr_labels) > 1:
                restrictions = [r for r in restrictions if r != "none"]

        profile_dict = {
            "age": int(age),
            "bmi": float(bmi),
            "goal": goal,
            "fitness_level": fitness_level,
            "equipment": equipment,
            "restrictions": restrictions,
        }

        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            save_profile = st.button("💾 Generate Recommendations", type="primary", use_container_width=True)

        if save_profile:
            errs = validate_profile(profile_dict)
            if errs:
                st.error("Validation failed:\n" + "\n".join(f"- {e}" for e in errs))
            else:
                st.session_state.user_profile = make_profile(profile_dict)
                st.success("Profile updated successfully!")

    # Active Profile Banner & Persona Discovery
    current_prof: UserProfile | None = st.session_state.user_profile
    if current_prof:
        # User Persona Clustering (PRD 5.4)
        clusterer: PersonaClusterer = st.session_state.clusterer
        prof_dict_raw = {
            "age": current_prof.age,
            "bmi": current_prof.bmi,
            "goal": current_prof.goal,
            "fitness_level": current_prof.fitness_level,
            "equipment": list(current_prof.equipment),
            "restrictions": list(current_prof.restrictions) if current_prof.restrictions else ["none"],
        }
        cluster_id = clusterer.predict_cluster(prof_dict_raw)
        persona = clusterer.personas.get(cluster_id)

        st.markdown("---")
        p_col1, p_col2 = st.columns([3, 2])
        with p_col1:
            st.subheader("🎯 Assigned User Persona Cohort")
            if persona:
                st.markdown(f"#### **{persona.name}**")
                st.write(persona.description)
                st.caption(f"Cluster #{persona.cluster_id} • Optimal K={clusterer.best_k} (Silhouette: {clusterer.best_silhouette:.3f})")
        with p_col2:
            st.subheader("📋 Active Filter Summary")
            st.markdown(
                f"- **Goal:** `{current_prof.goal.replace('_', ' ').title()}`\n"
                f"- **Level:** `{current_prof.fitness_level.title()}`\n"
                f"- **Equipment:** {', '.join(f'`{e}`' for e in current_prof.equipment)}\n"
                f"- **Restrictions:** {', '.join(f'`{r}`' for r in current_prof.restrictions) if current_prof.restrictions else '`None (Full Clearance)`'}"
            )

        # 2. FILTERING & SCORING PIPELINE (PRD 5.3, 5.5, 5.6)
        all_exercises = st.session_state.catalogue
        eligible, exclusions = filter_exercises(current_prof, all_exercises)

        st.markdown("---")
        st.subheader("⭐ Top Recommended Exercises")

        if not eligible:
            st.warning("No exercises match your strict equipment and restriction filters. Try adding equipment or adjusting restrictions.")
        else:
            # Score candidates
            content_ranks = {ex.exercise_id: content_score(current_prof, ex) for ex in eligible}
            vectors = [feature_vector(current_prof, ex) for ex in eligible]
            priors = [content_ranks[ex.exercise_id][0] for ex in eligible]

            # Re-rank with LinUCB
            bandit: LinUCB = st.session_state.bandit
            ranked_with_scores = bandit.rank(eligible, vectors, priors=priors)

            # Invariant check
            bad = violations(current_prof, [ex for ex, _ in ranked_with_scores])
            if bad:
                st.error("CRITICAL: Safety invariant violation detected!")
            else:
                top_k = ranked_with_scores[:6]

                for idx, (ex, final_score) in enumerate(top_k):
                    c_score, comps = content_ranks[ex.exercise_id]
                    expls = generate_explanation(
                        current_prof,
                        ex,
                        comps,
                        bandit_score=final_score,
                        feedback_history=st.session_state.feedback_history,
                    )

                    with st.container():
                        st.markdown(
                            f"""
                            <div class="rec-card">
                                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                                    <h3 style="margin: 0; color: #0F172A;">#{idx+1} {ex.name}</h3>
                                    <span style="font-size: 0.9rem; font-weight: 600; color: #2563EB;">Match Score: {final_score:.2f}</span>
                                </div>
                                <div style="margin-top: 0.5rem; margin-bottom: 0.8rem;">
                                    <span class="rec-badge badge-diff-{ex.difficulty}">{ex.difficulty.upper()}</span>
                                    <span class="rec-badge badge-type">{ex.exercise_type.upper()}</span>
                                    <span class="rec-badge badge-equipment">🎯 {', '.join(ex.muscle_groups)}</span>
                                    <span class="rec-badge badge-equipment">🛠️ {', '.join(ex.equipment)}</span>
                                </div>
                                <p style="font-size: 0.92rem; color: #475569; margin-bottom: 0.5rem;">{ex.instructions}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # Explanations
                        expl_md = " • ".join(expls)
                        st.markdown(f'<div class="expl-box">💡 <strong>Why this was chosen:</strong> {expl_md}</div>', unsafe_allow_html=True)

                        # Interactive Feedback Actions (PRD 5.6)
                        btn_c1, btn_c2, btn_c3, btn_space = st.columns([1.2, 1.2, 1.2, 4])
                        with btn_c1:
                            if st.button("✅ Completed", key=f"comp_{ex.exercise_id}_{idx}", help="+1.0 reward: marks completed and strengthens similar suggestions"):
                                vec = feature_vector(current_prof, ex)
                                bandit.update(vec, C.REWARDS["completion"])
                                st.session_state.feedback_history.append({
                                    "exercise_id": ex.exercise_id,
                                    "exercise_name": ex.name,
                                    "action": "completion",
                                    "reward": C.REWARDS["completion"],
                                })
                                st.toast(f"Completed '{ex.name}'! Bandit updated (+1.0)", icon="🎉")
                                st.rerun()

                        with btn_c2:
                            if st.button("👍 Like", key=f"like_{ex.exercise_id}_{idx}", help="+0.5 reward: prioritises this movement pattern"):
                                vec = feature_vector(current_prof, ex)
                                bandit.update(vec, C.REWARDS["like"])
                                st.session_state.feedback_history.append({
                                    "exercise_id": ex.exercise_id,
                                    "exercise_name": ex.name,
                                    "action": "like",
                                    "reward": C.REWARDS["like"],
                                })
                                st.toast(f"Liked '{ex.name}' (+0.5)", icon="👍")
                                st.rerun()

                        with btn_c3:
                            if st.button("⏭️ Skip", key=f"skip_{ex.exercise_id}_{idx}", help="-0.25 penalty: down-weights this type"):
                                vec = feature_vector(current_prof, ex)
                                bandit.update(vec, C.REWARDS["skip"])
                                st.session_state.feedback_history.append({
                                    "exercise_id": ex.exercise_id,
                                    "exercise_name": ex.name,
                                    "action": "skip",
                                    "reward": C.REWARDS["skip"],
                                })
                                st.toast(f"Skipped '{ex.name}' (-0.25)", icon="⏭️")
                                st.rerun()

                        st.markdown("<br>", unsafe_allow_html=True)

        # 3. TRANSPARENCY: "Why Not Recommended?" Filter Audit Table (PRD 5.7, 15)
        with st.expander(f"🛡️ Safety & Feasibility Filter Audit ({len(exclusions)} items excluded)"):
            st.caption("Explicit log of exercises filtered out before scoring, with exact reason codes:")
            if exclusions:
                audit_rows = []
                for excl in exclusions:
                    target_ex = next((e for e in all_exercises if e.exercise_id == excl.exercise_id), None)
                    name = target_ex.name if target_ex else excl.exercise_id
                    audit_rows.append({
                        "Exercise ID": excl.exercise_id,
                        "Exercise Name": name,
                        "Reason Code": excl.reason_code,
                        "Constraint Detail": excl.detail or "N/A",
                    })
                st.dataframe(pd.DataFrame(audit_rows), use_container_width=True)
            else:
                st.write("All exercises in catalogue were feasible for your profile.")
    else:
        st.info("👈 Please enter and submit your profile in Step 1 above to view tailored recommendations.")


# ----------------- VIEW 2: EVALUATION & ABLATIONS -----------------
elif nav_choice == "📊 Evaluation & Ablations":
    st.markdown('<div class="main-header">Evaluation Dashboard & Ablation Benchmarks</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Rigorous comparative benchmarks evaluating the hybrid pipeline against baselines and ablations (PRD Section 8).</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Baseline & Ablation Comparison", "👥 Clustering & Silhouette Analysis", "📜 Synthetic Population Explorer"])

    with tab1:
        st.subheader("Benchmark: Random vs Popularity vs No-Adaptation vs Hybrid")
        st.caption("Evaluating Precision@5, NDCG@5, Constraint-Violation Rate, and Cumulative Reward over interaction rounds across held-out synthetic test profiles.")

        eval_btn = st.button("🚀 Run Evaluation Benchmark (30 Test Profiles, 10 Rounds)", type="primary")

        if eval_btn or st.session_state.eval_cache is not None:
            if eval_btn or st.session_state.eval_cache is None:
                with st.spinner("Simulating user sessions, feedback policies, and metric aggregation..."):
                    # Generate test users
                    from src.synthetic_users import generate
                    test_dicts = generate(n=30, seed=123)
                    test_profiles = [make_profile(d) for d in test_dicts]
                    evaluator = Evaluator(catalogue=st.session_state.catalogue, seed=C.SEED)
                    df_metrics, reward_curves = evaluator.run_benchmark(test_profiles, k=5, n_rounds=10)
                    st.session_state.eval_cache = (df_metrics, reward_curves)
            else:
                df_metrics, reward_curves = st.session_state.eval_cache

            # Metrics Table
            st.markdown("#### Summary Metrics")
            st.dataframe(
                df_metrics.style.format({
                    "precision@k": "{:.3f}",
                    "ndcg@k": "{:.3f}",
                    "constraint_violation_rate": "{:.3f}",
                    "final_cumulative_reward": "{:.2f}",
                }).highlight_max(subset=["precision@k", "ndcg@k", "final_cumulative_reward"], color="#DCFCE7")
                .highlight_min(subset=["constraint_violation_rate"], color="#DCFCE7"),
                use_container_width=True,
            )

            # Cumulative Reward Plot
            st.markdown("#### Cumulative Adaptation Reward Across Interaction Rounds")
            fig, ax = plt.subplots(figsize=(8, 4))
            for method, curve in reward_curves.items():
                marker = "o" if method == "hybrid" else None
                lw = 2.5 if method == "hybrid" else 1.5
                ax.plot(range(1, len(curve) + 1), curve, label=method.replace("_", " ").title(), marker=marker, linewidth=lw)
            ax.set_xlabel("Interaction Round")
            ax.set_ylabel("Average Cumulative Reward")
            ax.set_title("Adaptation Trajectory (LinUCB vs Baselines)")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend()
            st.pyplot(fig)
            plt.close()

            st.info("💡 **Ablation Insight:** Notice that both Hybrid and No-Adaptation maintain a **0.000 Constraint Violation Rate** due to the hard safety gate. LinUCB adaptation dynamically elevates cumulative reward over successive interactions through positive feedback exploitation.")

    with tab2:
        st.subheader("K-Means Persona Clustering Analysis")
        clusterer: PersonaClusterer = st.session_state.clusterer

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown(f"**Selected Optimal K:** `{clusterer.best_k}` (Silhouette Score: `{clusterer.best_silhouette:.4f}`)")
            # Silhouette curve
            k_vals = list(clusterer.silhouette_scores.keys())
            s_vals = list(clusterer.silhouette_scores.values())
            fig_s, ax_s = plt.subplots(figsize=(6, 3.5))
            ax_s.plot(k_vals, s_vals, marker="s", color="#3B82F6", linewidth=2)
            ax_s.axvline(clusterer.best_k, color="#EF4444", linestyle="--", label=f"Best K={clusterer.best_k}")
            ax_s.set_xlabel("Number of Clusters (K)")
            ax_s.set_ylabel("Silhouette Score")
            ax_s.set_title("Silhouette Search Curve")
            ax_s.grid(True, linestyle="--", alpha=0.5)
            ax_s.legend()
            st.pyplot(fig_s)
            plt.close()

        with col_c2:
            st.markdown("**2D PCA Cluster Projection**")
            # PCA plot
            if clusterer.fitted_data_2d is not None:
                fig_p, ax_p = plt.subplots(figsize=(6, 3.5))
                scatter = ax_p.scatter(
                    clusterer.fitted_data_2d[:, 0],
                    clusterer.fitted_data_2d[:, 1],
                    c=clusterer.fitted_labels,
                    cmap="tab10",
                    alpha=0.6,
                    s=30,
                )
                ax_p.set_xlabel("PCA Component 1")
                ax_p.set_ylabel("PCA Component 2")
                ax_p.set_title("Simulated User Population Personas")
                st.pyplot(fig_p)
                plt.close()

        st.markdown("#### Identified Persona Cohorts")
        p_cards = []
        for cid, p in clusterer.personas.items():
            p_cards.append({
                "Cluster": cid,
                "Cohort Name": p.name,
                "Dominant Goal": p.dominant_goal.replace("_", " ").title(),
                "Fitness Level": p.dominant_level.title(),
                "Avg Age": p.avg_age,
                "Avg BMI": p.avg_bmi,
                "Common Equipment": ", ".join(p.common_equipment),
            })
        st.dataframe(pd.DataFrame(p_cards), use_container_width=True)

    with tab3:
        st.subheader("Synthetic Population Dataset Explorer")
        st.caption("Viewing data/simulated_users.csv generated under fixed seed (PRD 8.1).")
        st.dataframe(st.session_state.df_users.head(50), use_container_width=True)


# ----------------- VIEW 3: SYSTEM ARCHITECTURE & ETHICS -----------------
elif nav_choice == "ℹ️ System Architecture & Ethics":
    st.markdown('<div class="main-header">System Architecture & Responsible AI</div>', unsafe_allow_html=True)

    st.markdown(
        """
        ### 🏗️ Logical Recommendation Pipeline
        ```
        User Profile + Stated Restrictions
                     │
                     ▼
        [1. Hard Rule-Based Safety Filter]  ──► Excludes equipment conflicts & contraindications
                     │
                     ▼
        [2. K-Means Persona Clustering]     ──► Interpretable cohort alignment via Silhouette & PCA
                     │
                     ▼
        [3. Content-Based Scoring Scorer]   ──► Transparent goal, difficulty, and type matching
                     │
                     ▼
        [4. Contextual Bandit (LinUCB)]     ──► Dynamically adapts ranking based on like/skip/complete
                     │
                     ▼
        [5. Explainable Recommendations]    ──► Verifiable reason tags & feedback tracking
        ```
        
        ---
        ### ⚖️ Ethics, Privacy & Responsible Use
        1. **Non-Medical Boundary:** The system does not diagnose pathology, assess injury severity, or prescribe physical therapy.
        2. **Conservative Allow-List Policy:** Missing or unverified contraindication tags are treated as ineligible.
        3. **Zero-Knowledge Privacy:** No personal identifiers (names, emails, exact birthdays) are captured or stored.
        4. **Transparent Audit:** Users can view the exact reason codes for every excluded exercise.
        """
    )
