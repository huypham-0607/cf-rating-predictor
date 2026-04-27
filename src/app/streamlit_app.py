"""
Streamlit MVP interface for Codeforces rating prediction.

Run with:
  streamlit run src/app/streamlit_app.py
"""

import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd

from src.features.encoder import ALL_TAGS
from src.inference import ProblemInput, RatingPredictor

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Codeforces Rating Predictor",
    page_icon="💻",
    layout="wide",
)

st.title("💻 Codeforces Rating Predictor")
st.caption(
    "Estimates the difficulty rating (800-3500) of a Codeforces problem from its structured metadata."
)
st.divider()


# ── Load predictor (cached) ──────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model …")
def load_predictor() -> RatingPredictor:
    return RatingPredictor()


try:
    predictor = load_predictor()
    model_ready = True
except FileNotFoundError as e:
    st.error(
        "Model files not found. Run the full pipeline first:\n\n"
        "```\npython scripts/run_pipeline.py\n```"
    )
    st.code(str(e))
    model_ready = False


# ── Input form ───────────────────────────────────────────────────────────────

col_in, col_out = st.columns([1, 1], gap="large")

with col_in:
    st.subheader("Problem Details")

    options = [chr(65 + i) for i in range(26)]

    problem_index = st.selectbox(
        "Problem position in contest",
        options,
        index=1,
        help="Which slot does this problem occupy? A is typically easiest, F+ hardest.",
    )

    tags = st.multiselect(
        "Tags",
        options=sorted(ALL_TAGS),
        default=[],
        help="Select all applicable problem tags.",
    )

    contest_division = st.selectbox(
        "Contest division / round type",
        options=["div2", "div1", "div3", "div4", "div1+2", "educational", "global", "icpc", "other"],
        index=0,
    )

    contest_type = st.selectbox(
        "Contest type",
        options=["CF", "ICPC", "IOI"],
        index=0,
    )

    contest_year = st.number_input(
        "Contest year",
        min_value=2010,
        max_value=2030,
        value=2024,
        step=1,
    )

    contest_duration = st.number_input(
        "Contest duration (hours)",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.5,
    )

    st.divider()
    st.subheader("Optional (post-contest data)")

    use_solved = st.checkbox("Include solved count?", value=False)
    solved_count: int | None = None
    if use_solved:
        solved_count = st.number_input(
            "Solved count",
            min_value=0,
            max_value=500_000,
            value=5000,
            step=100,
            help="Number of users who solved this problem. Makes the prediction stronger but only available after the contest.",
        )

    predict_btn = st.button("Predict Rating", type="primary", use_container_width=True)


# ── Output panel ──────────────────────────────────────────────────────────────

with col_out:
    st.subheader("Prediction")

    if predict_btn and model_ready:
        problem = ProblemInput(
            problem_index=problem_index,
            tags=tags,
            contest_division=contest_division,
            contest_type=contest_type,
            contest_year=int(contest_year),
            contest_duration_hours=float(contest_duration),
            solved_count=solved_count,
        )

        with st.spinner("Predicting …"):
            result = predictor.predict(problem)

        # Rating display
        st.metric("Predicted Rating", result.predicted_rating)
        st.metric("Raw Predicted Rating", f"{result.predicted_rating_raw:.2f}")
        st.info(f"**Band:** {result.rating_band}")

        if result.is_cold_start:
            st.caption("Cold-start mode (no solved count). Using Variant B.")
        else:
            st.caption("Post-contest mode (solved count provided). Using Variant C.")

        # Feature importance
        if result.top_features:
            st.subheader("Top contributing features")
            fi_df = pd.DataFrame(result.top_features, columns=["Feature", "Importance"])
            fi_df["Importance"] = fi_df["Importance"] / fi_df["Importance"].max()  # normalise to 0–1
            st.bar_chart(fi_df.set_index("Feature"))

        st.divider()

    elif not model_ready:
        st.info("Run the pipeline to generate model files, then reload this page.")
    else:
        st.info("Fill in the problem details on the left and click **Predict Rating**.")