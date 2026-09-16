"""
app.py
------
Streamlit demo for the Brand-Aware AI Customer Support Agent.
Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
import sys
import os
from pathlib import Path

# ── Project root on sys.path ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

# ── Page config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Hiver | Intelligent Support Routing",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #0f0f1a; }
.stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%); }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: rgba(26, 26, 46, 0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.05);
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* Cards */
.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.4rem 0;
    backdrop-filter: blur(10px);
}
.intent-badge {
    display: inline-block;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white !important;
    padding: 4px 14px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85rem;
}
.auto-badge {
    display: inline-block;
    background: linear-gradient(135deg, #11998e, #38ef7d);
    color: #0f1a0f !important;
    padding: 4px 14px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.85rem;
}
.escalate-badge {
    display: inline-block;
    background: linear-gradient(135deg, #f7971e, #ffd200);
    color: #1a1000 !important;
    padding: 4px 14px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.85rem;
}
.response-box {
    background: rgba(102, 126, 234, 0.1);
    border-left: 4px solid #667eea;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.5rem;
    margin-top: 0.5rem;
}
h1, h2, h3, h4 { color: #ffffff !important; }
p, li, div { color: #cbd5e0; }
.stTextArea textarea {
    background-color: #1a1a2e !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-size: 1.05rem !important;
    font-weight: 500 !important;
}
div[data-baseweb="base-input"] {
    background-color: transparent !important;
}
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }
code {
    color: #667eea !important;
    background-color: rgba(0,0,0, 0.3) !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading AI pipeline…")
def load_pipeline():
    """Load or build the pipeline (cached across Streamlit reruns)."""
    from src.config import EMBEDDING_CLASSIFIER_PATH, FAISS_INDEX_PATH, FAISS_METADATA_PATH
    from src.pipeline import SupportPipeline

    if EMBEDDING_CLASSIFIER_PATH.exists() and FAISS_METADATA_PATH.exists():
        try:
            return SupportPipeline.load(), None
        except Exception as e:
            return None, str(e)

    # Build from sample data if artefacts not found
    try:
        from src.data_loader import load_brand_conversations
        from src.preprocessing import preprocess_dataframe
        from evaluation.create_golden_set import assign_intent

        df = load_brand_conversations()
        
        # SPEED OPTIMIZATION: Limit to 500 rows for instant Streamlit startup
        if len(df) > 500:
            df = df.sample(n=500, random_state=42)
            
        df = preprocess_dataframe(df)
        df[["intent", "_nr"]] = df["cleaned_text"].apply(
            lambda t: pd.Series(assign_intent(t))
        )
        df = df.drop(columns=["_nr"])

        pipeline = SupportPipeline.build(df, label_col="intent")
        try:
            pipeline.save()
        except Exception:
            pass
        return pipeline, None
    except Exception as e:
        return None, str(e)


def _intent_color(intent: str) -> str:
    colors = {
        "order_delivery": "#4299e1",
        "refund_return": "#48bb78",
        "account_login": "#ed8936",
        "payment_billing": "#e53e3e",
        "technical_issue": "#9f7aea",
        "cancellation": "#f6e05e",
        "product_inquiry": "#81e6d9",
        "complaint_other": "#fc8181",
    }
    return colors.get(intent, "#a0aec0")


def load_eval_results() -> dict:
    """Load saved evaluation results if available."""
    from src.config import RESULTS_DIR
    results = {}
    summary_path = RESULTS_DIR / "overall_summary.json"
    metrics_path = RESULTS_DIR / "metrics.json"
    esc_path     = RESULTS_DIR / "escalation_metrics.json"

    if summary_path.exists():
        with open(summary_path) as f:
            results["summary"] = json.load(f)
    if metrics_path.exists():
        with open(metrics_path) as f:
            results["metrics"] = json.load(f)
    if esc_path.exists():
        with open(esc_path) as f:
            results["escalation"] = json.load(f)

    csv_path = RESULTS_DIR / "evaluation_summary.csv"
    if csv_path.exists():
        results["summary_df"] = pd.read_csv(csv_path)
    return results


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Settings")

    from src.config import SELECTED_BRAND, TOP_K, LLM_PROVIDER, GROQ_API_KEY, GOOGLE_API_KEY
    st.markdown(f"**Brand:** `{SELECTED_BRAND}`")
    st.markdown(f"**LLM Provider:** `{LLM_PROVIDER.upper()}`")

    api_ok = bool(GROQ_API_KEY or GOOGLE_API_KEY)
    if api_ok:
        st.success("LLM API key loaded")
    else:
        st.warning(" No LLM API key — response generation disabled")

    top_k = st.slider("Historical examples (Top-K)", 1, 5, TOP_K)
    generate_response_flag = st.checkbox("Generate AI response", value=True)

    st.markdown("---")
    st.markdown("### Quick Examples")
    examples = [
        "My order hasn't arrived in 2 weeks, please help",
        "I want to return a damaged item and get a refund",
        "I can't log into my account, password reset not working",
        "Why was I charged twice for the same order?",
        "The Amazon app keeps crashing on my phone",
        "I want to cancel my Prime subscription",
        "Is the Kindle Paperwhite waterproof?",
        "This is fraud! I never authorized this charge",
    ]
    for ex in examples:
        if st.button(ex[:50] + ("…" if len(ex) > 50 else ""), key=f"ex_{ex[:20]}"):
            st.session_state["prefill"] = ex


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 2rem 0 1rem 0;">
  <h1 style="font-size:2.4rem; font-weight:700; margin:0;">
    Brand-Aware AI Support Agent
  </h1>
  <p style="color:#a0aec0; font-size:1.05rem; margin-top:0.5rem;">
    Intent Classification · Semantic Retrieval · AI Response Generation · Smart Escalation
  </p>
  <p style="color:#718096; font-size:0.9rem;">Brand: <strong style="color:#667eea;">AmazonHelp</strong></p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Main input ────────────────────────────────────────────────────────────────
col_input, col_results = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("### Customer Message")
    prefill = st.session_state.get("prefill", "")
    message = st.text_area(
        "Enter the customer's message:",
        value=prefill,
        height=140,
        placeholder="e.g. My order hasn't arrived yet and it's been 10 days…",
        label_visibility="collapsed",
    )
    analyze_btn = st.button("Analyze Message", use_container_width=True)

    # Pipeline status
    pipeline, pipeline_error = load_pipeline()
    if pipeline_error:
        st.error(f"Pipeline failed to load: {pipeline_error}")
    elif pipeline:
        st.success("Pipeline ready")

# ── Analysis results ─────────────────────────────────────────────────────────
with col_results:
    if analyze_btn and message.strip():
        if pipeline is None:
            st.error("Pipeline not available. Check the error above.")
        else:
            with st.spinner("Analyzing…"):
                result = pipeline.run(
                    message=message,
                    top_k=top_k,
                    generate=generate_response_flag,
                )

            st.markdown("### Analysis Results")

            # Intent + Confidence
            intent = result["intent"]
            conf   = result["confidence"]
            color  = _intent_color(intent)
            st.markdown(
                f"<div class='metric-card'>"
                f"<strong>Detected Intent</strong><br/>"
                f"<span class='intent-badge' style='background:linear-gradient(135deg,{color}99,{color});'>"
                f"{intent.replace('_',' ').title()}</span>"
                f"  <span style='color:#a0aec0; font-size:0.9rem;'>confidence: {conf:.1%}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Confidence bar
            st.progress(conf)

            # Escalation decision
            decision = result["decision"]
            reason   = result.get("decision_reason", "")
            badge_cls = "auto-badge" if decision == "AUTO-HANDLE" else "escalate-badge"
            icon = " " if decision == "AUTO-HANDLE" else " "
            st.markdown(
                f"<div class='metric-card'>"
                f"<strong>Escalation Decision</strong><br/>"
                f"<span class='{badge_cls}'>{icon} {decision}</span><br/>"
                f"<small style='color:#a0aec0;'>{reason}</small>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # AI response
            if result.get("response"):
                st.markdown("**Generated Response:**")
                st.markdown(
                    f"<div class='response-box'>{result['response']}</div>",
                    unsafe_allow_html=True,
                )

    elif analyze_btn:
        st.warning("Please enter a customer message first.")

# ── Historical examples ────────────────────────────────────────────────────────
if analyze_btn and message.strip() and pipeline:
    result_exists = "result" in dir()
    if result_exists and result.get("historical_examples"):
        st.markdown("---")
        st.markdown("### Similar Historical Conversations")
        for i, ex in enumerate(result["historical_examples"], 1):
            score = ex.get("similarity_score", 0)
            with st.expander(
                f"Example {i}  —  similarity: {score:.1%}  |  "
                f"{ex['customer_text'][:80]}…"
            ):
                st.markdown(f"**Customer:** {ex['customer_text']}")
                st.markdown(f"**{SELECTED_BRAND}:** {ex['brand_response']}")

# ── Evaluation Results ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Evaluation Results")

eval_data = load_eval_results()

if not eval_data:
    st.info(
        "No evaluation results found yet.\n\n"
        "Run `python evaluation/run_all.py` to generate metrics, then refresh this page."
    )
else:
    if "summary_df" in eval_data:
        st.markdown("**Intent Classifier Comparison**")
        df_show = eval_data["summary_df"].copy()
        df_show.columns = [c.replace("_", " ").title() for c in df_show.columns]
        st.dataframe(df_show, use_container_width=True, hide_index=True)

    if "escalation" in eval_data:
        esc = eval_data["escalation"]
        ec1, ec2, ec3, ec4 = st.columns(4)
        ec1.metric("Escalation Accuracy",   f"{esc.get('accuracy', 0):.1%}")
        ec2.metric("ESCALATE Precision",    f"{esc.get('escalation_precision', 0):.1%}")
        ec3.metric("ESCALATE Recall",       f"{esc.get('escalation_recall', 0):.1%}")
        ec4.metric("ESCALATE F1",           f"{esc.get('escalation_f1', 0):.1%}")

    if "metrics" in eval_data:
        emb = eval_data["metrics"].get("embedding_classifier", {})
        if emb:
            st.markdown("**Final Classifier (Embedding) — Key Metrics**")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Accuracy",   f"{emb.get('accuracy', 0):.1%}")
            mc2.metric("Macro F1",   f"{emb.get('macro_f1', 0):.1%}")
            mc3.metric("Weighted F1",f"{emb.get('weighted_f1', 0):.1%}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#4a5568; font-size:0.8rem;'>"
    "Hiver SDE Intern Take-Home Assignment · Brand-Aware AI Customer Support Agent"
    "</p>",
    unsafe_allow_html=True,
)
