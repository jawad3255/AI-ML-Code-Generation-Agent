"""
Streamlit UI — AI/ML Code Generation Agent
"""

import streamlit as st
from agents import run_pipeline

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI/ML Code Generation Agent",
    page_icon="🤖",
    layout="wide",
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🤖 AI/ML Code Generation Agent")
st.markdown(
    "Powered by **CrewAI · LangChain · Gemini** — "
    "describe your ML task and get a complete, reviewed Python script."
)
st.divider()

# ── Sidebar — Example Prompts ──────────────────────────────────────────────────
with st.sidebar:
    st.header("💡 Example Prompts")
    examples = [
        "Binary classification on the Titanic dataset using XGBoost",
        "House price prediction with linear regression and feature engineering",
        "Customer churn prediction using a Random Forest with SMOTE",
        "Sentiment analysis on tweets using a fine-tuned BERT model",
        "K-Means clustering on an e-commerce customer dataset",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["prompt"] = ex

    st.divider()
    st.markdown("**How it works**")
    st.markdown(
        "1. 🗺️ **Planner** — designs the ML blueprint\n"
        "2. 💻 **Coder** — writes the Python code\n"
        "3. 🔍 **Reviewer** — audits and corrects the code"
    )

# ── Main Input ─────────────────────────────────────────────────────────────────
prompt = st.text_area(
    "Describe your ML task:",
    value=st.session_state.get("prompt", ""),
    height=100,
    placeholder="e.g. Build a spam email classifier using Naive Bayes...",
)

col1, col2 = st.columns([1, 5])
with col1:
    generate = st.button("⚡ Generate Code", type="primary", use_container_width=True)
with col2:
    if st.button("🔄 Clear", use_container_width=False):
        st.session_state["prompt"] = ""
        st.rerun()

# ── Generation ─────────────────────────────────────────────────────────────────
if generate and prompt.strip():
    with st.spinner("🤖 Agents are working... this may take 30–60 seconds"):
        try:
            result = run_pipeline(prompt.strip())

            st.success("✅ Code generation complete!")
            st.divider()

            # Split review summary from code block if possible
            if "```python" in result:
                parts = result.split("```python", 1)
                summary = parts[0].strip()
                code_raw = parts[1].split("```")[0].strip()

                if summary:
                    st.subheader("📋 Review Summary")
                    st.markdown(summary)

                st.subheader("💻 Generated Code")
                st.code(code_raw, language="python")

                st.download_button(
                    label="⬇️ Download .py",
                    data=code_raw,
                    file_name="ml_solution.py",
                    mime="text/x-python",
                )
            else:
                st.subheader("📄 Output")
                st.markdown(result)

        except Exception as e:
            st.error(f"❌ Error: {e}")

elif generate and not prompt.strip():
    st.warning("⚠️ Please enter an ML task description.")
