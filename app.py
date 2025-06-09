import streamlit as st
from transformers import pipeline

# 1) Must be the very first Streamlit command:
st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="🧠",
    layout="centered",
)

# 2) Load model (still OK to cache below imports)
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

classifier = load_model()

# 3) Now all your UI code
st.markdown(
    "<h1 style='text-align: center;'>🧠 AI Sentiment Analyzer</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align: center; color: gray;'>Understand customer emotions from product reviews</h4>",
    unsafe_allow_html=True
)
st.markdown("---")

review = st.text_area("✍️ Enter a product review below:", height=150)

if st.button("🚀 Analyze Sentiment"):
    if not review.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            result = classifier(review)[0]
            label = result["label"]
            score = result["score"]
            emoji = {"POSITIVE": "😊", "NEGATIVE": "😞"}.get(label, "😐")
            color = "green" if label == "POSITIVE" else "red"
            st.markdown(f"""
                <div style='background-color:#f9f9f9; padding:20px; border-radius:10px;'>
                    <h3 style='color:{color};'>
                        Sentiment: {label.capitalize()} {emoji}
                    </h3>
                    <p><b>Confidence:</b> {score:.2%}</p>
                </div>
            """, unsafe_allow_html=True)
