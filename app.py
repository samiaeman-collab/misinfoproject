"""
app.py
------
This creates a simple website where someone can paste a message and see
whether the model thinks it looks like a "forward-style" viral message
or a normal message - plus WHY it thinks that.

Run this with: streamlit run app.py
(It will open automatically in your browser)
"""

import streamlit as st
import joblib
import pandas as pd
from scipy.sparse import hstack

from features import extract_features

# Load the trained model and helpers we saved in train_model.py
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")
scaler = joblib.load("scaler.joblib")

st.title("Message Style Checker")
st.write(
    "This tool checks whether a message *looks* like a forwarded viral "
    "message (based on writing style) or a normal message. "
    "**It does not check whether something is true or false** - only "
    "whether it uses common forward-style patterns."
)

message = st.text_area("Paste a message here:", height=120)

if st.button("Check message") and message.strip():
    # Turn the message into the same kind of numbers the model was trained on
    feats = extract_features(message)
    feat_df = pd.DataFrame([feats])
    scaled = scaler.transform(feat_df)
    tfidf_vec = vectorizer.transform([message])
    combined = hstack([tfidf_vec, scaled])

    # Get the model's prediction and its confidence
    prediction = model.predict(combined)[0]
    probabilities = model.predict_proba(combined)[0]
    class_names = model.classes_
    confidence = max(probabilities) * 100

    st.subheader("Result")
    if prediction == "forward":
        st.error(f"Looks like a FORWARD-style message ({confidence:.0f}% confidence)")
    else:
        st.success(f"Looks like a NORMAL message ({confidence:.0f}% confidence)")

    st.subheader("Why? (the clues it used)")
    st.write(f"- Exclamation marks: **{feats['exclamation_count']}**")
    st.write(f"- Capital letter ratio: **{feats['capital_ratio']}**")
    st.write(f"- Urgency-style words found: **{feats['urgency_word_count']}**")
    st.write(f"- Word count: **{feats['word_count']}**")

    st.caption(
        "Reminder: this flags writing STYLE patterns common in viral "
        "forwards. It is not a fact-checker and can be wrong - always "
        "verify claims through a real source."
    )
