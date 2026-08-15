# Message Style Checker (Forward-style Detector)

A small machine learning project that flags whether a message *looks* like
a viral WhatsApp/SMS forward (based on writing style - exclamation marks,
urgency words, capitalization) versus a normal message.

**Important framing:** this detects writing STYLE, not truth. It cannot
tell you if a claim is factually true or false - only whether it's written
the way viral forwards typically are. Say this clearly in any writeup.

## How this project works (in plain terms)

1. **`sample_data.csv`** - example messages labeled "forward" or "normal".
   This is placeholder data to get everything working. Replace with your
   own real, collected messages (see "Next steps" below).
2. **`features.py`** - reads a message and counts things like exclamation
   marks and urgency words. Machine learning models need numbers, not raw
   text, so this step turns words into measurable clues.
3. **`train_model.py`** - loads the data, extracts features, and trains a
   Logistic Regression model to learn the difference between the two
   classes. Run this first.
4. **`app.py`** - a simple website (using Streamlit) where you paste a
   message and see the model's prediction, live.

## How to run it yourself

```bash
pip install pandas scikit-learn streamlit joblib scipy

# Step 1: train the model (creates model.joblib etc.)
python3 train_model.py

# Step 2: launch the demo website
streamlit run app.py
```

## What "training a model" actually means (plain English)

You're not programming rules like "if it has 3+ exclamation marks, call it
a forward." Instead, you show the model many labeled examples, and it
works out on its own which patterns (a combination of exclamation count,
urgency words, specific vocabulary, etc.) best separate the two classes.
Logistic Regression specifically learns a "weight" for each feature - how
much that feature pushes a prediction toward "forward" or "normal."

## Why the results look "too good" right now

With this sample dataset, the model got 100% accuracy on the test set.
That's not because the model is amazing - it's because this sample data
is small (25 messages) and the two classes are written very differently
on purpose, to make it easy to test the pipeline works. Real messages
will be messier, more mixed, and the model will make mistakes. That's
normal and expected - and honestly discussing this in your writeup is a
sign of understanding, not weakness.

## Next steps (do these to make it a real project)

1. **Collect real data.** Ask friends/family to share (with permission,
   remove any personal info) 100-200 real forwarded messages they've
   received, plus 100-200 normal texts. Replace `sample_data.csv` with
   these. This is the single most important step - it's what makes the
   project genuinely yours instead of a tutorial clone.
2. **Re-run `train_model.py`** on the real data and look honestly at the
   results - expect accuracy to drop from 100%, and that's fine.
3. **Read the confusion matrix** printed by the script - it shows exactly
   which messages got misclassified, so you can look at *why*.
4. **Expand `URGENCY_WORDS`** in `features.py` based on patterns you
   actually notice in your real data (including Roman Urdu phrases if
   relevant, e.g. "abhi share karein").
5. **Push this to GitHub** with regular commits as you go (not one big
   upload at the end) and write a short explanation of your findings,
   including anything that surprised you or didn't work.
6. **Optional stretch goal:** try a Random Forest model too and compare
   which one performs better, and think about why.

## Ethical note to include in your writeup

Be explicit that this tool detects *style*, not *truth*. A real news
report can be written calmly, and a false claim can be written calmly
too. Flag this limitation openly - it shows you understand the boundaries
of your own project, which is exactly what admissions readers and
interviewers want to see.
