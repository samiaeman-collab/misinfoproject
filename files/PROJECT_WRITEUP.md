# Project Writeup: Message Style Checker (Forward-style Detector)

## The problem
Viral WhatsApp/SMS "forward" messages — health myths, urgent warnings,
chain messages — follow recognizable writing patterns: excessive
exclamation marks, urgency language ("share before it's deleted"), vague
authority claims ("a doctor told me"). This project tests whether a
machine learning model can learn to recognize these *style* patterns and
tell forward-style messages apart from normal conversational text.

Important framing: this detects writing STYLE, not truth. It is not a
fact-checker, and a message can be written calmly and still be false, or
written urgently and still be true. This limitation is discussed below.

## Data
- 62 messages total: 26 labeled "forward", 36 labeled "normal"
- Normal messages collected from a real tuition/study group chat
- Forward messages collected from real WhatsApp forwards received by
  myself and others, supplemented with additional realistic examples to
  balance the dataset, since genuine forwards were harder to find than
  normal messages
- Class imbalance (more normal than forward messages) is a known
  limitation, discussed below

## Method
1. **Hand-crafted features**: exclamation mark count, ratio of capital
   letters, count of urgency-related words/phrases, word count
2. **TF-IDF text vectors**: captures the actual vocabulary used in each
   message, on top of the style features above
3. **Model**: Logistic Regression, trained on 80% of the data, tested on
   the remaining 20% it had never seen
4. **Deployment**: a Streamlit web app where a user pastes a message and
   receives a prediction with a confidence score and the underlying
   feature values that drove the decision

## Results
On the held-out test set (13 messages):

| Metric | Forward | Normal |
|---|---|---|
| Precision | 1.00 | 0.80 |
| Recall | 0.60 | 1.00 |
| F1-score | 0.75 | 0.89 |

Overall accuracy: 85%

The model never misclassified a normal message as a forward (perfect
normal recall), but missed 2 of 5 real forward messages in the test set
(0.60 recall). Precision for the forward class was perfect (1.00) —
whenever it did flag something as a forward, it was always right — but
it was cautious, under-flagging borderline cases.

## Why forward recall is lower — the most interesting finding
Looking through the misclassified examples, the forward messages the
model missed tended to be calmer and less exaggerated than the more
obvious "URGENT!!! SHARE NOW!!!" style messages it was trained on. This
suggests two things:

1. Real-world forwards vary more in tone than the clearest examples —
   some rely on subtle emotional or authority appeals rather than
   shouting in capitals
2. The hand-crafted style features (exclamation counts, capital ratio)
   work well for the loud, obvious cases but may need to be weighted
   less heavily than the TF-IDF text content for subtler examples

A next step, if continued, would be to weight the TF-IDF vocabulary more
heavily, or specifically collect more "subtle" forward examples to
retrain on, since the current forward examples skew toward the more
obvious end of the spectrum.

## Limitations
- Small dataset (62 messages) by machine learning standards — a larger,
  more diverse sample would give more reliable results
- Class imbalance (26 forward vs 36 normal) likely affects the model's
  confidence and recall on the forward class
- Detects *style*, not *truth* — this is a fundamental scope limitation,
  not a bug. A genuinely true message can be written urgently, and a
  false claim can be written calmly
- Feature list (urgency words) was built by hand based on common
  patterns I observed, not exhaustively researched

## What I'd do next
- Collect more real forward messages, particularly calmer/subtler ones,
  to reduce the class imbalance and improve forward recall
- Compare against a Random Forest model to see if a different algorithm
  handles the imbalance better
- Expand the urgency word list based on patterns in the data rather than
  my initial guesses
