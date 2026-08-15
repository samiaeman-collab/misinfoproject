# Personal statement / interview reflection (draft)

Use or adapt this as a starting point — make it sound like you, not like
a template.

---

I built a machine learning tool that detects whether a WhatsApp message
looks like a viral "forward" — the health myths and urgent chain messages
that circulate constantly — based on its writing style, rather than
whether it is true or false. I collected real messages from a tuition
group chat and from forwards sent to family and friends, extracted
features like exclamation mark frequency and urgency language, combined
them with a text-based model, and trained a classifier to tell the two
apart.

The most interesting part wasn't getting it to work — it was
understanding why it didn't work perfectly. My model correctly identified
every normal message, but missed several real forward messages that were
calmer and less exaggerated than the obvious "URGENT!!! SHARE NOW!!!"
examples I trained it on. That taught me something concrete about how
class imbalance and feature choice shape a model's blind spots, and
pushed me to think about what "style" actually means in text a person
writes to manipulate rather than just to communicate.

I was also careful about the ethical framing throughout: the tool detects
patterns in how something is written, not whether it's true, and I made
sure the demo app states that limitation explicitly rather than
overclaiming what it can do.

---

Tips for adapting this:
- Swap in your own specific example of a misclassified message if you
  remember one, it makes it more concrete and memorable
- If asked in an interview "why logistic regression," your honest answer
  can be: it's simple, interpretable (you can see which features mattered
  most), and a reasonable starting point before trying more complex
  models
- If asked "what would you improve," the class imbalance and subtle vs
  obvious forward messages point above is a strong, genuine answer
