"""
features.py
------------
This file takes a raw text message and turns it into NUMBERS a machine
learning model can understand. Models can't read text directly - we have
to describe each message with measurable clues (called "features").

Think of it like describing a person: instead of a photo, you might say
"height: 180cm, age: 25, wears glasses: yes". We're doing the same thing
but for messages - "exclamation_count: 5, has_urgency_word: yes", etc.
"""

import re

# A small list of words/phrases common in "forward"-style urgent messages.
# You can (and should) expand this list once you look at your own real data.
URGENCY_WORDS = [
    "urgent", "share", "forward", "immediately", "before it's too late",
    "don't ignore", "warning", "alert", "secretly", "delete", "deleted",
    "breaking", "scientists say", "doctor", "government hiding", "now",
]


def count_exclamations(text):
    """Counts how many '!' characters appear in the message."""
    return text.count("!")


def capital_letter_ratio(text):
    """
    What fraction of the letters are capitalized?
    Forward-style messages often SHOUT in capitals.
    Returns a number between 0 and 1.
    """
    letters = [c for c in text if c.isalpha()]
    if len(letters) == 0:
        return 0.0
    capitals = [c for c in letters if c.isupper()]
    return len(capitals) / len(letters)


def urgency_word_count(text):
    """Counts how many urgency-related words/phrases appear in the message."""
    text_lower = text.lower()
    count = 0
    for word in URGENCY_WORDS:
        if word in text_lower:
            count += 1
    return count


def message_length(text):
    """Simple word count of the message."""
    return len(text.split())


def extract_features(text):
    """
    Takes one message (a string) and returns a dictionary of features.
    This is the main function other files will call.
    """
    return {
        "exclamation_count": count_exclamations(text),
        "capital_ratio": round(capital_letter_ratio(text), 3),
        "urgency_word_count": urgency_word_count(text),
        "word_count": message_length(text),
    }


# This block only runs if you execute this file directly (python features.py)
# It's a quick way to test that the functions work before using them elsewhere.
if __name__ == "__main__":
    test_message = "URGENT!!! Share this before it gets DELETED! Doctors don't want you to know!!!"
    result = extract_features(test_message)
    print("Test message:", test_message)
    print("Extracted features:", result)
