import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


REPLACEMENTS = {
    "fr33": "free",
    "m0ney": "money",
    "cl1ck": "click",
    "0ffer": "offer",
    "pr1ze": "prize",
    "w1n": "win",
    "c4sh": "cash",
    "u": "you",
    "ur": "your"
}

def preprocess(text):

    
    text = text.lower()

    
    text = re.sub(
        r'https?://\S+|www\.\S+',
        ' URL ',
        text
    )

    
    text = re.sub(
        r'\S+@\S+',
        ' EMAIL ',
        text
    )

    
    text = re.sub(
        r'\b\d{10,13}\b',
        ' PHONE ',
        text
    )

    
    text = re.sub(
        r'₹\s?\d+',
        ' MONEY ',
        text
    )

    text = re.sub(
        r'\$\s?\d+',
        ' MONEY ',
        text
    )

    
    text = re.sub(
        r'\d+',
        ' NUMBER ',
        text
    )

    
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    words = text.split()

    cleaned_words = []

    for word in words:

        if word in REPLACEMENTS:
            word = REPLACEMENTS[word]

        if word not in stop_words:

            cleaned_words.append(
                stemmer.stem(word)
            )

    return " ".join(cleaned_words)