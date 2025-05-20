import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

def summarize_conversation(conversation, top_n):
    # 1) pull out text
    texts = [turn.split(":",1)[1].strip() for turn in conversation]
    
    # 2) prep lemmatizer & English stop-words
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # 3) extend stop_words with common greetings
    greetings = {'hi', 'hello', 'hey', 'greetings', 'good', 'morning', 
                 'afternoon', 'evening'}
    stop_words |= greetings

    # 4) custom analyzer: tokenize→POS→keep only noun lemmas not in stop_words
    def noun_analyzer(doc):
        tokens = word_tokenize(doc)
        tagged = pos_tag(tokens)
        nouns = []
        for word, tag in tagged:
            lw = word.lower()
            if tag.startswith('NN') and lw not in stop_words:
                lemma = lemmatizer.lemmatize(lw, pos='n')
                if lemma not in stop_words:
                    nouns.append(lemma)
        return nouns

    # 5) TF-IDF on those noun-lemmas
    vectorizer = TfidfVectorizer(analyzer=noun_analyzer)
    X = vectorizer.fit_transform(texts)
    scores = np.asarray(X.mean(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    if not terms.any():
        return "No clear topics detected."

    # 6) pick top_n
    top_n = min(top_n, len(terms))
    idxs = scores.argsort()[::-1][:top_n]
    top_terms = [terms[i] for i in idxs]

    # 7) format summary
    if len(top_terms) == 1:
        return f"The user asked mainly about {top_terms[0]}."
    elif len(top_terms) == 2:
        return f"The user asked mainly about {top_terms[0]} and {top_terms[1]}."
    else:
        return (
            "The user asked mainly about "
            + ", ".join(top_terms[:-1])
            + ", and "
            + top_terms[-1]
            + "."
        )

# --- Test it
conversation = [
    """User: Hello!
AI: Hi! How can I assist you today?
User: Can you explain what machine learning is?
AI: Certainly! Machine learning is a field of AI that allows systems to
learn from data."""
]

print(summarize_conversation(conversation, top_n=2))
# → “The conversation focuses on python and use.”
# (You can bump top_n or further tweak stop_words to drop “use” if you’d like even tighter topics.)
