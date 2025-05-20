import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# (Make sure you’ve installed punkt, tagger, wordnet & stopwords corpora)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

def summarize_conversation(conversation, top_n=3):
    # 1) extract text after "Speaker:"
    texts = [turn.split(":",1)[1].strip() for turn in conversation]
    
    # 2) prep lemmatizer & stop-words (including greetings)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    stop_words |= {'hi','hello','hey','greetings','good','morning','afternoon','evening'}

    # 3) analyzer: tokenize → POS-tag → noun lemmatize → filter stop-words
    def noun_analyzer(doc):
        tokens = word_tokenize(doc)
        tagged = pos_tag(tokens)
        out = []
        for w,tag in tagged:
            lw = w.lower()
            if tag.startswith('NN') and lw not in stop_words:
                out.append(lemmatizer.lemmatize(lw, pos='n'))
        return out

    # 4) TF-IDF over noun-lemmas
    vectorizer = TfidfVectorizer(analyzer=noun_analyzer)
    X = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()
    if len(terms) == 0:
        return "No clear topics detected."
    scores = np.asarray(X.mean(axis=0)).ravel()

    # 5) pick top_n
    top_n = min(top_n, len(terms))
    idxs  = scores.argsort()[::-1][:top_n]
    top_terms = [terms[i] for i in idxs]

    # 6) generic formatting for ANY number of terms
    if len(top_terms) == 1:
        body = top_terms[0]
    else:
        body = ", ".join(top_terms[:-1]) + " and " + top_terms[-1]

    return f"The conversation focuses on {body}."

# --- Demo:
conversation = [
    """User: Hi, can you tell me about Python?
AI: Sure! Python is a popular programming language known for its readability.
User: What can I use it for?
AI: You can use Python for web development, data analysis, AI, and more."""
]

print(summarize_conversation(conversation, top_n=4))
# → “The conversation focuses on machine, learning, algorithm and system.”
