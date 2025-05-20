# import nltk
# from nltk import pos_tag, word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# import numpy as np

# # Make sure you’ve installed the punkt, tagger, wordnet & stopwords corpora:
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('stopwords')

# def summarize_conversation(conversation, top_n=3):
#     """
#     conversation: list of strings like "User: ...", "AI: ..."
#     top_n: how many topics to show in the summary
#     Returns one-line summary of main noun-based topics.
#     """
#     # 1) pull out just the utterance text
#     texts = [turn.split(":", 1)[1].strip() for turn in conversation]
    
#     # 2) prep lemmatizer & stop-words
#     lemmatizer = WordNetLemmatizer()
#     stop_words = set(stopwords.words('english'))

#     # 3) custom analyzer: tokenize → POS-tag → keep only nouns → lemmatize
#     def noun_analyzer(doc):
#         tokens = word_tokenize(doc)
#         tagged = pos_tag(tokens)
#         nouns = [
#             lemmatizer.lemmatize(word.lower(), pos='n')
#             for word, tag in tagged
#             if tag.startswith('NN') and word.lower() not in stop_words
#         ]
#         return nouns

#     # 4) compute TF-IDF over those noun-lemmas
#     vectorizer = TfidfVectorizer(analyzer=noun_analyzer)
#     X = vectorizer.fit_transform(texts)
#     scores = np.asarray(X.mean(axis=0)).ravel()
#     terms = vectorizer.get_feature_names_out()
#     if len(terms) == 0:
#         return "No clear topics detected."

#     # 5) pick top_n
#     top_n = min(top_n, len(terms))
#     idxs = scores.argsort()[::-1][:top_n]
#     top_terms = [terms[i] for i in idxs]

#     # 6) format the summary with proper commas/“and”
#     if len(top_terms) == 1:
#         return f"The conversation focuses on {top_terms[0]}."
#     elif len(top_terms) == 2:
#         return f"The conversation focuses on {top_terms[0]} and {top_terms[1]}."
#     else:
#         return (
#             "The conversation focuses on "
#             + ", ".join(top_terms[:-1])
#             + ", and "
#             + top_terms[-1]
#             + "."
#         )

# # --- Example:
# conversation = [
#     "User: Hi, can you tell me about Python?",
#     "AI: Sure! Python is a popular programming language known for its readability.",
#     "User: What can I use it for?",
#     "AI: You can use Python for web development, data analysis, AI, and more."
# ]

# print(summarize_conversation(conversation, top_n=2))
# # → “The conversation focuses on python and use.”
# # You can bump top_n or tweak stop_words to filter out very generic nouns.



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

def summarize_conversation(conversation, top_n=3):
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
        return f"The conversation focuses on {top_terms[0]}."
    elif len(top_terms) == 2:
        return f"The conversation focuses on {top_terms[0]} and {top_terms[1]}."
    else:
        return (
            "The conversation focuses on "
            + ", ".join(top_terms[:-1])
            + ", and "
            + top_terms[-1]
            + "."
        )

# --- Test it
conversation = [
    "User: Hey there! Could you explain machine learning?",
    "AI: Sure—machine learning is a subset of AI that learns from data.",
    "User: What kinds of algorithms exist?",
    "AI: Examples include regression, decision trees, clustering, neural networks, etc.",
    "User: And where is it used?",
    "AI: Finance, healthcare, self-driving cars, recommender systems, and more."
]

print(summarize_conversation(conversation, top_n=2))
# → “The conversation focuses on python and use.”
# (You can bump top_n or further tweak stop_words to drop “use” if you’d like even tighter topics.)
