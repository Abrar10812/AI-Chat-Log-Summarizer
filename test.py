# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from collections import Counter
# import ssl

# # Make sure you have downloaded the necessary NLTK data once in your environment:
# nltk.download('punkt_tab')
# # try:
# #     _create_unverified_https_context = ssl._create_unverified_context
# # except AttributeError:
# #     pass
# # else:
# #     ssl._create_default_https_context = _create_unverified_https_context

# nltk.download()
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('wordnet')
# nltk.download('stopwords')

# def extract_user_utterances(chatlog):
#     """
#     Extracts lines from the chatlog that start with 'User:'.
#     """
#     return [
#         line.replace("User:", "").strip()
#         for line in chatlog.splitlines()
#         if line.startswith("User:")
#     ]

# def extract_keywords(text, top_n=3):
#     """
#     Tokenizes text, filters stopwords, tags parts of speech,
#     lemmatizes, and returns the top_n most frequent nouns and verbs.
#     """
#     tokens = word_tokenize(text.lower())
#     words = [w for w in tokens if w.isalpha()]
#     stop_words = set(stopwords.words('english'))
#     filtered = [w for w in words if w not in stop_words]
    
#     tagged = nltk.pos_tag(filtered)
#     lemmatizer = WordNetLemmatizer()
    
#     nouns = [lemmatizer.lemmatize(w) for w, pos in tagged if pos.startswith('NN')]
#     verbs = [lemmatizer.lemmatize(w, 'v') for w, pos in tagged if pos.startswith('VB')]
    
#     noun_freq = Counter(nouns)
#     verb_freq = Counter(verbs)
    
#     top_nouns = [w for w, _ in noun_freq.most_common(top_n)]
#     top_verbs = [w for w, _ in verb_freq.most_common(top_n)]
#     return top_nouns, top_verbs

# def summarize(chatlog):
#     """
#     Generates a one-line summary of the user's questions in the chatlog.
#     """
#     user_lines = extract_user_utterances(chatlog)
#     combined = " ".join(user_lines)
#     nouns, verbs = extract_keywords(combined)
    
#     noun_part = ""
#     if nouns:
#         noun_part = (
#             ", ".join(nouns[:-1]) + " and " + nouns[-1]
#             if len(nouns) > 1
#             else nouns[0]
#         )
    
#     if verbs:
#         verb = verbs[0]
#         return f"The user asked about {noun_part} and its {verb}."
#     else:
#         return f"The user asked about {noun_part}."

# # To use:
# chatlog = "User: Hi, can you tell me about Python? \nAI: Sure! Python is a popular programming language known for its readability.\nUser: What can I use it for?\nAI: You can use Python for web development, data analysis, AI, and more."
# print(summarize(chatlog))



import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Make sure you’ve run this once on your machine:
# python3 -m nltk.downloader punkt averaged_perceptron_tagger stopwords

GREETING_RE = re.compile(r'^(hi|hello|hey)[,!\s]+', re.I)
STOPWORDS = set(stopwords.words('english'))

def extract_user_utterances(chatlog: str):
    return [
        GREETING_RE.sub('', line).strip()
        for line in chatlog.splitlines()
        if line.startswith("User:")
    ]

def summarize(chatlog: str) -> str:
    # 1) pull out all User: lines and strip greetings
    user_lines = extract_user_utterances(chatlog)
    text = " ".join(user_lines)

    # 2) tokenize and POS-tag
    tokens = word_tokenize(text)
    tagged = nltk.pos_tag(tokens)

    # 3) pick out nouns (NN, NNP, NNS) as topics, filter stopwords
    topics = [
        w for w, tag in tagged
        if tag in ("NN", "NNS", "NNP") and w.lower() not in STOPWORDS
    ]
    # normalize & dedupe, preserve order
    seen = set()
    topics = [t.capitalize() for t in topics if not (t.lower() in seen or seen.add(t.lower()))]

    # choose primary topic
    topic = topics[0] if topics else "it"

    # 4) detect if the user asked “use”/“using”
    uses = any(w.lower() in ("use", "using") for w, tag in tagged if tag.startswith("VB"))

    if uses:
        return f"The user asked about {topic} and how to use it."
    else:
        return f"The user asked about {topic}."

# — example —

chatlog = (
    "User: Hi, can you tell me about Python? \n"
    "AI: Sure! Python is a popular programming language known for its readability.\n"
    "User: What can I use it for?\n"
    "AI: You can use Python for web development, data analysis, AI, and more."
)

print(summarize(chatlog))
# → The user asked about Python and how to use it.
