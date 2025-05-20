import os
import re
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer



nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')



def load_chat_logs(folder):
    logs = []
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), 'r', encoding='utf-8') as f:
                logs.append(f.read())
    return logs

def parse_logs(log):
    user_msgs = re.findall(r'User: (.*)', log)
    ai_msgs = re.findall(r'AI: (.*)', log)
    return user_msgs, ai_msgs




def message_stats(user_msgs, ai_msgs):
    total = len(user_msgs) + len(ai_msgs)
    return total, len(user_msgs), len(ai_msgs)



def extract_keywords_nouns(texts, top_n):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english')) | {
        'hi', 'hello', 'hey', 'greetings', 'good', 'morning',
        'afternoon', 'evening', 'please', 'thanks', 'thank'
    }

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

    vectorizer = TfidfVectorizer(analyzer=noun_analyzer)
    X = vectorizer.fit_transform(texts)
    scores = np.asarray(X.mean(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    if not terms.any():
        return []

    top_n = min(top_n, len(terms))
    idxs = scores.argsort()[::-1][:top_n]
    return [(terms[i], scores[i]) for i in idxs]




def generate_summary(total, user_count, ai_count, keywords):
    print("\n--- Chat Log Summary ---")
    print(f"Total exchanges: {total}")
    print(f"User messages: {user_count}")
    print(f"AI messages: {ai_count}")

    if not keywords:
        print("- Unable to determine the main topic of the conversation.")
        return

    # Extract top keywords
    top_words = [word for word, _ in keywords]
    excluded_words = {'today', 'now', 'system', 'thing', 'example', 'something'}
    topic_terms = [word for word in top_words if word not in excluded_words]

    topic_terms = top_words[:2]

    # Generate topic sentence
    if len(topic_terms) == 1:
        topic_line = f"- The user asked mainly about {topic_terms[0]}."
    else:
        topic_line = f"- The user asked mainly about {topic_terms[0]} and {topic_terms[1]}."

    print(topic_line)
    print(f"- Most common keywords: {', '.join(top_words)}")





def main():
    logs = load_chat_logs('data/')
    for log in logs:
        user_msgs, ai_msgs = parse_logs(log)
        total, user_count, ai_count = message_stats(user_msgs, ai_msgs)
        keywords = extract_keywords_nouns(user_msgs + ai_msgs, top_n=5)
        generate_summary(total, user_count, ai_count, keywords)

if __name__ == "__main__":
    main()
