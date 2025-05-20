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



def extract_keywords(texts):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    stop_words |= {'hi','hello','hey','greetings','good','morning','afternoon','evening', 'thanks', 'thank you', 'please', 'ok', 'okay', 'sure', 'yes', 'no'}
    words = [word.lower() for text in texts for word in re.findall(r'\b\w+\b', text)]
    filtered_words = [word for word in words if word not in stop_words]
    counter = Counter(filtered_words)
    return counter.most_common(5)



def generate_summary(total, user_count, ai_count, keywords):
    print("\n--- Chat Log Summary ---")
    print(f"Total exchanges: {total}")
    print(f"User messages: {user_count}")
    print(f"AI messages: {ai_count}")

    # Determine the main topic of the conversation
    if keywords:
        main_topic = ", ".join([word for word, _ in keywords[:3]])
        print(f"- The conversation had {total} exchanges.")
        print(f"- The user asked mainly about {main_topic}.")
        print(f"- Most common keywords: {', '.join([word for word, _ in keywords])}")
    else:
        print("- Unable to determine the main topic of the conversation.")




def main():
    logs = load_chat_logs('data/')
    for log in logs:
        user_msgs, ai_msgs = parse_logs(log)
        total, user_count, ai_count = message_stats(user_msgs, ai_msgs)
        keywords = extract_keywords(user_msgs + ai_msgs)
        generate_summary(total, user_count, ai_count, keywords)

if __name__ == "__main__":
    main()
