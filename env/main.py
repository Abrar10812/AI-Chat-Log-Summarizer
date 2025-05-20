import os
import re
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords
import nltk
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
    stop_words = set(stopwords.words('english'))
    stop_words |= {'hi','hello','hey','greetings','good','morning','afternoon','evening'}
    words = [word.lower() for text in texts for word in re.findall(r'\b\w+\b', text)]
    filtered_words = [word for word in words if word not in stop_words]
    counter = Counter(filtered_words)
    return counter.most_common(5)

def tfidf_keywords(texts):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5)
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer.get_feature_names_out()



def generate_summary(total, user_count, ai_count, keywords):
    """Generate a readable summary of the chat log."""
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
