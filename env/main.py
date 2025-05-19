import os
import re
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords
import nltk
nltk.download('stopwords')




def load_chat_logs(folder):
    """Load all .txt files from the data folder."""
    logs = []
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), 'r', encoding='utf-8') as f:
                logs.append(f.read())
    return logs

# Test
logs = load_chat_logs('data/')
print("Loaded chat logs:")
for log in logs:
    print(log[:100], "...")  


def parse_logs(log):
    """Separate User and AI messages."""
    user_msgs = re.findall(r'User: (.*)', log)
    ai_msgs = re.findall(r'AI: (.*)', log)
    return user_msgs, ai_msgs

# Test2
for log in logs:
    user_msgs, ai_msgs = parse_logs(log)
    print(f"User Messages: {len(user_msgs)} | AI Messages: {len(ai_msgs)}")
    
    
def message_stats(user_msgs, ai_msgs):
    total = len(user_msgs) + len(ai_msgs)
    user_count = len(user_msgs)
    ai_count = len(ai_msgs)
    return total, user_count, ai_count

# Test3
for log in logs:
    user_msgs, ai_msgs = parse_logs(log)
    total, user_count, ai_count = message_stats(user_msgs, ai_msgs)
    print(f"Total messages: {total} | User: {user_count} | AI: {ai_count}")