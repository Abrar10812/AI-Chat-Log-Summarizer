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

# Testing log loading
logs = load_chat_logs('data/')
print("Loaded chat logs:")
for log in logs:
    print(log[:100], "...")  # Print first 100 characters of each log
