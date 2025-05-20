import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

def one_line_summary(user_msgs, ai_msgs):
    """Generate a one-line summary of the conversation based on main topics."""
    all_texts = user_msgs + ai_msgs
    stop_words = set(stopwords.words('english'))
    # Use TF-IDF to extract top keywords
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3)
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    keywords = vectorizer.get_feature_names_out()
    if keywords.size > 0:
        topics = ", ".join(keywords)
        return f"The user asked mainly about {topics}."
    else:
        return "Unable to determine the main topic of the conversation."

# Example usage:
if __name__ == "__main__":
    # Example conversation
    log = """User: Hi, can you tell me about Python?
AI: Sure! Python is a popular programming language known for its readability.
User: What can I use it for?
AI: You can use Python for web development, data analysis, AI, and more."""
    user_msgs = re.findall(r'User: (.*)', log)
    ai_msgs = re.findall(r'AI: (.*)', log)
    print(one_line_summary(user_msgs, ai_msgs))