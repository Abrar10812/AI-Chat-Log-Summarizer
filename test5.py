# import nltk     # pip install nltk
# import heapq    ## built-in library

# ## nltk.download('punkt')
# ## nltk.download('stopwords')


# text = '''Introduction

# Trees, those silent giants that populate our landscapes, have always been an integral part of our natural world. They provide us with oxygen, shade on a hot summer day, and a refuge for countless creatures. Yet, beyond their physical presence, trees have a secret life that is truly fascinating. One of the most intriguing aspects of their existence is their ability to communicate with each other and form intricate networks beneath the soil. In this article, we'll delve into this extraordinary fact about trees – their ability to communicate and share vital information with their neighbors.

# The Wood Wide Web

# Imagine a vast underground network, similar to the World Wide Web, but made of roots and fungi. This is what scientists have aptly named the "Wood Wide Web." This underground system enables trees to exchange a wealth of information, such as warnings about pests and diseases, sharing nutrients, and even sending distress signals during times of danger.

# The Mycorrhizal Connection

# At the heart of the Wood Wide Web are mycorrhizal fungi, which have a symbiotic relationship with trees. These fungi attach themselves to the tree's roots and extend their thread-like structures, known as hyphae, far and wide through the soil. This intricate network of hyphae connects multiple trees, forming a mycorrhizal network.

# Sharing Nutrients

# One of the most remarkable aspects of this tree communication system is the sharing of nutrients. Trees, through their roots, provide sugars and carbohydrates to the fungi. In return, the fungi scavenge the soil for essential nutrients, such as phosphorus and nitrogen, which are often scarce. The fungi then transport these nutrients back to the trees, ensuring their mutual survival.

# Warning Signals

# When a tree is under attack by insects or disease, it can release chemical signals into the air and soil. These signals are picked up by neighboring trees through their roots and can trigger a defensive response. For example, when a tree is infested with aphids, it can release chemicals that repel aphids or attract predators of aphids. This warning system can help neighboring trees prepare for an impending threat.

# Sharing Resources

# In a dense forest, where trees compete for sunlight, the Wood Wide Web allows for a form of cooperation. Larger, older trees with more access to sunlight can share some of their energy with smaller, shaded trees. This cooperative behavior ensures the survival of the entire forest ecosystem.

# Conclusion

# The fact that trees communicate with each other through the Wood Wide Web is not only fascinating but also a testament to the interconnectedness of life on Earth. It highlights the complexity of ecosystems and the importance of preserving our forests. Understanding this hidden world of tree communication can lead to more sustainable forestry practices and a deeper appreciation for the remarkable lives of these silent giants. The next time you walk through a forest, take a moment to marvel at the incredible network of communication happening beneath your feet, where trees share their secrets and support one another in the silent language of nature.'''

# ## Initializing Stop words (so, but , in, on, and etc.)

# stopwords = nltk.corpus.stopwords.words('english')

# ## spliting article into sentences

# sentence_list = nltk.sent_tokenize(text)

# ## Making a dictonary of frequency scores to words {word : frequency}

# frequency_map = {}

# word_list = nltk.word_tokenize(text)  ## Spliting article into words

# for i in word_list:
#     if i not in stopwords:
#         if i not in frequency_map:
#             frequency_map[i] = 1
#         else:
#             frequency_map[i] += 1

# max_frequency = max(frequency_map.values()) ## Checking for the maximum frequency

# for word in frequency_map:
#     frequency_map[word] = frequency_map[word] / max_frequency ## reassigning the scores in proportion to max frequency

# sent_scores = {}

# ## Setting sentence scores based on word scores

# for sent in sentence_list:
#     for word in word_list:
#         if word in frequency_map and len(sent.split(' ')) < 35:
#             if sent not in sent_scores:
#                 sent_scores[sent] = frequency_map[word]
#             else:
#                 sent_scores[sent] += frequency_map[word]

# ## Finding top 10 sentences based on scores

# summary = heapq.nlargest( 10,sent_scores, key=sent_scores.get)

# for a in summary: ## Final output           
#        print(a) 

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.tree import Tree
from nltk import download, FreqDist
from nltk import pos_tag, ne_chunk

# Downloads
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

# Example conversation
conversation = """
User: Hi, can you tell me about Python?
AI: Sure! Python is a popular programming language known for its readability.
User: What can I use it for?
AI: You can use Python for web development, data analysis, AI, and more.
"""

# Step 1: Extract User lines only
user_lines = [line.split("User:")[1].strip() for line in conversation.split('\n') if line.lower().startswith("user:")]
user_text = " ".join(user_lines)

# Step 2: Named Entity Recognition (NER) using NLTK's chunker
tokens = word_tokenize(user_text)
pos_tags = pos_tag(tokens)
named_entities = ne_chunk(pos_tags, binary=False)

# Step 3: Extract Named Entities (e.g., Python, Machine Learning)
topics = []
for subtree in named_entities:
    if isinstance(subtree, Tree):  # This is a named entity
        entity = " ".join([token for token, pos in subtree.leaves()])
        topics.append(entity)

# Fallback: Use proper nouns if no named entities found
if not topics:
    topics = [word for word, pos in pos_tags if pos == 'NNP' and word.lower() not in stopwords.words('english')]

# Frequency count (to prioritize most mentioned topics)
freq_dist = FreqDist(topics)
top_topics = [topic for topic, freq in freq_dist.most_common(2)]

# Final summary
if top_topics:
    if len(top_topics) == 1:
        summary = f"The user asked mainly about {top_topics[0]}."
    else:
        summary = f"The user asked mainly about {top_topics[0]} and {top_topics[1]}."
else:
    summary = "The user did not ask about a specific topic."

print(summary)
