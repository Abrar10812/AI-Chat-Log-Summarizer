# AI Chat Log Summarizer

&#x20;

## Overview

The AI Chat Log Summarizer is a Python-based tool that reads chat logs between a user and an AI, parses the conversation, and generates a concise summary. It provides message statistics, frequently used keywords, and the nature of the conversation.

## Features

* Reads and parses chat logs from .txt files.
* Separates user and AI messages for analysis.
* Provides message statistics (total, user, and AI counts).
* Extracts top 5 frequently used keywords (excluding stop words).
* Supports basic keyword extraction with NLTK and TF-IDF.
* Supports processing multiple chat logs from a folder.

## Requirements

* Python 3.8+
* NLTK
* Scikit-learn

## Setup/Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Abrar10812/AI-Chat-Log-Summarizer.git
   ```
2. Create a virtual environment:

   ```bash
   python -m venv env
   ```
3. Activate the virtual environment:

   * Windows:

     ```bash
     .\env\Scripts\activate
     ```
   * Mac/Linux:

     ```bash
     source env/bin/activate
     ```
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your chat logs (.txt files) in the `data/` folder.
2. Run the summarizer:

   ```bash
   python main.py
   ```
3. View the summary output in the console.

## Demo

**Input:**

```
User: Hi, can you tell me about Python?
AI: Sure! Python is a popular programming language known for
its readability.
User: What can I use it for?
AI: You can use Python for web development, data analysis,
AI, and more.
```

**Output:**

```
User messages: 2
AI messages: 2
Summary:
- The conversation had 4 exchanges.
- The user asked mainly about python and programming.
- Most common keywords: python, programming, language, web, development
```

## Contact & Contributing

* Fork the repository and create a new branch.
* Make your changes and submit a pull request.
* Follow the existing code style and structure.
* For issues or improvements, contact me via [GitHub Issues](https://github.com/Abrar10812/AI-Chat-Log-Summarizer/issues).

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
