import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from langdetect import detect
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import datetime
import pytz
import joblib
import os

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Freshdesk API configuration
FRESHDESK_DOMAIN = "https://.freshdesk.com"
API_KEY = ""  # Replace with your actual API key

def get_tickets(start_date, end_date):
    url = f"{FRESHDESK_DOMAIN}/api/v2/tickets"
    headers = {"Content-Type": "application/json"}
    params = {
        "order_by": "created_at",
        "order_type": "desc",
        "per_page": 100,
        "page": 1
    }
    all_tickets = []
    
    print(f"Fetching tickets from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    while True:
        response = requests.get(url, headers=headers, auth=(API_KEY, 'X'), params=params)
        if response.status_code == 200:
            tickets = response.json()
            print(f"Retrieved {len(tickets)} tickets on page {params['page']}")
            if not tickets:
                break
            all_tickets.extend(tickets)
            if len(tickets) < 100:
                break
            params["page"] += 1
        else:
            print(f"Failed to retrieve tickets. Status code: {response.status_code}")
            print(f"Response content: {response.text}")
            break
    
    filtered_tickets = [
        ticket for ticket in all_tickets
        if start_date <= datetime.datetime.fromisoformat(ticket['created_at'].replace('Z', '+00:00')) < end_date
    ]
    
    print(f"Total tickets found within the date range: {len(filtered_tickets)}")
    return filtered_tickets

def get_conversation(ticket_id):
    url = f"{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}/conversations"
    headers = {"Content-Type": "application/json"}
    response = requests.get(url, headers=headers, auth=(API_KEY, 'X'))
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to retrieve conversation for ticket {ticket_id}. Status code: {response.status_code}")
        return []

def detect_language(text):
    if isinstance(text, str):
        try:
            return detect(text)
        except:
            return 'unknown'
    return 'unknown'

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)

def analyze_tickets(tickets):
    ticket_data = []
    for ticket in tickets:
        conversations = get_conversation(ticket['id'])
        full_text = f"{ticket['subject']} {ticket.get('description_text', '')}"
        for conv in conversations:
            full_text += f" {conv.get('body_text', '')}"
        
        ticket_data.append({
            'id': ticket['id'],
            'created_at': ticket['created_at'],
            'subject': ticket['subject'],
            'status': ticket['status'],
            'priority': ticket['priority'],
            'tags': ', '.join(ticket.get('tags', [])),
            'full_text': full_text,
            'preprocessed_text': preprocess_text(full_text)
        })
    
    df = pd.DataFrame(ticket_data)
    df['language'] = df['full_text'].apply(detect_language)
    df['ticket_length'] = df['full_text'].apply(len)
    
    return df

def plot_ticket_length_distribution(df):
    plt.figure(figsize=(10, 6))
    plt.hist(df['ticket_length'], bins=50)
    plt.title('Distribution of Ticket Lengths')
    plt.xlabel('Ticket Length (characters)')
    plt.ylabel('Frequency')
    plt.savefig('ticket_length_distribution.png')
    plt.close()

def plot_common_words(word_freq):
    plt.figure(figsize=(12, 6))
    words, counts = zip(*word_freq.most_common(20))
    sns.barplot(x=list(words), y=list(counts))
    plt.title('20 Most Common Words in Tickets')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('common_words.png')
    plt.close()

def extract_features_and_labels(df):
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_features = tfidf.fit_transform(df['preprocessed_text'])
    
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(df['tags'].fillna('').str.split(', '))
    
    return tfidf_features, labels, tfidf, mlb

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred))
    
    return model

# Main execution
start_date = datetime.datetime(2024, 1, 1, tzinfo=pytz.UTC)
end_date = datetime.datetime.now(pytz.UTC)

# Fetch and analyze tickets
tickets = get_tickets(start_date, end_date)
df = analyze_tickets(tickets)

# Display basic information about the dataset
print(df.info())

# Check for missing values
print("\nMissing values:\n", df.isnull().sum())

# Display a few sample rows
print("\nSample rows:\n", df.head())

# Analyze language distribution
language_dist = df['language'].value_counts()
print("\nLanguage distribution:\n", language_dist)

# Analyze ticket length
print("\nTicket length statistics:\n", df['ticket_length'].describe())

# Plot ticket length distribution
plot_ticket_length_distribution(df)

# Analyze common words
all_words = ' '.join(df['preprocessed_text']).split()
word_freq = Counter(all_words)
print("\nMost common words:\n", word_freq.most_common(20))

# Plot common words
plot_common_words(word_freq)

# Analyze priority distribution
priority_dist = df['priority'].value_counts()
print("\nPriority distribution:\n", priority_dist)

# Analyze tag distribution
all_tags = [tag.strip() for tags in df['tags'].dropna() for tag in tags.split(',')]
tag_freq = Counter(all_tags)
print("\nMost common tags:\n", tag_freq.most_common(20))

# Extract features and labels
X, y, tfidf, mlb = extract_features_and_labels(df)

# Train the model
model = train_model(X, y)

# Classify tickets
predictions = model.predict(X)
df['Predicted_Tags'] = mlb.inverse_transform(predictions)

print("\nSample of classified tickets:")
for _, ticket in df.head().iterrows():
    print(f"Ticket ID: {ticket['id']}")
    print(f"Subject: {ticket['subject']}")
    print(f"Actual Tags: {ticket['tags']}")
    print(f"Predicted Tags: {ticket['Predicted_Tags']}")
    print("---")

# Identify recurring issues
print("\nPotential recurring issues:")
word_pairs = [' '.join(pair) for pair in zip(df['preprocessed_text'].str.split().sum()[:-1], df['preprocessed_text'].str.split().sum()[1:])]
pair_freq = Counter(word_pairs)
recurring_issues = pair_freq.most_common(10)
for issue, count in recurring_issues:
    print(f"{issue}: {count}")

# Analyze potential knowledge base confusion
print("\nPotential knowledge base confusion:")
kb_related_words = ['guide', 'documentation', 'manual', 'instructions', 'help', 'confused', 'unclear']
kb_mentions = df[df['preprocessed_text'].str.contains('|'.join(kb_related_words))]
print(f"Number of tickets potentially related to knowledge base confusion: {len(kb_mentions)}")

if len(kb_mentions) > 0:
    print("Sample tickets with potential knowledge base confusion:")
    for _, ticket in kb_mentions.head().iterrows():
        print(f"Ticket ID: {ticket['id']}")
        print(f"Subject: {ticket['subject']}")
        print("---")

# Save the results to a file
with open('analysis_results.txt', 'w') as f:
    f.write("Data Analysis Results\n\n")
    f.write(str(df.info()) + "\n\n")
    f.write("Missing values:\n" + str(df.isnull().sum()) + "\n\n")
    f.write("Language distribution:\n" + str(language_dist) + "\n\n")
    f.write("Ticket length statistics:\n" + str(df['ticket_length'].describe()) + "\n\n")
    f.write("Most common words:\n" + str(word_freq.most_common(20)) + "\n\n")
    f.write("Priority distribution:\n" + str(priority_dist) + "\n\n")
    f.write("Most common tags:\n" + str(tag_freq.most_common(20)) + "\n\n")
    f.write("Potential recurring issues:\n" + str(recurring_issues) + "\n\n")
    f.write("Number of tickets potentially related to knowledge base confusion: " + str(len(kb_mentions)) + "\n\n")

# Save the model and related objects
joblib.dump(model, 'ticket_classification_model.joblib')
joblib.dump(tfidf, 'tfidf_vectorizer.joblib')
joblib.dump(mlb, 'multilabel_binarizer.joblib')

print("\nAnalysis complete. Results saved to 'analysis_results.txt'.")
print("Visualizations saved as 'ticket_length_distribution.png' and 'common_words.png'.")
print("Model, vectorizer, and label binarizer saved for future use.")
