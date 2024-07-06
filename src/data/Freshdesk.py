import requests
import pandas as pd
import joblib
import os
import datetime
import pytz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Print current working directory
print("Current working directory:", os.getcwd())

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Freshdesk API configuration
FRESHDESK_DOMAIN = "https://cintoo.freshdesk.com"
API_KEY = "AYrGLqYvCFrlwBTMEFb"  # Replace with your actual API key

def get_tickets(start_date, end_date):
    url = f"{FRESHDESK_DOMAIN}/api/v2/tickets"
    headers = {
        "Content-Type": "application/json"
    }
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
    
    # Filter tickets based on the date range
    filtered_tickets = [
        ticket for ticket in all_tickets
        if start_date <= datetime.datetime.fromisoformat(ticket['created_at'].replace('Z', '+00:00')) < end_date
    ]
    
    print(f"Total tickets found within the date range: {len(filtered_tickets)}")
    return filtered_tickets

def get_conversation(ticket_id):
    url = f"{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}/conversations"
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers, auth=(API_KEY, 'X'))
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to retrieve conversation for ticket {ticket_id}. Status code: {response.status_code}")
        return []

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)

def analyze_tickets(tickets):
    # Prepare data for analysis
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
            'full_text': full_text,
            'preprocessed_text': preprocess_text(full_text)
        })
    
    df = pd.DataFrame(ticket_data)
    
    # Analyze common words
    all_words = ' '.join(df['preprocessed_text']).split()
    word_freq = Counter(all_words)
    common_words = word_freq.most_common(20)
    
    # Analyze ticket creation over time
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['date'] = df['created_at'].dt.date
    daily_counts = df['date'].value_counts().sort_index()
    
    # Analyze status and priority distribution
    status_counts = df['status'].value_counts()
    priority_counts = df['priority'].value_counts()
    
    # Visualizations
    plt.figure(figsize=(12, 6))
    plt.plot(daily_counts.index, daily_counts.values)
    plt.title('Ticket Creation Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Tickets')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('ticket_creation_over_time.png')
    plt.close()
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x=[word for word, count in common_words], y=[count for word, count in common_words])
    plt.title('Most Common Words in Tickets')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('common_words.png')
    plt.close()
    
    # Print analysis results
    print("\nMost common words:")
    for word, count in common_words:
        print(f"{word}: {count}")
    
    print("\nTicket status distribution:")
    print(status_counts)
    
    print("\nTicket priority distribution:")
    print(priority_counts)
    
    return df

# Load the trained model and related objects
model_path = os.path.join(current_dir, 'rf_model_multi_label.joblib')
vectorizer_path = os.path.join(current_dir, 'tfidf_vectorizer.joblib')
mlb_path = os.path.join(current_dir, 'multi_label_binarizer.joblib')

try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    mlb = joblib.load(mlb_path)
    print("Model, vectorizer, and binarizer loaded successfully.")
except Exception as e:
    print(f"Error loading model files: {str(e)}")
    raise

# Set date range for ticket retrieval (January 1, 2024 to current date)
start_date = datetime.datetime(2024, 1, 1, tzinfo=pytz.UTC)
end_date = datetime.datetime.now(pytz.UTC)

# Fetch tickets
tickets = get_tickets(start_date, end_date)

# Analyze tickets
df = analyze_tickets(tickets)

# Classify tickets
def classify_ticket(ticket_text, model, vectorizer, mlb):
    X = vectorizer.transform([ticket_text])
    predictions = model.predict(X)
    predicted_tags = mlb.inverse_transform(predictions)[0]
    return predicted_tags

# Classify and update tickets
for _, ticket in df.iterrows():
    predicted_tags = classify_ticket(ticket['preprocessed_text'], model, vectorizer, mlb)
    print(f"Ticket ID: {ticket['id']}")
    print(f"Subject: {ticket['subject']}")
    print(f"Predicted Tags: {predicted_tags}")
    print("---")

# Identify recurring issues
print("\nPotential recurring issues:")
word_pairs = [' '.join(pair) for pair in zip(all_words[:-1], all_words[1:])]
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

print("\nAnalysis completed. Check 'ticket_creation_over_time.png' and 'common_words.png' for visualizations.")