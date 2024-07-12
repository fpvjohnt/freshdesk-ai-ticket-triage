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
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Print current working directory
logger.info(f"Current working directory: {os.getcwd()}")

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Freshdesk API configuration
FRESHDESK_DOMAIN = os.getenv('FRESHDESK_DOMAIN', 'https://your-domain.freshdesk.com')
API_KEY = os.getenv('FRESHDESK_API_KEY', '')

# Check if matplotlib and seaborn are available
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logger.warning("Matplotlib or seaborn not available. Plotting functions will be disabled.")

class FreshdeskAPI:
    def __init__(self):
        self.base_url = f"{FRESHDESK_DOMAIN}/api/v2"
        self.headers = {
            "Content-Type": "application/json",
        }

    def fetch_new_tickets(self, start_date, end_date):
        url = f"{self.base_url}/tickets"
        params = {
            "order_by": "created_at",
            "order_type": "desc",
            "per_page": 100,
            "page": 1
        }
        all_tickets = []
        
        logger.info(f"Fetching tickets from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        while True:
            response = requests.get(url, headers=self.headers, auth=(API_KEY, 'X'), params=params)
            if response.status_code == 200:
                tickets = response.json()
                logger.info(f"Retrieved {len(tickets)} tickets on page {params['page']}")
                if not tickets:
                    break
                all_tickets.extend(tickets)
                if len(tickets) < 100:
                    break
                params["page"] += 1
            else:
                logger.error(f"Failed to retrieve tickets. Status code: {response.status_code}")
                logger.error(f"Response content: {response.text}")
                break
        
        # Filter tickets based on the date range
        filtered_tickets = [
            ticket for ticket in all_tickets
            if start_date <= datetime.datetime.fromisoformat(ticket['created_at'].replace('Z', '+00:00')) < end_date
        ]
        
        logger.info(f"Total tickets found within the date range: {len(filtered_tickets)}")
        return filtered_tickets

    def get_conversation(self, ticket_id):
        url = f"{self.base_url}/tickets/{ticket_id}/conversations"
        response = requests.get(url, headers=self.headers, auth=(API_KEY, 'X'))
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to retrieve conversation for ticket {ticket_id}. Status code: {response.status_code}")
            return []

    def get_ticket_details(self, ticket_id):
        url = f"{self.base_url}/tickets/{ticket_id}"
        response = requests.get(url, headers=self.headers, auth=(API_KEY, 'X'))
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to retrieve details for ticket {ticket_id}. Status code: {response.status_code}")
            return None

    def update_ticket_tags(self, ticket_id, tags):
        url = f"{self.base_url}/tickets/{ticket_id}"
        data = {"tags": tags}
        response = requests.put(url, headers=self.headers, auth=(API_KEY, 'X'), json=data)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to update tags for ticket {ticket_id}. Status code: {response.status_code}")
            return None

    def create_ticket(self, subject, description, email):
        url = f"{self.base_url}/tickets"
        data = {
            "subject": subject,
            "description": description,
            "email": email,
            "status": 2,
            "priority": 1
        }
        response = requests.post(url, headers=self.headers, auth=(API_KEY, 'X'), json=data)
        if response.status_code == 201:
            return response.json()
        else:
            logger.error(f"Failed to create ticket. Status code: {response.status_code}")
            return None

    def get_ticket_fields(self):
        url = f"{self.base_url}/ticket_fields"
        response = requests.get(url, headers=self.headers, auth=(API_KEY, 'X'))
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to retrieve ticket fields. Status code: {response.status_code}")
            return None

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)

def analyze_tickets(freshdesk_api, tickets):
    # Prepare data for analysis
    ticket_data = []
    for ticket in tickets:
        conversations = freshdesk_api.get_conversation(ticket['id'])
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
    
    # Print analysis results
    logger.info("\nMost common words:")
    for word, count in common_words:
        logger.info(f"{word}: {count}")
    
    logger.info("\nTicket status distribution:")
    logger.info(status_counts.to_string())
    
    logger.info("\nTicket priority distribution:")
    logger.info(priority_counts.to_string())
    
    return df, all_words

def plot_ticket_analysis(df, common_words):
    if not PLOTTING_AVAILABLE:
        logger.warning("Plotting is not available. Skipping visualizations.")
        return

    # Analyze ticket creation over time
    daily_counts = df['date'].value_counts().sort_index()

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

    logger.info("Visualizations saved as 'ticket_creation_over_time.png' and 'common_words.png'")

def load_model():
    model_path = os.path.join(current_dir, 'rf_model_multi_label.joblib')
    vectorizer_path = os.path.join(current_dir, 'tfidf_vectorizer.joblib')
    mlb_path = os.path.join(current_dir, 'multi_label_binarizer.joblib')

    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        mlb = joblib.load(mlb_path)
        logger.info("Model, vectorizer, and binarizer loaded successfully.")
        return model, vectorizer, mlb
    except Exception as e:
        logger.error(f"Error loading model files: {str(e)}")
        raise

def classify_ticket(ticket_text, model, vectorizer, mlb):
    X = vectorizer.transform([ticket_text])
    predictions = model.predict(X)
    predicted_tags = mlb.inverse_transform(predictions)[0]
    return predicted_tags

def main():
    freshdesk_api = FreshdeskAPI()

    # Set date range for ticket retrieval (January 1, 2024 to current date)
    start_date = datetime.datetime(2024, 1, 1, tzinfo=pytz.UTC)
    end_date = datetime.datetime.now(pytz.UTC)

    # Fetch tickets
    tickets = freshdesk_api.fetch_new_tickets(start_date, end_date)

    # Analyze tickets
    df, all_words = analyze_tickets(freshdesk_api, tickets)

    # Load model
    model, vectorizer, mlb = load_model()

    # Classify and update tickets
    for _, ticket in df.iterrows():
        predicted_tags = classify_ticket(ticket['preprocessed_text'], model, vectorizer, mlb)
        logger.info(f"Ticket ID: {ticket['id']}")
        logger.info(f"Subject: {ticket['subject']}")
        logger.info(f"Predicted Tags: {predicted_tags}")
        logger.info("---")

    # Identify recurring issues
    logger.info("\nPotential recurring issues:")
    word_pairs = [' '.join(pair) for pair in zip(all_words[:-1], all_words[1:])]
    pair_freq = Counter(word_pairs)
    recurring_issues = pair_freq.most_common(10)
    for issue, count in recurring_issues:
        logger.info(f"{issue}: {count}")

    # Analyze potential knowledge base confusion
    logger.info("\nPotential knowledge base confusion:")
    kb_related_words = ['guide', 'documentation', 'manual', 'instructions', 'help', 'confused', 'unclear']
    kb_mentions = df[df['preprocessed_text'].str.contains('|'.join(kb_related_words))]
    logger.info(f"Number of tickets potentially related to knowledge base confusion: {len(kb_mentions)}")

    if len(kb_mentions) > 0:
        logger.info("Sample tickets with potential knowledge base confusion:")
        for _, ticket in kb_mentions.head().iterrows():
            logger.info(f"Ticket ID: {ticket['id']}")
            logger.info(f"Subject: {ticket['subject']}")
            logger.info("---")

    # Generate visualizations
    plot_ticket_analysis(df, recurring_issues)

    logger.info("Analysis completed.")

if __name__ == "__main__":
    main()
