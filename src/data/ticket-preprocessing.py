import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load the CSV file
df = pd.read_csv('FreshDesk_Analytics.csv')

# Function to clean and preprocess text
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers, but keep important terms
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Join tokens back into a string
    return ' '.join(tokens)

# Apply preprocessing to Conversation Summary
df['processed_text'] = df['Conversation Summary'].apply(preprocess_text)

# Remove empty tickets
df = df[df['processed_text'] != ""]

# Save preprocessed data
df.to_csv('preprocessed_tickets.csv', index=False)

print("Preprocessing complete. Preprocessed data saved to 'preprocessed_tickets.csv'.")
print(f"Number of tickets after preprocessing: {len(df)}")
print("\nSample of preprocessed text:")
print(df['processed_text'].head())