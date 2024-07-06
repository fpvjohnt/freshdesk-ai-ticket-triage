import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
from langdetect import detect

# Load the CSV file
df = pd.read_csv('FreshDesk_Analytics.csv')

# Display basic information about the dataset
print(df.info())

# Check for missing values
print("\nMissing values:\n", df.isnull().sum())

# Display a few sample rows
print("\nSample rows:\n", df.head())

# Analyze language distribution
def detect_language(text):
    if isinstance(text, str):
        try:
            return detect(text)
        except:
            return 'unknown'
    return 'unknown'

df['language'] = df['Conversation Summary'].apply(detect_language)
language_dist = df['language'].value_counts()
print("\nLanguage distribution:\n", language_dist)

# Analyze ticket length
df['ticket_length'] = df['Conversation Summary'].apply(lambda x: len(x) if isinstance(x, str) else 0)
print("\nTicket length statistics:\n", df['ticket_length'].describe())

# Plot ticket length distribution
plt.figure(figsize=(10, 6))
plt.hist(df['ticket_length'], bins=50)
plt.title('Distribution of Ticket Lengths')
plt.xlabel('Ticket Length (characters)')
plt.ylabel('Frequency')
plt.savefig('ticket_length_distribution.png')
plt.close()

# Analyze common words (simple approach)
def simple_preprocess(text):
    if not isinstance(text, str):
        return []
    # Convert to lowercase and split into words
    words = text.lower().split()
    # Remove punctuation and numbers
    words = [re.sub(r'[^\w\s]', '', word) for word in words]
    # Remove empty strings
    words = [word for word in words if word]
    return words

all_words = [word for ticket in df['Conversation Summary'] for word in simple_preprocess(ticket)]
word_freq = Counter(all_words)
print("\nMost common words:\n", word_freq.most_common(20))

# Analyze priority distribution
priority_dist = df['Priority'].value_counts()
print("\nPriority distribution:\n", priority_dist)

# Analyze tag distribution
if df['Tags'].dtype == 'object':
    all_tags = [tag.strip() for tags in df['Tags'].dropna() for tag in tags.split(',')]
    tag_freq = Counter(all_tags)
    print("\nMost common tags:\n", tag_freq.most_common(20))
else:
    print("\nTags column is not in the expected format.")

# Save the results to a file
with open('analysis_results.txt', 'w') as f:
    f.write("Data Analysis Results\n\n")
    f.write(str(df.info()) + "\n\n")
    f.write("Missing values:\n" + str(df.isnull().sum()) + "\n\n")
    f.write("Language distribution:\n" + str(language_dist) + "\n\n")
    f.write("Ticket length statistics:\n" + str(df['ticket_length'].describe()) + "\n\n")
    f.write("Most common words:\n" + str(word_freq.most_common(20)) + "\n\n")
    f.write("Priority distribution:\n" + str(priority_dist) + "\n\n")
    if df['Tags'].dtype == 'object':
        f.write("Most common tags:\n" + str(tag_freq.most_common(20)) + "\n\n")
    else:
        f.write("Tags column is not in the expected format.\n\n")

print("Analysis complete. Results saved to 'analysis_results.txt' and ticket length distribution plot saved as 'ticket_length_distribution.png'.")