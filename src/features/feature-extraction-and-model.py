import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer

# Load the preprocessed data
df = pd.read_csv('preprocessed_tickets.csv')

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=1000)
text_features = tfidf.fit_transform(df['processed_text'])

# Process tags
df['Tags'] = df['Tags'].fillna('').apply(lambda x: [tag.strip() for tag in str(x).split(',')])

# Get the most common tags (e.g., top 20)
all_tags = [tag for tags in df['Tags'] for tag in tags]
top_tags = pd.Series(all_tags).value_counts().nlargest(20).index.tolist()

# Filter tags to keep only top tags
df['filtered_tags'] = df['Tags'].apply(lambda x: [tag for tag in x if tag in top_tags])

# Multi-label binarization
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['filtered_tags'])

# Combine TF-IDF features with Priority
X = np.hstack((text_features.toarray(), df['Priority'].values.reshape(-1, 1)))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Multi-output Random Forest model
rf_model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Print classification report for each tag
for i, tag in enumerate(mlb.classes_):
    print(f"\nClassification report for {tag}:")
    print(classification_report(y_test[:, i], y_pred[:, i]))

# Feature importance
feature_importance = np.mean([estimator.feature_importances_ for estimator in rf_model.estimators_], axis=0)
word_importance = dict(zip(tfidf.get_feature_names(), feature_importance[:1000]))
sorted_word_importance = sorted(word_importance.items(), key=lambda x: x[1], reverse=True)

print("\nTop 20 most important words:")
for word, importance in sorted_word_importance[:20]:
    print(f"{word}: {importance}")

# Save the model, vectorizer, and multi-label binarizer
import joblib
joblib.dump(rf_model, 'rf_model_multi_label.joblib')
joblib.dump(tfidf, 'tfidf_vectorizer.joblib')
joblib.dump(mlb, 'multi_label_binarizer.joblib')

print("\nModel, vectorizer, and multi-label binarizer saved.")