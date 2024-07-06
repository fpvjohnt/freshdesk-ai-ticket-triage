import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Load the preprocessed data
df = pd.read_csv('preprocessed_tickets.csv')

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

# TF-IDF Vectorization with n-grams
tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X = tfidf.fit_transform(df['processed_text'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the pipeline
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
])

# Define the parameter grid
param_grid = {
    'clf__estimator__n_estimators': [100, 200],
    'clf__estimator__max_depth': [None, 10, 20],
    'clf__estimator__min_samples_split': [2, 5, 10]
}

# Perform grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Print the best parameters
print("Best parameters:", grid_search.best_params_)

# Make predictions
y_pred = grid_search.predict(X_test)

# Print classification report for each tag
for i, tag in enumerate(mlb.classes_):
    print(f"\nClassification report for {tag}:")
    print(classification_report(y_test[:, i], y_pred[:, i]))

# Feature importance
feature_importance = grid_search.best_estimator_.named_steps['clf'].estimators_[0].feature_importances_
word_importance = dict(zip(tfidf.get_feature_names(), feature_importance))
sorted_word_importance = sorted(word_importance.items(), key=lambda x: x[1], reverse=True)

print("\nTop 20 most important words/phrases:")
for word, importance in sorted_word_importance[:20]:
    print(f"{word}: {importance}")

# Save the model and vectorizer
import joblib
joblib.dump(grid_search.best_estimator_, 'improved_model.joblib')
joblib.dump(tfidf, 'improved_tfidf_vectorizer.joblib')
joblib.dump(mlb, 'improved_multi_label_binarizer.joblib')

print("\nImproved model, vectorizer, and multi-label binarizer saved.")
