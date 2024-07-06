import os
from dotenv import load_dotenv
from src.data import ticket_preprocessing, Freshdesk
from src.features import feature_extraction
from src.models import train_model, predict

# Load environment variables
load_dotenv()

def main():
    # Fetch new data from Freshdesk API
    print("Fetching new tickets from Freshdesk...")
    Freshdesk.fetch_new_tickets()

    # Preprocess data
    print("Preprocessing tickets...")
    ticket_preprocessing.preprocess_tickets()

    # Extract features
    print("Extracting features...")
    features, labels = feature_extraction.extract_features()

    # Train model
    print("Training model...")
    model = train_model.train(features, labels)

    # Fetch a new ticket for prediction
    print("Fetching latest ticket for prediction...")
    new_ticket = Freshdesk.get_latest_ticket()

    # Make prediction
    print("Making prediction...")
    predicted_tags = predict.predict(model, new_ticket['description'])

    print(f"Predicted tags for new ticket: {predicted_tags}")

    # Update the ticket in Freshdesk with predicted tags
    print("Updating ticket with predicted tags...")
    Freshdesk.update_ticket_tags(new_ticket['id'], predicted_tags)

    print("Process completed successfully!")

if __name__ == "__main__":
    main()