
from sentiment_service import SentimentAnalyzer


def main():

    # Create the prediction service
    analyzer = SentimentAnalyzer()

    # Test sentences
    sentences = [
        "I absolutely love this experience!",
        "This is the worst product I have ever purchased.",
        "The product arrived yesterday."
    ]

    # Perform predictions
    for sentence in sentences:

        result = analyzer.predict(sentence)

        print("\nText:", result["text"])
        print("Sentiment:", result["sentiment"])
        print("Confidence:", round(result["confidence"] * 100, 2), "%")
        print("All scores:", result["scores"])


if __name__ == "__main__":
    main()
