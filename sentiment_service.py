
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class SentimentAnalyzer:

    MODEL_NAME = (
        "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )

    def __init__(self):
        # Load the tokenizer and pretrained model once
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_NAME
        )

        # Set model to evaluation mode
        self.model.eval()

    def predict(self, text: str) -> dict:

        # Validate input
        if not isinstance(text, str):
            raise TypeError("Input must be a string.")

        if not text.strip():
            raise ValueError("Text cannot be empty.")

        # Convert text into numerical tensors
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True
        )

        # Perform inference
        with torch.inference_mode():
            outputs = self.model(**inputs)

        # Extract logits
        logits = outputs.logits

        # Convert logits into normalized class scores
        probabilities = torch.softmax(logits, dim=-1)

        # Find predicted class
        predicted_class = torch.argmax(
            probabilities, dim=-1
        ).item()

        # Get predicted label
        label = self.model.config.id2label[predicted_class]

        # Get prediction confidence score
        confidence = probabilities[0][predicted_class].item()

        return {
            "text": text,
            "sentiment": label,
            "confidence": confidence,
            "scores": {
                self.model.config.id2label[i]: probabilities[0][i].item()
                for i in range(len(probabilities[0]))
            }
        }
