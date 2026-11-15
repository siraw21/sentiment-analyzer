
import gradio as gr

from sentiment_service import SentimentAnalyzer


# Load the model once when the application starts.
analyzer = SentimentAnalyzer()


def analyze_sentiment(text):
    """Analyze the sentiment of user-provided text."""

    if not isinstance(text, str) or not text.strip():
        raise gr.Error("Please enter some text before analyzing.")

    result = analyzer.predict(text)

    return (
        result["scores"],
        result["sentiment"],
        f'{result["confidence"]:.2%}'
    )


demo = gr.Interface(
    fn=analyze_sentiment,

    inputs=gr.Textbox(
        lines=4,
        label="Enter your text",
        placeholder="For example: I really enjoy learning machine learning."
    ),

    outputs=[
        gr.Label(
            num_top_classes=2,
            label="Sentiment scores"
        ),
        gr.Textbox(label="Predicted sentiment"),
        gr.Textbox(label="Confidence")
    ],

    title="Sentiment Analyzer",
    description=(
        "Analyze English text using a pretrained "
        "DistilBERT sentiment classification model."
    ),

    examples=[
        ["I absolutely love this product!"],
        ["This is the worst experience I have ever had."],
        ["The product arrived yesterday."]
    ]
)


if __name__ == "__main__":
    demo.launch()
