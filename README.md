# Sentiment Analyzer

A simple sentiment analysis application built with Python and a pretrained DistilBERT model from Hugging Face. It classifies English text as either positive or negative.

## Model

This project uses [`distilbert/distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english), a DistilBERT model fine-tuned on the Stanford Sentiment Treebank (SST-2) dataset.

The model returns two class scores:

- `POSITIVE`
- `NEGATIVE`

It does not have a neutral class. As a result, neutral or ambiguous text is still assigned to one of the two classes.

## Features

- Sentiment prediction using a pretrained transformer model.
- Confidence scores for both sentiment classes.
- Reusable prediction service.
- Input validation.
- Automated tests using pytest.
- Browser-based interface built with Gradio.

## Project Structure

```text
sentiment-analyzer/
├── app.py
├── sentiment_service.py
├── sentiment.py
├── experiment.py
├── main.py
├── requirements.txt
├── README.md
└── tests/
    └── test_sentiment.py
```

## Installation

Clone the repository:

```bash
git clone https://github.com/siraw21/sentiment-analyzer.git
cd sentiment-analyzer
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

The pretrained model is downloaded automatically from Hugging Face the first time the application runs.

## Usage

### Run the web application

```bash
python app.py
```

Open the local URL displayed in the terminal to use the interface.

### Run the command-line application

```bash
python main.py
```

### Run the tests

```bash
pytest -v
```

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- DistilBERT
- Gradio
- pytest
