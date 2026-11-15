from transformers import AutoTokenizer

MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

text = "I really love this product."

# Convert text into model inputs
inputs = tokenizer(text, return_tensors="pt")

print("Original text:", text)
print("Tokenized words:", tokenizer.tokenize(text))
print("Token IDs:", inputs["input_ids"])
print("Attention mask:", inputs["attention_mask"])