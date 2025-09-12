import torch, sys
from train import SentimentModel, tokenize, numericalize, pad_sequence, MAX_LEN

MODEL_PATH = "models/hausa_sentiment_custom.pt"
checkpoint = torch.load(MODEL_PATH, map_location="cpu")

stoi, itos = checkpoint["stoi"], checkpoint["itos"]
label2id, id2label = checkpoint["label2id"], checkpoint["id2label"]

model = SentimentModel(len(itos), 128, 128, len(label2id))
model.load_state_dict(checkpoint["model_state"])
model.eval()

def predict(text):
    seq = pad_sequence(numericalize(text))
    X = torch.tensor([seq])
    with torch.no_grad():
        outputs = model(X)
        pred = torch.argmax(outputs, dim=1).item()
    return id2label[pred]

if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = sys.argv[1]
        sentiment = predict(text)
        print(f"Text: {text}\nPredicted Sentiment: {sentiment}")
    else:
        print("⚠️ Usage: python src/predict.py 'Ina jin dadi sosai.'")
