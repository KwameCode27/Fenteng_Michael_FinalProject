import os, torch, pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from train import HausaDataset, SentimentModel, pad_sequence, numericalize, MAX_LEN

DATA_PATH = os.path.join("data", "train_hausa.csv")
MODEL_PATH = "models/hausa_sentiment_custom.pt"

checkpoint = torch.load(MODEL_PATH, map_location="cpu")
stoi, itos = checkpoint["stoi"], checkpoint["itos"]
label2id, id2label = checkpoint["label2id"], checkpoint["id2label"]

df = pd.read_csv(DATA_PATH)
test_df = df.sample(frac=0.2, random_state=42)

test_ds = HausaDataset(test_df)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32)

model = SentimentModel(len(itos), 128, 128, len(label2id))
model.load_state_dict(checkpoint["model_state"])
model.eval()

preds, labels = [], []
with torch.no_grad():
    for X, y in test_loader:
        outputs = model(X)
        pred = torch.argmax(outputs, dim=1)
        preds.extend(pred.numpy())
        labels.extend(y.numpy())

print(classification_report(labels, preds, target_names=[id2label[i] for i in range(len(id2label))]))
print(confusion_matrix(labels, preds))
