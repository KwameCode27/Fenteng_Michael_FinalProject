import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
import re

# Paths
DATA_PATH = os.path.join( "data/train_hausa.csv")
SAVE_DIR = "models/hausa_sentiment_custom.pt"

# Hyperparameters
MAX_VOCAB_SIZE = 20000
EMBED_DIM = 128
HIDDEN_DIM = 128
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Step 1: Load Data ---
df = pd.read_csv(DATA_PATH)
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

label2id = {label: idx for idx, label in enumerate(df['label'].unique())}
id2label = {idx: label for label, idx in label2id.items()}

# --- Step 2: Tokenizer & Vocab ---
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", "", text)  # keep letters & numbers
    return text.split()

counter = Counter()
for text in train_df['text']:
    counter.update(tokenize(text))

most_common = counter.most_common(MAX_VOCAB_SIZE-2)
itos = ["<PAD>", "<UNK>"] + [word for word, _ in most_common]
stoi = {word: i for i, word in enumerate(itos)}

def numericalize(text):
    return [stoi.get(tok, stoi["<UNK>"]) for tok in tokenize(text)]

MAX_LEN = 50  # truncate/pad length

def pad_sequence(seq, max_len=MAX_LEN):
    if len(seq) < max_len:
        seq = seq + [stoi["<PAD>"]] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq

# --- Step 3: Dataset ---
class HausaDataset(Dataset):
    def __init__(self, df):
        self.texts = [pad_sequence(numericalize(text)) for text in df['text']]
        self.labels = [label2id[label] for label in df['label']]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx]), torch.tensor(self.labels[idx])

train_ds = HausaDataset(train_df)
val_ds = HausaDataset(val_df)
test_ds = HausaDataset(test_df)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# --- Step 4: Model ---
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)  # biLSTM
        return self.fc(hidden)

model = SentimentModel(len(itos), EMBED_DIM, HIDDEN_DIM, len(label2id)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# --- Step 5: Training Loop ---
def train_epoch(loader):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            pred = torch.argmax(outputs, dim=1)
            preds.extend(pred.cpu().numpy())
            labels.extend(y.cpu().numpy())
    return accuracy_score(labels, preds), f1_score(labels, preds, average="macro")

for epoch in range(EPOCHS):
    train_loss = train_epoch(train_loader)
    acc, f1 = eval_epoch(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f}, Val Acc: {acc:.4f}, Val F1: {f1:.4f}")

# --- Step 6: Save Model ---
torch.save({
    "model_state": model.state_dict(),
    "stoi": stoi,
    "itos": itos,
    "label2id": label2id,
    "id2label": id2label
}, SAVE_DIR)

print(f"✅ Model saved to {SAVE_DIR}")
