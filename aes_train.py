import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import re
import pickle

# ==========================
# 1. Load Dataset
# ==========================
data = pd.read_csv("Data/train.csv")
data = data[["essay_id", "full_text", "score"]]

train_texts, val_texts, train_labels, val_labels = train_test_split(
    data["full_text"].values, data["score"].values, test_size=0.2, random_state=42
)

# ==========================
# 2. Tokenizer + Vocab
# ==========================
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()

# Build vocab from training data
counter = Counter()
for txt in train_texts:
    counter.update(tokenize(txt))

vocab = {word: idx + 2 for idx, (word, _) in enumerate(counter.most_common(20000))}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

def encode(text, max_len=256):
    tokens = tokenize(text)
    ids = [vocab.get(t, 1) for t in tokens]  # 1 = <UNK>
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

# ==========================
# 3. Dataset
# ==========================
class EssayDataset(Dataset):
    def __init__(self, texts, labels, max_len=256):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        ids = encode(text, self.max_len)

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.float32),
        }

train_dataset = EssayDataset(train_texts, train_labels)
val_dataset = EssayDataset(val_texts, val_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ==========================
# 4. LSTM Model
# ==========================
class EssayScoringLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super(EssayScoringLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        lstm_out, _ = self.lstm(x)
        pooled = torch.mean(lstm_out, dim=1)
        out = self.fc(self.dropout(pooled))
        return out.squeeze(-1)

# ==========================
# 5. Training Setup
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EssayScoringLSTM(vocab_size=len(vocab)).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ==========================
# 6. Training Loop
# ==========================
EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs, labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids)
            loss = criterion(outputs, labels.view(-1))
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

# ==========================
# 7. Save Model + Vocab
# ==========================
torch.save(model.state_dict(), "essay_scoring_lstm_only.pt")
with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

print("✅ Model and vocab saved (essay_scoring_lstm_only.pt, vocab.pkl)")
