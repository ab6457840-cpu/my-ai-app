import torch
import torch.nn as nn
import torch.optim as optim

# 📂 данные
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

tokens = text.split()
words = list(set(tokens))

word_to_ix = {w: i for i, w in enumerate(words)}
ix_to_word = {i: w for w, i in word_to_ix.items()}

VOCAB_SIZE = len(words)
SEQ_LEN = 6
EMB = 128

# 📦 dataset
data = []
for i in range(len(tokens) - SEQ_LEN):
    x = tokens[i:i+SEQ_LEN]
    y = tokens[i+SEQ_LEN]
    data.append((x, y))

# 🧠 Transformer модель
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.emb = nn.Embedding(VOCAB_SIZE, EMB)

        self.attn = nn.MultiheadAttention(EMB, num_heads=4, batch_first=True)

        self.ff = nn.Sequential(
            nn.Linear(EMB, 256),
            nn.ReLU(),
            nn.Linear(256, VOCAB_SIZE)
        )

    def forward(self, x):
        x = self.emb(x)

        attn_out, _ = self.attn(x, x, x)

        out = self.ff(attn_out[:, -1, :])

        return out

model = Model()

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# 🔥 обучение
for epoch in range(5):
    total = 0

    for x, y in data:
        model.zero_grad()

        x = torch.tensor([[word_to_ix[w] for w in x]])
        y = torch.tensor([word_to_ix[y]])

        out = model(x)
        loss = loss_fn(out, y)

        loss.backward()
        optimizer.step()

        total += loss.item()

    print("epoch", epoch, "loss", total)

# 🚀 генерация
def generate(text, length=15):
    words_input = text.lower().split()

    for _ in range(length):
        seq = words_input[-SEQ_LEN:]

        if any(w not in word_to_ix for w in seq):
            break

        x = torch.tensor([[word_to_ix[w] for w in seq]])

        out = model(x)

        logits = out / 1.2
        probs = torch.softmax(logits, dim=1)

        pred = torch.multinomial(probs, 1).item()

        words_input.append(ix_to_word[pred])

    return " ".join(words_input)
