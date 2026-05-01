import torch
import torch.nn as nn
import torch.optim as optim

# 📂 читаем данные прямо в Streamlit Cloud
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

tokens = text.split()
words = list(set(tokens))

word_to_ix = {w: i for i, w in enumerate(words)}
ix_to_word = {i: w for w, i in word_to_ix.items()}

SEQ_LEN = 6

data = []
for i in range(len(tokens) - SEQ_LEN):
    seq = tokens[i:i+SEQ_LEN]
    tgt = tokens[i+SEQ_LEN]
    data.append((seq, tgt))

# 🧠 Transformer (упрощённый)
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(words), 128)
        self.attn = nn.MultiheadAttention(128, 4, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, len(words))
        )

    def forward(self, x):
        x = self.emb(x)
        x, _ = self.attn(x, x, x)
        return self.ff(x[:, -1])

model = Model()

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# 🔥 ОБУЧЕНИЕ ПРИ ЗАПУСКЕ (Streamlit Cloud)
for epoch in range(3):  # мало эпох = быстрее запуск
    total = 0

    for seq, tgt in data:
        model.zero_grad()

        x = torch.tensor([[word_to_ix[w] for w in seq]])
        y = torch.tensor([word_to_ix[tgt]])

        out = model(x)
        loss = loss_fn(out, y)

        loss.backward()
        optimizer.step()

        total += loss.item()

    print("epoch", epoch, "loss", total)

# 🚀 генерация
def generate(text, length=12):
    words_input = text.lower().split()

    for _ in range(length):
        seq = words_input[-SEQ_LEN:]

        if any(w not in word_to_ix for w in seq):
            break

        x = torch.tensor([[word_to_ix[w] for w in seq]])

        out = model(x)

        probs = torch.softmax(out / 1.3, dim=1)

        pred = torch.multinomial(probs, 1).item()

        words_input.append(ix_to_word[pred])

    return " ".join(words_input)
