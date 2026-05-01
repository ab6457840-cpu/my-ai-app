import torch
import torch.nn as nn
import torch.optim as optim
import os

# 📂 данные
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

tokens = text.split()

# 🔥 СТАБИЛЬНЫЙ словарь (ВАЖНО!)
words = sorted(list(set(tokens)))

word_to_ix = {w: i for i, w in enumerate(words)}
ix_to_word = {i: w for w, i in word_to_ix.items()}

VOCAB_SIZE = len(words)
SEQ_LEN = 10
EMB = 128

# 📦 dataset
data = []
for i in range(len(tokens) - SEQ_LEN):
    seq = tokens[i:i+SEQ_LEN]
    tgt = tokens[i+SEQ_LEN]
    data.append((seq, tgt))

# 🧠 модель
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, EMB)

        self.attn = nn.MultiheadAttention(EMB, 4, batch_first=True)

        self.ff = nn.Sequential(
            nn.Linear(EMB, 256),
            nn.ReLU(),
            nn.Linear(256, VOCAB_SIZE)
        )

    def forward(self, x):
        x = self.emb(x)
        x, _ = self.attn(x, x, x)
        return self.ff(x[:, -1])

model = Model()

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# 🔥 ФИКС 1: обучение только если нет сохранённой модели
if not os.path.exists("model.pt"):

    print("🔄 Обучение модели...")

    for epoch in range(3):
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

    torch.save(model.state_dict(), "model.pt")

else:
    print("📦 Загружаем модель...")
    model.load_state_dict(torch.load("model.pt"))
    model.eval()

# 🚀 ГЕНЕРАЦИЯ (исправленная)

def generate(text, length=15, temperature=1.2, top_k=10):
    words_input = text.lower().split()

    for _ in range(length):

        seq = words_input[-SEQ_LEN:]

        if len(seq) < SEQ_LEN:
            break

        if any(w not in word_to_ix for w in seq):
            break

        x = torch.tensor([[word_to_ix[w] for w in seq]])

        out = model(x)

        logits = out / temperature

        values, indices = torch.topk(logits, top_k)
        probs = torch.softmax(values, dim=1).squeeze()

        choice = torch.multinomial(probs, 1).item()
        word = ix_to_word[indices[0][choice].item()]

        # 🔥 ФИКС 2: жёсткая защита от повторов
        if word in words_input[-5:]:
            continue

        words_input.append(word)

    return " ".join(words_input)
