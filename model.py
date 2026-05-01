import torch
import torch.nn as nn
import torch.optim as optim

# 📂 загрузка датасета
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

tokens = text.split()
words = list(set(tokens))

word_to_ix = {w: i for i, w in enumerate(words)}
ix_to_word = {i: w for w, i in word_to_ix.items()}

VOCAB_SIZE = len(words)
SEQ_LEN = 6
EMB = 128

# 📦 подготовка данных
data = []
for i in range(len(tokens) - SEQ_LEN):
    seq = tokens[i:i+SEQ_LEN]
    tgt = tokens[i+SEQ_LEN]
    data.append((seq, tgt))

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
        x, _ = self.attn(x, x, x)
        return self.ff(x[:, -1])

model = Model()

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# 🔥 обучение (Streamlit Cloud запускает каждый раз)
for epoch in range(3):
    total_loss = 0

    for seq, tgt in data:
        model.zero_grad()

        x = torch.tensor([[word_to_ix[w] for w in seq]])
        y = torch.tensor([word_to_ix[tgt]])

        out = model(x)
        loss = loss_fn(out, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("epoch", epoch, "loss", total_loss)

# 🚀 ГЕНЕРАЦИЯ (TOP-K + TEMPERATURE)
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

        # 🔥 temperature
        logits = out / temperature

        # 🔥 top-k фильтрация
        values, indices = torch.topk(logits, top_k)

        probs = torch.softmax(values, dim=1).squeeze()

        choice = torch.multinomial(probs, 1).item()
        next_word = ix_to_word[indices[0][choice].item()]

        # 🔥 защита от повторов
        if next_word in words_input[-3:]:
            continue

        words_input.append(next_word)

    return " ".join(words_input)
