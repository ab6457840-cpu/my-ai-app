import torch
import torch.nn as nn
import torch.optim as optim

# 📂 ЧТЕНИЕ ДАННЫХ ИЗ ФАЙЛА
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

tokens = text.split()
words = list(set(tokens))

word_to_ix = {w: i for i, w in enumerate(words)}
ix_to_word = {i: w for w, i in word_to_ix.items()}

SEQ_LEN = 4

# 📦 подготовка данных
data = []
for i in range(len(tokens) - SEQ_LEN):
    seq = tokens[i:i+SEQ_LEN]
    target = tokens[i+SEQ_LEN]
    data.append((seq, target))

# 🧠 модель
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(words), 64)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.fc = nn.Linear(128, len(words))

    def forward(self, x):
        x = self.emb(x)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = Model()

optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# 🔥 ОБУЧЕНИЕ
for epoch in range(10):  # можно увеличить
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

# 🚀 ГЕНЕРАЦИЯ (убрали жёсткое повторение)
def generate(text, length=10):
    words_input = text.lower().split()

    for _ in range(length):
        if len(words_input) < SEQ_LEN:
            break

        seq = words_input[-SEQ_LEN:]

        if any(w not in word_to_ix for w in seq):
            break

        x = torch.tensor([[word_to_ix[w] for w in seq]])

        out = model(x)
        probs = torch.softmax(out / 1.2, dim=1)

        pred = torch.multinomial(probs, 1).item()
        next_word = ix_to_word[pred]

        # 🔥 защита от повторов
        if next_word in words_input[-3:]:
            continue

        words_input.append(next_word)

    return " ".join(words_input)
