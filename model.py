import torch
import torch.nn as nn
import torch.optim as optim

text = """
привет друг как ты сегодня
привет как дела у тебя
как ты поживаешь сегодня
что ты делаешь сейчас
у меня всё хорошо
"""

tokens = text.split()
words = list(set(tokens))

word_to_ix = {w: i for i, w in enumerate(words)}
ix_to_word = {i: w for w, i in word_to_ix.items()}

# 🔥 последовательности
seq_len = 3

data = []
for i in range(len(tokens) - seq_len):
    seq = tokens[i:i+seq_len]
    target = tokens[i+seq_len]
    data.append((seq, target))

# 🧠 LSTM модель (ВАЖНО)
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(words), 32)
        self.lstm = nn.LSTM(32, 64, batch_first=True)
        self.fc = nn.Linear(64, len(words))

    def forward(self, x):
        x = self.emb(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = LSTMModel()

optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# 🔥 обучение
for _ in range(300):
    for seq, tgt in data:
        model.zero_grad()

        x = torch.tensor([[word_to_ix[w] for w in seq]])
        y = torch.tensor([word_to_ix[tgt]])

        out = model(x)
        loss = loss_fn(out, y)

        loss.backward()
        optimizer.step()

# 🚀 генерация
def generate(text, length=10):
    words_input = text.split()

    for _ in range(length):
        if len(words_input) < seq_len:
            break

        seq = words_input[-seq_len:]
        x = torch.tensor([[word_to_ix[w] for w in seq]])

        out = model(x)
        probs = torch.softmax(out, dim=1)

        pred = torch.multinomial(probs, 1).item()
        words_input.append(ix_to_word[pred])

    return " ".join(words_input)
