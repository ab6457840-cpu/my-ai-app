import torch
import torch.nn as nn
import torch.optim as optim

# 🔥 нормальный текст (обучение)
text = """
привет друг как ты
привет как дела
как ты сегодня
как жизнь идет
друг как жизнь
"""

tokens = text.split()
words = list(set(tokens))

word_to_ix = {w: i for i, w in enumerate(words)}
ix_to_word = {i: w for w, i in word_to_ix.items()}

CONTEXT = 2

data = []
for i in range(len(tokens) - CONTEXT):
    ctx = tokens[i:i+CONTEXT]
    tgt = tokens[i+CONTEXT]
    data.append((ctx, tgt))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(words), 32)
        self.linear = nn.Linear(32 * CONTEXT, len(words))

    def forward(self, x):
        x = self.emb(x).view(1, -1)
        return self.linear(x)

model = Model()

# 🔥 ОБУЧЕНИЕ (ВАЖНО!)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

for _ in range(300):
    for ctx, tgt in data:
        model.zero_grad()

        ctx_idx = torch.tensor([word_to_ix[w] for w in ctx])
        tgt_idx = torch.tensor([word_to_ix[tgt]])

        out = model(ctx_idx)
        loss = loss_fn(out, tgt_idx)

        loss.backward()
        optimizer.step()

# 🔥 ГЕНЕРАЦИЯ (ИСПРАВЛЕНО)
def generate(text):
    words_input = text.split()

    if len(words_input) < CONTEXT:
        return "Напиши хотя бы 2 слова"

    ctx = words_input[-CONTEXT:]

    # 🔥 заменяем неизвестные слова на случайные известные
    ctx = [w if w in word_to_ix else list(word_to_ix.keys())[0] for w in ctx]

    ctx_idx = torch.tensor([word_to_ix[w] for w in ctx])

    out = model(ctx_idx)
    probs = torch.softmax(out, dim=1)

    pred = torch.multinomial(probs, 1).item()

    return ix_to_word[pred]
