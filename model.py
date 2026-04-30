import torch
import torch.nn as nn
import torch.optim as optim

# 📚 ДАННЫЕ (расширь сколько хочешь)
text = """
привет друг как ты
привет друг как дела
привет как жизнь идет
привет как ты сегодня
как дела друг
как дела сегодня
как ты поживаешь
как жизнь идет хорошо
жизнь идет нормально
жизнь идет отлично
друг как твои дела
друг как ты сегодня
ты как поживаешь
ты как жизнь
что ты делаешь
как у тебя дела
"""

tokens = text.split()
words = list(set(tokens))

word_to_ix = {w: i for i, w in enumerate(words)}
ix_to_word = {i: w for w, i in word_to_ix.items()}

CONTEXT = 2

# 📦 создаём обучающие пары
data = []
for i in range(len(tokens) - CONTEXT):
    ctx = tokens[i:i+CONTEXT]
    tgt = tokens[i+CONTEXT]
    data.append((ctx, tgt))

# 🧠 модель
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(words), 32)
        self.linear = nn.Linear(32 * CONTEXT, len(words))

    def forward(self, x):
        x = self.emb(x).view(1, -1)
        return self.linear(x)

model = Model()

# 🔥 обучение
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

# 🚀 ГЕНЕРАЦИЯ МНОГИХ СЛОВ
def generate(text, length=8):
    words_input = text.split()

    if len(words_input) < CONTEXT:
        return "Напиши минимум 2 слова"

    result = words_input[:]

    for _ in range(length):
        ctx = result[-CONTEXT:]

        # защита от ошибок
        if any(w not in word_to_ix for w in ctx):
            break

        ctx_idx = torch.tensor([word_to_ix[w] for w in ctx])

        out = model(ctx_idx)
        probs = torch.softmax(out, dim=1)

        pred = torch.multinomial(probs, 1).item()
        next_word = ix_to_word[pred]

        result.append(next_word)

    return " ".join(result)
