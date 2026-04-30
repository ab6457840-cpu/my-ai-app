import torch
import torch.nn as nn

text = """
привет друг как ты
привет как дела
как ты сегодня
как жизнь идет
"""

tokens = text.split()
words = list(set(tokens))

word_to_ix = {word: i for i, word in enumerate(words)}
ix_to_word = {i: word for word, i in word_to_ix.items()}

CONTEXT = 2

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(words), 30)
        self.linear = nn.Linear(30 * CONTEXT, len(words))

    def forward(self, x):
        x = self.emb(x).view(1, -1)
        return self.linear(x)

model = Model()

def generate(text):
    words_input = text.split()

    if len(words_input) < CONTEXT:
        return "Слишком мало текста"

    ctx = words_input[-CONTEXT:]
    ctx_idx = torch.tensor([word_to_ix[w] for w in ctx])

    out = model(ctx_idx)
    probs = torch.softmax(out, dim=1)
    pred = torch.multinomial(probs, 1).item()

    return ix_to_word[pred]
