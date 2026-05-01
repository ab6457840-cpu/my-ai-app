import torch
import torch.nn as nn
import torch.optim as optim

# 📂 читаем диалоги
pairs = []

with open("data.txt", "r", encoding="utf-8") as f:
    for line in f:
        if "|" in line:
            inp, out = line.strip().split("|")
            pairs.append((inp.lower().split(), out.lower().split()))

# словарь
words = sorted(list(set(w for p in pairs for s in p for w in s)))

word_to_ix = {w: i for i, w in enumerate(words)}
ix_to_word = {i: w for w, i in word_to_ix.items()}

VOCAB = len(words)

SEQ_LEN = 6

def encode(seq):
    return [word_to_ix[w] for w in seq if w in word_to_ix]

# 🧠 модель
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, 128)
        self.lstm = nn.LSTM(128, 256, batch_first=True)
        self.fc = nn.Linear(256, VOCAB)

    def forward(self, x):
        x = self.emb(x)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

model = Model()

opt = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# 🔥 обучение
for epoch in range(5):
    total = 0
    for inp, out in pairs:

        x = torch.tensor([encode(inp[:SEQ_LEN])])
        y = torch.tensor([word_to_ix[out[0]]])  # берём первое слово ответа

        outp = model(x)
        loss = loss_fn(outp, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total += loss.item()

    print("epoch", epoch, total)

# 🚀 генерация ответа
def generate(text, length=10):
    words_input = text.lower().split()

    if len(words_input) == 0:
        return "напиши что-нибудь"

    x = torch.tensor([encode(words_input[:SEQ_LEN])])

    result = []

    for _ in range(length):
        out = model(x)
        probs = torch.softmax(out / 1.2, dim=1)

        pred = torch.multinomial(probs, 1).item()
        word = ix_to_word[pred]

        result.append(word)

        x = torch.tensor([[pred]])

    return " ".join(result)
