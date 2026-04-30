import torch
import torch.nn as nn
import torch.optim as optim

# 📚 расширенные данные (ВАЖНО: больше разнообразия)
text = """
Утренняя заря медленно окрашивала небо в нежно-розовые тона, когда я вошел в старый сосновый бор. Воздух здесь был особенный — густой, смолистый, напоенный ароматом хвои и лесных трав. Казалось, что время остановилось, а городская суета, оставшаяся далеко позади, потеряла всякое значение. Высокие сосны, словно величавые стражи, уходили вершинами в небо, а между ними пробивались золотистые лучи солнца.Я шел по мягкой тропинке, усыпанной прошлогодними иголками, и слушал, как просыпается лес. Сначала послышалось робкое пение маленькой птички, затем к нему присоединился неугомонный стук дятла, где-то высоко на суку искавшего пропитание. Вдали, за густыми зарослями папоротника, послышался шорох — это ежик спешил по своим делам, не обращая внимания на редкого гостя.Каждый раз, бывая в лесу, я удивляюсь, насколько сложна и удивительна природа. Под ногами, в тени раскидистого дуба, притаились маленькие ежики, их черные глазки-бусинки внимательно следят за происходящим. А на поляне, освещенной солнцем, расцвели первые весенние цветы, радуя глаз своей хрупкой красотой.Важно научиться ценить эти моменты тишины и уединения. В современной жизни, полной стрессов и постоянной спешки, человек часто забывает о том, что он — часть этого огромного, дышащего мира. Лес помогает восстановить душевное равновесие, дарит энергию и вдохновение.Когда я наконец выбрался из чащи на опушку, солнце уже стояло высоко. Я остановился, глубоко вдохнул свежий воздух и почувствовал, как усталость уходит, а на смену ей приходят спокойствие и уверенность. Природа учит нас мудрости, терпению и красоте — нужно лишь уметь слушать и видеть.
"""

tokens = text.split()
words = list(set(tokens))

word_to_ix = {w: i for i, w in enumerate(words)}
ix_to_word = {i: w for w, i in word_to_ix.items()}

# 🔥 увеличили контекст
CONTEXT = 4

# 📦 обучающие пары
data = []
for i in range(len(tokens) - CONTEXT):
    ctx = tokens[i:i+CONTEXT]
    tgt = tokens[i+CONTEXT]
    data.append((ctx, tgt))

# 🧠 модель
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(words), 64)
        self.linear1 = nn.Linear(64 * CONTEXT, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, len(words))

    def forward(self, x):
        x = self.emb(x).view(1, -1)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

model = Model()

# 🔥 обучение
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

for _ in range(400):
    for ctx, tgt in data:
        model.zero_grad()

        ctx_idx = torch.tensor([word_to_ix[w] for w in ctx])
        tgt_idx = torch.tensor([word_to_ix[tgt]])

        out = model(ctx_idx)
        loss = loss_fn(out, tgt_idx)

        loss.backward()
        optimizer.step()

# 🚀 генерация (УБРАЛИ ПОВТОРЫ)
def generate(text, length=10):
    words_input = text.split()

    if len(words_input) < CONTEXT:
        return "Напиши больше слов (минимум 4)"

    result = words_input[:]

    for _ in range(length):
        ctx = result[-CONTEXT:]

        # защита от неизвестных слов
        if any(w not in word_to_ix for w in ctx):
            break

        ctx_idx = torch.tensor([word_to_ix[w] for w in ctx])

        out = model(ctx_idx)
        probs = torch.softmax(out, dim=1).squeeze()

        # 🔥 сглаживание (меньше повторов)
        probs = probs + 1e-8
        probs = probs / probs.sum()

        # 🔥 выбор слова
        pred = torch.multinomial(probs, 1).item()

        next_word = ix_to_word[pred]

        # 🔥 защита от зацикливания
        if next_word in result[-3:]:
            continue

        result.append(next_word)

    return " ".join(result)
