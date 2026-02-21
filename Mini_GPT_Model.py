import torch 
import torch.nn as nn
import torch.nn.functional as f
import random

from Transformers_Block import Block

corpus = ["Hello my name is champ",
"The quick brown fox jumps over the lazy dog.",
"Artificial intelligence is transforming the way we solve complex problems.",
"Graph neural networks are powerful tools for modeling relational data.",
"Diffusion models generate images by progressively removing noise.",
"Effective machine learning systems require both strong theory and practical experimentation."
]

corpus = [s + " <END> " for s in corpus]
text = " ".join(corpus)
print(text)

words = list(set(text.split()))
print(words)

vocab_size = len(words)
print(vocab_size)

word2idx = {w: i for i, w in enumerate(words)}
print('word2idx : ',word2idx)

idx2words = {i: w for w, i in word2idx.items()}
print('idx2words : ', idx2words)

data = torch.tensor([word2idx[w] for w in text.split()], dtype = torch.long)
print('data : ', data)
print(len(data))

block_size = 6
embedding_dim = 32
n_heads = 2
n_layers = 2
lr = 1e-3
epochs = 1500

def get_batch(batch_size = 16):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i:i+block_size] for i in ix])
    return x, y


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.Sequential(*[Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets = None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)

        pos_emb = self.position_embedding(torch.arange(T, device = idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = f.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = f.softmax(logits, dim =-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_idx), dim = 1)
        return idx
    

model = TinyGPT()
optimizer = torch.optim.AdamW(model.parameters(), lr = lr)

for step in range(epochs):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.step()
    if step % 300==0:
        print(f'Step {step}, loss = {loss.item():.4f}')



context = torch.tensor([[word2idx["Hello"]]], dtype = torch.long)
out = model.generate(context, max_new_tokens = 15)


print('\nGenerated text: \n')
print(' '.join(idx2words[int(i)] for i in out[0]))
    

        