""" converting the google colab notebook to a python script """

import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32
block_size = 8
learning_rate = 1e-3
max_iters = 5000
eval_interval = 300
device = 'cuda' if torch.cuda.is_available() else 'cpu' # no cuda available
eval_iters = 100
n_embd = 32


torch.manual_seed(1337)

# extracting input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# creating mapping for the tokenization of charecters in the dataset
s_to_i = { ch:i for i,ch in enumerate(chars) }
i_to_s = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [s_to_i[c] for c in s]
decode = lambda l: ''.join([i_to_s[i] for i in l])

# train test data split
data = torch.tensor(encode(text), dtype=torch.long)
split_fraction = int(0.9*len(data)) 
train_data = data[:split_fraction]
val_data = data[split_fraction:]


def get_batch(split):
  data = train_data if split == 'train' else val_data # depending on wheter we are working on train data or validation data
  # we are generating 4 different numbers which will be the index of the first element of the 4 batches we are going to hand to the GPU.
  # [len(data) - block_size] beacuse if we take only len(data) we will observe overflow of data and hence an indexing error. 
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i : i + block_size] for i in ix]) # this is the tensor of the 4 batches' features
  y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]) # this is the tensor of the 4 batches' labels
  x, y = x.to(device), y.to(device)
  return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """one head of self attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)

        # calculating affinities
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) #(B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)

        # clip the weights
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    """multiple heads of self attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
        

class FeedForward(nn.Module):
    """"a linear layer followed by a nonlinearity"""
    def __init__(self, expansion):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """transformer block: communication then computation"""
    
    def __init__(self,n_embd, n_head, *args, **kwargs):
        super().__init__(*args, **kwargs)
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # n_embd is the number of embeddings dimensions
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        self.blocks = nn.Sequential(
            Block(n_embd,n_head=4),
            Block(n_embd,n_head=4),
            Block(n_embd,n_head=4),
            nn.LayerNorm(n_embd)
        )
        #self.sa_head = MultiHeadAttention(4, n_embd//4) # self attention head
        #self.ffwd = FeedForward(n_embd) # feed forward head
        self.lm_head = nn.Linear(n_embd, vocab_size) # language modelling head


    def forward(self, idx, targets=None):
        B, T = idx.shape


        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        #x = self.sa_head(x) 
        #x = self.ffwd(x)
        logits = self.lm_head(x) # (B,T,V) V is the vocab size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size elements
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


model = BigramLanguageModel()
m = model.to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))