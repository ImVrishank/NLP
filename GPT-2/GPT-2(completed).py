import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # number of sequences we will run in parallel. Works as B in logits
block_size = 256 # max context length. Works as T in the logits
max_iters = 5000 # number of times we will run across the training dataset
eval_interval = 500 # this is used in the function estimate_loss. every eval_interval, we will evaluate the loss on the train and val sets
learning_rate = 3e-4 # learning rate for the optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu' # use GPU if available
eval_iters = 200 # it prints the loss of the train and val sets every eval_iters number of batches. This drastically reduces noise. It averages out the loss of all the batches passed and then gives it out as the loss of the set.
n_embd = 384 # works as the C in the logits. It is the number of channels in the input data. More channels means more features to learn from.
n_head = 6 # number of heads we will compute parallelly in the MultiHeadAttention. Explain in detail in the blog.
n_layer = 6 # the number of layers we have in the transformer, written when cleaning up the code. 
dropout = 0.2 # the percentage of neuron we will drop in the dropout layer. It is used to prevent overfitting. Explained in detail in the blog, under the Cleaning up the code section's Dropout.
# ------------

torch.manual_seed(1337) # for reporducability when following along.

# loading in the text file (tiny shakespeare)
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text))) # sorted list of all unique characters, we will use this as tokens as this is a character based model
vocab_size = len(chars) # number of unique tokens

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) } # mapping from character to integer
itos = { i:ch for i,ch in enumerate(chars) } # mapping from integer to character
encode = lambda s: [stoi[c] for c in s] # encoder: a function that takes in a string, outputs a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: function that takes in a list of integers, outputs a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long) # encoding the whole text into integers using the encode function
n = int(0.9*len(data)) # split fraction. 90% of the data will be training data and the rest 10% is the validation data
train_data = data[:n] 
val_data = data[n:]

# This function is used to make batches of features and labels. It takes in the split as an argument. This is explained in detail in the blog and the colab file. 
def get_batch(split):
    data = train_data if split == 'train' else val_data # what kinda data are we making batches of? train or val
    ix = torch.randint(len(data) - block_size, (batch_size,)) # we are generating batch_size different numbers which will be the index of the first element of the batch_size number of batches we are going to hand to the GPU.
    x = torch.stack([data[i:i+block_size] for i in ix]) # stacking each of the batches' features over each other, using the indexes that we generated earlier
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # stackinge each of the batches' labels over each other, using the indexes that we generated earlier corresponding to the features.
    x, y = x.to(device), y.to(device) # converting the features and labels to the device (GPU or CPU)
    return x, y

@torch.no_grad() # telling Pytorch that we are not going to be updating the weights in this function. This is used in the function estimate_loss. This betters memory usage and computation time.

# This function is used to estimate the loss of the model on the train and val sets. It is used to evaluate the model's performance on the train and val sets.
# We will only estimate the loss once every every eval_iters batches have passed. This is done to reduce noise in the loss.
# We will average out the loss of all the batches passed and then give it out as the loss of the set.
def estimate_loss():
    out = {}
    model.eval() # setting the model to evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters) 
        for k in range(eval_iters):
            X, Y = get_batch(split) # getting the features and labels of the batch
            logits, loss = model(X, Y) # getting the logits and loss of the model
            losses[k] = loss.item() # storing the loss in the losses tensor
        out[split] = losses.mean() # averaging out the loss of all the batches we have calculated above.
    model.train() # setting the model back to training mode
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size): 
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) # explained in great detail in the blog
        self.query = nn.Linear(n_embd, head_size, bias=False) # explained in great detail in the blog
        self.value = nn.Linear(n_embd, head_size, bias=False) # explained in great detail in the blog
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # register_buffer ensures it is not affected by optimizer.step()

        self.dropout = nn.Dropout(dropout) # explained in cleaning up of code (dropout) of the blog.

    def forward(self, x):
        # input x of size (batch, time-step, channels) ---> (B, T, C) defined in hyperparameters
        # output of size (batch, time-step, head size) ---> (B, T, head_size)
        B,T,C = x.shape 
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        # self attention: compute the weight matrix. Explained in great detail in the blog.
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei) # more dropuout, explained in the blog
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs) also explained in the blog
        out = wei @ v # (B, T, T) @ (B, T, hs) ---> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ As per documentation, this is basically running multiple heads of SelfAttention parallelly. Then we concatenate the heads at the end.  """
    # Here we are processing 256 tokens per block, and we are running 6 heads in parallel. so each head will process 256/6 = 42 tokens. and all 6 of them will be concatenated in the end.
    def __init__(self, num_heads, head_size):
        super().__init__()
        # defining a list of all layers of the neural network. We use this instead of a python list because pytorch can analize this in real time unlike in a python list.
        # This list now contains num_heads number of items. where for each cell we call the Head class and pass it head_size for its constructor. 
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # residual connections. Explained in the blog under the Residual connections section.
        self.proj = nn.Linear(head_size * num_heads, n_embd) # concatenating all the heads back together.
        self.dropout = nn.Dropout(dropout) # dropout again, explained in the blog.

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concatenating.
        out = self.dropout(self.proj(out)) # projecting it then passing it through a dropout layer.
        return out 

class FeedFoward(nn.Module):
    """ Explained in great detain the blog. I have explained why we try to make the model non linear and how we go about doing it here. """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ This is basically an amalgamation of all The entire transformer architecture. Here we are basically connecting all the dots and connections for all the other parts of the architecture that we have worked on till now.  """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # residual connections by using x = x + (deviated value)
        """LayerNorm by altering deviated value by (self.sa(self.ln1(x))) before the Self attention and altering deviated value by (self.ffwd(ln2(x))) before the FeedForward sections. 
        Not similar to the architecture diagram in blog as we are implemeting LayerNorm before FeedForward as in case of ln1 and before Self Attention in case of ln2. """
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    """ This is the final model that we will be using to train and generate text. used to train and generate text. used to train and generate text. """
    # all the details can be found in the blog.
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from the token embedding table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # interspacing communication multiple times.
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) # language model head

        # for reporducibility
        self.apply(self._init_weights)
    
    # this block doesnt do anything important. It is so that building on this code is easier for later on
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

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
            # crop idx to the last block_size tokens
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

model = GPTLanguageModel()
m = model.to(device) # moving the model to the device (GPU or CPU)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# We are using AdamW for this model
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# this is used to reduce noise. It averages out the loss of all the batches passed and then gives it out as the loss of the set.
# We will only estimate the loss once every every eval_iters batches have passed. This is done to reduce noise in the loss.
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
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

# uncomment the following line to save the generated text to a file of 10000 tokens. 
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))