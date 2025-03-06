- Get the dataset you want to train your model on.
  _ `!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
  _ read the data:
  _ `with open('input.txt', 'r', encoding='utf-8') as f:
  _ `text = f.read()`

  - Now _text_ is a long string containing all the data from the dataset.
     * Take a good look at your dataset and also observe how you could tokenize your data.
     * Define a few necessary hyperparameters like _vocab_size_, which contain the number of chars in the sequence _text_ and _chars_ containing all the unique characters, this is very important as we are building a character based model, that is, each character is a token.

  * `chars = sorted(list(set()))`

  * `vocab_size = len(chars)`

- Tokenize your dataset. Here, since we are working on a character based model, each unique character is given a unique integer and mapped to it. _char_to_int_ and _int_to_char_ will be our two maps. The first to encode and the second to decode.
  - Dictionary to go from char to int: `stio = {ch:i for i, ch in ennumerate(chars)}`
  - Dictionary to go from int to char: `itos = { i:ch for i,ch in enumerate(chars)}`
  - function to encode: `encode = lambda s: [stoi[c] for c in s]`
  - function to decode: `decode = lambda l: ''.join([itos[i] for i in l])`
- We also define our first hyperparameter here: _vocab_size_. This hyperparameter basically specifies the number of unique characters we have in our dataset.
- Divide your dataset into training and validation sets. All further work will include working on only the training dataset.
- Set your hyperparameters:
  - block-size - size of each chunk of data you're sending to the compute at a time.
  - batch-size - number of chunks of data you will process parallelly
- Now that you have your hyperparameters, let us work on making features and labels out the data we have.
  - **Features**
    - Pick _batch-size_ number of random indices on the data and using those indices get _block-size_ number of continuous values and stack them on top of each other. This gives you your _features_ vector.
    - Since we are stacking them on top of each other, we end up with a tensor of dimensions **(_batch-size_ , _block-size_)** for our _features_ vector.
  - **Labels**
    - In the model we are working on, we take all the previous characters in the batch and these are the features/context. As for labels, we take the character right after the features/context. This is how we get out _labels_ vector. As for the labels vector, we again stack them up as we did for the _features_ vector.
    - As explained we end up with a tensor of dimensions **(_batch-size_, _block-size_)** for our labels vector.
- We can now get started with building the model.

  - We now need a tensor of size **(_vocab_size_, _vocab_size_)**. This is going to be called the _token_embedding_table_. We initialize it in the constructor of the _BigramLanguageModel_ subclass, which is an extension of the nn.Module class.
  - `class BigramLanguageModel(nn.Module)`

    - We also need the following libraries:

      - `import torch`
      - `import torch.nn as nn`
      - `from torch.nn import functional as F`

    - constructor which takes the _vocab_size_ as a parameter
      - `super().__init__()` because it is a daughter class
      - `self.token_embedding_table = nn.Embedding(num_embeddings = vocab_size, embedding_dim = vocab_size)`
        - This defines an embedding table and initializes it with random values.

    ``

  - We also define a special function in Pytorch called _forward()_.
    - `def forward(self, idx, tagets = None)` _idx_ is the features vector we calculated earlier, _targets_ is the labels vector we calculated earlier.
    - This function basically tells the neural network what activation function to apply as we pass data along the layers. It is automatically called whenever we create an object of the same class and initialize the object. So, in simple terms, when we create and initialize an object, we automatically call the constructor and the forward functions.
  - To the _forward()_ function. We pass the _features_ and the _labels_. We then calculate the _logits_. _logits_ is a vector of **(B, T, C)** dimensions. B stands for batch, T stands for time and C stands for channel. I shall explain this working in great detail.
    - We import the torch.nn library as nn
    - We then make an object of the class nn.Embedding. We are going to call this object the _token_embedding_table_. While making this object we need to pass 2 arguments to the constructor of the nn.Embedding class. These arguements are:
      - num_embeddings --> the number of unique token we have in the dataset
      - embedding*dim --> c value. This defines how we want to store the data of each token. When we say data of each token, we refer to the connections of this token with the other tokens in the same batch. We can choose a high number for \_C*. This would increase the clarity of the connections between the different tokens, but at the same time, it would increase computation time. It we used a small number for C, this would mean the computation time would be lesser, but, the clarity in the connections between tokens would be smaller.
    - At the constructor, it initialized random values in all the cells of the _token_embedding_table_.
    - When the _forward()_ function runs, it takes in the _features_ vector, and passes this as the argument to the object of the nn.Embedding class. That is, the _token_embedding_table_. `logits = self.token_embedding_table(x)` takes in the _features_ vector and makes a table of **(B, T, C)** with the (B, T) table we have seen in the form of _features_ vector. It just extends each of these cells of the (B, T) table and makes it store information in _C_ amount of cells right behind it.
    - The main takeaway is that, it takes in the features table and then makes sure that every token in the _features_ table has _C_ number of cells which store it's information with respect to other tokens in the same batch.
    - We now need to evaluate the loss. A good way of measuring the loss is the negative log likelihood. This is done using an inbuilt function called the _cross_entropy_. To use _cross_entropy_:
      - We need to import `from torch.nn import functional as F`
      - We will need to reshape the _logits_ and _targets_ so make sure to preserve the shape `B, T, C = logits.shape`
      - We also need to reshape the _logits_ because _cross_entropy_ expects a 2d array, with the first dimension containing tokens and the second containing channels. That is it wants a **(B x T , C)** dimension _features_ array and a **(B x T)** dimension _labels_ array. So we run the commands `logits = logits.view(B*T, C)` `targets = targets.view(B*T, C)` (view is basically reshape, but when using tensors, view is more reliable).
      - Then we get the loss by `loss = F.cross_entropy(logits, target)`
    - The _forward()_ function returns _loss_ and the _logits_.
    - Generating output: `def generate(self, idx, max_new_tokens): `
      - _idx_ is the _features_ vector, _max_new_tokens_ is the maximum number of tokens you want the model to generate.
      - The _generate_ function takes in _idx_ (A **(B x T)** array and makes that a **(B x T+1)**, then a **(B x T+2)** array and so on till it becomes **(B x max_new_tokens)**.
      - We loop around _max_new_tokens_ times to produce each token at a time. `for _ in range(max_new_tokens)`
        - This just means that, initially, we have _idx_ as the context/features. we guess the next token (a char) then, during the next iteration of the loop, it takes the old context and the next token as the new context/features to make a new guess. this keeps happening until the loop terminates
        - Inside the for loop, we first get the logits and the loss for the initial _idx_ that was submitted to the _generate()_ function as context. `logits, loss = self(idx)`
        - Now we alter the logits. We only take into consideration the channels and the batches of the last target. That is, we only want the part of the context that will be used to give predictions. We only take the last row of the time section while retaining all of the last row's channels and batches. `logits = logits[:, -1, :]`
        - We then use Softmax as a matrix trimming method. Matrix trimming is a wide topic which i shall discuss soon. `prob = F.softmax(logits, dim = -1)`
        - Now that we have the probabilities of the next character's pick from the bucket of tokens. We pick an index of the token from the bucket of tokens keeping in mind the probability of each token occurring with respect to the context provided. `torch` provides a very helpful function to do the same. `idx_next = torch.multinomial(prob, num_samples = 1)`
        - This above code tells the torch library, now that we have given the probability of each token ouccering in the bucket of tokens, pick _num_of_samples_ (an int) to give. The output is going to be an index, which here could just be converted to char by using the _encode_ and _decode_ functions we defined earlier.
        - Then we just concatenate the _idx_new_ to _idx_ `idx = torch.cat((idx, idx_new), dim = 1)`
      - Now we return the _idx_ value that we have concatenated to. During input, _idx_ was **(B x T)**, now during the termination of the function _generate()_, it is of size **(B x T + max_new_tokens)** `return idx`

- To make use of the code we have written above, you make an instance/object of this class `m = BigramLanguageModel(vocal_size)`
- Get the logits and loss `logits, loss = m(xb, yb)`
