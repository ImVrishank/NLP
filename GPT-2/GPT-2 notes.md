- Get the dataset you want to train your model on.
  _ `!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
  _ read the data:
  _ `with open('input.txt', 'r', encoding='utf-8') as f:
  _ `text = f.read()`
  * Now *text* is a long string containing all the data from the dataset.
   * Take a good look at your dataset and also observe how you could tokenize your data.
   * Define a few necessary hyperparameters like *vocab_size*, which contain the number of chars in the sequence *text* and *chars* containing all the unique characters, this is very important as we are building a character based model, that is, each character is a token.
   * `chars = sorted(list(set()))`
   \* `vocab_size = len(chars)`
- Tokenize your dataset. Here, since we are working on a character based model, each unique character is given a unique integer and mapped to it. _char_to_int_ and _int_to_char_ will be our two maps. The first to encode and the second to decode.
  - Dictionary to go from char to int: `stio = {ch:i for i, ch in ennumerate(chars)}`
  - Dictionary to go from int to char: `itos = { i:ch for i,ch in enumerate(chars)}`
  - function to encode: `encode = lambda s: [stoi[c] for c in s]`
  - function to decode: `decode = lambda l: ''.join([itos[i] for i in l])`
- Now that we have the functions to encode and decode our text, we can go ahead and encode the whole text.
  - You need to encode it and store the data in a torch tensor. So we need to import torch
    - `import torch`
  - Now you could encode it and store it in _data_:
    - `data = torch.tensor(encode(text), dtype = torch.long)`
- Divide your dataset into training and validation sets. All further work will include working on only the training dataset. In this model we choose to use a 90:10 split for train and validation.
  - `split_fraction = 0.9`
  - `n = int(split_fraction * len(data))`
  - `train_data = data[:n]`
  - `val_data = data[n:]`
- Set your hyperparameters:
  - block-size - size of each chunk of data you're sending to the compute at a time.
    - `block_size = 8`
  - batch-size - number of chunks of data you will process parallelly
    - `batch_size = 4`
- Now that you have your hyperparameters, let us work on making features and labels out the data we have.

  - **Features**
    - Pick _batch-size_ number of random indices on the data and using those indices get _block-size_ number of continuous values and stack them on top of each other. This gives you your _features_ vector.
    - Since we are stacking them on top of each other, we end up with a tensor of dimensions **(_batch-size_ , _block-size_)** for our _features_ vector.
  - **Labels**
    - In the model we are working on, we take all the previous characters in the batch and these are the features/context. As for labels, we take the character right after the features/context. This is how we get out _labels_ vector. As for the labels vector, we again stack them up as we did for the _features_ vector.
    - As explained we end up with a tensor of dimensions **(_batch-size_, _block-size_)** for our labels vector.
  - Code of the above:
    - We make a function which takes in the type of split you want to put into batches and blocks. _split_ is a string that takes in whether the split you want to work on is the train split or the test split.
      - `def get_batch(split):`
    - We then get the data we are going to splitting into batches and blocks depending on _split_.
      - `data = train_data if split == "train" else val_data`
    - We can now work on constructing the features array.
      - `x = torch.stack([data[i:i+block_size] for i in ix])`
    - Similarly the labels array.
      - `y = torch.stack([data[i+1:i+block_size+1] for i in ix])`
    - Then return the values we have calculated.
      - `return x, y`
  - We can now get batches and blocks from either the _train_data_ or the _val_data_. Here i am going to take a batch from the _train_data_:
    - `xb, yb = get_batch('train')`
    - _xb_ and _yb_ now look somewhat like this: both are of dimensions (_batch_size_ = 4, _block_size_ = 8)
      - _xb_ = ` tensor([[24, 43, 58, 5, 57, 1, 46, 43], 
          [44, 53, 56, 1, 58, 46, 39, 58],
          [52, 58, 1, 58, 46, 39, 58, 1], 
          [25, 17, 27, 10, 0, 21, 1, 54]])`
      - _yb_ = ` tensor([[43, 58, 5, 57, 1, 46, 43, 39], 
		  [53, 56, 1, 58, 46, 39, 58, 1], 
		  [58, 1, 58, 46, 39, 58, 1, 46],
		   [17, 27, 10, 0, 21, 1, 54, 39]])`
    - We can observe that, when _{24}_ is the context(at _xb_), we have target(at _yb_) as _{43}_, which is the next element in _xb_'s first batch. When we have _{24, 43}_ as the context, we have _{58}_ as the target, and so on. We can also observe that when we have _{24, 43, 58, 5, 57, 1, 46, 43}_ as the context, we have _{39}_ as the target which comes right after the sequence of context in the _train_data_ but is not quiet visible to us in _xb_.
    -

- Visualizing all that we have done till now:

  - `train_data[:block_size + 1]`
  - we have (_block_size_ + 1) because we want to see the target when we have all eight values as context.
  - Output: `tensor([18, 47, 56, 57, 58,  1, 15, 47, 58])`
  - In this block,
    - `when input is tensor([18]) the target: 47`
    - `when input is tensor([18, 47]) the target: 56`
    - `when input is tensor([18, 47, 56]) the target: 57`
    - `when input is tensor([18, 47, 56, 57]) the target: 58`
    - `when input is tensor([18, 47, 56, 57, 58]) the target: 1`
    - `when input is tensor([18, 47, 56, 57, 58, 1]) the target: 15`
    - `when input is tensor([18, 47, 56, 57, 58, 1, 15]) the target: 47`
    - `when input is tensor([18, 47, 56, 57, 58, 1, 15, 47]) the target: 58`
  - This is the same for each block.
  - We implemented the same and made out features and labels vectors.

- We can now get started with building the model.

  - We now need a tensor of size **(_vocab_size_, _vocab_size_)**. This is going to be called the _token_embedding_table_. We initialize it in the constructor of the _BigramLanguageModel_ subclass, which is an extension of the nn.Module class.
  - `class BigramLanguageModel(nn.Module)`
    _ We also need the following libraries:
    _ `import torch`
    _ `import torch.nn as nn`
    _ `from torch.nn import functional as F`

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
      - embedding_dim --> c value. This defines how we want to store the data of each token. When we say data of each token, we refer to the connections of this token with the other tokens in the same batch. We can choose a high number for _C_. This would increase the clarity of the connections between the different tokens, but at the same time, it would increase computation time. It we used a small number for C, this would mean the computation time would be lesser, but, the clarity in the connections between tokens would be smaller.
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
        - Then we just concatenate the _idx_new_ to _idx_ `idx = torch.cat((idx, idx_new), dim = 1)` Then we end the loop.
      - Outside the loop, we return the _idx_ value that we have concatenated to. During input, _idx_ was **(B x T)**, now during the termination of the function _generate()_, it is of size **(B x T + max_new_tokens)**
        - `return idx`

- To make use of the code we have written above, you make an instance/object of this class `m = BigramLanguageModel(vocal_size)`
- Get the logits and loss `logits, loss = m(xb, yb)`
- Now we are ready to get out first output:
  - We are basically going to create a _(B x T)_ array of _(1 x 1)_ which has a zero. We then send it to the _generate()_ function and generate 100 new tokens.
    - `idx = torch.zeros((1, 1)`
    - `print(decode(m.generate(idx, dtype=torch.long), max_new_tokens=100)[0].tolist()))`
  - Output looks somewhat weird now:
    - `Sr?qP-QWktXoL&jLDJgOLVz'RIoDqHdhsV&vLLxatjscMpwLERSPyao.qfzs$Ys$zF-w,;eEkzxjgCKFChs!iWW.ObzDnxA Ms$3`
  - Lets look at why this is happening. Our model now is a typical **Bigram** model. This type of model only looks at the previous token (which is a char here) as context. Using this context, it predicts the next char. What we actually need to be doing is looking at all of the char before the prediction char and then predict a char. Lets work on that.
- Lets first choose an optimizer. Our best bet for this model is **AdamW**. This optimizer is not too simple, its not too complex and gets the job done well.
  - `optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)`
- Also before we make use of this optimizer, let us get the _batch_size_ up to a more reasonable value. We can be doing way more calculations in parallel.
  - `batch_size = 32`
- Let us also define the number of epochs we want to run.
  - An epoch is basically the number of times we run through every example of the whole dataset.
  - `epochs = 1000`
- Now we can explain what we do in every iteration of every epoch
- `for steps in range(epochs):`
  - sample a batch of data using the _get_batch(split)_ function we made earlier.
    - `xb, yb = get_batch('train')`
  - Generate the loss
    - `logits, loss = m(xb, yb)`
  - Set the gradients from previous iteration to zero
    - `optimizer.zero_grad(set_to_none=True)`
  - Get the gradients for all the parameters
    - `loss.backward()`
  - Use these gradients to update parameters
    - `optimizer.step()`
  - This is the end of the for loop
- Now we can try to get the output again:
  - `  print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=300)[0].tolist()))`
  - Output looks like:
    - `Iyoteng h hasbe pave pirance Rie hicomyonthar's Plinseard ith henoure wounonthioneir thondy, y heltieiengerofo'dsssit ey KIN d pe wither vouprrouthercc. hathe; d! My hind tt hinig t ouchos tes; st yo hind wotte grotonear 'so it t jod weancotha: h hay.JUCle n prids, r loncave w hollular s O: HIs; ht`
- This looks significantly better then the previous output.
- MATHEMATICAL TRICK IN SELF ATTENTION

  - We want to work on a mathematical trick that helps us make a lower triangular matrix where each row elements adds up to 1.
    - For example:
      - from input tensor:
      - ````tensor(
         [1,2,5,7],
         [1,1,1,9],
         [1,1,1,1],
         [0,1,4,5]
        ) ```
        ````
      - to output tensor:
      - ````tensor(
        [1,   0,   0,   0], --> adds up to 1
        [0.3, 0.7, 0,   0],-->  adds up to 1
        [0.1, 0.1, 0.8, 0], --> adds up to 1
        [0.7, 0.1, 0.1, 0.1] ) --> adds up to 1```
        ````
  - We do this for a very specific reason. In the **(B, T, C)** array we had. You could recall that we had a _batch_, _time_ matrix and the each value corresponding to the cell in this matrix stored values about that corresponding token's relations with the other tokens in the same column (that is, the same batch). We have details of the token's relations with the previous tokens and the next tokens in the same batch. In this model, we only want it to store relations of that token with the previous tokens and not the latter tokens. This is because the latter tokens are to come in the future and they haven't been predicted yet. So, in simple terms, the numbers per column are just the connections to the previous tokens, and it is in the form of a probability, meaning, we want it to add up to one.
  - METHOD 1:
  - METHOD 2:
  - METHOD 3:

  - In simple terms, each token in the **(B, T)** array emits 3 values, a **Query**, **Key** and **Value**. **Query** is what the token is looking for and the **Key** is the the array telling us what the token is. We now take each **Query** and then dot product it with each **Key** of that batch. This gives us the **Weights** matrix which we call _wei_. Now lets say the **Query** and the **Key** align very well. As in the connection between those two tokens is very high, then they tend to interact drastically. This gives us a huge value in the _wei_ matrix. **Value** is the array that each token emits which tells us in detail what each token contains. To sum it all up, in perspective of the token:
    - **Query** --> "what am i looking for?"
    - **Key** --> "What does this token mean?"
    - **Value** --> "What information does this token convey?"
      - Imagine you going to a book store. What kind of book you're looking for is the **Query**, The book's title or the front page is the **Key** and the contents of the book is the **Value**.
      - So **Value** of the token is only comes into play when the interaction between the **Key** and the **Query** is good.
    - **Weight** --> "How well does this token interact with the other tokens?"
  - In this code block we are initializing 3 simple neural networks called key, query and the value.
    `head_size = 16`
    `key = nn.Linear(C, head_size, bias=False)`
    `query = nn.Linear(C, head_size, bias=False)`
    `value = nn.Linear(C, head_size, bias=False)`

  - Let us talk about the `nn.Linear` neural network.
    - This is a neural network that takes in **Number of input features**, **Number of output features** and **Bias**(true or false) to initialize.
    - This basically performs the operation: **Y = X . W$^T$+ b**
    - Lets say we initialize the layer with the number of input features as 3, the number of output features as 1 and the bias as True.
      - `torch.nn.Linear(3,1,True)`
    - Now we are set to give this layer input data and the layer will perform the operation as mentioned above and give out a singular output value for each training example.
    - In this example, let us take the number of training examples as 2. So we get an **X** matrix like below:
      - $$X = \begin{vmatrix}1 & 2 & 3 \\ 2 & 4 & 1 \end{vmatrix} $$
    - Now our weight matrix **W** looks somewhat like this (It is randomly initialized by pytorch):
      - $$W =  \begin{vmatrix}-0.020 & 0.5852 & -0.2880 \end{vmatrix} $$
    - And our Bias will also be randomly initialized singular value on a 1 x 1 matrix.
      - $$ Bias = \begin{vmatrix}0.5594 \end{vmatrix} $$
    - Here is how our calculation takes place:
      - $$ Y = X.W^T + Bias$$
      - Since, X.W$^T$ dimensions are (2 x 1), bias also is made into a (2 x 1) matrix.
      - $$Y =  \begin{vmatrix}1 & 2 & 3 \\ 2 & 4 & 1 \end{vmatrix} .\begin{vmatrix}-0.020 \\ 0.5852 \\ -0.2880 \end{vmatrix} + \begin{vmatrix} 0.5594 \\ 0.5594 \end{vmatrix}$$
      - $$ Y = \begin{vmatrix}0.2414 \\1.4072 \end{vmatrix}$$
      - This is the output.
    - Each training example gives in N number of features and gets M number of outputs. This operation could be done with or without bias.
  - So now coming back to the **Key**, **Query** and **Value** neural networks. **Key**, **Query** and **Value**are 3 neural networks that takes in _C_ number of features and converts it into _head_size_ number of values(that is 32 to 16).
  - Now we pass in the _x_ to the _key_ and the _query_ neural networks.
    - `k = key(x)`
    - `q = query(x)`
  - So now, _k_ is a **(B, T, 16)** array, where each token of the **(B, T)** matrix contains a 16 cell array as its key. Similarly, _q_ is a **(B, T, 16)** array, where each token of the (B, T) matrix contains a 16 cell array as its query.
  - Now we can dot product each of the token's keys and queries to get the weights, that is the _wei_ array.
    - `wei =  q @ k.transpose(-2, -1)`
  - _wei_ array is a **(B, T, T)** array.
  - Now we use the **MATHEMATICAL TRICK IN SELF ATTENTION.** We know that the 3rd method is the best method to do this. Hence we choose it.
    - `tril = torch.tril(torch.ones(T, T))`
    - `wei = wei.masked_fill(tril == 0, float('-inf'))`
    - `wei = F.softmax(wei, dim=-1)`
  - This gives us _wei_.
  - Now we take the dot product of the _wei_ and **Value**:
    - `v = value(x)`
    - `out = wei @ v`
  -
