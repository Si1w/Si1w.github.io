---
title: Attention
date: 2025-12-09
tag: Neural Networks, Machine Translation
mathjax: true
---
# Papers & Blogs

- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215)

- [NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE](https://arxiv.org/pdf/1409.0473)

- [The Mathematics of Statistical Machine Translation: Parameter Estimation](https://aclanthology.org/J93-2003.pdf)

- [Statistical Phrase-Based Translation](https://aclanthology.org/N03-1017.pdf)

- [Neural Machine Translation](https://arxiv.org/pdf/1709.07809)

- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

- [king - man + woman is queen; but why?](https://p.migdal.pl/blog/2017/01/king-man-woman-queen-why/)

- [Word2Vec Tutorial - The Skip-Gram Model](https://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)

- [Word2Vec Tutorial Part 2 - Negative Sampling](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/)

# Implementation

- [Pytorch-seq2seq](https://github.com/bentrevett/pytorch-seq2seq)

# Machine Translation

## Statistical Machine Translation

The intuition of **Statistical Machine Translation (SMT)** is to treat translation as a probability problem: finding the most probable target sentence $e$ given a source sentence $c$.

For example, given a source sentence in Chinese: `"我喜欢玩游戏" (I like playing games)`.
From a machine's perspective, it doesn't inherently recognize Chinese characters or grammar rules. Instead, it relies on the **Noisy Channel Model**. It assumes that the observed Chinese sentence $c$ was originally a clean English sentence $e$, which passed through a "noisy channel" and got distorted into Chinese. The task of the machine is to decode this signal and recover the most probable original English sentence $e$.

Essentially, translation here is not about "understanding" linguistics, but calculating probabilities to find the best match.

### Noisy Channel Model

Formally, given a source sentence $c$ (Chinese), we need to find the target sentence $e$ (English) that maximizes the conditional probability $P(e \mid c)$.

Based on Bayes' Theorem, we can decompose this as:

$$
P(e \mid c) = \frac{P(c \mid e)P(e)}{P(c)}
$$

Since $P(c)$ is constant for any given source sentence, we can ignore it during the search. Thus, our goal becomes:

$$
	ext{argmax}_{e} P(e \mid c) \propto \text{argmax}_{e} \underbrace{P(c \mid e)}_{\text{Translation Model}} \times \underbrace{P(e)}_{\text{Language Model}}
$$

Where:
- $P(c \mid e)$ is the **Translation Model**. It measures **fidelity**, or how likely the English sentence $e$ can generate the Chinese sentence $c$ (ensuring the meaning is preserved).
- $P(e)$ is the **Language Model**. It measures **fluency**, or how likely $e$ is to be a valid, natural English sentence (ensuring the grammar is correct).

To find the specific sentence $e$ that maximizes this product, we use a search algorithm known as the **Decoder**.

### Alignment

In SMT, **Alignment** is the process of finding the correspondence between words in the source sentence and the target sentence.

Since manual alignment is impossible at scale, we need the machine to learn these correspondences automatically from a large parallel corpus (without dictionaries). **IBM Models 1-5** were proposed to model this alignment process mathematically.

We define an alignment variable $a$, where $a_j = i$ means the $j$-th word in the Chinese sentence aligns to the $i$-th word in the English sentence. Also, $e= (e_0, e_1, \ldots, e_l)$ is the English sentence of length $l$, and $c = (c_1, c_2, \ldots, c_m)$ is the Chinese sentence of length $m$. Here, $e_0$ is a special `NULL` word that allows for words in $c$ to not align to any word in $e$.

#### Expectation-Maximization Algorithm

Before diving into the specific models, it is crucial to understand the **Expectation-Maximization (EM) Algorithm**.
In our case, the alignment $a$ is a **latent variable** (hidden data we cannot observe directly from the text). The EM algorithm helps us estimate parameters when data is missing.

1.  **Initialization**: Initialize the probability parameters randomly (or uniformly).
2.  **E-step (Expectation)**: Estimate the expected alignment counts using the current parameters. (i.e., Probability of `apple` aligning to `苹果`).
3.  **M-step (Maximization)**: Update the model parameters to maximize the likelihood of the data based on the counts from the E-step.
4.  **Convergence**: Repeat steps 2 and 3 until the parameters stabilize.

#### Lexical Translation Model (IBM Model 1)

IBM Model 1 is the simplest model. It assumes that **word order does not matter** and treats sentences as "bags of words." It focuses purely on **Lexical Translation Probability** $t(c \mid e)$.

For example, to determine if `apple` maps to `苹果`, we need to learn the probability $t(\text{苹果} \mid \text{apple})$.

For every word position $j$ in the Chinese sentence:
1.  Select an English word $e_i$ to align with (uniformly).
2.  Translate $e_i$ into Chinese word $c_j$ based on the probability $t(c_j \mid e_i)$.

Mathematically, the probability of a Chinese sentence $c$ and an alignment $a$ given an English sentence $e$ is:

$$
\begin{aligned}
P(c, a \mid e) &\propto \prod_{j=1}^{m} t(c_j \mid e_{a_j}) \\
P(c \mid e) &\propto \sum_{a_{1}=0}^{l} \sum_{a_{2}=0}^{l} \ldots \sum_{a_{m}=0}^{l} P(c, a \mid e) \\
&\propto \sum_{a_{1}=0}^{l} \sum_{a_{2}=0}^{l} \ldots \sum_{a_{m}=0}^{l} \prod_{j=1}^{m} t(c_j \mid e_{a_j})
\end{aligned}
$$

We use the EM algorithm to train this:
1.  Initially assume `apple` aligns to any Chinese word with equal probability.
2.  In the E-step, calculate alignment weights.
3.  In the M-step, if `apple` and `苹果` frequently appear in the same sentence pairs, increase $t(\text{苹果} \mid \text{apple})$.
4.  Repeat until convergence.

**Limitation:** It fails to distinguish structure. `I eat apple` and `apple eat I` have the same probability in Model 1.

#### Absolute Alignment Model (IBM Model 2)

To address the structure issue, IBM Model 2 assumes that **position matters**.

Intuitively, words in similar positions are more likely to align. The first word in English usually aligns to the beginning of the Chinese sentence, not the end. To model this, we introduce an **Alignment Probability** (or Distortion parameter) $q(i \mid j, l, m)$.

This parameter represents the probability that the $j$-th Chinese word aligns to the $i$-th English word, given the sentence lengths $l$ and $m$.

The joint probability now includes this position term:

$$
P(c, a \mid e) \propto \prod_{j=1}^{m} \left( t(c_j \mid e_{a_j}) \times q(a_j \mid j, l, m) \right)
$$

Visually, this encourages alignment points to cluster around the diagonal:

| en \ zh | 我 | 喜欢 | 玩 | 游戏 |
| :---: | :---: | :---: | :---: | :---: |
| **I** | x |   |   |   |
| **like** |   | x |   |   |
| **playing** |   |   | x |   |
| **games** |   |   |   | x |

#### Fertility Model (IBM Model 3)

Models 1 and 2 assume a one-to-one mapping (or many-to-one), but real languages are complex.
- **One-to-Many:** English `not` might become Chinese `不是` (two words).
- **Null Generation:** Some words like `do` (auxiliary) might disappear in Chinese.

IBM Model 3 introduces **Fertility**, denoted as $\phi$. It models **how many Chinese words** are generated by a single English word $e_i$.

The generative story becomes more realistic:
1.  **Fertility:** For each English word $e_i$, choose a fertility count $\phi_i$ with probability $n(\phi_i \mid e_i)$.
2.  **Translation:** Generate $\phi_i$ Chinese words using $t(c \mid e)$.
3.  **Distortion:** Place the generated words in positions according to distortion probabilities. Here, we use $d(j \mid i, l, m)$ to denote the probability of placing a Chinese word at position $j$ given it was generated from English word position $i$.

Mathematically, the formulation becomes:

$$
P(c, a \mid e) \propto \prod_{i=1}^{l} n(\phi_i \mid e_i) \times \prod_{k=1}^{\phi_i} \Bigl( t(c_{j_k} \mid e_i) \times d(j \mid i, l, m) \Bigr)
$$

Here:

- $n(\phi_i \mid e_i)$ is the number of Chinese words generated by English word $e_i$.
- $t(c_{j_k} \mid e_i)$ is the lexical translation probability.
- $d(j \mid i, l, m)$ is the distortion probability for placing words.

Because we now have to sum over all possible fertility combinations, exact calculation is impossible. We must use **sampling techniques** to estimate parameters.

#### Relative Distortion Model (IBM Model 4)

Models 2 and 3 use "Absolute Position," which creates a problem for phrases.
If an English phrase moves (e.g., `I [always] go` vs `I go [always]`), the absolute positions of all words inside it change drastically, leading to a high penalty.

We can solves this by modeling **Relative Distortion**. It assumes that if word $e_i$ follows $e_{i-1}$, then the translation of $e_i$ should likely be placed **near** the translation of $e_{i-1}$ in the Chinese sentence.

Mathematically, instead of predicting the absolute position $j$, we predict the displacement relative to the center of the previous word's translation $\odot_{i-1}$:

$$
d(j - \odot_{i-1})
$$

#### Deficiency Model (IBM Model 5)

Models 3 and 4 have a problem called **deficiency**, which means that $\sum_{c} \sum_{a} P(c, a \mid e) < 1$ or will allocate some probability mass to impossible events.

For example, two English words generating Chinese words that overlap in position.

Therefore, Model 5 modifies the distortion model to ensure that words are placed only in unoccupied positions, fixing the deficiency issue.

$$
d(j - \odot_{i-1} \mid \text{vacancies})
$$

It will track which positions are already filled and only allow placing new words in the remaining vacancies.

### Phrase-based SMT (TBD)

While IBM Models focus on word-level translation, **Phrase-based SMT** translates multi-word phrases as single units. This approach captures local context and idiomatic expressions better than word-by-word translation.

### Decoding/Beam Search (TBD)

The possibilities of target sentences are enormous. To efficiently search for the best translation, we use **Beam Search** to find out the most probable $k$ candidate translations at each step, pruning less likely options.

## Neural Machine Translation

### Vector Representation

In SMT, words are treated as discrete symbols. In **Neural Machine Translation (NMT)**, we represent words as continuous vectors (embeddings) in a high-dimensional space. This allows the model to capture semantic relationships between words.

### Encoder & Decoder

Neural Machine Translation (NMT) uses neural networks to model the entire translation process end-to-end. The core components are:
- **Encoder**: A neural network that processes the source sentence and encodes it into a fixed-length context vector.
- **Decoder**: Another neural network that generates the target sentence word by word, conditioned on the context vector from the encoder.

### Seq2Seq Model

Given a source sentence $(x_1, x_2, \ldots, x_T)$, a standard **Recurrent Neural Network (RNN)** computes a sequence of outputs $(y_1, y_2, \ldots, y_T)$ by iterating the following equations:

$$
\begin{aligned}
h_{t} &= \sigma(W_{hx} x_{t} + W_{hh} h_{t-1}) \\
y_{t} &= W_{yh} h_{t}
\end{aligned}
$$

Where:
- $h_t$ is the hidden state at time $t$.
- $W_{hx}, W_{hh}, W_{yh}$ are weight matrices.
- $\sigma$ is an activation function (e.g., tanh or ReLU).

The RNN can easily map sequences to sequences whenever the alignment between the inputs the outputs is known ahead of time. However, it is not clear how to apply an RNN to problems whose input and the output sequences have different lengths with complicated and non-monotonic relationships.

Long Short-Term Memory (LSTM) is known to learn problems with long range dependencies. The goal of LSTM is to estimate the conditional probability $P(y_1, \ldots, y_{T'} \mid x_1, \ldots, x_T)$ where the input sequence $(x_1, \ldots, x_T)$ and the output sequence $(y_1, \ldots, y_{T'})$ may have different lengths $T$ and $T'$.

The LSTM computes this probability by first obtaining the fixed-dimensional representation $v$ of the input sequence $(x_1, \ldots, x_T)$ by iterating the following equations from $t=1$ to $T$:

$$
p(y1, \ldots, y_{T'} \mid x_1, \ldots, x_T) = \prod_{t=1}^{T'} p(y_t \mid v, y_1, \ldots, y_{t-1})
$$

In Seq2Seq model implementation, it uses two separate LSTMs with four layers: one for the input sequence (encoder) and one for the output sequence (decoder). Also, to improve performance, the model reverses the order of the words in the input sequence. For example, instead of mapping the $a$, $b$, $c$ to $\alpha$, $\beta$, $\gamma$, the LSTM is asked to map $c$, $b$, $a$ to $\alpha$, $\beta$, $\gamma$. In this way, $a$ is in close proximity to $\alpha$, $b$ to $\beta$ and so on, which makes SGD easily to establish communication between the input and output.

$$
(h^1_t, c^1_t) = \text{EncoderLSTM}(e(x_t), (h^1_{t-1}, c^1_{t-1}))
(h^2_t, c^2_t) = \text{EncoderLSTM}(h^1_t, (h^2_{t-1}, c^2_{t-1}))
$$

```python
class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src length, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src length, batch size, embedding dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [src length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # outputs are always from the top hidden layer
        return hidden, cell
```

- `input_dim` is the size/dimensionality of the one-hot vectors that will be input to the encoder. This is equal to the input (source) vocabulary size.

- `embedding_dim` is the dimensionality of the embedding layer. This layer converts the one-hot vectors into dense vectors with `embedding_dim` dimensions.

- `hidden_dim` is the dimensionality of the hidden and cell states.

- `n_layers` is the number of layers in the RNN.

- `dropout` is the amount of dropout to use. This is a regularization parameter to prevent overfitting.

In the `forward` method, we pass in the source sentence, $X$, which is converted into dense vectors using the `embedding` layer, and then dropout is applied. These embeddings are then passed into the RNN. As we pass a whole sequence to the RNN, it will automatically do the recurrent calculation of the hidden states over the whole sequence for us! Notice that we do not pass an initial hidden or cell state to the RNN. 

$$
(s^1_t, c^1_t) = \text{DecoderLSTM}(e(y_{t-1}), (s^1_{t-1}, c^1_{t-1}))
(s^2_t, c^2_t) = \text{DecoderLSTM}(s^1_t, (s^2_{t-1}, c^2_{t-1}))
$$

Then pass the hidden state from the top layer of the RNN, $s^2_t$, through a linear layer, $f$, to make a prediction of what the next token in the target (output) sequence should be, $\hat{y}_{t+1}$.

$$
\hat{y}_{t+1} = f(s^L_t)
$$

```python
class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hidden dim]
        # context = [n layers, batch size, hidden dim]
        input = input.unsqueeze(0)
        # input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, embedding dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output = [seq length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # seq length and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, hidden dim]
        # hidden = [n layers, batch size, hidden dim]
        # cell = [n layers, batch size, hidden dim]
        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch size, output dim]
        return prediction, hidden, cell
```

For the final part of the implemenetation, we'll implement the seq2seq model. This will handle:

1. receiving the input/source sentence
2. using the encoder to produce the context vectors
3. using the decoder to produce the predicted output/target sentence

Our `forward` method takes the source sentence, target sentence and a teacher-forcing ratio. The teacher forcing ratio is used when training our model. When decoding, at each time-step we will predict what the next token in the target sequence will be from the previous tokens decoded, $\hat{y}_{t+1} = f(s^L_t)$.With probability equal to the teaching forcing ratio (`teacher_forcing_ratio`) we will use the actual ground-truth next token in the sequence as the input to the decoder during the next time-step. However, with probability `1 - teacher_forcing_ratio`, we will use the token that the model predicted as the next input to the model, even if it doesn't match the actual next token in the sequence

During each iteration of the loop, we:

1. pass the input, previous hidden and previous cell states $(y_t, s_{t-1}, c_{t-1})$ into the decoder
2. receive a prediction, next hidden state and next cell state $(\hat{y}_{t+1}, s_t, c_t)$ from the decoder
3. place our prediction, $\hat{y}_{t+1}$ (`output`) in our tensor of predictions, $\hat{Y}$ (`outputs`)
4. decide if we are going to "teacher force" or not
    - if we do, the next `input` is the ground-truth next token in the sequence, $y_{t+1}$ (`trg[t]`)
    - if we don't, the next `input` is the predicted next token in the sequence, $\hat{y}_{t+1}$ (`top1`), which we get by doing an `argmax` over the output tensor

Once we've made all of our predictions, we return our tensor full of predictions, $\hat{Y}$ (`outputs`).

```python
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert (
            encoder.hidden_dim == decoder.hidden_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        assert (
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio):
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[1]
        trg_length = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # first input to the decoder is the <sos> tokens
        input = trg[0, :]
        # input = [batch size]
        for t in range(1, trg_length):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            # output = [batch size, output dim]
            # hidden = [n layers, batch size, hidden dim]
            # cell = [n layers, batch size, hidden dim]
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1
            # input = [batch size]
        return outputs
```

### Attention