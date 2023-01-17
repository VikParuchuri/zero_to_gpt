# Zero to GPT

This course will get you from no knowledge of deep learning to training a GPT model.  We'll start with the basics, and build up to more complex networks.

- The `explanations` folder has lessons, starting with the basics.  There are video walkthroughs of some of the explanations (linked below).
- The `notebooks` folder has full implementations.

# Installation

- Clone this repository.
- Run `pip install -r requirements.txt` (you should have python 3.8+ installed).

# Lessons

## 1. Gradient Descent

Gradient descent is an important building block for neural networks.  It's how networks train their parameters to fit the data.

- [Gradient descent tutorial](explanations/linreg.ipynb)
- [Video explanation](https://youtu.be/-cs5D91eBLE)
- [Notebook implementation](notebooks/linreg/linreg.ipynb)
- [Clean implementation](nnets/dense.py)

## 2. Dense networks

Dense networks are networks where every input is connected to an output.  They're the most general form of a neural network.  These can also be called fully connected networks.

- Dense network tutorial coming soon
- [Notebook implementation](notebooks/dense/dense.ipynb)
- [Clean implementation](nnets/dense.py)

## 3. Convolutional networks

Convolutional neural networks are used for working with images and time series.

- Convolutional network tutorial coming soon
- [Notebook implementation](notebooks/cnn/cnn.ipynb)
- [Clean implementation](nnets/conv.py)

## 4. Recurrent networks

Recurrent neural networks can process sequences of data.  They are used for time series and natural language processing.

- Recurrent network tutorial coming soon
- [Notebook implementation](notebooks/rnn/rnn.ipynb)
- Clean implementation coming soon

## 5. Gated recurrent networks

Gated recurrent networks help RNNs process long sequences by helping networks forget irrelevant information.  LSTM and GRU are two popular types of gated networks.

- GRU tutorial coming soon
- [Notebook implementation](notebooks/gru/gru.ipynb)
- Clean implementation coming soon

## 6.  Encoder/Decoder RNNs

Encoder/decoders are used for NLP tasks when the output isn't the same length as the input.  For example, if you want to use questions/answers as training data, the answers may be a different length than the question.

- Tutorial coming soon
- [Notebook implementation](notebooks/rnnencoder/encoder.ipynb)
- Clean implementation coming soon

## 7. Transformers

Transformers fix the problem vanishing/exploding gradients in RNNs by using attention.  Attention allows the network to focus on the most relevant parts of the input.

- Tutorial coming soon
- Notebook implementation coming soon
- Clean implementation coming soon