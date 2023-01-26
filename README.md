# Zero to GPT

This course will get you from no knowledge of deep learning to training a GPT model.  We'll start with the basics, then build up to complex networks.

To use this course, go through each chapter from the beginning.  Read the lessons, or watch the optional videos.  Then look through the implementations to solidify your understanding.  I also recommend implementing each algorithm on your own.

# Course Outline

## 0. Introduction

Get an overview of the course and what we'll learn.  Includes some math and NumPy fundamentals you'll need for deep learning.

- Lesson: Read the [intro](explanations/intro.ipynb)

## 1. Gradient Descent

Gradient descent is how neural networks train their parameters to match the data.  It's the "learning" part of deep learning.

- Lesson: Read the [gradient descent tutorial](explanations/linreg.ipynb) and watch the optional [video](https://youtu.be/-cs5D91eBLE)
- Implementation: [Notebook](notebooks/linreg/linreg.ipynb) and [class](nnets/dense.py)

## 2. Dense networks

Dense networks are the basic form of a neural network, where every input is connected to an output.  These can also be called fully connected networks.

- Lesson: Read the [dense network tutorial](explanations/dense.ipynb)
- Implementation: [Notebook](notebooks/dense/dense.ipynb) and [class](nnets/dense.py)

## 3. Classification with neural networks

In the last two lessons, we learned how to perform regression with neural networks.  Now, we'll learn how to perform classification.

- Lesson: Read the [classification tutorial](explanations/classification.ipynb)

## 4. Recurrent networks

Recurrent neural networks can process sequences of data.  They're used for time series and natural language processing.

- Lesson: Read the [recurrent network tutorial](explanations/rnn.ipynb)
- Implementation: [Notebook](notebooks/rnn/rnn.ipynb)

## 5. Regularization

Regularization prevents overfitting to the training set.  This means that the network can generalize well to new data.

- Lesson: Read the regularization tutorial (coming soon)

## 6. PyTorch

PyTorch is a framework for deep learning that automates the backward pass of neural networks.  This makes it simpler to implement complex networks.

- Lesson: Read the PyTorch tutorial (coming soon)

## 7. Gated recurrent networks

Gated recurrent networks help RNNs process long sequences by helping networks forget irrelevant information.  LSTM and GRU are two popular types of gated networks.

- Lesson: Read the GRU tutorial (coming soon)
- Implementation: [Notebook](notebooks/gru/gru.ipynb)

## 8.  Encoders and Decoders

Encoder/decoders are used for NLP tasks when the output isn't the same length as the input.  For example, if you want to use questions/answers as training data, the answers may be a different length than the question.

- Lesson: Read the encoder/decoder tutorial (coming soon)
- Implementation: [Notebook](notebooks/rnnencoder/encoder.ipynb)

## 9. Transformers

Transformers fix the problem of vanishing/exploding gradients in RNNs by using attention.  Attention allows the network to process the whole sequence at once, instead of iteratively.

- Lesson: Read the transformer tutorial (coming soon)
- Implementation: [Notebook](notebooks/transformer/transformer.ipynb)

## 10. Efficient Transformers

GPT models take a long time to train.  We can reduce that time by using more GPUs, but we don't all have access to GPU clusters.  To reduce training time, we'll incorporate some recent advances to make the transformer model more efficient.

- Lesson: Read the efficient transformer tutorial (coming soon)
- Implementation: Notebook coming soon

## More Chapters Coming Soon

# Optional Chapters

## Convolutional networks

Convolutional neural networks are used for working with images and time series.

- Lesson: Read the convolutional network tutorial (coming soon)
- Implementation: [Notebook](notebooks/cnn/cnn.ipynb) and [class](nnets/conv.py)

# Installation

If you want to run these notebooks locally, you'll need to install some Python packages.

- Make sure you have Python 3.8 or higher installed.
- Clone this repository.
- Run `pip install -r requirements.txt`

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.