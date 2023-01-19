# Zero to GPT

This course will get you from no knowledge of deep learning to training a GPT model.  We'll start with the basics, then build up to complex networks.

To use this course, go through each chapter from the beginning.  Read the lessons, or watch the optional videos.  Then, look through the implementations to solidify your understanding.  I also recommend implementing each algorithm on your own afterwards.

# Course Outline

## 1. Gradient Descent

Gradient descent is how neural networks train their parameters to match the data.  It's the "learning" part of deep learning.

- Lesson: Read the [gradient descent tutorial](explanations/linreg.ipynb) and watch the optional [video](https://youtu.be/-cs5D91eBLE)
- Implementation: [Notebook](notebooks/linreg/linreg.ipynb) and [class](nnets/dense.py)

## 2. Dense networks

Dense networks are the basic form of a neural network, where every input is connected to an output.  These can also be called fully connected networks.

- Lesson: Read the [dense network tutorial](explanations/dense.ipynb)
- Implementation: [Notebook](notebooks/dense/dense.ipynb) and [class](nnets/dense.py)

## 3. Classifiction with neural networks

In the last two lessons, we learned how to perform regression with neural networks.  Now, we'll learn how to perform classification.

- Lesson: Read the [classification tutorial](explanations/classification.ipynb)

## 4. Recurrent networks

Recurrent neural networks can process sequences of data.  They are used for time series and natural language processing.

- Lesson: Read the recurrent network tutorial (coming soon)
- Implementation: [Notebook](notebooks/rnn/rnn.ipynb)

## 5. Regularization

Regularization prevents overfitting to the training set.  This means that the network can generalize well to new data.

- Lesson: Read the regularization tutorial (coming soon)

## 6. Gated recurrent networks

Gated recurrent networks help RNNs process long sequences by helping networks forget irrelevant information.  LSTM and GRU are two popular types of gated networks.

- Lesson: Read the GRU tutorial (coming soon)
- Implementation: [Notebook](notebooks/gru/gru.ipynb)

## 7.  Encoder/Decoder RNNs

Encoder/decoders are used for NLP tasks when the output isn't the same length as the input.  For example, if you want to use questions/answers as training data, the answers may be a different length than the question.

- Lesson: Read the encoder/decoder tutorial (coming soon)
- Implementation: [Notebook](notebooks/rnnencoder/encoder.ipynb)

## 8. Transformers

Transformers fix the problem vanishing/exploding gradients in RNNs by using attention.  Attention allows the network to focus on the most relevant parts of the input.

- Read the transformer tutorial (coming soon)
- Look through the notebook implementation (coming soon)

## More Chapters Coming Soon

# Optional Chapters

You don't have to look through these, but it is recommended.

## Convolutional networks

Convolutional neural networks are used for working with images and time series.

- Lesson: Read the convolutional network tutorial (coming soon)
- Implementation: [Notebook](notebooks/cnn/cnn.ipynb) and [class](nnets/conv.py)

# Installation

If you want to run these notebooks locally, you'll need to install some Python packages.

- Make sure you have Python 3.8 or higher installed.
- Clone this repository.
- Run `pip install -r requirements.txt`