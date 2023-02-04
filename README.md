# Zero to GPT

This course will get you from no knowledge of deep learning to training a GPT model.  As AI moves out of the research lab, the world needs more people who can understand and apply it.  If you want to be one of them, this course is for you.

We’ll start with the fundamentals - the basics of neural networks, how they work, and how to tune them.  You need some math to understand deep learning, but we won’t get bogged down in it.  

This course focuses on understanding concepts over theory.  We’ll solve real problems, like predicting the weather and translating languages.  Over time, we'll move to more complex topics, like transformers, GPU programming, and distributed training.

To use this course, go through each chapter sequentially.  Read the lessons or watch the optional videos.  Look through the implementations to solidify your understanding, and try to recreate them on your own.

# Course Outline

## 0. Introduction

An overview of the course and topics we'll cover.  Includes some math and NumPy fundamentals you'll need for deep learning.

- Lesson: Read the [intro](explanations/intro.ipynb)

## 1. Gradient descent

Gradient descent is how neural networks train their parameters to match the data.  It's the "learning" part of deep learning.

- Lesson: Read the [gradient descent tutorial](explanations/linreg.ipynb) and watch the optional [video](https://youtu.be/-cs5D91eBLE)
- Implementation: [Notebook](notebooks/linreg/linreg.ipynb) and [class](nnets/dense.py)

## 2. Dense networks

Dense networks are the basic form of a neural network, where every input is connected to an output.  These can also be called fully connected networks.

- Lesson: Read the [dense network tutorial](explanations/dense.ipynb) and watch the optional [video](https://youtu.be/MQzG1hfhow4)
- Implementation: [Notebook](notebooks/dense/dense.ipynb) and [class](nnets/dense.py)

## 3. Classification with neural networks

Classification is how we get neural networks to categorize data for us.  Classification is used by language models like GPT to predict the next word in a sequence.

- Lesson: Read the [classification tutorial](explanations/classification.ipynb)

## 4. Recurrent networks

Recurrent neural networks are optimized to process sequences of data.  They're used for tasks like translation and text classification.

- Lesson: Read the [recurrent network tutorial](explanations/rnn.ipynb)
- Implementation: [Notebook](notebooks/rnn/rnn.ipynb)

## 5. Backpropagation in depth

So far, we've taken a somewhat loose look at backpropagation to let us focus on understanding neural network architecture.  We'll build a computational graph, and use it to take a deeper look at how backpropagation works.

- Lesson: Read the in-depth backpropagation tutorial (coming soon)
- Implementation: [Notebook](notebooks/comp_graph/comp_graph.ipynb)

## 6. PyTorch

PyTorch is a framework for deep learning that automatically differentiates functions.  It's widely used to create cutting-edge models.

- Lesson: Read the PyTorch tutorial (coming soon)

## 7. Regularization

Regularization prevents overfitting to the training set.  This means that the network can generalize well to new data.

- Lesson: Read the regularization tutorial (coming soon)

## 8. Data

If you want to train a deep learning model, you need data.  Gigabytes of it.  We'll discuss how you can get this data and process it.

- Lesson: Read the data tutorial (coming soon)
- Implementation: Notebook coming soon

## 9.  Encoders and decoders

Encoder/decoders are used for NLP tasks when the output isn't the same length as the input.  For example, if you want to use questions/answers as training data, the answers may be a different length than the question.

- Lesson: Read the encoder/decoder tutorial (coming soon)
- Implementation: [Notebook](notebooks/rnnencoder/encoder.ipynb)

## 10. Transformers

Transformers fix the problem of vanishing/exploding gradients in RNNs by using attention.  Attention allows the network to process the whole sequence at once, instead of iteratively.

- Lesson: Read the transformer tutorial (coming soon)
- Implementation: [Notebook](notebooks/transformer/transformer.ipynb)

## 11. GPU programming with triton

To train a large neural network, we'll need to use GPUs.  PyTorch can automatically use GPUs, but not all operators are fused and optimized.  For example, [flash attention](https://github.com/HazyResearch/flash-attention) can speed up transformers by 2x or more.  We'll use [OpenAI Triton](https://github.com/openai/triton) to implement GPU kernels.

- Lesson: Read the GPU programming tutorial (coming soon)
- Implementation: Notebook coming soon

## 12. Efficient transformers

GPT models take a long time to train.  We can reduce that time by using more GPUs, but we don't all have access to GPU clusters.  To reduce training time, we'll incorporate some recent advances to make the transformer model more efficient.

- Lesson: Read the efficient transformer tutorial (coming soon)
- Implementation: [Notebook](notebooks/eff_transformer/eff_transformer.ipynb)

## More Chapters Coming Soon

# Optional Chapters

## Convolutional networks

Convolutional neural networks are used for working with images and time series.

- Lesson: Read the convolutional network tutorial (coming soon)
- Implementation: [Notebook](notebooks/cnn/cnn.ipynb) and [class](nnets/conv.py)

## Gated recurrent networks

Gated recurrent networks help RNNs process long sequences by helping networks forget irrelevant information.  LSTM and GRU are two popular types of gated networks.

- Lesson: Read the GRU tutorial (coming soon)
- Implementation: [Notebook](notebooks/gru/gru.ipynb)

# Installation

If you want to run these notebooks locally, you'll need to install some Python packages.

- Make sure you have Python 3.8 or higher installed.
- Clone this repository.
- Run `pip install -r requirements.txt`

# License

You can use and adapt this material for your own courses, but not commercially.  You must provide attribution to `Vik Paruchuri, Dataquest` if you use this material.

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.