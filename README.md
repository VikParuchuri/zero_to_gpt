# Zero to GPT

This course will take you from no knowledge of deep learning to training your own GPT model.  As AI moves out of the research lab, the world needs more people who can understand and apply it.  If you want to be one of them, this course is for you.

We’ll start with the fundamentals - the basics of neural networks, how they work, and how to tune them.  You need some math to understand deep learning, but we won’t get bogged down in it.  

This course focuses on understanding concepts over theory.  We’ll solve real problems, like predicting the weather and translating languages.  Over time, we'll move to more complex topics, like transformers, GPU programming, and distributed training.

To use this course, go through each chapter sequentially.  Read the lessons or watch the optional videos - they have the same information.  Look through the implementations to solidify your understanding, and try to recreate them on your own.

## Course Outline

**0. Introduction**

An overview of the course and topics we'll cover.

- [Course intro](explanations/intro.ipynb)
- Video coming soon

**1. Math fundamentals**

This is an optional lesson with a basic refresher on linear algebra and calculus for deep learning.  We'll use NumPy to apply the concepts.  If you're already familiar with these topics, you can skip this lesson.

- [Lesson](explanations/linalg.ipynb)
- Video coming soon

**2. Gradient descent**

Gradient descent is how neural networks train their parameters to match the data.  It's the "learning" part of deep learning.

- [Lesson](explanations/linreg.ipynb) 
- [Video](https://youtu.be/-cs5D91eBLE) (optional)
- Implementation: [Notebook](notebooks/linreg/linreg.ipynb) and [class](nnets/dense.py)

**3. Dense networks**

Dense networks are the basic form of a neural network, where every input is connected to an output.  These can also be called fully connected networks.

- [Lesson](explanations/dense.ipynb)
- [Video](https://youtu.be/MQzG1hfhow4) (optional)
- Implementation: [Notebook](notebooks/dense/dense.ipynb) and [class](nnets/dense.py)

**4. Classification with neural networks**

Classification is how we get neural networks to categorize data for us.  Classification is used by language models like GPT to predict the next word in a sequence.

- [Lesson](explanations/classification.ipynb)
- [Video](https://youtu.be/71GtdWmznok) (optional)

**5. Recurrent networks**

Recurrent neural networks are optimized to process sequences of data.  They're used for tasks like translation and text classification.

- [Lesson](explanations/rnn.ipynb)
- [Video](https://youtu.be/4wuIOcD1LLI) (optional)
- [Implementation](notebooks/rnn/rnn.ipynb)

**6. Backpropagation in depth**

So far, we've taken a loose look at backpropagation to let us focus on understanding neural network architecture.  We'll build a miniature version of PyTorch, and use it to understand backpropagation better.

- [Lesson](explanations/comp_graph.ipynb)
- Video coming soon

**7. PyTorch**

PyTorch is a framework for deep learning that automatically differentiates functions.  It's widely used to create cutting-edge models.

- Lesson coming soon

**8. Regularization**

Regularization prevents overfitting to the training set.  This means that the network can generalize well to new data.

- Lesson coming soon

**9. Data**

If you want to train a deep learning model, you need data.  Gigabytes of it.  We'll discuss how you can get this data and process it.

- Lesson coming soon

**10.  Encoders and decoders**

Encoder/decoders are used for NLP tasks when the output isn't the same length as the input.  For example, if you want to use questions/answers as training data, the answers may be a different length than the question.

- Lesson coming soon
- [Implementation](notebooks/rnnencoder/encoder.ipynb)

**11. Transformers**

Transformers fix the problem of vanishing/exploding gradients in RNNs by using attention.  Attention allows the network to process the whole sequence at once, instead of iteratively.

- Lesson coming soon
- [Implementation](notebooks/transformer/transformer.ipynb)

**12. GPU programming with Triton**

To train a large neural network, we'll need to use GPUs.  PyTorch can automatically use GPUs, but not all operators are fused and optimized.  For example, [flash attention](https://github.com/HazyResearch/flash-attention) can speed up transformers by 2x or more.  We'll use [OpenAI Triton](https://github.com/openai/triton) to implement GPU kernels.

- Lesson coming soon
- Implementation coming soon

**13. Efficient transformers**

GPT models take a long time to train.  We can reduce that time by using more GPUs, but we don't all have access to GPU clusters.  To reduce training time, we'll incorporate some recent advances to make the transformer model more efficient.

- Lesson coming soon
- [Implementation](notebooks/eff_transformer/eff_transformer.ipynb)

### More Chapters Coming Soon

## Optional Chapters

**Convolutional networks**

Convolutional neural networks are used for working with images and time series.

- Implementation: [Notebook](notebooks/cnn/cnn.ipynb) and [class](nnets/conv.py)

**Gated recurrent networks**

Gated recurrent networks help RNNs process long sequences by helping networks forget irrelevant information.  LSTM and GRU are two popular types of gated networks.

- [Implementation](notebooks/gru/gru.ipynb)

## Installation

If you want to run these notebooks locally, you'll need to install some Python packages.

- Make sure you have Python 3.8 or higher installed.
- Clone this repository.
- Run `pip install -r requirements.txt`

## License

You can use and adapt this material for your own courses, but not commercially.  You must provide attribution to `Vik Paruchuri, Dataquest` if you use this material.

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.