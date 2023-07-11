# Zero to GPT

This course will take you from no knowledge of deep learning to training your own GPT model.  As AI moves out of the research lab, the world needs more people who can understand and apply it.  If you want to be one of them, this course is for you.

This course balances theory and application.  We’ll solve real problems, like predicting the weather and translating languages.  As we do so, we'll extensively cover theoretical building blocks like gradient descent and backpropagation.  This will prepare you to successfully train and use models in the real world.

We’ll start with the fundamentals - neural network architectures and training methods. Later in the course, we'll move to complex topics like transformers, GPU programming, and distributed training.

You'll need to understand Python to take this course, including for loops, functions, and classes.  The first part of [this Dataquest path](https://www.dataquest.io/path/data-scientist/) will teach you what you need.

To use this course, go through each chapter sequentially.  Read the lessons or watch the optional videos - they have the same information.  Look through the implementations to solidify your understanding, and recreate them on your own.

## Course Outline

**0. Introduction**

An overview of the course and topics we'll cover.

- [Course intro](explanations/intro.ipynb)
- [Video](https://youtu.be/C9FORlAlByo) (optional)

**1. Math and NumPy fundamentals**

This is an optional lesson with a basic refresher on linear algebra and calculus for deep learning.  We'll use NumPy to apply the concepts.  If you're already familiar with these topics, you can skip this lesson.

- [Lesson](explanations/linalg.ipynb)
- [Video](https://youtu.be/5zbTnOd_53g) (optional)

**2. Gradient descent**

Gradient descent is how neural networks train their parameters to match the data.  It's the "learning" part of deep learning.

- [Lesson](explanations/linreg.ipynb) 
- [Video](https://youtu.be/-cs5D91eBLE) (optional)
- [Implementation](notebooks/linreg/linreg.ipynb)

**3. Dense networks**

Dense networks are the basic form of a neural network, where every input is connected to an output.  These can also be called fully connected networks.

- [Lesson](explanations/dense.ipynb)
- [Video](https://youtu.be/MQzG1hfhow4) (optional)
- [Implementation](notebooks/dense/dense.ipynb)

**4. Classification with neural networks**

Classification is how we get neural networks to categorize data for us.  Classification is used by language models like GPT to predict the next word in a sequence.

- [Lesson](explanations/classification.ipynb)
- [Video](https://youtu.be/71GtdWmznok) (optional)

**5. Recurrent networks**

Recurrent neural networks (RNNs) are optimized to process sequences of data.  They're used for tasks like translation and text classification.

- [Lesson](explanations/rnn.ipynb)
- [Video](https://youtu.be/4wuIOcD1LLI) (optional)
- [Implementation](notebooks/rnn/rnn.ipynb)

**6. Backpropagation in depth**

So far, we've taken a loose look at backpropagation to let us focus on understanding neural network architecture.  We'll build a miniature version of PyTorch, and use it to understand backpropagation better.

- [Lesson](explanations/comp_graph.ipynb)
- [Video](https://youtu.be/RyKrG8rTGUY) (optional)

**7. Optimizers**

We've used SGD to update model parameters so far.  We'll learn about other optimizers that have better convergence properties.

- [Lesson](explanations/optimizers.ipynb)
- Video coming soon

**8. Regularization**

Regularization prevents overfitting to the training set.  This means that the network can generalize well to new data.

- [Lesson](explanations/regularization.ipynb)
- Video coming soon

**9. PyTorch**

PyTorch is a framework for deep learning that automatically differentiates functions.  It's widely used to create cutting-edge models.

- [Lesson](explanations/pytorch.ipynb)
- Video coming soon

**10. Working with Text**

GPT models are trained on text.  We'll learn how to process text data for use in deep learning.

- [Lesson](explanations/text.ipynb)
- Video coming soon

**11. Transformers**

Transformers fix the problem of vanishing/exploding gradients in RNNs by using attention.  Attention allows the network to process the whole sequence at once, instead of iteratively.

- Lesson coming soon
- [Implementation](notebooks/transformer/transformer.ipynb)

**12. Cleaning Text Data**

If you want to train a deep learning model, you need data.  Gigabytes of it.  We'll discuss how you can get this data and process it.

- Lesson coming soon

**13. Distributed Training**

To train large models, we need to use multiple GPUs.

- Lesson coming soon

**14. GPT-2**

We'll train a version of the popular GPT-2 model.

- Lesson coming soon

**15. GPU kernels**

PyTorch can automatically use GPUs for training, but not all operators are fused and optimized.  For example, [flash attention](https://github.com/HazyResearch/flash-attention) can speed up transformers by 2x or more.  We'll use [OpenAI Triton](https://github.com/openai/triton) to implement GPU kernels.

- Lesson coming soon
- Implementation coming soon

**16. Efficient Transformers**

GPT models take a long time to train.  We can reduce that time by using more GPUs, but we don't all have access to GPU clusters.  To reduce training time, we'll incorporate some recent advances to make the transformer model more efficient.

- Lesson coming soon
- [Implementation](notebooks/eff_transformer/eff_transformer.ipynb)

**17. Training GPT-X**

We'll train GPT-X, a version of a GPT model with some optimizations and improvements.

- Lesson coming soon
- Implementation coming soon

### More Chapters Coming Soon

## Optional Chapters

**Convolutional networks**

Convolutional neural networks are used for working with images and time series.

- Implementation: [Notebook](notebooks/cnn/cnn.ipynb) and [class](nnets/conv.py)

**Gated recurrent networks**

Gated recurrent networks help RNNs process long sequences by helping networks forget irrelevant information.  LSTM and GRU are two popular types of gated networks.

- [Implementation](notebooks/gru/gru.ipynb)

**Encoders and decoders**

Encoder/decoders are used for NLP tasks when the output isn't the same length as the input.  For example, if you want to use questions/answers as training data, the answers may be a different length than the question.

- [Implementation](notebooks/rnnencoder/encoder.ipynb)

## Installation

If you want to run these notebooks locally, you'll need to install some Python packages.

- Make sure you have Python 3.8 or higher installed.
- Clone this repository.
- Run `pip install -r requirements.txt`

## License

You can use and adapt this material for your own courses, but not commercially.  You must provide attribution to `Vik Paruchuri, Dataquest` if you use this material.

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.