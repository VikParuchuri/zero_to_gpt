# Neural Networks From Scratch

This course will help you master neural networks from the ground up by implementing them from scratch.

- The `explanations` folder has writeups of different networks, starting with the basics.
- The `nnets` folder has clean implementations that are best for someone who understands the high level concept.
- The `exploration` folder has implementations that are more exploratory and easier to understand for beginners.

I recommend reading the writeup, then looking at the exploratory implementation, then looking at the clean implementation.

## 1. Gradient Descent

Gradient descent is an important building block for neural networks.  It's how networks train their parameters to fit the data.

- [Gradient descent with linear regression tutorial](explanations/linreg.ipynb)
- [Clean implementation](nnets/dense.py) - linear regression is equivalent to a dense network with no activation function.
- [Notebook implementation](exploration/linreg/linreg.ipynb)

## 2. Dense networks

Dense networks are networks where every input is connected to an output.  They're the most general form of a neural network.  These can also be called fully connected networks.

- Dense network tutorial coming soon
- [Clean implementation](nnets/dense.py)
- [Notebook implementation](exploration/dense/dense.ipynb)

## 3. Convolutional networks

Convolutional neural networks are used for working with images and time series.

- Convolutional network tutorial coming soon
- [Clean implementation](nnets/conv.py)
- [Notebook implementation](exploration/cnn/cnn.ipynb)

## 4. Recurrent networks

Recurrent neural networks can process sequences of data.  They are used for time series and natural language processing.

- Recurrent network tutorial coming soon
- Clean implementation coming soon
- [Notebook implementation](exploration/rnn/rnn.ipynb)


# Installation

- `pip install -r requirements.txt`
- You can download the data [here](https://drive.google.com/drive/folders/1uchDw57-lJ_lA7gqLvUZ9mOy4Ig0rH5y?usp=share_link).  Put it in the `data` folder.