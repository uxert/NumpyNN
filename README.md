# Just numpy NN
This is an educational project meant to show that neural networks are just maths.
Of course, they can be deep, convolutional, recurrent... but they all boil down to just maths.

In this project we implement one of the simplest neural networks to classify MNIST handwritten digits.
**Everything is performed using simple algebra** - not even calculus :) You just have to know how to calculate derivative of the ReLU function,
but that isn't very challenging, is it? Everything else is just pure matrix algebra.   

## What do you need?
**Numpy**, of course. 

Also, **Pandas** is required to load data from CSVs, but that is all. 

**No machine learning libraries, like tensorflow or pytorch.** We implement everything ourselves.

Even with this simple setup, without using some fancy optimizers like ADAM or complicated architectures like convolutional nets, 
**our most-basic-ever network achieves over 90% accuracy on test data.**
Of course, MNIST digits classification is one of the simplest tasks available, but everything has to start somewhere, right?

## Useful materials
Articles, that will help you dive deeper into topic of neural networks.

- Multi layer perceptrons. [More](https://www.datacamp.com/tutorial/multilayer-perceptrons-in-machine-learning) about architecture(s) of neural networks.
- [Loss functions](https://towardsdatascience.com/loss-functions-and-their-use-in-neural-networks-a470e703f1e9/) - what are they, why does a neural net need one, and how to choose one appropriate for your task.
- *(A bit more advanced)* [Kaiming He initialization](https://www.geeksforgeeks.org/kaiming-initialization-in-deep-learning/), and why one would want to use it instead of uniform or standard Gaussian distribution

Also, I could never recommend [Andrej Karpathy](https://github.com/karpathy) enough.
He even created his own "zero to hero" series on neural networks: 
[Neural Networks: Zero to Hero](https://youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&si=ipT_MXsCOI6L-fVQ).

