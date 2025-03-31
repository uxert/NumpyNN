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
