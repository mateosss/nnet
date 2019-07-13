# Introduction

The idea of this project is to implement the classic example of a network that
recognizes characters from the `mnist` database, based on intuition of how
neural networks should work, and thus being probably a very naive and simplistic
approach.

These are the main resources as of now (and I want this list to be as small as
possible):

- [3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [ai-junkie](http://www.ai-junkie.com/ann/evolved/nnt1.html)
- [Backpropagation for dummies](https://codesachin.wordpress.com/2015/12/06/backpropagation-for-dummies/)

# Roadmap

The roadmap of this project is as follows:

1. [ ] Implement forward propagation
2. [ ] Implement batch or similar way of measuring the cost of the network
3. [ ] Improve the network with a genetic algorithm
4. [ ] Improve the network with proper backpropagation

## 1 - Forward Propagation

`nnet` should be able to run a forward propagation and put a score on a single
run like this:

```python
from nnet import nnet
nnet.set_layers([784, 16, 16, 10]) # Amount of neurons per layer
input = [...] # Pixels (0-255) values of a 28x28 (784) grayscaled image
expected = [...] # Expected output layer based on the real label of the input
params = [...] # Initially, enough random numbers for weights and biases
guess = nnet(params) # Runs the forward propagation and returns the last layer
score = sum((g - e)**2 for g, e in zip(guess, expected)) # Fitness score
```

## 2 - Batch Scoring

(*I'm creating this title names as I go, they probably have a proper name
already, I'll update them when I learn how they're really called*)

Next, we should run a batch of examples for determining the fitness of the
network, it will be something like. The main concern here is that it should be
*fast*, last time I implemented everything in plain python, and it was *very*
slow, should probably use matrices and `numpy`.

```python
scores = [score_from(input, expected) for input, expected in batch]
total_score = avg(scores)
```

## 3 - Genetic Algorithm

After having the score, we should try to improve it, first we will
implement a genetic algorithm for this, the logic for it will be implemented
in a generic separate set of classes that will serve as mixins for the
concrete individuals.

# 4 - Backpropagation

And finally, the core of the project. Once the genetic algorithm version is
done, we will finally try to implement proper backpropagation, dumping out the
genetic algorithms and understanding the equations, solving the necessary ones,
and finally implementing them.

# Nice Extras

- [ ] MyPy
- [ ] Perfect pylint score
- [ ] Good testing support
