# Introduction

Let's define a *neural network* as a directed weighted graph where its nodes are
called *neurons* which hold a numerical value (the *activation value*). Neurons
are organized in *layers*, where every neuron in layer $k$ is connected to every
neuron in layer $k+1$ and $k-1$ if that layer exists.

This particular network will be trained to recognize digits in images from the
[mnist database](http://yann.lecun.com/exdb/mnist/), those images are an array
of $28 * 28 = 784$ pixels with values from 0 to 255. The network output will
tell how certain it is that a given image is one of the ten decimal digits.

The input will be represented by the first layer with its 784 neurons. The
output, by the last layer, which is called the *output layer* and its ten
neurons represent how much does the network think the input is each of the ten
digits. Our objective is to tweak the weights of the edges between neurons so
that when given.

# Definitions and Notation

- $L = \text{output layer index}$

- $\#k = \text{Number of neurons in layer } k \space (\forall k = 0, \cdots, L)$

- $\text{neuron i in layer k} = a^k_i = \sigma(in^k_i)$ with

  - $\sigma(x) = \frac{1}{1+e^{-x}}$ and therefore
    $\sigma'(x) = \sigma(x)(1 - \sigma(x))$

  - $in^k_i = \sum_{j = 1}^{\#(k - 1)} a^{k - 1}_j w^{k - 1}_{ji}$

- $w^k_{ji} = \text{weight from } a^k_j \text{ to } a^{k + 1}_i$

- $t_i = \text{expected value of }a^L_i$

# Summarizing

- $\nabla E^k = \sum_{q=1}^{\#L} (a_q^L - t_q) A_k^{Lq}$

- $A_k^{(k+1)q} = g'(in^{k+1}_q) [0 + (A^k)^{(q)}]$

- $A_k^{(k+2)q} = g'(in^{k+1}_j)[w_{jq}^{k+1}g'(in^{k+1}_j)a_i^k]_{ij}$

- $A_k^{lq} = g'(in_q^l) \sum_{r=1}^{\#(l-1)} w_{rq}^{l-1} A_k^{(l-1)r}$

Con $\nabla E^k, A_k^{lq} \in M_{\#k \times \#(k + 1)}$

# Vectorizing the Summations

- $\nabla E^k = [a_q^L - t_q]_{1(q={1,...,\#L})} [A_k^{Lq}]_{(q=1, \ldots,\#L)1}$

- $A_k^{lq} = g'(in_q^l) [w_{rq}^{l - 1}]_{1 (r=1, \ldots, \#(l - 1))} [A_k^{(l - 1)r}]_{(r=1, \ldots, \#(l - 1)) 1}$

- $\begin{bmatrix}a_{11}&a_{12}&\cdots &a_{1n} \\a_{21}&a_{22}&\cdots &a_{2n} \\\vdots & \vdots & \ddots & \vdots\\a_{n1}&a_{n2}&\cdots &a_{nn}\end{bmatrix}$

\end{equation}
$$
