Some definitions and notation:

- $L = \text{output layer index}$

- $\#k = \text{Number of neurons in layer } k \space (\forall k = 0, \cdots, L)$

- $a^k_i = \sigma(in^k_i)$

  - $\sigma(x) = \frac{1}{1+e^{-x}}$ and therefore
    $\sigma'(x) = \sigma(x)(1 - \sigma(x))$

  - $in^k_i = \sum_{j = 1}^{\#(k - 1)} a^{k - 1}_j w^{k - 1}_{ji}$

- $w^k_{ji} = \text{weight from } a^k_j \text{ to } a^{k + 1}_i$

- $t_i = \text{expected value of }a^L_i$
