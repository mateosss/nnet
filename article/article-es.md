---
# title: "Modelo de Neurona Integrate and Fire"
# subtitle: "Redes Neuronales 2020 - FaMAF, UNC - Octubre 2020"
# author: Mateo de Mayo (mateodemayo@mi.unc.edu.com)
# linkcolor=[rgb]{0 1 1} works as well
date: November 29, 2020
output: pdf_document
header-includes:
  - \hypersetup{
      colorlinks=true,
      linkcolor=cyan,
      allbordercolors={1 1 0},
      pdfborderstyle={/S/U/W 1},
      filecolor=magenta
      urlcolor=cyan
      urlstyle{same}}
---

# Implementación Red Feedforward

*Por Mateo de Mayo - <mateodemayo@gmail.com>*

Redes Neuronales 2020 - FaMAF, UNC - Febrero 2021

## Introducción

<!-- TODO: es PyTorch o Pytorch? Tensorflow o TensorFlow? corregir si necesario-->

Se implementa una perceptrón multicapa sin utilizar librerías especializadas
como lo son PyTorch o TensorFlow. El propósito de la implementación es
interiorizarse con estas redes de una forma más directa que permita entender con
mayor profundidad las complicaciones de las mismas. Las abstracciones que las
librerías antes mencionadas otorgan, a pesar de ser de gran valor en la práctica
común, pueden amplificar la visión de que este tipo de algoritmos son de "caja
negra". Pudiendo experimentar los pormenores que suceden en su implementación se
logran desmitificar algunas de estas ideas. En un contexto en dónde la
explicabilidad e interpretabilidad de estos modelos cobra cada vez más
importancia es vital interiorizar estos algoritmos. La implementación con gran
nivel de detalle da al programador un mejor entendimiento del problema y le
permite moldearlo a su gusto. Probar nuevas ideas, para las cuales las librerías
especializadas pueden no estar pensadas, es de vital importancia para evitar el
estancamiento en la ciencia.

<!-- TODO: Referenciar el paper de hardware lottery en lo del estancamiento -->

## Derivación

Gran parte del funcionamiento de las redes feedforward es relativamente intuitivo, el mayor desafío está en la correcta derivación e implementación del paso de actualización de pesos (backpropagation).

Inicialmente utilizaremos como función de costo el error cuadrático medio (MSE) de la capa de salida contra el objetivo esperado:

$$
E(\vec s, \vec t) = \frac 1 {\#L} \sum^{*L}_{q=0}{(O_q - t_q)^2}
$$

Con

- $\vec s$: entrada
- $\vec t$: objetivo
- $L$: índice de última capa
- $*L$: índice de la última neurona de la capa $L$
- $\# L$: tamaño de la capa $L$ (incluyendo neurona con activación constante para el bias)

Expresamos el gradiente de la función de error con respecto a un peso específico.

$$
\nabla E^k_{ij}
= \frac {\partial E(\vec s, \vec t)} {\partial w^k_{ij}}
= \frac 2 {\# L} \sum_{q=0}^{*L} (O_q - t_q) \frac {\partial a^L_q} {\partial w^k_{ij}}
$$

Con

- $w^k_{ij}$: peso de neurona $i$ de capa $k$ a neurona $j$ de capa $k+1$
- $a^L_q$: valor de activación de neurona $q$ de capa $L$

Como podría verse, necesitaremos analizar $\frac {\partial a^l_q} {\partial w^k_{ij}}$ con $l = 0, \ldots, L$, veamos por casos.

Si $l = 0$ (capa de entrada) $\Rightarrow \frac {\partial a^0_q} {\partial w^k_{ij}}$ ya que $a^0_q \equiv 0$

Sino, si $q = \#l$ (neurona de bias) $\Rightarrow \frac {\partial a^l_q} {\partial w^k_{ij}} = 0$ ya que la neurona bias es constantemente 1.

Sino, si $k \ge l$ (peso posterior a neurona) $\Rightarrow \frac {\partial a^l_q} {\partial w^k_{ij}} = 0$ ya que un peso posterior no influye en el valor de activación.

Sino, si $k = l - 1$ (peso inmediato a la neurona) $\Rightarrow \frac {\partial a^l_q} {\partial w^k_{ij}} = g'(h^l_q) \frac {\partial h^l_q} {\partial w^k_{ij}}$ con

- $g'$: derivada de la función de activación elegida
- $h^l_q$: sumatoria pesada de la neurona $q$ en la capa $l$ (también llamado el fanin)

En este caso como

$$h^l_q = \sum_{r=0}^{\# (l - 1)} a^{l-1}_r w^{l-1}_{rq}$$

Dividimos en dos subcasos para determinar la derivada parcial $\frac {\partial h^l_q} {\partial w^k_{ij}}$, cuando el peso llega a la neurona $q$ y cuando no, es decir:

Si $j = q \Rightarrow \frac {\partial a^l_q} {\partial w^k_{ij}} = g'(h^l_q) \cdot a^k_i$

Si $j \ne q \Rightarrow \frac {\partial a^l_q} {\partial w^k_{ij}} = g'(h^l_q) \cdot 0 = 0$

Sino, si $k \lt l - 1$ (peso no inmediato a la neurona) $\Rightarrow \frac {\partial a^l_q} {\partial w^k_{ij}} = g'(h^l_q) \cdot \sum^{*(l-1)}_{r=0}{w^{l-1}_{rq} \frac {\partial a^{l-1}_r} {\partial w^k_{ij}}}$

Con esto ya es suficiente para tener una fórmula recursiva bien definida. En la implementación además se define el caso $k = l - 2$, es decir cuando el peso es casi inmediato a la neurona, con el fin de reducir la computación extra generada por la recursividad.

En conclusión, tenemos que:

\[
\frac {\partial a^l_q} {\partial w^k_{ij}}=g'(h^l_q) \cdot \left\{
    \begin{array}{ll}
        0 \text{ if l = 0}\\
        a^k_i \text{ if } k = l - 1 \text{ and } j = q\\
        \sum^{*(l-1)}_{r=0}{w^{l-1}_{rq} \frac {\partial a^{l-1}_r} {\partial w^k_{ij}}}
    \end{array}
\right.
\]

## Implementación

El desarrolo anterior puede verse reflejado en la implementación del método `NeuralNetwork.py_dadw(l, q, k, i, j)`.
<!-- TODO: Link al github -->

Con este método es ahora posible calcular los gradientes y actualizar los pesos, y si bien es aún muy ineficiente es bueno realizar tests numéricos para corroborar que el cálculo del gradiente es correcto.

<!-- TODO: Estoy manteniendo bien los tiempos verbales a lo largo del texto? -->

Se replantea el problema en función de matrices para aprovecharse de las mejoras de rendimiento proporcionadas por `NumPy`, ver métodos con prefijo `np_`.
<!-- TODO: Linkear a los métodos np_DADW, np_get_gradients, etc -->
<!-- TODO: Hablar más de las matrices DADW? y las otras matrices? -->

Además, como esperamos utilizar minibatches, prepararemos a la red para tomar como entrada el tamaño del minibatch y devolver la misma cantidad de salidas.
<!-- TODO: Linkear a algo que muestre lo de los batches -->

Se utiliza Cython, un superconjunto de Python que transpila a C, para reescribir y paralelizar con OpenMP porciones del código que lo ameritan para mayor rendimiento. En el proceso se utilizan diversos profilers como `line_profile`, `cProfile` y `perf` para entender mejor los hotspots del código.

Luego de estas mejoras, la implementación gana un rendimiento sustancial que, pese a no estar completamente optimizado, es suficiente como para no quedarse tan atrás de otras implementaciones como las de PyTorch o TensorFlow. Como para tener una idea a grandes rasgos en una arquitectura `784 x 16 x 16 x 10`, la versión en `NumPy` unicamente toma en cierto hardware 40 segundos por época mientras que en el mismo hardware las optimizaciones con Cython la reducen a 0.6 segundos por época, no tan lejano a los 0.1 segundos ofrecido por Pytorch.

<!-- TODO: Ser más formal con las mediciones de tiempo, tener en cuenta los cores y demás -->

### Otras Ideas

<!-- TODO
Genetic algorithms
custom initialization -->

## Mejoras Progresivas

Ahora que tenemos nuestra implementación funcionando

<!-- TODO:

red con
1. inizializacion random
2. activaciones sigmoide
3. minibatches no random
4. gd sin momentum
5. mse loss function

luego como mejora cuando le cambias 1, 2, 3, 4 y 5 respectivamente

five stages:
    1. weight initialization
        random vs autoencoder, mio, xavier, kaiming
    2. activation functions
        sigmoid vs relu/clamp
    3. randomness in gradient descent (minibatch, dropout)
        batched vs uno por uno vs minibatch vs shuffle
        no dropout vs dropout
    4. improve weight update: momentum, nesterov, lr adaptativo, rprop, rmsprop, adam
        no momentum vs momentum
    5. better loss function: regularization L1 L2, cross entropy -->

<!-- TODO: Performance autoencoder, clasificador, y mencionar que anda igual a pytorch -->
