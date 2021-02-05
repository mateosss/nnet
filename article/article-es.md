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

<!-- TODO: Estoy manteniendo bien los tiempos verbales a lo largo del texto? -->
<!-- TODO: Spellcheck -->
<!-- TODO: 80 column wrap -->

# Implementación Red Feedforward

*Por Mateo de Mayo - <mateodemayo@gmail.com>*

Redes Neuronales 2020 - FaMAF, UNC - Febrero 2021

## Introducción

Se explora el proceso de derivación e implementación de una red neuronal
feedforward multicapa y se analizan sus complicaciones para lograr un mejor
entendimiento de los modelos actuales. Se deriva e implementa una fórmula para
el cálculo de los gradientes de forma ingenua y se la contrasta con el estándar
de backpropagation ofrecido por PyTorch. La implementación se realiza
inicialmente en Python con la ayuda de la librería NumPy y tanto por el método
ineficiente de actualización de gradientes como por el costo adicional del
lenguaje interpretado surgen varios desafíos de performance que son abordados
mediante el uso de Cython y paralelismo en CPU. El resultado final, lejos de ser
óptimo, es adecuado para el entrenamiento de un clasificador y un autoencoder
sobre los datos de MNIST en tiempos razonables.

<!-- TODO: Linkear mnist -->

## Derivación

<!-- TODO: Agregar gráfico de la red para visualizar neurona bias y indices k, q, i, j -->

<!-- TODO: Referenciar nnet.svgz como una derivación handwritten más detallada -->

Gran parte del funcionamiento de las redes feedforward es relativamente intuitivo, el mayor desafío está en la correcta derivación e implementación del paso de actualización de pesos.

Inicialmente utilizaremos como función de costo el error cuadrático medio (MSE) de la capa de salida contra el objetivo esperado:

$$
E(\vec s, \vec t) = \frac 1 {\#L} \sum^{*L}_{q=0}{(O_q - t_q)^2}
$$

Con

- $\vec s$: entrada
- $\vec t$: objetivo
- $L$: índice de última capa
- $*L$: índice de la última neurona de la capa $L$
- $\# L$: tamaño de la capa $L$ (incluyendo neurona con salida constante para el bias)

Expresamos el gradiente de la función de error con respecto a un peso específico.

$$
\tag{1}
\nabla E^k_{ij}
:= \frac {\partial E(\vec s, \vec t)} {\partial w^k_{ij}}
= \frac 2 {\# L} \sum_{q=0}^{*L} (O_q - t_q) \frac {\partial a^L_q} {\partial w^k_{ij}}
$$

Con

- $w^k_{ij}$: peso de neurona $i$ de capa $k$ a neurona $j$ de capa $k+1$
- $O_q$: salida $q$ de la red
- $a^L_q$: salida de la neurona $q$ de la capa $L$. Al ser la última capa es equivalente a $O_q$.

---

Es en este punto en dónde se ha divergido de la derivación estándar
que llevaría a la implementación del algoritmo de backpropagation. La diferencia
reside en plantear $\nabla E^k_{ij}$ de la siguiente manera
$$
\nabla E^k_{ij} = \frac {\partial E(\vec s, \vec t)} {\partial w^k_{ij}} = \frac
1 {\# L} \sum_{q=0}^{*L} \delta^k_j \frac {\partial h^k_{j}} {\partial w^k_{ij}}
$$

En donde

- $h^k_j$: la entrada de la neurona $j$ de la capa $k$ que es una suma pesada
  (también llamada el fanin de la neurona).
- $\delta^k_j := \frac {(O_q - t_q)^2} {\partial h^k_{j}}$: el llamado término
  de error.

Siguiendo la derivación desde este punto se llega a plantear los
gradientes de la capa $k$ en función de los términos de error de la
capa posterior $\delta^{k+1}_j$ y es de esta forma que barriendo desde la
salida hacia la entrada propagando los términos de error es posible calcular
todos los gradientes, de aquí el nombre *backpropagation*. Veremos que, por el
contrario, la derivación presentada aquí dependerá de capas previas y por lo
tanto hará un barrido desde la capa de entrada a la de salida. Llamaremos
coloquialmente a su implementación *frontpropagation* (no confundir con el
*forward pass* de la red).

---

Continuando desde $(1)$ es posible ver que se necesitará analizar $\frac
{\partial a^l_q} {\partial w^k_{ij}}$ con $l = 0, \ldots, L$, es decir cómo
afecta el peso $w^k_{ij}$ a la neurona $a^l_q$, para poder calcular $\frac
{\partial a^L_q} {\partial w^k_{ij}}$. Con

- $g$: función de activación utilizada en todas las capas.

Estamos ahora en posición de analizar por casos el valor de $\frac {\partial
a^l_q} {\partial w^k_{ij}}$.

- Si $l = 0$ (capa de entrada) $\Rightarrow \frac {\partial a^0_q} {\partial
  w^k_{ij}}$ ya que $a^0_q \equiv 0$

- Sino, si $q = \#l$ (neurona de bias) $\Rightarrow \frac {\partial a^l_q}
  {\partial w^k_{ij}} = 0$ ya que la neurona bias es constantemente 1.

- Sino, si $k \ge l$ (peso posterior a neurona) $\Rightarrow \frac {\partial
  a^l_q} {\partial w^k_{ij}} = 0$ ya que un peso posterior no influye en el
  valor de activación.

- Sino, si $k = l - 1$ (peso inmediato a la neurona) $\Rightarrow \frac
  {\partial a^l_q} {\partial w^k_{ij}} = g'(h^l_q) \frac {\partial h^l_q}
  {\partial w^k_{ij}}$

    En este caso como

    $$h^l_q = \sum_{r=0}^{\# (l - 1)} a^{l-1}_r w^{l-1}_{rq}$$

    Dividimos en dos subcasos para determinar la derivada parcial $\frac {\partial h^l_q} {\partial w^k_{ij}}$, cuando el peso llega a la neurona $q$ y cuando no, lo que resulta en:

    - Si $j = q \Rightarrow \frac {\partial a^l_q} {\partial w^k_{ij}} =
      g'(h^l_q) \cdot a^k_i$

    - Si $j \ne q \Rightarrow \frac {\partial a^l_q} {\partial w^k_{ij}} =
      g'(h^l_q) \cdot 0 = 0$

- Sino, si $k < l - 1$ (peso no inmediato a la neurona) $\Rightarrow \frac
  {\partial a^l_q} {\partial w^k_{ij}} = g'(h^l_q) \cdot
  \sum^{*(l-1)}_{r=0}{w^{l-1}_{rq} \frac {\partial a^{l-1}_r} {\partial
  w^k_{ij}}}$

Con esto ya es suficiente para tener una fórmula recursiva bien definida. En la
implementación además se define el caso $k = l - 2$, es decir cuando el peso es
casi inmediato a la neurona, con el fin de reducir la computación extra generada
por la recursividad.

En conclusión, según los valores de $l$, $q$, $k$, $i$ y $j$, tenemos que:

$$
\frac {\partial a^l_q} {\partial w^k_{ij}} = g'(h^l_q) \cdot \left\{
    \begin{array}{ll}
        0\\
        a^k_i\\
        \sum^{*(l-1)}_{r=0}{w^{l-1}_{rq}
            \frac {\partial a^{l-1}_r} {\partial w^k_{ij}}}
    \end{array}
\right.
$$

## Implementación

Al implementar tanto la fórmula anterior en este algoritmo de frontpropagation
como la red en general surgen distintas particularidades que se detallan a
continuación.

La derivación anterior de $\frac {\partial a^l_q} {\partial w^k_{ij}}$ puede
verse reflejada en la implementación en los métodos con nombre conteniendo la
palabra `dadw`, en particular la definición más directa se encuentra en
`NeuralNetwork.py_dadw(l, q, k, i, j)`.
<!-- TODO: Link al github -->

Con este método es ahora posible calcular los gradientes y actualizar los pesos,
y si bien es aún muy ineficiente es bueno realizar *chequeos numéricos* para
corroborar que el cálculo del gradiente es correcto.

<!-- TODO: Link tests -->

Se replantea el problema en función de *matrices* para aprovecharse de las
mejoras de rendimiento proporcionadas por `NumPy`, ver métodos con prefijo
`np_`. La explicación sobre las distintas transformaciones que le suceden al
problema escapan al alcance de este trabajo y pueden verse en detalle en las
notas manuscritas, en la sección de *"matricization"*.

<!--
TODO: Referenciar handwritten notes on matricization
TODO: Linkear a los métodos np_DADW, np_get_gradients, etc
TODO: Hablar más de las matrices DADW? y las otras matrices?
TODO: una vez que haga los TODO de arriba, revisar y reescribir el párrafo
-->

Además, como esperamos utilizar minibatches, prepararemos a la red para *correr
de a batches*. En lugar de tomar una única entrada y devolver una única capa de
salida, vectorizaremos la corrida para que la red reciba y devuelva la misma
cantidad de entradas y salidas que el tamaño del minibatch.

<!-- TODO: Linkear a algo que muestre lo de los batches -->

Para lograr mejoras en los tiempos de ejecución se utilizan técnicas de
*programación dinámica* para que la recursividad de la definición y *Cython*, un
superconjunto de Python que transpila a C, para reescribir y paralelizar con
OpenMP porciones del código que lo ameritan para mayor rendimiento. En el
proceso se utilizan diversos profilers como `line_profile`, `cProfile` y `perf`
para entender mejor los hotspots del código. Por cada cuello de botella
detectado con un profiler se lo reescribe utilizando de forma extensiva el
conocimiento de las ecuaciones y transformaciones de aritmética matricial que
puedan ser de beneficio computacional. Y si bien la versión más performante
pierda legibilidad en contraste con la original, es escencial entender que no es
más que una transformación aritmética.

Luego de estas mejoras, la implementación gana un rendimiento sustancial que,
pese a no estar completamente optimizado, es suficiente como para entrenar en
tiempos razonables las arquitecturas deseadas. El tiempo de entrenamiento por
época del clasificador de dígitos MNIST con arquitectura `28² x 16 x 16 x 10`
puede verse reflejado en la siguiente tabla comparándose las versiones que
calculan los gradientes en Python puro `py`, con NumPy `np`, con Cython `cy` y
la comparación con el estado del arte que se asume está implementado en PyTorch
`tr` (recordar que este implementa backpropagation y no el algoritmo más ingenuo
de frontpropagation presentado en este trabajo).

| *Versión* | i7 7700HQ x1 | E5 2680 x1 | i7 7700HQ x4 | E5 2680 x24 |
|---|---|---|---|---|
| `py` | 405  | 467  | — | — |
| `np` | 6.31 | 7.14 | — | — |
| `cy` | 3.97 | 4.23 | 1.71 | 0.63 |
| `tr` | 0.08 | 0.10 | — | — |

*<center>Tabla 1: Tiempos de entrenamiento por época en segundos</center>*

<!-- TODO: Checkear si ese tag center anda en el pdf de pandoc -->

## Desempeño de la Red

Se utiliza la implementación para modelar dos redes sobre el conjunto de dígitos manuscritos [MNIST](http://yann.lecun.com/exdb/mnist/). Un clasificador con arquitectura `28² x 16 x 16 x 10` que reconoce el dígito escrito y un autoencoder `28² x 64 x 28²` que imita la función identidad de la entrada. Se implementan también las mismas redes en PyTorch. Estos son los resultados luego de 16 épocas.

<!-- TODO: Hacer gráficos -->

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
