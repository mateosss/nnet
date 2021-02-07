---
# title: "Modelo de Neurona Integrate and Fire"
# subtitle: "Redes Neuronales 2020 - FaMAF, UNC - Octubre 2020"
# author: Mateo de Mayo (mateodemayo@mi.unc.edu.com)
# linkcolor=[rgb]{0 1 1} works as well
# date: November 29, 2020
output: pdf_document
geometry: "left=3cm,right=3cm,top=3cm,bottom=3cm"
header-includes:

  # Color of links
  - \hypersetup{
      colorlinks=true,
      linkcolor=cyan,
      allbordercolors={1 1 0},
      pdfborderstyle={/S/U/W 1},
      filecolor=magenta
      urlcolor=cyan
      urlstyle{same}}

  # For wrapping text around figures
  - \usepackage{wrapfig}

  # Multicolumn definitions
  - \usepackage{multicol}

  # Blockquotes background color
  - \usepackage{framed}
  - \usepackage{xcolor}
  - \let\oldquote=\quote
  - \let\endoldquote=\endquote
  - \definecolor{shadecolor}{HTML}{ECEFF1}
  - \renewenvironment{quote}{\begin{shaded*}\begin{oldquote}}{\end{oldquote}\end{shaded*}}

  # Caption prefixes in spanish
  - \renewcommand{\figurename}{Figura}
  - \renewcommand{\tablename}{Tabla}
---

<!-- TODO: Tiene el color correcto los links? -->
<!-- TODO: Estoy manteniendo bien los tiempos verbales a lo largo del texto? -->
<!-- TODO: Spellcheck -->
<!-- TODO: 80 column wrap -->
<!-- Ver que los nombres de figura y tablas matcheen en el pdf y el html -->

# Implementación Red Feedforward

*Por Mateo de Mayo - <mateodemayo@gmail.com>*

Redes Neuronales 2020 - FaMAF, UNC - Febrero 2021

## Introducción

Se explora el proceso de derivación e implementación de una red neuronal
feedforward multicapa y se analizan sus complicaciones para lograr un mejor
entendimiento de los modelos actuales. Se deriva e implementa un algoritmo para
el cálculo de los gradientes de forma ingenua y se lo contrasta con el estándar
de backpropagation ofrecido por PyTorch. La implementación se realiza
inicialmente en Python con la ayuda de la librería NumPy y tanto por el método
ineficiente de actualización de gradientes como por el costo adicional del
lenguaje interpretado surgen varios desafíos de performance que son abordados
mediante el uso de Cython y paralelismo en CPU. El resultado final, lejos de ser
óptimo, es adecuado para el entrenamiento de un clasificador y un autoencoder
sobre los datos de [MNIST] en tiempos razonables.

## Derivación

<!-- TODO: Agregar gráfico de la red para visualizar neurona bias y indices k, q, i, j -->

<!-- TODO: Referenciar nnet.svgz como una derivación handwritten más detallada -->

Gran parte del funcionamiento de las redes feedforward es relativamente
intuitivo, el mayor desafío está en la correcta derivación e implementación del
paso de actualización de pesos. La forma usual de esta actualización es mediante
el descenso por el gradiente, en particular mediante el algoritmo de
backpropagation. Vale aclarar que hay muchas formas de minimizar una función
como la de costo, se implementó una versión que actualiza los pesos mediante
algoritmos genéticos que, si bien es subóptima no logrando superar el 25% de
precisión en el clasificador MNIST, muestra que incluso algoritmos tan sencillos
logran hacer que la red aprenda ciertos patrones. El modelo desarrollado en este
trabajo utilizará descenso por el gradiente pero con un algoritmo distinto a
backpropagation que se deriva a continuación.

 <!-- HTML BEGIN  -->
<!-- <center><img src="res/network-diagram.svg" width="65%"/></center>
*<center>Figura 1: Diagrama y notación de la red. Los colores serán referenciados a continuación.</center>* -->
 <!-- HTML END  -->

<!-- LATEX BEGIN -->
\begin{figure}[h]
  \centering
  \includegraphics[width=0.65\textwidth]{res/network-diagram.pdf}
  \caption{\emph{Diagrama y notación de la red. Los colores serán referenciados a continuación.}}
\end{figure}
<!-- LATEX END -->

<!--
TODO: explicación del diagrama, agregar el código de colores a los casos -->

Utilizaremos como función de costo el error cuadrático medio (MSE) de la capa de
salida contra el objetivo esperado:

$$
E(\vec s, \vec t) = \frac 1 {\#L} \sum^{*L}_{q=0}{(O_q - t_q)^2}
$$

En donde

<!-- HTML BEGIN -->
<!-- - $\vec s$: entrada
- $\vec t$: objetivo
- $O_q$: salida $q$ de la red
- $L$: índice de última capa
- $*L$: índice de la última neurona de la capa $L$
- $\# L$: tamaño de la capa $L$ -->
<!-- HTML END -->

<!-- LATEX BEGIN -->
\begin{multicols}{3}
\begin{itemize}
\item $\vec s$: entrada
\item $\vec t$: objetivo
\item $O_q$: salida $q$ de la red
\item $L$: índice de última capa
\item $*L$: índice de la última neurona de la capa $L$
\item $\# L$: tamaño de la capa $L$
\end{itemize}
\end{multicols}
<!-- END BEGIN -->

Expresamos el gradiente de la función de error con respecto a un peso
específico.

$$
\tag{1}
\nabla E^k_{ij}
:= \frac {\partial E(\vec s, \vec t)} {\partial w^k_{ij}}
= \frac 2 {\# L} \sum_{q=0}^{*L} (O_q - t_q)
\frac {\partial a^L_q} {\partial w^k_{ij}}
$$

Con

<!-- HTML BEGIN -->
<!-- - $w^k_{ij}$: peso de neurona $i$ de capa $k$ a neurona $j$ de capa $k+1$
- $a^L_q$: salida de la neurona $q$ de la capa $L$. Al ser la última capa es
  $O_q$. -->
<!-- HTML END -->

<!-- LATEX BEGIN -->
\begin{multicols}{2}
\begin{itemize}
\item $w^k_{ij}$: peso de neurona $i$ de capa $k$ a neurona $j$ de capa $k+1$
\item $a^L_q$: salida de la neurona $q$ de la capa $L$. Al ser la última capa es
  $O_q$.
\end{itemize}
\end{multicols}
<!-- LATEX END -->

> **Nota:** Es en este punto en dónde se ha divergido de la derivación estándar
> que llevaría a la implementación del algoritmo de backpropagation. La diferencia
> reside en plantear $\nabla E^k_{ij}$ de la siguiente manera
> $$
> \nabla E^k_{ij} = \frac {\partial E(\vec s, \vec t)} {\partial w^k_{ij}} = \frac
> 1 {\# L} \sum_{q=0}^{*L} \delta^k_j \frac {\partial h^k_{j}} {\partial w^k_{ij}}
> $$
>
> En donde
>
> - $h^k_j$: la entrada de la neurona $j$ de la capa $k$ que es una suma pesada
>   (también llamada el fanin de la neurona).
>
> - $\delta^k_j := \frac {(O_q - t_q)^2} {\partial h^k_{j}}$: el llamado término
>   de error.
>
> Siguiendo la derivación desde este punto se llega a plantear los
> gradientes de la capa $k$ en función de los términos de error de la
> capa posterior $\delta^{k+1}_j$ y es de esta forma que barriendo desde la
> salida hacia la entrada propagando los términos de error es posible calcular
> todos los gradientes, de aquí el nombre *backpropagation*. Veremos que, por el
> contrario, la derivación presentada aquí dependerá de capas previas y por lo
> tanto hará un barrido desde la capa de entrada a la de salida. Llamaremos
> coloquialmente a su implementación *frontpropagation* (no confundir con el
> *forward pass* de la red).
>
> ​ <!-- This line has a hidden whitespace character for padding -->

Continuando desde $(1)$ es posible ver que se necesitará analizar $\frac
{\partial a^l_q} {\partial w^k_{ij}}$ con $l = 0, \ldots, L$, es decir cómo
afecta el peso $w^k_{ij}$ a la neurona $a^l_q$, para poder calcular $\frac
{\partial a^L_q} {\partial w^k_{ij}}$. Con

- $g$: función de activación utilizada en todas las capas.

Estamos ahora en posición de analizar por casos el valor de $\frac {\partial
a^l_q} {\partial w^k_{ij}}$.

- Si $l = 0$ (capa de entrada) $\Rightarrow \frac {\partial a^0_q} {\partial
  w^k_{ij}} = 0$

  ya que $a^0_q \equiv 0$

- Sino, si $q = \#l$ (neurona de bias) $\Rightarrow \frac {\partial a^l_q}
  {\partial w^k_{ij}} = 0$

  ya que la neurona bias es constantemente 1.

- Sino, si $k \ge l$ (peso posterior a neurona) $\Rightarrow \frac {\partial
  a^l_q} {\partial w^k_{ij}} = 0$

  ya que un peso posterior no influye en el valor de activación.

- Sino, si $k = l - 1$ (peso inmediato a la neurona) $\Rightarrow \frac
  {\partial a^l_q} {\partial w^k_{ij}} = g'(h^l_q) \frac {\partial h^l_q}
  {\partial w^k_{ij}}$

    En este caso como

    $$h^l_q = \sum_{r=0}^{\# (l - 1)} a^{l-1}_r w^{l-1}_{rq}$$

    Dividimos en dos subcasos para determinar la derivada parcial $\frac
    {\partial h^l_q} {\partial w^k_{ij}}$, cuando el peso llega a la neurona $q$
    y cuando no, lo que resulta en:

    - Si $j = q \Rightarrow \frac {\partial a^l_q} {\partial w^k_{ij}} =
      g'(h^l_q) \cdot a^k_i$

    - Si $j \ne q \Rightarrow \frac {\partial a^l_q} {\partial w^k_{ij}} =
      g'(h^l_q) \cdot 0 = 0$

- Sino, si $k < l - 1$ (peso no inmediato a la neurona)

  $\Rightarrow \frac
  {\partial a^l_q} {\partial w^k_{ij}} = g'(h^l_q) \cdot
  \sum^{*(l-1)}_{r=0}{w^{l-1}_{rq} \frac {\partial a^{l-1}_r} {\partial
  w^k_{ij}}}$

Con esto ya es suficiente para tener una fórmula recursiva bien definida. En la
implementación además se define el caso $k = l - 2$, es decir cuando el peso es
casi inmediato a la neurona, con el fin de reducir la computación extra generada
por la recursividad.

En conclusión, según los valores de $l$, $q$, $k$, $i$ y $j$, tenemos que:

$$
\tag{2}
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
como el resto de los aspectos de la red surgen distintas particularidades que se
detallan a continuación.

La derivación anterior de $\frac {\partial a^l_q} {\partial w^k_{ij}}$ puede
verse reflejada en la implementación en los métodos con nombre conteniendo la
palabra `dadw`, en particular la definición más directa se encuentra en
`NeuralNetwork.py_dadw(l, q, k, i, j)`.
<!-- TODO: Link al github -->

Con este método es ahora posible calcular los gradientes y actualizar los pesos,
y si bien es aún muy ineficiente es bueno realizar *chequeos numéricos* para
corroborar que el cálculo del gradiente es correcto.

<!-- TODO: Link tests -->
<!-- TODO: Link stanford cs231n -->

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
*programación dinámica* para aliviar el trabajo extra que significa la
recursividad en la actualización de pesos. También se utiliza *Cython*, un
superconjunto de Python que transpila a C, para reescribir y paralelizar con
OpenMP porciones del código que lo ameritan para mayor rendimiento. En el
proceso se utilizan diversos profilers como `line_profile`, `cProfile` y `perf`
para entender mejor los hotspots del código. Por cada cuello de botella
detectado con un profiler se lo reescribe utilizando de forma extensiva el
conocimiento de las ecuaciones y transformaciones de aritmética matricial que
puedan ser de beneficio computacional. Si bien la versión más performante pierde
legibilidad en contraste con la original, es escencial entender que no es más
que una transformación aritmética.

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

<!-- HTML BEGIN -->
<!-- *<center>Tiempos de entrenamiento por época en segundos</center>* -->
<!-- HTML END -->

<!-- LATEX BEGIN -->
\begin{center}
\emph{Tiempos de entrenamiento de una época en segundos sobre distintas CPUs}
\end{center}
<!-- LATEX END -->

<!-- TODO: Checkear si la descripcion de la tabla se ve bien -->

## Desempeño de la Red

Se utiliza la implementación para modelar dos redes sobre el conjunto de dígitos
manuscritos [MNIST]. Un clasificador con arquitectura `28² x 16 x 16 x 10` que
reconoce el dígito escrito y un autoencoder `28² x 64 x 28²` que imita la
función identidad de la entrada. Se implementan también las mismas redes en
PyTorch. Los siguientes resultados son sobre el conjunto de validación, los
resultados sobre el conjunto de entrenamiento son virtualmente identicos.

![Clasificador MNIST - Error y Precisión](res/class_losses_hitrate_es.svg)

Cabe aclarar que si bien tanto la implementación de este trabajo `Nnet` como la
versión en `PyTorch` utilizan la [misma][kaiming-pytorch-docs] inicialización de
pesos basada en [Kaiming][kaiming-paper] sobre una distribución uniforme, los
bias en la versión `PyTorch` son inicializados de forma levemente distinta. A
esto y al azar inherente a la corrida se le atribuyen las pequeñas diferencias
de performance de las redes. Para mostrar este punto se grafica además una
versión `Nnet*` que utiliza una forma de inicialización más ingenua (i.e. $\frac
{\mathcal{N}(0, 1)} {\sqrt{fanin}}$) que en estas arquitecturas parece mejorar
el rendimiento.

<!-- HTML BEGIN -->
<!-- ![Clasificador MNIST - Error y Precisión](res/auto_losses_es.svg) -->
<!-- HTML END -->

<!-- LATEX BEGIN -->
\begin{center}
  \includegraphics[width=200px]{res/auto_losses_es.pdf}
\end{center}
<!-- LATEX END -->

Notar la similitud de las versiones `Nnet` y `PyTorch`. Esto muestra que nuestro
algoritmo de frontpropagation parece estar dando los mismos resultados que el
backpropagation clásico. Efectivamente si se inicializan las dos redes con los
mismos pesos los resultados no presentan diferencia significativa, incluso con
las aleatoriedades propias del descenso estocástico.

## Conclusiones

Se ha implementado una red neuronal desde los conceptos básicos derivando e
implementando en el proceso un algoritmo de actualización de pesos que es
correcto. Se ha mostrado que a pesar de ser poco eficiente, logra entrenar redes
de tamaños significativos en tiempos aceptables. El conocimiento obtenido en el
proceso es de gran valor y permite obtener una mayor solvencia en la
implementación de nuevas técnicas sobre las redes.

A futuro es posible implementar otras características como regularizaciones,
dropout, actualización ADAM, función de error cross entropy las cuales será
posible poner a punto para la aplicación deseada modificando no solo los
parámetros pero, de ser necesario, los algoritmos en sí. Esto trae mucha
libertad de exploración para plantear nuevas técnicas. Otra tecnología que no se
alcanzó a profundizar, es el uso de autodiferenciación. Notar que cambios en la
función de error necesitarían volver a derivar e implementar los gradientes,
librerías como PyTorch o TensorFlow utilizan en cambio autodiferenciación
sobrecargando operadores para, regla de la cadena mediante, calcular los
gradientes de forma automática para cualquier combinación de operaciones.

<!-- Referencias -->

[MNIST]: http://yann.lecun.com/exdb/mnist/
[kaiming-pytorch-docs]: https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_
[kaiming-paper]: https://arxiv.org/abs/1502.01852v1
