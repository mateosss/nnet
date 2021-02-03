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

<!-- TODO:
derivacion
vectorizacion/matrizizacion
performance numpy (batches)
tests/numchecks
-->

## Implementación

<!-- TODO:
Genetic algorithms
custom initialization
performance cython/openmp/line_profiler
comparacion con pytorch
-->

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

##
