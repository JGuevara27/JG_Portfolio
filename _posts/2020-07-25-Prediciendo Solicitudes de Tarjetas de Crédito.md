---
title: "Prediciendo Solicitudes de Tarjetas de Crédito"
excerpt: "Construyendo un modelo de Aprendizaje Automático que prediga las aprobaciones de las solicitudes de las tarjetas de crédito"
layout: single
header:
  overlay_image: "https://images.unsplash.com/photo-1430276084627-789fe55a6da0?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=752&q=80"
  overlay_filter: 0.4
  caption: ""
  cta_label: "Switch to English version"
  cta_url: "https://jguevara27.github.io/JG_Portfolio/finance/python/Predicting-Credit-Card-Approvals"
categories:
  - finanzas
  - python
tags:
  - finanzas
  - aprendizaje automático
  - python
author: "Josue Guevara"
date: "25 Julio 2020"
hidden: False
---


## 1. Solicitudes de Tarjetas de Crédito
<p align="justify">Los bancos comerciales reciben <em>muchísimas</em> solicitudes de tarjetas de crédito. Muchos de ellos son rechazados por diversas razones tales como altos saldos de préstamo, bajos niveles de ingresos, o tantas consultas sobre el informe de crédito de un individuo (solicitudes). Analizar manualmente estas solicitudes es mundano, propenso a errores y requiere mucho tiempo (y el tiempo es dinero!). Por suerte, esta tarea puede ser automatizada con el poder del aprendizaje automatico y casi todos los bancos comerciales lo hacen hoy en día. En este proyecto, construiremos un predictor automatizado de aprobación de tarjetas de crédito usando técnicas de Machine Learning, tal como los bancos reales lo hacen.</p>

<div align="center"><img src="https://images.unsplash.com/photo-1578670812003-60745e2c2ea9?ixlib=rb-1.2.1&auto=format&fit=crop&w=334&q=80" alt="Credit card being held in hand"></div>

<p align="justify">Usaremos el <a href="http://archive.ics.uci.edu/ml/datasets/credit+approval">conjunto de datos de <em>Credit Card Approval</em> </a> del repositorio UCI Machine Learning. La estructura del proyecto es el siguiente: </p>
<ul>
<li>Primero, iniciaremos cargando y viendo el conjunto de datos.</li>
<li>Veremos que el conjunto de datos tiene una mixtura de variables numéricas y no numéricas, que contienen valores de diferentes rangos  y que además contiene un número de datos vacíos.</li>
<li> Tendremos que preprocesar el conjunto de datos para asegurarnos de que el modelo de Machine Learning que escojamos pueda hacer buenas predicciones.</li>
<li>Después de que nuestros datos estén en buena forma, haremos un análisis exploratorio de datos para construir nuestras intuiciones.</li>
<li>Finalmente, crearemos un modelo de aprendizaje automático que pueda predecir si se aceptará la solicitud de una persona para una tarjeta de crédito.</li>
</ul>
<p align="justify">Primero, cargamos y vemos el conjunto de datos. Descubriremos que el contribuyente del conjunto de datos ha anonimizado los nombres de las variables puesto que estos datos son confidencionales</p>


```python
# Import pandas
import pandas as pd
# Load dataset
cc_apps = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data", header=None)

# Inspect data
cc_apps.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b</td>
      <td>30.83</td>
      <td>0.000</td>
      <td>u</td>
      <td>g</td>
      <td>w</td>
      <td>v</td>
      <td>1.25</td>
      <td>t</td>
      <td>t</td>
      <td>1</td>
      <td>f</td>
      <td>g</td>
      <td>00202</td>
      <td>0</td>
      <td>+</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>58.67</td>
      <td>4.460</td>
      <td>u</td>
      <td>g</td>
      <td>q</td>
      <td>h</td>
      <td>3.04</td>
      <td>t</td>
      <td>t</td>
      <td>6</td>
      <td>f</td>
      <td>g</td>
      <td>00043</td>
      <td>560</td>
      <td>+</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a</td>
      <td>24.50</td>
      <td>0.500</td>
      <td>u</td>
      <td>g</td>
      <td>q</td>
      <td>h</td>
      <td>1.50</td>
      <td>t</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>00280</td>
      <td>824</td>
      <td>+</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b</td>
      <td>27.83</td>
      <td>1.540</td>
      <td>u</td>
      <td>g</td>
      <td>w</td>
      <td>v</td>
      <td>3.75</td>
      <td>t</td>
      <td>t</td>
      <td>5</td>
      <td>t</td>
      <td>g</td>
      <td>00100</td>
      <td>3</td>
      <td>+</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b</td>
      <td>20.17</td>
      <td>5.625</td>
      <td>u</td>
      <td>g</td>
      <td>w</td>
      <td>v</td>
      <td>1.71</td>
      <td>t</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>s</td>
      <td>00120</td>
      <td>0</td>
      <td>+</td>
    </tr>
  </tbody>
</table>
</div>



## 2. Analizando las solicitudes
<p align="justify">El output puede parecer un poco confuso a primera vista, pero intentemos encontrar las variables más importantes de las solicitudes de tarjetas de crédito. Las variables han sido anonimizadas para proteger la privacidad, pero <a href="http://rstudio-pubs-static.s3.amazonaws.com/73039_9946de135c0a49daa7a0a9eda4a67a72.html">este blog</a> nos da una muy buena aproximación de las probables variables. Las variables en una solicitud de tarjeta de crédito podrían ser <code>Gender</code>, <code>Age</code>, <code>Debt</code>, <code>Married</code>, <code>BankCustomer</code>, <code>EducationLevel</code>, <code>Ethnicity</code>, <code>YearsEmployed</code>, <code>PriorDefault</code>, <code>Employed</code>, <code>CreditScore</code>, <code>DriversLicense</code>, <code>Citizen</code>, <code>ZipCode</code>, <code>Income</code> y finalmente <code>ApprovalStatus</code>. Esto nos da un muy buen punto de inicio, y podemos podemos relacionar estas variables con respecto a las columnas en el output.   </p>
<p align="justify"> Como podemos ver desde nuestro primer vistazo a los datos, el conjunto de datos tiene una combinación de características numéricas y no numéricas. Esto se puede solucionar con algo de preprocesamiento, pero antes de hacerlo, aprendamos un poco más sobre el conjunto de datos para ver si hay otros problemas del conjunto de datos que deben corregirse.</p>


```python
# Print summary statistics
cc_apps_description = cc_apps.describe()
print(cc_apps_description)

print("\n")

# Print DataFrame information
cc_apps_info = cc_apps.info()
print(cc_apps_info)

print("\n")

# Inspect missing values in the dataset
cc_apps.tail()
```

                   2           7          10             14
    count  690.000000  690.000000  690.00000     690.000000
    mean     4.758725    2.223406    2.40000    1017.385507
    std      4.978163    3.346513    4.86294    5210.102598
    min      0.000000    0.000000    0.00000       0.000000
    25%      1.000000    0.165000    0.00000       0.000000
    50%      2.750000    1.000000    0.00000       5.000000
    75%      7.207500    2.625000    3.00000     395.500000
    max     28.000000   28.500000   67.00000  100000.000000
    
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 690 entries, 0 to 689
    Data columns (total 16 columns):
    0     690 non-null object
    1     690 non-null object
    2     690 non-null float64
    3     690 non-null object
    4     690 non-null object
    5     690 non-null object
    6     690 non-null object
    7     690 non-null float64
    8     690 non-null object
    9     690 non-null object
    10    690 non-null int64
    11    690 non-null object
    12    690 non-null object
    13    690 non-null object
    14    690 non-null int64
    15    690 non-null object
    dtypes: float64(2), int64(2), object(12)
    memory usage: 86.3+ KB
    None
    
    
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>685</th>
      <td>b</td>
      <td>21.08</td>
      <td>10.085</td>
      <td>y</td>
      <td>p</td>
      <td>e</td>
      <td>h</td>
      <td>1.25</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>00260</td>
      <td>0</td>
      <td>-</td>
    </tr>
    <tr>
      <th>686</th>
      <td>a</td>
      <td>22.67</td>
      <td>0.750</td>
      <td>u</td>
      <td>g</td>
      <td>c</td>
      <td>v</td>
      <td>2.00</td>
      <td>f</td>
      <td>t</td>
      <td>2</td>
      <td>t</td>
      <td>g</td>
      <td>00200</td>
      <td>394</td>
      <td>-</td>
    </tr>
    <tr>
      <th>687</th>
      <td>a</td>
      <td>25.25</td>
      <td>13.500</td>
      <td>y</td>
      <td>p</td>
      <td>ff</td>
      <td>ff</td>
      <td>2.00</td>
      <td>f</td>
      <td>t</td>
      <td>1</td>
      <td>t</td>
      <td>g</td>
      <td>00200</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <th>688</th>
      <td>b</td>
      <td>17.92</td>
      <td>0.205</td>
      <td>u</td>
      <td>g</td>
      <td>aa</td>
      <td>v</td>
      <td>0.04</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>00280</td>
      <td>750</td>
      <td>-</td>
    </tr>
    <tr>
      <th>689</th>
      <td>b</td>
      <td>35.00</td>
      <td>3.375</td>
      <td>u</td>
      <td>g</td>
      <td>c</td>
      <td>h</td>
      <td>8.29</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>t</td>
      <td>g</td>
      <td>00000</td>
      <td>0</td>
      <td>-</td>
    </tr>
  </tbody>
</table>
</div>



## 3. Manejando los datos vacíos (part I)
<p align="justify">Hemos descubierto algunos problemas que afectarán el rendimiento de nuestros modelos de aprendizaje automático si no cambian:</p>
<ul>
<li>Nuestro conjunto de datos contiene datos numéricos y no numéricos (específicamente datos que son <code>float64</code>, <code>int64</code> y <code>object</code> types). Específicamente, las columnass 2, 7, 10 y 14 contienen valores numéricos (de tipos float64, float64, int64 y int64 respectivamente) y todas las otras columnas contienen valores no-numéricos.</li>
<li>El conjunto de datos también contiene valores de varios rangos. Algunas columnas tienen un rango de valores de 0 a 28, algunas tienen un rango de 2 a 67 y otras tienen un rango de 1017 a 100000. Aparte de estas, podemos obtener información estadística útil (como <code>mean</code>, <code>max</code>, y <code>min</code>) sobre las características que tienen valores numéricos. </li>
<li> Finalmente, el conjunto de datos tiene valores faltantes, de los cuales nos ocuparemos en esta tarea. Los valores que faltan en el conjunto de datos están etiquetados con '?', Que se puede ver en la salida de la última celda. </li>
</ul>
<p align="justify">Ahora, reemplacemos temporalmente estos signos de interrogación de valor perdido con NaN.</p>


```python
# Import numpy
import numpy as np
# Inspect missing values in the dataset
print(cc_apps.tail(17))

# Replace the '?'s with NaN
cc_apps = cc_apps.replace('?', np.nan)

# Inspect the missing values again
cc_apps.tail(17)
```

        0      1       2  3  4   5   6      7  8  9   10 11 12     13   14 15
    673  ?  29.50   2.000  y  p   e   h  2.000  f  f   0  f  g  00256   17  -
    674  a  37.33   2.500  u  g   i   h  0.210  f  f   0  f  g  00260  246  -
    675  a  41.58   1.040  u  g  aa   v  0.665  f  f   0  f  g  00240  237  -
    676  a  30.58  10.665  u  g   q   h  0.085  f  t  12  t  g  00129    3  -
    677  b  19.42   7.250  u  g   m   v  0.040  f  t   1  f  g  00100    1  -
    678  a  17.92  10.210  u  g  ff  ff  0.000  f  f   0  f  g  00000   50  -
    679  a  20.08   1.250  u  g   c   v  0.000  f  f   0  f  g  00000    0  -
    680  b  19.50   0.290  u  g   k   v  0.290  f  f   0  f  g  00280  364  -
    681  b  27.83   1.000  y  p   d   h  3.000  f  f   0  f  g  00176  537  -
    682  b  17.08   3.290  u  g   i   v  0.335  f  f   0  t  g  00140    2  -
    683  b  36.42   0.750  y  p   d   v  0.585  f  f   0  f  g  00240    3  -
    684  b  40.58   3.290  u  g   m   v  3.500  f  f   0  t  s  00400    0  -
    685  b  21.08  10.085  y  p   e   h  1.250  f  f   0  f  g  00260    0  -
    686  a  22.67   0.750  u  g   c   v  2.000  f  t   2  t  g  00200  394  -
    687  a  25.25  13.500  y  p  ff  ff  2.000  f  t   1  t  g  00200    1  -
    688  b  17.92   0.205  u  g  aa   v  0.040  f  f   0  f  g  00280  750  -
    689  b  35.00   3.375  u  g   c   h  8.290  f  f   0  t  g  00000    0  -
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>673</th>
      <td>NaN</td>
      <td>29.50</td>
      <td>2.000</td>
      <td>y</td>
      <td>p</td>
      <td>e</td>
      <td>h</td>
      <td>2.000</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>00256</td>
      <td>17</td>
      <td>-</td>
    </tr>
    <tr>
      <th>674</th>
      <td>a</td>
      <td>37.33</td>
      <td>2.500</td>
      <td>u</td>
      <td>g</td>
      <td>i</td>
      <td>h</td>
      <td>0.210</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>00260</td>
      <td>246</td>
      <td>-</td>
    </tr>
    <tr>
      <th>675</th>
      <td>a</td>
      <td>41.58</td>
      <td>1.040</td>
      <td>u</td>
      <td>g</td>
      <td>aa</td>
      <td>v</td>
      <td>0.665</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>00240</td>
      <td>237</td>
      <td>-</td>
    </tr>
    <tr>
      <th>676</th>
      <td>a</td>
      <td>30.58</td>
      <td>10.665</td>
      <td>u</td>
      <td>g</td>
      <td>q</td>
      <td>h</td>
      <td>0.085</td>
      <td>f</td>
      <td>t</td>
      <td>12</td>
      <td>t</td>
      <td>g</td>
      <td>00129</td>
      <td>3</td>
      <td>-</td>
    </tr>
    <tr>
      <th>677</th>
      <td>b</td>
      <td>19.42</td>
      <td>7.250</td>
      <td>u</td>
      <td>g</td>
      <td>m</td>
      <td>v</td>
      <td>0.040</td>
      <td>f</td>
      <td>t</td>
      <td>1</td>
      <td>f</td>
      <td>g</td>
      <td>00100</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <th>678</th>
      <td>a</td>
      <td>17.92</td>
      <td>10.210</td>
      <td>u</td>
      <td>g</td>
      <td>ff</td>
      <td>ff</td>
      <td>0.000</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>00000</td>
      <td>50</td>
      <td>-</td>
    </tr>
    <tr>
      <th>679</th>
      <td>a</td>
      <td>20.08</td>
      <td>1.250</td>
      <td>u</td>
      <td>g</td>
      <td>c</td>
      <td>v</td>
      <td>0.000</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>00000</td>
      <td>0</td>
      <td>-</td>
    </tr>
    <tr>
      <th>680</th>
      <td>b</td>
      <td>19.50</td>
      <td>0.290</td>
      <td>u</td>
      <td>g</td>
      <td>k</td>
      <td>v</td>
      <td>0.290</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>00280</td>
      <td>364</td>
      <td>-</td>
    </tr>
    <tr>
      <th>681</th>
      <td>b</td>
      <td>27.83</td>
      <td>1.000</td>
      <td>y</td>
      <td>p</td>
      <td>d</td>
      <td>h</td>
      <td>3.000</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>00176</td>
      <td>537</td>
      <td>-</td>
    </tr>
    <tr>
      <th>682</th>
      <td>b</td>
      <td>17.08</td>
      <td>3.290</td>
      <td>u</td>
      <td>g</td>
      <td>i</td>
      <td>v</td>
      <td>0.335</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>t</td>
      <td>g</td>
      <td>00140</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <th>683</th>
      <td>b</td>
      <td>36.42</td>
      <td>0.750</td>
      <td>y</td>
      <td>p</td>
      <td>d</td>
      <td>v</td>
      <td>0.585</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>00240</td>
      <td>3</td>
      <td>-</td>
    </tr>
    <tr>
      <th>684</th>
      <td>b</td>
      <td>40.58</td>
      <td>3.290</td>
      <td>u</td>
      <td>g</td>
      <td>m</td>
      <td>v</td>
      <td>3.500</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>t</td>
      <td>s</td>
      <td>00400</td>
      <td>0</td>
      <td>-</td>
    </tr>
    <tr>
      <th>685</th>
      <td>b</td>
      <td>21.08</td>
      <td>10.085</td>
      <td>y</td>
      <td>p</td>
      <td>e</td>
      <td>h</td>
      <td>1.250</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>00260</td>
      <td>0</td>
      <td>-</td>
    </tr>
    <tr>
      <th>686</th>
      <td>a</td>
      <td>22.67</td>
      <td>0.750</td>
      <td>u</td>
      <td>g</td>
      <td>c</td>
      <td>v</td>
      <td>2.000</td>
      <td>f</td>
      <td>t</td>
      <td>2</td>
      <td>t</td>
      <td>g</td>
      <td>00200</td>
      <td>394</td>
      <td>-</td>
    </tr>
    <tr>
      <th>687</th>
      <td>a</td>
      <td>25.25</td>
      <td>13.500</td>
      <td>y</td>
      <td>p</td>
      <td>ff</td>
      <td>ff</td>
      <td>2.000</td>
      <td>f</td>
      <td>t</td>
      <td>1</td>
      <td>t</td>
      <td>g</td>
      <td>00200</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <th>688</th>
      <td>b</td>
      <td>17.92</td>
      <td>0.205</td>
      <td>u</td>
      <td>g</td>
      <td>aa</td>
      <td>v</td>
      <td>0.040</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>00280</td>
      <td>750</td>
      <td>-</td>
    </tr>
    <tr>
      <th>689</th>
      <td>b</td>
      <td>35.00</td>
      <td>3.375</td>
      <td>u</td>
      <td>g</td>
      <td>c</td>
      <td>h</td>
      <td>8.290</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>t</td>
      <td>g</td>
      <td>00000</td>
      <td>0</td>
      <td>-</td>
    </tr>
  </tbody>
</table>
</div>



## 4.  Manejando los datos vacíos (part II)
<p align="justify"> Reemplazamos todos los signos de interrogación con NaNs. Esto nos ayudará en el próximo tratamiento de los valores perdidos que vamos a realizar. </p>
<p align="justify"> Una pregunta importante que surge aquí es <em> ¿por qué le damos tanta importancia a los valores perdidos </em>? ¿No pueden ser ignorados? Ignorar los valores perdidos puede afectar en gran medida el rendimiento de un modelo de aprendizaje automático. Si ignoramos los valores faltantes, nuestro modelo de aprendizaje automático puede perder información sobre el conjunto de datos que puede ser útil para su capacitación. </p>
<p align="justify"> Entonces, para evitar este problema, vamos a imputar los valores faltantes con una estrategia llamada imputación media. </p>


```python
# Impute the missing values with mean imputation
cc_apps.fillna(cc_apps.mean(), inplace=True)

# Count the number of NaNs in the dataset to verify
cc_apps.isnull().sum()
```




    0     12
    1     12
    2      0
    3      6
    4      6
    5      9
    6      9
    7      0
    8      0
    9      0
    10     0
    11     0
    12     0
    13    13
    14     0
    15     0
    dtype: int64



## 5.  Manejando los datos vacíos (part III)
<p align="justify"> Nos hemos ocupado con éxito de los valores perdidos que están en las columnas numéricas. Todavía hay algunos datos vacíos para ser imputados en las columnas 0, 1, 3, 4, 5, 6 y 13. Todas estas columnas contienen datos no numéricos y es por eso que la estrategia de imputación media no funcionaría aquí. Esto necesita un tratamiento diferente. </p>
<p align="justify"> Vamos a imputar estos valores perdidos con los valores más frecuentes que están presentes en las columnas respectivas. Esto es <a href="https://www.datacamp.com/community/tutorials/categorical-data"> una buena práctica</a> cuando se trata de imputar valores perdidos para datos categóricos en general. </p>


```python
# Iterate over each column of cc_apps
for col in cc_apps:
    # Check if the column is of object type
    if cc_apps[col].dtype == 'object':
        # Impute with the most frequent value
        cc_apps = cc_apps.fillna(cc_apps[col].value_counts()[0])

# Count the number of NaNs in the dataset and print the counts to verify
cc_apps.isna().sum()
```




    0     0
    1     0
    2     0
    3     0
    4     0
    5     0
    6     0
    7     0
    8     0
    9     0
    10    0
    11    0
    12    0
    13    0
    14    0
    15    0
    dtype: int64



## 6. Preprocesando los datos (part I)
<p align="justify"> Los valores perdidos ahora son manejados con éxito. </p>
<p align="justify"> Todavía es necesario un preprocesamiento de datos menor pero esencial antes de proceder a construir nuestro modelo de aprendizaje automático. Vamos a dividir estos pasos de preprocesamiento restantes en tres tareas principales: </p>
<ol>
<li> Convertir los datos no numéricos en numéricos. </li>
<li> Dividir los datos en dos conjuntos de entrenamiento y prueba. </li>
<li> Escalar los valores de la característica a un rango uniforme. </li>
</ol>
<p align="justify"> Primero, convertiremos todos los valores no numéricos en valores numéricos. Hacemos esto porque no solo da como resultado un cálculo más rápido, sino que también muchos modelos de aprendizaje automático (como XGBoost) (y especialmente los que son desarrollados usando scikit-learn) requieren que los datos estén en un formato estrictamente numérico. Lo haremos utilizando una técnica llamada <a href="http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html"> <em>label encoding</em> </a>. </p>


```python
# Import LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Instantiate LabelEncoder

le = LabelEncoder()

# Iterate over all the values of each column and extract their dtypes
for col in cc_apps.columns.values:
    # Compare if the dtype is object
    if cc_apps[col].dtype=='object':
    # Use LabelEncoder to do the numeric transformation
        cc_apps[col]=le.fit_transform(cc_apps[col].astype(str))
        
cc_apps.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 690 entries, 0 to 689
    Data columns (total 16 columns):
    0     690 non-null int64
    1     690 non-null int64
    2     690 non-null float64
    3     690 non-null int64
    4     690 non-null int64
    5     690 non-null int64
    6     690 non-null int64
    7     690 non-null float64
    8     690 non-null int64
    9     690 non-null int64
    10    690 non-null int64
    11    690 non-null int64
    12    690 non-null int64
    13    690 non-null int64
    14    690 non-null int64
    15    690 non-null int64
    dtypes: float64(2), int64(14)
    memory usage: 86.3 KB
    

## 7. Separando el conjunto de datos en dos: entrenamiento y prueba
<p align="justify"> Hemos convertido con éxito todos los valores no numéricos en valores numéricos. </p>
<p align="justify"> Ahora, dividiremos nuestros datos en un conjunto de entrenamiento y un conjunto de prueba para preparar nuestros datos para dos fases diferentes de modelado de aprendizaje automático: capacitación y pruebas. Idealmente, ninguna información de los datos de la prueba debería usarse para escalar los datos de entrenamiento o para dirigir el proceso de entrenamiento de un modelo de aprendizaje automático. Por lo tanto, primero dividimos los datos y luego aplicamos la escala. </p>
<p align="justify"> Además, características como <code> DriversLicense </code> y <code> ZipCode </code> no son tan importantes como las otras características del conjunto de datos para predecir las aprobaciones de tarjetas de crédito. No deberíamos considerarlos para diseñar nuestro modelo de aprendizaje automático con el mejor conjunto de variables. En la literatura de Data Science, esto a menudo se conoce como <em> feature selection </em>. </p>


```python
# Import train_test_split
from sklearn.model_selection import train_test_split
# Drop the features 11 and 13 and convert the DataFrame to a NumPy array
cc_apps = cc_apps.drop([11, 13], axis=1)
cc_apps = cc_apps.values

# Segregate features and labels into separate variables
X,y = cc_apps[:,0:12] , cc_apps[:,13]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                y,
                                test_size=0.33,
                                random_state=42)
```

## 8. Preprocesando los datos (part II)
<p align="justify"> Los datos ahora se dividen en dos conjuntos separados: conjuntos de entrenamiento y prueba, respectivamente. Solo nos queda un paso final que es el escalado antes de que podamos aplicar un modelo de aprendizaje automático a los datos. </p>
<p align="justify"> Ahora, intentemos comprender qué significan estos valores escalados en el mundo real. Usemos la variable <code> CreditScore </code> como ejemplo. La calificación crediticia de una persona es su solvencia en función de su historial crediticio. Cuanto mayor sea su solvencia, más confiable financieramente se considera que una persona es. Entonces, un <code> CreditScore </code> de 1 es el más alto ya que estamos reescalando todos los valores al rango de 0-1. </p>


```python
# Import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
# Instantiate MinMaxScaler and use it to rescale X_train and X_test
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.fit_transform(X_test)
```

## 9. Ajustando un modelo de regresión logística al conjunto de entrenamiento.
<p align="justify"> Esencialmente, predecir si una solicitud de tarjeta de crédito será aprobada o no es una tarea <a href="https://en.wikipedia.org/wiki/Statistical_classification">  de clasificación </a>. <a href="http://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.names"> Según el UCI </a>, nuestro conjunto de datos contiene más instancias que corresponden al estado "Denegado" que las instancias correspondientes al estado "Aprobado". Específicamente, de 690 instancias, hay 383 (55.5%) solicitudes que fueron rechazadas y 307 (44.5%) solicitudes que fueron aprobadas. </p>
<p align="justify"> Esto nos da un punto de referencia. Un buen modelo de aprendizaje automático debería poder predecir con precisión el estado de las solicitudes con respecto a estas estadísticas. </p>
<p align="justify"> ¿Qué modelo debemos elegir? Una pregunta que debe hacerse es: <em> ¿Las variables que afectan el proceso de decisión de aprobación de la tarjeta de crédito están correlacionadas entre sí? </em> Aunque podemos medir la correlación, eso está fuera del alcance de este proyecto, por lo que confiaremos en nuestra intuición de que de hecho están correlacionadas por ahora. Debido a esta correlación, aprovecharemos el hecho de que los modelos lineales generalizados funcionan bien en estos casos. Comencemos nuestro proceso de modelación de aprendizaje automático con un modelo de Regresión Logística (un modelo lineal generalizado). </p>


```python
# Import LogisticRegression
from sklearn.linear_model import LogisticRegression

# Instantiate a LogisticRegression classifier with default parameter values
logreg = LogisticRegression()

# Fit logreg to the train set
logreg.fit(rescaledX_train, y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



## 10. Haciendo predicciones y evaluando el rendimiento
<p align="justify"> ¿Pero qué tan bien funciona nuestro modelo? </p>
<p align="justify"> Ahora evaluaremos nuestro modelo en el conjunto de prueba con la <a href="https://developers.google.com/machine-learning/crash-course/classification/accuracy"> precisión de clasificación </a> . Pero también analizaremos la <a href="http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/"> matriz de confusión </a> del modelo. En el caso de la predicción de solicitudes de tarjetas de crédito, es igualmente importante ver si nuestro modelo de aprendizaje automático puede predecir la aprobación de las solicitudes que originalmente fueron denegadas. Si nuestro modelo no funciona bien en este aspecto, entonces podría terminar denegando la solicitud que debería haber sido aprobada. La matriz de confusión nos ayuda a ver el rendimiento de nuestro modelo desde estos aspectos. </p>


```python
# Import confusion_matrix
from sklearn.metrics import confusion_matrix
# Use logreg to predict instances from the test set and store it
y_pred = logreg.predict(rescaledX_test)

# Get the accuracy score of logreg model and print it
print("Accuracy of logistic regression classifier: ", logreg.score(rescaledX_test,y_test))

# Print the confusion matrix of the logreg model
confusion_matrix(y_test, y_pred)
```

    Accuracy of logistic regression classifier:  0.8377192982456141
    
    array([[92, 11],
           [26, 99]])



## 11. Grid searching y haciendo que el modelo funcione mejor
<p align="justify"> ¡Nuestro modelo es bastante bueno! Es capaz de obtener una precisión de casi el 84%. </p>
<p align="justify"> Para la matriz de confusión, el primer elemento de la primera fila denota los verdaderos negativos que significan el número de instancias negativas (solicitudes denegadas) predichas por el modelo correctamente. Y el último elemento de la segunda fila de la matriz de confusión denota los verdaderos positivos que significan el número de instancias positivas (solicitudes aprobadas) predichas por el modelo correctamente. </p>
<p align="justify"> Veamos si podemos hacerlo mejor. Podemos realizar un <a href="https://machinelearningmastery.com/how-to-tune-algorithm-parameters-with-scikit-learn/"> Grid searching </a> de los parámetros del modelo para mejorar su capacidad de predecir aprobaciones de tarjetas de crédito. </p>
<p align="justify"> <a href="http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html"> La implementación de regresión logística de scikit-learn </a> consta de diferentes hiperparámetros, pero nosotros buscaremos en el Grid Search los dos siguientes: </p>
<ul>
<li>tol</li>
<li>max_iter</li>
</ul>


```python
# Import GridSearchCV
from sklearn.model_selection import GridSearchCV
# Define the grid of values for tol and max_iter
tol = [0.01,0.001,0.0001]
max_iter = [100,150,200]

# Create a dictionary where tol and max_iter are keys and the lists of their values are corresponding values
param_grid = dict(tol=tol, max_iter=max_iter)
```

## 12. Encontrando el modelo con el mejor rendimiento
<p align="justify"> Hemos definido la matriz de valores de hiperparámetros y los hemos convertido a un único formato de diccionario que <code> GridSearchCV () </code> espera como uno de sus parámetros. Ahora, comenzaremos la búsqueda en la matriz para ver qué valores funcionan mejor. </p>
<p align="justify"> Aplicaremos <code> GridSearchCV () </code> con nuestro modelo anterior <code> logreg </code> con todos los datos que tenemos. En lugar de pasar el entrenamiento y el conjunto de prueba por separado, suministraremos <code> X </code> (versión a escala) y la variable explicada <code> y </code>. También le indicaremos a <code> GridSearchCV () </code> que realice una <a href="https://www.dataschool.io/machine-learning-with-scikit-learn/"> validación cruzada </a> de cinco instancias. </p>
<p align="justify"> Terminaremos el proyecto almacenando el puntaje mejor logrado y los mejores parámetros respectivos. </p>
<p align="justify"> Al construir este predictor de tarjeta de crédito, abordamos algunos de los pasos de preprocesamiento más conocidos, como <strong> scaling </strong>, <strong> label encoding </strong> y <strong> missing value imputation </strong>. Terminamos con algo de <strong> aprendizaje automático </strong> para predecir si la solicitud de una tarjeta de crédito de una persona sería aprobada o no dado la información sobre esa persona.</p> 


```python
# Instantiate GridSearchCV with the required parameters
grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)

# Use scaler to rescale X and assign it to rescaledX
rescaledX = scaler.fit_transform(X)

# Fit data to grid_model
grid_model_result = grid_model.fit(rescaledX, y)

# Summarize results
best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_
print("Best: %f using %s" % (best_score, best_params))
```

    Best: 0.850725 using {'tol': 0.01, 'max_iter': 100}
    
<p align="justify">Como vemos, terminamos con una precisión del 85% y que los mejores parámetros para <code>tol</code> y <code>max_iter</code> son 0.01 y 100, respectivamente</p>
