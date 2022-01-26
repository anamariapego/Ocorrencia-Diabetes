<h1><center>Projeto Prevendo a Ocorrência de Diabetes</center></h1>

<center>
    <img src="diabetes.png" width = 500px/> 
 </center> 

<br>

Aplicar técnicas de Machine Learning que possa prever se as pessoas têm diabetes quando suas características são especificadas. 
O conjunto de dados faz parte do grande conjunto de dados realizado no National Institutes of Diabetes-Digestive-Kidney Diseases nos EUA. Dados usados para pesquisa de diabetes em mulheres indianas Pima com 21 anos ou mais que vivem em Phoenix, a 5ª maior cidade do Estado do Arizona, nos EUA. A variável de destino é especificada como "resultado"; 1 indica resultado positivo do teste de diabetes, 0 indica negativo.

O conjunto de dados foram coletados do Repositório de Machine Learning da UCI / Kaggle
https://www.kaggle.com/uciml/pima-indians-diabetes-database/data

Descrição das colunas:

- **Pregnancies**: Número de vezes que está grávida

- **Glucose**: Concentração de glicose plasmática a 2 horas em um teste oral de tolerância à glicose

- **BloodPressure**: Pressão arterial diastólica (mm Hg)

- **SkinThickness**: Espessura da dobra cutânea do tríceps (mm)

- **Insulin**: Insulina sérica de 2 horas (mu U/ml)

- **BMI**: Índice de massa corporal (peso em kg/(altura em m)^2)

- **DiabetesPedigreeFunction**: Função hereditária do diabetes

- **Age**: Anos de idade)

- **Outcome**: Variável de classe (0 ou 1) 268 de 768 são 1, as outras são 0




```python
# Carregando as bibliotecas

import pandas as pd      
import matplotlib as mat
import matplotlib.pyplot as plt    
import numpy as np   
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
import pickle
%matplotlib inline   
```


```python
# Carregando o dataset

dataset = pd.read_csv("pima-data.csv")   
```


```python
# Primeiras linhas

dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_preg</th>
      <th>glucose_conc</th>
      <th>diastolic_bp</th>
      <th>thickness</th>
      <th>insulin</th>
      <th>bmi</th>
      <th>diab_pred</th>
      <th>age</th>
      <th>skin</th>
      <th>diabetes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1.3780</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>1.1426</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>0.0000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89</td>
      <td>66</td>
      <td>23</td>
      <td>94</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
      <td>0.9062</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137</td>
      <td>40</td>
      <td>35</td>
      <td>168</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
      <td>1.3790</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Informações gerais dos dados

print("------------ Shape ------------")
print(dataset.shape)

print("------------ Types ------------")
print(dataset.dtypes)

print("------------ NAs ------------")
print(dataset.isnull().sum())
```

    ------------ Shape ------------
    (768, 10)
    ------------ Types ------------
    num_preg          int64
    glucose_conc      int64
    diastolic_bp      int64
    thickness         int64
    insulin           int64
    bmi             float64
    diab_pred       float64
    age               int64
    skin            float64
    diabetes           bool
    dtype: object
    ------------ NAs ------------
    num_preg        0
    glucose_conc    0
    diastolic_bp    0
    thickness       0
    insulin         0
    bmi             0
    diab_pred       0
    age             0
    skin            0
    diabetes        0
    dtype: int64
    


```python
# Definindo as clases 

classes = {True : 1, False : 0}

dataset["diabetes"] = dataset["diabetes"].map(classes)

dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_preg</th>
      <th>glucose_conc</th>
      <th>diastolic_bp</th>
      <th>thickness</th>
      <th>insulin</th>
      <th>bmi</th>
      <th>diab_pred</th>
      <th>age</th>
      <th>skin</th>
      <th>diabetes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1.3780</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>1.1426</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>0.0000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89</td>
      <td>66</td>
      <td>23</td>
      <td>94</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
      <td>0.9062</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137</td>
      <td>40</td>
      <td>35</td>
      <td>168</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
      <td>1.3790</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Identificando a correlação entre as variáveis via tabela

dataset.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_preg</th>
      <th>glucose_conc</th>
      <th>diastolic_bp</th>
      <th>thickness</th>
      <th>insulin</th>
      <th>bmi</th>
      <th>diab_pred</th>
      <th>age</th>
      <th>skin</th>
      <th>diabetes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>num_preg</th>
      <td>1.000000</td>
      <td>0.129459</td>
      <td>0.141282</td>
      <td>-0.081672</td>
      <td>-0.073535</td>
      <td>0.017683</td>
      <td>-0.033523</td>
      <td>0.544341</td>
      <td>-0.081673</td>
      <td>0.221898</td>
    </tr>
    <tr>
      <th>glucose_conc</th>
      <td>0.129459</td>
      <td>1.000000</td>
      <td>0.152590</td>
      <td>0.057328</td>
      <td>0.331357</td>
      <td>0.221071</td>
      <td>0.137337</td>
      <td>0.263514</td>
      <td>0.057326</td>
      <td>0.466581</td>
    </tr>
    <tr>
      <th>diastolic_bp</th>
      <td>0.141282</td>
      <td>0.152590</td>
      <td>1.000000</td>
      <td>0.207371</td>
      <td>0.088933</td>
      <td>0.281805</td>
      <td>0.041265</td>
      <td>0.239528</td>
      <td>0.207371</td>
      <td>0.065068</td>
    </tr>
    <tr>
      <th>thickness</th>
      <td>-0.081672</td>
      <td>0.057328</td>
      <td>0.207371</td>
      <td>1.000000</td>
      <td>0.436783</td>
      <td>0.392573</td>
      <td>0.183928</td>
      <td>-0.113970</td>
      <td>1.000000</td>
      <td>0.074752</td>
    </tr>
    <tr>
      <th>insulin</th>
      <td>-0.073535</td>
      <td>0.331357</td>
      <td>0.088933</td>
      <td>0.436783</td>
      <td>1.000000</td>
      <td>0.197859</td>
      <td>0.185071</td>
      <td>-0.042163</td>
      <td>0.436785</td>
      <td>0.130548</td>
    </tr>
    <tr>
      <th>bmi</th>
      <td>0.017683</td>
      <td>0.221071</td>
      <td>0.281805</td>
      <td>0.392573</td>
      <td>0.197859</td>
      <td>1.000000</td>
      <td>0.140647</td>
      <td>0.036242</td>
      <td>0.392574</td>
      <td>0.292695</td>
    </tr>
    <tr>
      <th>diab_pred</th>
      <td>-0.033523</td>
      <td>0.137337</td>
      <td>0.041265</td>
      <td>0.183928</td>
      <td>0.185071</td>
      <td>0.140647</td>
      <td>1.000000</td>
      <td>0.033561</td>
      <td>0.183927</td>
      <td>0.173844</td>
    </tr>
    <tr>
      <th>age</th>
      <td>0.544341</td>
      <td>0.263514</td>
      <td>0.239528</td>
      <td>-0.113970</td>
      <td>-0.042163</td>
      <td>0.036242</td>
      <td>0.033561</td>
      <td>1.000000</td>
      <td>-0.113973</td>
      <td>0.238356</td>
    </tr>
    <tr>
      <th>skin</th>
      <td>-0.081673</td>
      <td>0.057326</td>
      <td>0.207371</td>
      <td>1.000000</td>
      <td>0.436785</td>
      <td>0.392574</td>
      <td>0.183927</td>
      <td>-0.113973</td>
      <td>1.000000</td>
      <td>0.074750</td>
    </tr>
    <tr>
      <th>diabetes</th>
      <td>0.221898</td>
      <td>0.466581</td>
      <td>0.065068</td>
      <td>0.074752</td>
      <td>0.130548</td>
      <td>0.292695</td>
      <td>0.173844</td>
      <td>0.238356</td>
      <td>0.074750</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Correlação entre as variáveis via gráfico

dataset.corr().style.background_gradient(cmap='coolwarm').set_precision(2)
```




<style  type="text/css" >
#T_94b54_row0_col0,#T_94b54_row1_col1,#T_94b54_row2_col2,#T_94b54_row3_col3,#T_94b54_row3_col8,#T_94b54_row4_col4,#T_94b54_row5_col5,#T_94b54_row6_col6,#T_94b54_row7_col7,#T_94b54_row8_col3,#T_94b54_row8_col8,#T_94b54_row9_col9{
            background-color:  #b40426;
            color:  #f1f1f1;
        }#T_94b54_row0_col1{
            background-color:  #516ddb;
            color:  #000000;
        }#T_94b54_row0_col2{
            background-color:  #5a78e4;
            color:  #000000;
        }#T_94b54_row0_col3,#T_94b54_row0_col8,#T_94b54_row7_col4{
            background-color:  #4358cb;
            color:  #f1f1f1;
        }#T_94b54_row0_col4,#T_94b54_row0_col5,#T_94b54_row0_col6,#T_94b54_row2_col9,#T_94b54_row3_col0,#T_94b54_row3_col1,#T_94b54_row3_col7,#T_94b54_row6_col2,#T_94b54_row7_col3,#T_94b54_row7_col8,#T_94b54_row8_col0,#T_94b54_row8_col1,#T_94b54_row8_col7{
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }#T_94b54_row0_col7{
            background-color:  #f1cdba;
            color:  #000000;
        }#T_94b54_row0_col9,#T_94b54_row1_col6{
            background-color:  #6f92f3;
            color:  #000000;
        }#T_94b54_row1_col0{
            background-color:  #799cf8;
            color:  #000000;
        }#T_94b54_row1_col2,#T_94b54_row6_col9{
            background-color:  #5e7de7;
            color:  #000000;
        }#T_94b54_row1_col3,#T_94b54_row1_col8{
            background-color:  #6b8df0;
            color:  #000000;
        }#T_94b54_row1_col4{
            background-color:  #b9d0f9;
            color:  #000000;
        }#T_94b54_row1_col5,#T_94b54_row3_col6,#T_94b54_row8_col6{
            background-color:  #7ea1fa;
            color:  #000000;
        }#T_94b54_row1_col7{
            background-color:  #abc8fd;
            color:  #000000;
        }#T_94b54_row1_col9{
            background-color:  #c9d7f0;
            color:  #000000;
        }#T_94b54_row2_col0,#T_94b54_row7_col2{
            background-color:  #7da0f9;
            color:  #000000;
        }#T_94b54_row2_col1{
            background-color:  #5977e3;
            color:  #000000;
        }#T_94b54_row2_col3,#T_94b54_row2_col8{
            background-color:  #9abbff;
            color:  #000000;
        }#T_94b54_row2_col4{
            background-color:  #6a8bef;
            color:  #000000;
        }#T_94b54_row2_col5,#T_94b54_row6_col3,#T_94b54_row6_col8{
            background-color:  #93b5fe;
            color:  #000000;
        }#T_94b54_row2_col6{
            background-color:  #506bda;
            color:  #000000;
        }#T_94b54_row2_col7{
            background-color:  #a5c3fe;
            color:  #000000;
        }#T_94b54_row3_col2,#T_94b54_row5_col1,#T_94b54_row8_col2{
            background-color:  #7295f4;
            color:  #000000;
        }#T_94b54_row3_col4,#T_94b54_row8_col4{
            background-color:  #d6dce4;
            color:  #000000;
        }#T_94b54_row3_col5,#T_94b54_row8_col5{
            background-color:  #bad0f8;
            color:  #000000;
        }#T_94b54_row3_col9,#T_94b54_row8_col9{
            background-color:  #3d50c3;
            color:  #f1f1f1;
        }#T_94b54_row4_col0{
            background-color:  #3c4ec2;
            color:  #f1f1f1;
        }#T_94b54_row4_col1{
            background-color:  #9bbcff;
            color:  #000000;
        }#T_94b54_row4_col2{
            background-color:  #4961d2;
            color:  #f1f1f1;
        }#T_94b54_row4_col3,#T_94b54_row4_col8{
            background-color:  #dbdcde;
            color:  #000000;
        }#T_94b54_row4_col5{
            background-color:  #7597f6;
            color:  #000000;
        }#T_94b54_row4_col6{
            background-color:  #80a3fa;
            color:  #000000;
        }#T_94b54_row4_col7,#T_94b54_row7_col6{
            background-color:  #4e68d8;
            color:  #000000;
        }#T_94b54_row4_col9{
            background-color:  #4f69d9;
            color:  #000000;
        }#T_94b54_row5_col0{
            background-color:  #5673e0;
            color:  #000000;
        }#T_94b54_row5_col2,#T_94b54_row5_col4{
            background-color:  #8db0fe;
            color:  #000000;
        }#T_94b54_row5_col3,#T_94b54_row5_col8{
            background-color:  #d1dae9;
            color:  #000000;
        }#T_94b54_row5_col6,#T_94b54_row9_col3,#T_94b54_row9_col8{
            background-color:  #7093f3;
            color:  #000000;
        }#T_94b54_row5_col7{
            background-color:  #6485ec;
            color:  #000000;
        }#T_94b54_row5_col9{
            background-color:  #8badfd;
            color:  #000000;
        }#T_94b54_row6_col0{
            background-color:  #485fd1;
            color:  #f1f1f1;
        }#T_94b54_row6_col1{
            background-color:  #5470de;
            color:  #000000;
        }#T_94b54_row6_col4{
            background-color:  #89acfd;
            color:  #000000;
        }#T_94b54_row6_col5{
            background-color:  #6282ea;
            color:  #000000;
        }#T_94b54_row6_col7{
            background-color:  #6384eb;
            color:  #000000;
        }#T_94b54_row7_col0{
            background-color:  #efcfbf;
            color:  #000000;
        }#T_94b54_row7_col1{
            background-color:  #81a4fb;
            color:  #000000;
        }#T_94b54_row7_col5{
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }#T_94b54_row7_col9{
            background-color:  #7699f6;
            color:  #000000;
        }#T_94b54_row9_col0,#T_94b54_row9_col5{
            background-color:  #97b8ff;
            color:  #000000;
        }#T_94b54_row9_col1{
            background-color:  #cbd8ee;
            color:  #000000;
        }#T_94b54_row9_col2{
            background-color:  #4257c9;
            color:  #f1f1f1;
        }#T_94b54_row9_col4{
            background-color:  #779af7;
            color:  #000000;
        }#T_94b54_row9_col6{
            background-color:  #7b9ff9;
            color:  #000000;
        }#T_94b54_row9_col7{
            background-color:  #a3c2fe;
            color:  #000000;
        }</style><table id="T_94b54_" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >num_preg</th>        <th class="col_heading level0 col1" >glucose_conc</th>        <th class="col_heading level0 col2" >diastolic_bp</th>        <th class="col_heading level0 col3" >thickness</th>        <th class="col_heading level0 col4" >insulin</th>        <th class="col_heading level0 col5" >bmi</th>        <th class="col_heading level0 col6" >diab_pred</th>        <th class="col_heading level0 col7" >age</th>        <th class="col_heading level0 col8" >skin</th>        <th class="col_heading level0 col9" >diabetes</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_94b54_level0_row0" class="row_heading level0 row0" >num_preg</th>
                        <td id="T_94b54_row0_col0" class="data row0 col0" >1.00</td>
                        <td id="T_94b54_row0_col1" class="data row0 col1" >0.13</td>
                        <td id="T_94b54_row0_col2" class="data row0 col2" >0.14</td>
                        <td id="T_94b54_row0_col3" class="data row0 col3" >-0.08</td>
                        <td id="T_94b54_row0_col4" class="data row0 col4" >-0.07</td>
                        <td id="T_94b54_row0_col5" class="data row0 col5" >0.02</td>
                        <td id="T_94b54_row0_col6" class="data row0 col6" >-0.03</td>
                        <td id="T_94b54_row0_col7" class="data row0 col7" >0.54</td>
                        <td id="T_94b54_row0_col8" class="data row0 col8" >-0.08</td>
                        <td id="T_94b54_row0_col9" class="data row0 col9" >0.22</td>
            </tr>
            <tr>
                        <th id="T_94b54_level0_row1" class="row_heading level0 row1" >glucose_conc</th>
                        <td id="T_94b54_row1_col0" class="data row1 col0" >0.13</td>
                        <td id="T_94b54_row1_col1" class="data row1 col1" >1.00</td>
                        <td id="T_94b54_row1_col2" class="data row1 col2" >0.15</td>
                        <td id="T_94b54_row1_col3" class="data row1 col3" >0.06</td>
                        <td id="T_94b54_row1_col4" class="data row1 col4" >0.33</td>
                        <td id="T_94b54_row1_col5" class="data row1 col5" >0.22</td>
                        <td id="T_94b54_row1_col6" class="data row1 col6" >0.14</td>
                        <td id="T_94b54_row1_col7" class="data row1 col7" >0.26</td>
                        <td id="T_94b54_row1_col8" class="data row1 col8" >0.06</td>
                        <td id="T_94b54_row1_col9" class="data row1 col9" >0.47</td>
            </tr>
            <tr>
                        <th id="T_94b54_level0_row2" class="row_heading level0 row2" >diastolic_bp</th>
                        <td id="T_94b54_row2_col0" class="data row2 col0" >0.14</td>
                        <td id="T_94b54_row2_col1" class="data row2 col1" >0.15</td>
                        <td id="T_94b54_row2_col2" class="data row2 col2" >1.00</td>
                        <td id="T_94b54_row2_col3" class="data row2 col3" >0.21</td>
                        <td id="T_94b54_row2_col4" class="data row2 col4" >0.09</td>
                        <td id="T_94b54_row2_col5" class="data row2 col5" >0.28</td>
                        <td id="T_94b54_row2_col6" class="data row2 col6" >0.04</td>
                        <td id="T_94b54_row2_col7" class="data row2 col7" >0.24</td>
                        <td id="T_94b54_row2_col8" class="data row2 col8" >0.21</td>
                        <td id="T_94b54_row2_col9" class="data row2 col9" >0.07</td>
            </tr>
            <tr>
                        <th id="T_94b54_level0_row3" class="row_heading level0 row3" >thickness</th>
                        <td id="T_94b54_row3_col0" class="data row3 col0" >-0.08</td>
                        <td id="T_94b54_row3_col1" class="data row3 col1" >0.06</td>
                        <td id="T_94b54_row3_col2" class="data row3 col2" >0.21</td>
                        <td id="T_94b54_row3_col3" class="data row3 col3" >1.00</td>
                        <td id="T_94b54_row3_col4" class="data row3 col4" >0.44</td>
                        <td id="T_94b54_row3_col5" class="data row3 col5" >0.39</td>
                        <td id="T_94b54_row3_col6" class="data row3 col6" >0.18</td>
                        <td id="T_94b54_row3_col7" class="data row3 col7" >-0.11</td>
                        <td id="T_94b54_row3_col8" class="data row3 col8" >1.00</td>
                        <td id="T_94b54_row3_col9" class="data row3 col9" >0.07</td>
            </tr>
            <tr>
                        <th id="T_94b54_level0_row4" class="row_heading level0 row4" >insulin</th>
                        <td id="T_94b54_row4_col0" class="data row4 col0" >-0.07</td>
                        <td id="T_94b54_row4_col1" class="data row4 col1" >0.33</td>
                        <td id="T_94b54_row4_col2" class="data row4 col2" >0.09</td>
                        <td id="T_94b54_row4_col3" class="data row4 col3" >0.44</td>
                        <td id="T_94b54_row4_col4" class="data row4 col4" >1.00</td>
                        <td id="T_94b54_row4_col5" class="data row4 col5" >0.20</td>
                        <td id="T_94b54_row4_col6" class="data row4 col6" >0.19</td>
                        <td id="T_94b54_row4_col7" class="data row4 col7" >-0.04</td>
                        <td id="T_94b54_row4_col8" class="data row4 col8" >0.44</td>
                        <td id="T_94b54_row4_col9" class="data row4 col9" >0.13</td>
            </tr>
            <tr>
                        <th id="T_94b54_level0_row5" class="row_heading level0 row5" >bmi</th>
                        <td id="T_94b54_row5_col0" class="data row5 col0" >0.02</td>
                        <td id="T_94b54_row5_col1" class="data row5 col1" >0.22</td>
                        <td id="T_94b54_row5_col2" class="data row5 col2" >0.28</td>
                        <td id="T_94b54_row5_col3" class="data row5 col3" >0.39</td>
                        <td id="T_94b54_row5_col4" class="data row5 col4" >0.20</td>
                        <td id="T_94b54_row5_col5" class="data row5 col5" >1.00</td>
                        <td id="T_94b54_row5_col6" class="data row5 col6" >0.14</td>
                        <td id="T_94b54_row5_col7" class="data row5 col7" >0.04</td>
                        <td id="T_94b54_row5_col8" class="data row5 col8" >0.39</td>
                        <td id="T_94b54_row5_col9" class="data row5 col9" >0.29</td>
            </tr>
            <tr>
                        <th id="T_94b54_level0_row6" class="row_heading level0 row6" >diab_pred</th>
                        <td id="T_94b54_row6_col0" class="data row6 col0" >-0.03</td>
                        <td id="T_94b54_row6_col1" class="data row6 col1" >0.14</td>
                        <td id="T_94b54_row6_col2" class="data row6 col2" >0.04</td>
                        <td id="T_94b54_row6_col3" class="data row6 col3" >0.18</td>
                        <td id="T_94b54_row6_col4" class="data row6 col4" >0.19</td>
                        <td id="T_94b54_row6_col5" class="data row6 col5" >0.14</td>
                        <td id="T_94b54_row6_col6" class="data row6 col6" >1.00</td>
                        <td id="T_94b54_row6_col7" class="data row6 col7" >0.03</td>
                        <td id="T_94b54_row6_col8" class="data row6 col8" >0.18</td>
                        <td id="T_94b54_row6_col9" class="data row6 col9" >0.17</td>
            </tr>
            <tr>
                        <th id="T_94b54_level0_row7" class="row_heading level0 row7" >age</th>
                        <td id="T_94b54_row7_col0" class="data row7 col0" >0.54</td>
                        <td id="T_94b54_row7_col1" class="data row7 col1" >0.26</td>
                        <td id="T_94b54_row7_col2" class="data row7 col2" >0.24</td>
                        <td id="T_94b54_row7_col3" class="data row7 col3" >-0.11</td>
                        <td id="T_94b54_row7_col4" class="data row7 col4" >-0.04</td>
                        <td id="T_94b54_row7_col5" class="data row7 col5" >0.04</td>
                        <td id="T_94b54_row7_col6" class="data row7 col6" >0.03</td>
                        <td id="T_94b54_row7_col7" class="data row7 col7" >1.00</td>
                        <td id="T_94b54_row7_col8" class="data row7 col8" >-0.11</td>
                        <td id="T_94b54_row7_col9" class="data row7 col9" >0.24</td>
            </tr>
            <tr>
                        <th id="T_94b54_level0_row8" class="row_heading level0 row8" >skin</th>
                        <td id="T_94b54_row8_col0" class="data row8 col0" >-0.08</td>
                        <td id="T_94b54_row8_col1" class="data row8 col1" >0.06</td>
                        <td id="T_94b54_row8_col2" class="data row8 col2" >0.21</td>
                        <td id="T_94b54_row8_col3" class="data row8 col3" >1.00</td>
                        <td id="T_94b54_row8_col4" class="data row8 col4" >0.44</td>
                        <td id="T_94b54_row8_col5" class="data row8 col5" >0.39</td>
                        <td id="T_94b54_row8_col6" class="data row8 col6" >0.18</td>
                        <td id="T_94b54_row8_col7" class="data row8 col7" >-0.11</td>
                        <td id="T_94b54_row8_col8" class="data row8 col8" >1.00</td>
                        <td id="T_94b54_row8_col9" class="data row8 col9" >0.07</td>
            </tr>
            <tr>
                        <th id="T_94b54_level0_row9" class="row_heading level0 row9" >diabetes</th>
                        <td id="T_94b54_row9_col0" class="data row9 col0" >0.22</td>
                        <td id="T_94b54_row9_col1" class="data row9 col1" >0.47</td>
                        <td id="T_94b54_row9_col2" class="data row9 col2" >0.07</td>
                        <td id="T_94b54_row9_col3" class="data row9 col3" >0.07</td>
                        <td id="T_94b54_row9_col4" class="data row9 col4" >0.13</td>
                        <td id="T_94b54_row9_col5" class="data row9 col5" >0.29</td>
                        <td id="T_94b54_row9_col6" class="data row9 col6" >0.17</td>
                        <td id="T_94b54_row9_col7" class="data row9 col7" >0.24</td>
                        <td id="T_94b54_row9_col8" class="data row9 col8" >0.07</td>
                        <td id="T_94b54_row9_col9" class="data row9 col9" >1.00</td>
            </tr>
    </tbody></table>




```python
# Verificando a distribuição dos dados

num_true = len(dataset.loc[dataset['diabetes'] == True])
num_false = len(dataset.loc[dataset['diabetes'] == False])
print("Número de Casos Verdadeiros: {0} ({1:2.2f}%)".format(num_true, (num_true/ (num_true + num_false)) * 100))
print("Número de Casos Falsos     : {0} ({1:2.2f}%)".format(num_false, (num_false/ (num_true + num_false)) * 100))
```

    Número de Casos Verdadeiros: 268 (34.90%)
    Número de Casos Falsos     : 500 (65.10%)
    

### Modelo Rondom Forest


```python
# Separando as variáveis independentes e a variável target

X = dataset.iloc[:, 0:9]
y = dataset.iloc[:, -1]
  
print(y.shape)
print("---------")
print(X.shape)
```

    (768,)
    ---------
    (768, 9)
    


```python
# Dividindo 70% dos dados para treino e 30% para teste
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, y, test_size = 0.30, random_state = 50)
```


```python
modelo = RandomForestClassifier(random_state = 50)
modelo.fit(X_treino, Y_treino)
```




    RandomForestClassifier(random_state=50)




```python
# Verificando os dados de treino
predict_train = modelo.predict(X_treino)
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(Y_treino, predict_train)))
```

    Accuracy: 1.0000
    


```python
# Verificando os dados de teste
predict_test = modelo.predict(X_teste)
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(Y_teste, predict_test)))
```

    Accuracy: 0.7532
    


```python
print("Confusion Matrix")

print(metrics.confusion_matrix(Y_teste, predict_test, labels = [1, 0]))
```

    Confusion Matrix
    [[ 46  37]
     [ 20 128]]
    


### Feature Selection (Seleção de Recursos)

Ter muitos recursos irrelevantes em nossos dados pode não ser tão interessante, além de diminuir a precisão dos modelos. E para que isso 
não ocorra existe uma tecnica de **seleção de recursos**.

**Seleção de Recursos**  é o processo de seleção de um subconjunto de recursos relevantes para uso na construção de modelos. 

Principais vantagens de realizar a seleção de recursos antes de modelar os dados:

- *Reduz o excesso de condições*: dados menos redundantes significam menos oportunidade de tomar decisões com base no ruído.
- *Melhora a precisão*: Dados menos enganosos significam que a precisão da modelagem melhora.
- *Reduz o treinamento Tempo*: Menos dados significam que os algoritmos treinam mais rápido.

Existem vários métodos diferentes de seleção de recursos, nesse projeto irei utilizar a método de *Classificação de Importância de Recursos* 
fornecido pela biblioteca *Scikit-Learn*. 


```python
# Seleção de variáveis preditoras (Feature Selection)

# Construindo o modelo

extra_tree_forest = ExtraTreesClassifier(criterion ='entropy')
```


```python
# Treinando o modelo
extra_tree_forest.fit(X, y)
```




    ExtraTreesClassifier(criterion='entropy')




```python
# importância de cada recurso
feature_importance = extra_tree_forest.feature_importances_
```


```python
# Plotando um Gráfico de Barras para comparar os modelos
plt.barh(X.columns, feature_importance)
plt.xlabel('Importância Features')
plt.ylabel('Features')
plt.title('Comparação de Importância de Features')
plt.show()
```


    
![png](output_20_0.png)
    



```python
# Seleção das variáveis preditoras

variaveis = ["glucose_conc", "bmi", "age", "diab_pred", "num_preg", "diastolic_bp"]

# Seleção da variável target

target = ["diabetes"]
```


```python
# Criando objetos

X_var = dataset[variaveis].values
y_targ = dataset[target].values
```


```python
# Dividindo 70% dos dados para treino e 30% para teste

X_treino, X_teste, Y_treino, Y_teste = train_test_split(X_var, y_targ, test_size = 0.30, random_state = 150)
```


```python
modelo_v2 = RandomForestClassifier(random_state = 150)
modelo_v2.fit(X_treino, Y_treino)
```

    <ipython-input-54-ec6d8fd6b442>:2: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      modelo_v2.fit(X_treino, Y_treino)
    




    RandomForestClassifier(random_state=150)




```python
# Verificando os dados de treino
predict_train = modelo_v2.predict(X_treino)
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(Y_treino, predict_train)))
```

    Accuracy: 1.0000
    


```python
# Verificando os dados de teste
predict_test = modelo_v2.predict(X_teste)
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(Y_teste, predict_test)))
```

    Accuracy: 0.7532
    


```python
print("Confusion Matrix")

print(metrics.confusion_matrix(Y_teste, predict_test, labels = [1, 0]))

```

    Confusion Matrix
    [[ 44  35]
     [ 22 130]]
    


```python
# Salvando o modelo treinado

filename = "modelo_treinado.sav"
pickle.dump(modelo_v2, open(filename, "wb"))
```
