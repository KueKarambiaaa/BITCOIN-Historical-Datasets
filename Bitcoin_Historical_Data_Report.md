NAMA : RAHMI AMILIA.A

EMAIL : amyliarahmi@gmail.com

ID DICODING : cakekarambiaa

<div style="text-align: center;">
  <img src="https://stormgain.com/sites/default/files/2024-03/bitcoin-price-predction-main.jpg" alt="Deskripsi Gambar" style="max-width: 100%; height: auto;">
</div>

# 📈 Proyek Machine Learning: Prediksi Harga Bitcoin

Proyek ini bertujuan untuk memprediksi harga penutupan harian Bitcoin menggunakan teknik regresi. Kita akan menggunakan dataset historical Bitcoin dari Kaggle dan membangun model machine learning dengan algoritma Linear Regression dan Random Forest.

## Domain & Business Understanding

Prediksi harga aset kripto seperti Bitcoin penting untuk membantu pengambilan keputusan dalam investasi dan trading. Harga yang sangat volatil membuat prediksi ini menjadi tantangan sekaligus peluang.

Dataset yang digunakan berasal dari Kaggle [Kaggle - Bitcoin Historical Dataset](https://www.kaggle.com/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024)
Dataset ini berisi data perdagangan Bitcoin, dengan beberapa atribut penting, antara lain:


*   Open Time
*   Open
*   High
*   Low
*   Close
*   Volume
*   Quote Asset Volume
*   Number of trades
*   Take Buy base asset volume
*   Take buy quote asset volume

Dari data tersebut, beberapa fitur yang relevan dipilih untuk membangun model prediksi harga Bitcoin.



## Import Library


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
```

## Load Dataset


```python
df = pd.read_csv('/content/btc_1d_data_2018_to_2025.csv')
```


```python
df.head()
```





  <div id="df-918da09c-f368-401a-a30c-6cd1ae17f5c0" class="colab-df-container">
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
      <th>Open time</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Close time</th>
      <th>Quote asset volume</th>
      <th>Number of trades</th>
      <th>Taker buy base asset volume</th>
      <th>Taker buy quote asset volume</th>
      <th>Ignore</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-01-01</td>
      <td>13715.65</td>
      <td>13818.55</td>
      <td>12750.00</td>
      <td>13380.00</td>
      <td>8609.915844</td>
      <td>2018-01-01 23:59:59.999</td>
      <td>1.147997e+08</td>
      <td>105595</td>
      <td>3961.938946</td>
      <td>5.280975e+07</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-01-02</td>
      <td>13382.16</td>
      <td>15473.49</td>
      <td>12890.02</td>
      <td>14675.11</td>
      <td>20078.092111</td>
      <td>2018-01-02 23:59:59.999</td>
      <td>2.797171e+08</td>
      <td>177728</td>
      <td>11346.326739</td>
      <td>1.580801e+08</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-01-03</td>
      <td>14690.00</td>
      <td>15307.56</td>
      <td>14150.00</td>
      <td>14919.51</td>
      <td>15905.667639</td>
      <td>2018-01-03 23:59:59.999</td>
      <td>2.361169e+08</td>
      <td>162787</td>
      <td>8994.953566</td>
      <td>1.335873e+08</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-01-04</td>
      <td>14919.51</td>
      <td>15280.00</td>
      <td>13918.04</td>
      <td>15059.54</td>
      <td>21329.649574</td>
      <td>2018-01-04 23:59:59.999</td>
      <td>3.127816e+08</td>
      <td>170310</td>
      <td>12680.812951</td>
      <td>1.861168e+08</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-01-05</td>
      <td>15059.56</td>
      <td>17176.24</td>
      <td>14600.00</td>
      <td>16960.39</td>
      <td>23251.491125</td>
      <td>2018-01-05 23:59:59.999</td>
      <td>3.693220e+08</td>
      <td>192969</td>
      <td>13346.622293</td>
      <td>2.118299e+08</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-918da09c-f368-401a-a30c-6cd1ae17f5c0')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-918da09c-f368-401a-a30c-6cd1ae17f5c0 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-918da09c-f368-401a-a30c-6cd1ae17f5c0');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-46cf8b2e-3361-4554-9a91-372938ad1839">
      <button class="colab-df-quickchart" onclick="quickchart('df-46cf8b2e-3361-4554-9a91-372938ad1839')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-46cf8b2e-3361-4554-9a91-372938ad1839 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>





```python
df.tail()
```





  <div id="df-f3be9782-7180-497f-83b5-1480c9ff4083" class="colab-df-container">
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
      <th>Open time</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Close time</th>
      <th>Quote asset volume</th>
      <th>Number of trades</th>
      <th>Taker buy base asset volume</th>
      <th>Taker buy quote asset volume</th>
      <th>Ignore</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2670</th>
      <td>2025-04-24</td>
      <td>93691.07</td>
      <td>93787.65</td>
      <td>93060.75</td>
      <td>93607.99</td>
      <td>1498.31243</td>
      <td>2025-04-24 23:59:59.999</td>
      <td>1.399920e+08</td>
      <td>173779</td>
      <td>429.75126</td>
      <td>4.014164e+07</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2671</th>
      <td>2025-04-25</td>
      <td>93980.47</td>
      <td>94444.00</td>
      <td>93520.00</td>
      <td>93664.14</td>
      <td>2699.07820</td>
      <td>2025-04-25 23:59:59.999</td>
      <td>2.533782e+08</td>
      <td>307401</td>
      <td>1518.80459</td>
      <td>1.425713e+08</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2672</th>
      <td>2025-04-26</td>
      <td>94638.68</td>
      <td>95199.00</td>
      <td>94527.84</td>
      <td>95055.48</td>
      <td>1878.89983</td>
      <td>2025-04-26 23:59:59.999</td>
      <td>1.781277e+08</td>
      <td>150054</td>
      <td>789.03707</td>
      <td>7.481943e+07</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2673</th>
      <td>2025-04-27</td>
      <td>94628.00</td>
      <td>95369.00</td>
      <td>94041.60</td>
      <td>94222.29</td>
      <td>3179.08570</td>
      <td>2025-04-27 23:59:59.999</td>
      <td>3.014499e+08</td>
      <td>394263</td>
      <td>1534.29285</td>
      <td>1.456140e+08</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2674</th>
      <td>2025-04-28</td>
      <td>93749.29</td>
      <td>93798.71</td>
      <td>92800.01</td>
      <td>93130.44</td>
      <td>2221.76369</td>
      <td>2025-04-28 23:59:59.999</td>
      <td>2.071204e+08</td>
      <td>436996</td>
      <td>991.34663</td>
      <td>9.240970e+07</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-f3be9782-7180-497f-83b5-1480c9ff4083')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-f3be9782-7180-497f-83b5-1480c9ff4083 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-f3be9782-7180-497f-83b5-1480c9ff4083');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-b18600af-96d2-4fb4-becf-f15c576fc452">
      <button class="colab-df-quickchart" onclick="quickchart('df-b18600af-96d2-4fb4-becf-f15c576fc452')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-b18600af-96d2-4fb4-becf-f15c576fc452 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>





```python
df.describe()
```





  <div id="df-b9f167af-3463-4026-ae43-7eb9e9b650ed" class="colab-df-container">
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
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Quote asset volume</th>
      <th>Number of trades</th>
      <th>Taker buy base asset volume</th>
      <th>Taker buy quote asset volume</th>
      <th>Ignore</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2675.000000</td>
      <td>2675.000000</td>
      <td>2675.000000</td>
      <td>2675.000000</td>
      <td>2675.000000</td>
      <td>2.675000e+03</td>
      <td>2.675000e+03</td>
      <td>2675.000000</td>
      <td>2.675000e+03</td>
      <td>2675.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>30840.578019</td>
      <td>31528.766520</td>
      <td>30104.435275</td>
      <td>30871.815563</td>
      <td>68932.035975</td>
      <td>1.783905e+09</td>
      <td>1.719947e+06</td>
      <td>34264.056822</td>
      <td>8.832812e+08</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>25302.219752</td>
      <td>25767.079921</td>
      <td>24820.256355</td>
      <td>25331.323460</td>
      <td>80006.389942</td>
      <td>2.002799e+09</td>
      <td>2.109577e+06</td>
      <td>39815.096251</td>
      <td>9.958256e+08</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3211.710000</td>
      <td>3276.500000</td>
      <td>3156.260000</td>
      <td>3211.720000</td>
      <td>300.986860</td>
      <td>1.177017e+07</td>
      <td>1.241700e+04</td>
      <td>151.876190</td>
      <td>6.532639e+06</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>9178.505000</td>
      <td>9347.500000</td>
      <td>8949.030000</td>
      <td>9183.355000</td>
      <td>28887.841936</td>
      <td>3.669396e+08</td>
      <td>4.016555e+05</td>
      <td>14434.566106</td>
      <td>1.847062e+08</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>23554.850000</td>
      <td>24199.720000</td>
      <td>23060.000000</td>
      <td>23628.970000</td>
      <td>43882.924625</td>
      <td>1.090742e+09</td>
      <td>1.003615e+06</td>
      <td>21963.773537</td>
      <td>5.292412e+08</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>46357.010000</td>
      <td>47490.840000</td>
      <td>44745.365000</td>
      <td>46385.005000</td>
      <td>71682.501594</td>
      <td>2.499919e+09</td>
      <td>1.912282e+06</td>
      <td>35556.832016</td>
      <td>1.248836e+09</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>106143.820000</td>
      <td>108353.000000</td>
      <td>105321.490000</td>
      <td>106143.820000</td>
      <td>760705.362783</td>
      <td>1.746531e+10</td>
      <td>1.522359e+07</td>
      <td>374775.574085</td>
      <td>8.783916e+09</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-b9f167af-3463-4026-ae43-7eb9e9b650ed')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-b9f167af-3463-4026-ae43-7eb9e9b650ed button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-b9f167af-3463-4026-ae43-7eb9e9b650ed');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-d6dca38d-1a45-4df6-9cd4-0d3130814129">
      <button class="colab-df-quickchart" onclick="quickchart('df-d6dca38d-1a45-4df6-9cd4-0d3130814129')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-d6dca38d-1a45-4df6-9cd4-0d3130814129 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>




## EDA - Exploratory Data Analysis


```python
print(df.columns)
```

    Index(['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
           'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
           'Taker buy quote asset volume', 'Ignore'],
          dtype='object')



```python
df['Open time'] = pd.to_datetime(df['Open time'])
```

Statistik deskriptif


```python
print(df.describe())
```

                     Open time           Open           High            Low  \
    count                 2675    2675.000000    2675.000000    2675.000000   
    mean   2021-08-30 00:00:00   30840.578019   31528.766520   30104.435275   
    min    2018-01-01 00:00:00    3211.710000    3276.500000    3156.260000   
    25%    2019-10-31 12:00:00    9178.505000    9347.500000    8949.030000   
    50%    2021-08-30 00:00:00   23554.850000   24199.720000   23060.000000   
    75%    2023-06-29 12:00:00   46357.010000   47490.840000   44745.365000   
    max    2025-04-28 00:00:00  106143.820000  108353.000000  105321.490000   
    std                    NaN   25302.219752   25767.079921   24820.256355   
    
                   Close         Volume  Quote asset volume  Number of trades  \
    count    2675.000000    2675.000000        2.675000e+03      2.675000e+03   
    mean    30871.815563   68932.035975        1.783905e+09      1.719947e+06   
    min      3211.720000     300.986860        1.177017e+07      1.241700e+04   
    25%      9183.355000   28887.841936        3.669396e+08      4.016555e+05   
    50%     23628.970000   43882.924625        1.090742e+09      1.003615e+06   
    75%     46385.005000   71682.501594        2.499919e+09      1.912282e+06   
    max    106143.820000  760705.362783        1.746531e+10      1.522359e+07   
    std     25331.323460   80006.389942        2.002799e+09      2.109577e+06   
    
           Taker buy base asset volume  Taker buy quote asset volume  Ignore  
    count                  2675.000000                  2.675000e+03  2675.0  
    mean                  34264.056822                  8.832812e+08     0.0  
    min                     151.876190                  6.532639e+06     0.0  
    25%                   14434.566106                  1.847062e+08     0.0  
    50%                   21963.773537                  5.292412e+08     0.0  
    75%                   35556.832016                  1.248836e+09     0.0  
    max                  374775.574085                  8.783916e+09     0.0  
    std                   39815.096251                  9.958256e+08     0.0  



```python
print(df.isnull().sum())
```

    Open time                       0
    Open                            0
    High                            0
    Low                             0
    Close                           0
    Volume                          0
    Close time                      0
    Quote asset volume              0
    Number of trades                0
    Taker buy base asset volume     0
    Taker buy quote asset volume    0
    Ignore                          0
    dtype: int64





```python
plt.figure(figsize=(8,4))
sns.histplot(df['Close'], bins=50, kde=True)
plt.title('Distribusi Harga Penutupan Bitcoin')
plt.show()
```


    
![png](output_18_0.png)
    



```python
plt.figure(figsize=(12,6))
plt.plot(df['Open time'], df['Close'])
plt.title('Harga Bitcoin dari Waktu ke Waktu')
plt.xlabel('Waktu')
plt.ylabel('Harga')
plt.show()
```


    
![png](output_19_0.png)
    



```python
numeric_cols = df.select_dtypes(include=[np.number])

plt.figure(figsize=(10,8))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap Korelasi Fitur')
plt.show()
```


    
![png](output_20_0.png)
    


## Data Preparation


```python
df = df[['Open', 'High', 'Low', 'Volume', 'Close']]
```


```python
print(df.isnull().sum())
```

    Open      0
    High      0
    Low       0
    Volume    0
    Close     0
    dtype: int64



```python
X = df.drop('Close', axis=1)
y = df['Close']
```


```python
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```


```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=False
)

print(f"Jumlah data train: {len(X_train)}")
print(f"Jumlah data test: {len(X_test)}")
```

    Jumlah data train: 2140
    Jumlah data test: 535


## Modeling


```python
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
```


```python
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
```


```python
svr_model = SVR(kernel='rbf', C=100)
svr_model.fit(X_train, y_train)
svr_pred = svr_model.predict(X_test)
```

## Evaluation



```python
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f" {model_name} Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.2f}")
    print("-"*30)
```


```python
evaluate_model(y_test, rf_pred, "Random Forest")
evaluate_model(y_test, xgb_pred, "XGBoost")
evaluate_model(y_test, svr_pred, "SVR")
```

     Random Forest Performance:
    MAE: 9179.32
    RMSE: 15520.82
    R2 Score: 0.33
    ------------------------------
     XGBoost Performance:
    MAE: 10122.21
    RMSE: 16573.84
    R2 Score: 0.23
    ------------------------------
     SVR Performance:
    MAE: 32494.02
    RMSE: 41278.32
    R2 Score: -3.76
    ------------------------------


## Visualization


```python
plt.figure(figsize=(15,6))
plt.plot(y_test.values, label='Actual', color='black')
plt.plot(rf_pred, label='Random Forest Prediction')
plt.plot(xgb_pred, label='XGBoost Prediction')
plt.plot(svr_pred, label='SVR Prediction')
plt.legend()
plt.title('Perbandingan Harga Asli vs Prediksi')
plt.xlabel('Sample')
plt.ylabel('Harga Bitcoin')
plt.show()
```


    
![png](output_35_0.png)
    



```python
sns.pairplot(df[['Open', 'High', 'Low', 'Close', 'Volume']])
plt.suptitle('Pairplot Antar Fitur Harga Bitcoin', y=1.02)
plt.show()
```


    
![png](output_36_0.png)
    

