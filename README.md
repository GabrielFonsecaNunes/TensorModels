# DeepModels: Easy-to-Use Deep Learning Models for Time Series Analysis

**DeepModels** is a Python library that provides simple and intuitive access to advanced deep learning architectures, such as:

- **Recurrent Neural Network (RNN)**
- **GRU (Gated Recurrent Unit)**
- **LSTM (Long Short-Term Memory)**
- **Multi-Head Attention (Transformer Models)**

Designed for **time series forecasting and modeling**, DeepModels offers a **user-friendly interface** inspired by the **statsmodels** library. With straightforward methods for **fitting**, **predicting**, and **evaluating models**, it simplifies the application of deep learning techniques, making it accessible to both beginners and professionals.

## **Installation**

You can install the library via pip (once it's available on PyPI):

```bash
pip install git+https://github.com/GabrielFonsecaNunes/deepmodels
```

## **Example Usage**

Below is an example of how to use the DeepModels library to implement LSTM, GRU, RNN, and Multi-Head Attention models for time series forecasting.

### 1. **Import Required Libraries**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import yfinance as yf
from datetime import datetime, timedelta
```

### 2. **Load Data and Prepare It**

You can load your data from any source, here we use **Bitcoin data** from a CSV file as an example.

```python
btc_data = pd.read_csv("./BTCUSDT.csv", sep=";")

# Prepare the data for training and testing
btc_data["DS"] = btc_data.index
btc_data.reset_index(drop=True, inplace=True)

train_index = int(btc_data.shape[0] * 0.95)

data_train = btc_data["DS"][:train_index]
data_out = btc_data["DS"][train_index:]

y_train = btc_data["Close"][:train_index]
y_out = btc_data["Close"][train_index:]

time_step_in = 3  # Number of time steps for the model to use for prediction
time_step_out = 1  # Number of time steps to predict ahead
normalize = True
```

### 3. **LSTM Model**

#### **Initialization and Training**
```python
from deepmodels import LSTM_Regressor

# Initialize the LSTM model
lstm_model = LSTM_Regressor(
                endog=y_train, 
                exog=None,  # Add exogenous variables if needed
                time_step_in=time_step_in, 
                time_step_out=time_step_out, 
                random_state=1,
                normalize=True
)

# Fit the model to the data
lstm_model.fit(epochs=100, batch_size=16, patience=10)
```

#### **Get Fitted Values (Training Predictions)**
```python
# Get the fitted values (predictions on the training data)
y_pred_train = lstm_model.fittedvalues()
```

#### **Forecasting (Out-of-Sample Predictions)**
```python
# Forecast future values (out-of-sample data)
y_pred_oot = lstm_model.get_forescating(steps=len(y_out), exog=None)
```

---

### 4. **GRU Model**

#### **Initialization and Training**
```python
from deepmodels import GRU_Regressor

# Initialize the GRU model
gru_model = GRU_Regressor(
                endog=y_train, 
                exog=None,  # Add exogenous variables if needed
                time_step_in=time_step_in, 
                time_step_out=time_step_out, 
                random_state=1,
                normalize=True
)

# Fit the model to the data
gru_model.fit(epochs=150, batch_size=16, patience=10)
```

#### **Get Fitted Values (Training Predictions)**
```python
# Get the fitted values (predictions on the training data)
y_pred_train = gru_model.fittedvalues()
```

#### **Forecasting (Out-of-Sample Predictions)**
```python
# Forecast future values (out-of-sample data)
y_pred_oot_gru = gru_model.get_forescating(steps=len(y_out), exog=None)
```

---

### 5. **RNN Model**

#### **Initialization and Training**
```python
from deepmodels import RNN_Regressor

# Initialize the RNN model
rnn_model = RNN_Regressor(
                endog=y_train, 
                exog=None,  # Add exogenous variables if needed
                time_step_in=time_step_in, 
                time_step_out=time_step_out, 
                random_state=42,
                normalize=True
)

# Fit the model to the data
rnn_model.fit(epochs=150, batch_size=16, patience=10)
```

#### **Get Fitted Values (Training Predictions)**
```python
# Get the fitted values (predictions on the training data)
y_pred_train_rnn = rnn_model.fittedvalues()
```

#### **Forecasting (Out-of-Sample Predictions)**
```python
# Forecast future values (out-of-sample data)
y_pred_oot_rnn = rnn_model.get_forescating(steps=len(y_out), exog=None)
```

---

### 6. **Multi-Head Attention Model**

#### **Initialization and Training**
```python
from deepmodels import MultiHeadAttention_Regressor

# Initialize the MultiHead Attention model
multheadattention_model = MultiHeadAttention_Regressor(
                endog=y_train, 
                exog=None,  # Add exogenous variables if needed
                time_step_in=time_step_in, 
                time_step_out=time_step_out, 
                random_state=2,
                normalize=True
)

# Fit the model to the data
multheadattention_model.fit(epochs=150, batch_size=16, patience=10)
```

#### **Get Fitted Values (Training Predictions)**
```python
# Get the fitted values (predictions on the training data)
y_pred_train_attention = multheadattention_model.fittedvalues()
```

#### **Forecasting (Out-of-Sample Predictions)**
```python
# Forecast future values (out-of-sample data)
y_pred_oot_attention = multheadattention_model.get_forescating(steps=len(y_out), exog=None)
```