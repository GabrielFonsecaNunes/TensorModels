```markdown
# DeepModels: Easy-to-Use Deep Learning Models for Time Series Analysis

**DeepModels** is a Python library that provides simple and intuitive access to advanced deep learning architectures, such as:

- **LSTM (Long Short-Term Memory)**
- **GRU (Gated Recurrent Unit)**
- **Transformer Models**
- **Recurrent Neural Network (RNN)**

Designed for **time series forecasting and modeling**, DeepModels offers a **user-friendly interface** inspired by the **statsmodels** library. With straightforward methods for **fitting**, **predicting**, and **evaluating models**, it simplifies the application of deep learning techniques, making it accessible to both beginners and professionals.

## **Installation**

You can install the library via pip (once it's available on PyPI):

```bash
pip install git+https://github.com/GabrielFonsecaNunes/deepmodels
```

## **Example Usage**

Below is an example of how to use the DeepModels library to implement LSTM, GRU, and RNN models for time series forecasting.

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

### 3. **Data Normalization**

```python
# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()

y_train_normalized = scaler.fit_transform(np.array(y_train).reshape(-1, 1)).flatten()
y_train_normalized_series = pd.Series(y_train_normalized)

y_out_normalized = scaler.transform(np.array(y_out).reshape(-1, 1)).flatten()
y_out_normalized_series = pd.Series(y_out_normalized)
```

---

### 4. **LSTM Model**

#### **Initialization and Training**
```python
from deepmodels.deepmodels import LSTM_Regressor

# Initialize the LSTM model
lstm_model = LSTM_Regressor(
endog=y_train_normalized_series, 
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
y_pred_train_normalized = lstm_model.fittedvalues()

# Inverse transform the normalized predictions back to the original scale
y_pred_train_lstm = scaler.inverse_transform(y_pred_train_normalized.reshape(-1, 1)).flatten()
```

#### **Forecasting (Out-of-Sample Predictions)**
```python
# Forecast future values (out-of-sample data)
y_pred_out_normalized = lstm_model.get_forescating(steps=len(y_out), exog=None)

# Inverse transform the normalized forecast back to the original scale
y_pred_out_lstm = scaler.inverse_transform(y_pred_out_normalized.reshape(-1, 1)).flatten()
```

---

### 5. **GRU Model**

#### **Initialization and Training**
```python
from deepmodels.deepmodels import GRU_Regressor

# Initialize the GRU model
gru_model = GRU_Regressor(
endog=y_train_normalized_series, 
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
y_pred_train_normalized_gru = gru_model.fittedvalues()

# Inverse transform the normalized predictions back to the original scale
y_pred_train_gru = scaler.inverse_transform(y_pred_train_normalized_gru.reshape(-1, 1)).flatten()
```

#### **Forecasting (Out-of-Sample Predictions)**
```python
# Forecast future values (out-of-sample data)
y_pred_out_normalized_gru = gru_model.get_forescating(steps=len(y_out), exog=None)

# Inverse transform the normalized forecast back to the original scale
y_pred_out_gru = scaler.inverse_transform(y_pred_out_normalized_gru.reshape(-1, 1)).flatten()
```

---

### 6. **RNN Model**

#### **Initialization and Training**
```python
from deepmodels.deepmodels import RNN_Regressor

# Initialize the RNN model
rnn_model = RNN_Regressor(
endog=y_train_normalized_series, 
exog=None,  # Add exogenous variables if needed
time_step_in=time_step_in, 
time_step_out=time_step_out, 
random_state=1,
normalize=True
)

# Fit the model to the data
rnn_model.fit(epochs=150, batch_size=16, patience=10)
```

#### **Get Fitted Values (Training Predictions)**
```python
# Get the fitted values (predictions on the training data)
y_pred_train_normalized_rnn = rnn_model.fittedvalues()

# Inverse transform the normalized predictions back to the original scale
y_pred_train_rnn = scaler.inverse_transform(y_pred_train_normalized_rnn.reshape(-1, 1)).flatten()
```

#### **Forecasting (Out-of-Sample Predictions)**
```python
# Forecast future values (out-of-sample data)
y_pred_out_normalized_rnn = rnn_model.get_forescating(steps=len(y_out), exog=None)

# Inverse transform the normalized forecast back to the original scale
y_pred_out_rnn = scaler.inverse_transform(y_pred_out_normalized_rnn.reshape(-1, 1)).flatten()
```

---

### 7. **Evaluation**

Evaluate the models using **Mean Absolute Percentage Error (MAPE)**:

```python
# Evaluate the models using MAPE
mape_lstm = mape(y_out, y_pred_out_lstm)
mape_gru = mape(y_out, y_pred_out_gru)
mape_rnn = mape(y_out, y_pred_out_rnn)

print(f"LSTM MAPE: {mape_lstm:.2f}%")
print(f"GRU MAPE: {mape_gru:.2f}%")
print(f"RNN MAPE: {mape_rnn:.2f}%")
```

---


## **Conclusion**

This example demonstrates how to use the `DeepModels` library for time series forecasting with **LSTM**, **GRU**, and **RNN** models. The library offers simple implementations for these models, and the process for training, obtaining fitted values, and forecasting is streamlined with intuitive methods such as `fit()`, `fittedvalues()`, and `get_forescating()`. You can easily modify the parameters and data to apply these models to your own time series data.
```