import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from typing import Optional, Union

import statsmodels.api as sm
from statsmodels.stats.diagnostic import (
    acorr_ljungbox
)

from datetime import (
    datetime as dt
)

from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import breakvar_heteroskedasticity_test
from scipy.stats import jarque_bera, skew, kurtosis

class GRU_Regressor(Sequential):
    """
    GRU model for time series regression,
    with support for exogenous variables and optional normalization.
    """
    
    def __init__(self, endog, exog: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None, time_step_in: int = 10, 
                 time_step_out: int = 1, random_state: int = 42, normalize: Optional[bool] = True, **kwargs):
        """
        GRU Model
        
        Args:
            endog (np.array | pd.DataFrame | pd.Series): Time series data
            exog (np.array | pd.DataFrame | pd.Series): Exogenous variables
            time_step_in (int): Number of time steps for input
            time_step_out (int): Number of time steps for output
            normalize (bool): If True, normalizes the data,
            random_state (int): Seed for deterministic training,
            default is 42.
        """
        super().__init__(**kwargs)
        self.endog = endog
        self.exog = exog
        self.time_step_in = time_step_in
        self.time_step_out = time_step_out
        self.random_state = random_state
        self.set_random_seed()
        self.normalize = normalize
        self.scaler = MinMaxScaler() if normalize else None
        self.input_shape_model = self.get_input_shape_model()
        self.model = self.set_model()
        self.date_fit = dt.now()
        self.trained = False
        
    def set_random_seed(self):
        """
        Function to set the random seed to ensure reproducibility.
        """
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        tf.random.set_seed(self.random_state)
        
    def get_input_shape_model(self):
        """
        Returns the input shape for the GRU model
        
        Returns:
            tuple: The input shape
        """
        n = 1
        m = 0 if self.exog is None else self.exog.shape[1]
        shape = n + m 
        return (self.time_step_in, shape)
        
    def set_model(self): 
        """
        Defines the architecture of the GRU model
        
        Returns:
            self: The configured GRU model.
        """
        # Weight initialization with fixed seed
        self.add(GRU(units = 32, activation = 'relu', return_sequences = True, input_shape = self.input_shape_model,
                      kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.random_state),
                      recurrent_initializer=tf.keras.initializers.GlorotUniform(seed=self.random_state),
                      bias_initializer=tf.keras.initializers.Zeros()))
        self.add(Dropout(0.2, seed=self.random_state))  # Controls the randomness of Dropout
        self.add(GRU(units = 16, return_sequences = False,
                      kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.random_state),
                      recurrent_initializer=tf.keras.initializers.GlorotUniform(seed=self.random_state),
                      bias_initializer=tf.keras.initializers.Zeros()))
        self.add(Dense(self.time_step_out, activation = 'linear',
                       kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.random_state),
                       bias_initializer=tf.keras.initializers.Zeros()))
        self.compile(optimizer= 'adam', loss = 'mean_squared_error')
        
    def create_dataset(self):
        """
        Creates time windows for GRU input
        
        Returns:
            tuple: Input and output arrays for the GRU model.
        """
        
        dataX, dataY = [], []
        y = self.endog
        X = self.exog
        
        if self.normalize and self.exog is not None:
            X = self.scaler.fit_transform(X.copy())
        
        # Autoregressive method without exogenous variables
        if X is None:
            for i in range(len(y) - self.time_step_in):
                
                # Input Sequence (Series values in a sliding window)
                dataX.append(np.array(y[i:self.time_step_in + i]))
                
                # Output Sequence (Series values in a sliding window)
                dataY.append(y[self.time_step_in + i: self.time_step_in + self.time_step_out + i])
                
            dataX = np.array(dataX).reshape(len(y) - self.time_step_in, self.time_step_in, 1)
            
        # Autoregressive method with exogenous variables
        else:
            for i in range(len(y) - self.time_step_in):
                # Sequence with both endogenous and exogenous variables starting from index (i = 0)
                # to the defined number of time steps for input
                
                arrays = np.concatenate(np.array(y[i: self.time_step_in + 1].reshape(-1, 1)), np.array(X[i:self.time_step_in + 1]), axis = 1)
                dataX.append(arrays)
                
                # Sequence with endogenous variable from the defined input time steps
                # to the defined number of output time steps
                dataY.append(y[self.time_step_in + i: self.time_step_in + self.time_step_out + i])
            
        return np.array(dataX), np.array(dataY)
    
    def fit(self, epochs = 100, batch_size = 16, patience = 10):
        """
        Trains the GRU model
        
        Args:
            epochs (int): Number of epochs for training
            batch_size (int): Batch size for training
            patience (int): Number of epochs without improvement to stop training
        """
        # If time series and exogenous variables are passed
        
        dataX, dataY = self.create_dataset()
        
        early_stopping = EarlyStopping(monitor = 'loss', patience = patience)

        super().fit(dataX, dataY, epochs = epochs, batch_size = batch_size, verbose = 1, callbacks = [early_stopping])
        self.trained = True

    def get_forecasting(self, exog: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None, steps: Optional[int] = None):
        """
        Method to make predictions with the trained model
        
        Args:
            exog (np.ndarray | pd.DataFrame | pd.Series | None, optional): Exogenous variables
            steps (int): Number of steps to forecast ahead
        """
        if not self.trained:
            raise ValueError("Training is required. Train the model with model.fit()")
        
        else:
            # Number of steps for projection
            # If the model only has its own series as input, adjustments are necessary
            
            y_pred = []
            
            if self.exog is None:
                if steps is None:
                    raise ValueError("Define the number of steps to make the forecast, steps = ")
                
                for step in range(steps):
                    if step == 0:
                        # Make the first prediction after training using only endogenous variables
                        array = np.array(self.endog[-self.time_step_in:])
                        array = array.reshape(1, self.time_step_in, 1)
                        y_pred_value = self.predict(array).flatten()
                        y_pred.append(y_pred_value)
                    else:
                        if step < self.time_step_in:
                            array1 = np.array(self.endog[-self.time_step_in + step:]).reshape(-1, 1)
                            array2 = np.array(y_pred[:step])

                            array = np.concatenate((array1, array2), axis = 0)
                            array = array.reshape(1, self.time_step_in, 1)
                            
                            y_pred_value = self.predict(array).flatten()
                            y_pred.append(y_pred_value)
                            
                        else:
                            array = np.array(y_pred[-self.time_step_in:])
                            array = array.reshape(1, self.time_step_in, 1)
                            
                            y_pred_value = self.predict(array).flatten()
                            y_pred.append(y_pred_value)
                            
            else:
                if self.normalize and exog is not None:
                    exog = self.scaler.transform(exog)
                    
                steps = exog.shape[0] if steps is None else steps
                
                for step in range(steps):
                    if step == 0:
                        array = np.concatenate((np.array(self.endog[-self.time_step_in + step:]).reshape(-1, 1), np.array(self.exog[-self.time_step_in:])), axis = 1)
                        array = array.reshape(1, array.reshape[0], array.shape[1])
                        
                        y_pred_value = self.predict(array).flatten()
                        y_pred.append(y_pred_value)
                        
                    else:
                        if step < self.time_step_in:
                            array1 = np.concatenate((np.array(self.endog[-self.time_step_in + step:]).reshape(-1, 1), self.exog[-self.time_step_in + step:]), axis = 1)
                            array2 = np.concatenate(np.array(y_pred).reshape(-1, 1), np.array(exog[:step]), axis = 1)
                            
                            array = np.concatenate((array1, array2), axis = 0)
                            array = array.reshape(1, array.shape[0], array.shape[1])
                            
                            y_pred_value = self.predict(array).flatten()
                            y_pred.append(y_pred_value)
                            
                        else:
                            array1 = np.array(y_pred[-self.time_step_in:]).reshape(-1, 1)
                            array2 = np.array(np.array(exog[step - self.time_step_in:step]))
                            
                            array = np.concatenate((array1, array2), axis = 1)
                            array = array.reshape(1, array.shape[0], array.shape[1])
                            
                            y_pred_value = self.predict(array).flatten()
                            y_pred.append(y_pred_value)
        
        return np.array(y_pred).flatten()
    
    def fittedvalues(self):
        """
        Returns the predictions for the training values of the model
        """
        if not self.trained:
            raise ValueError("Training is required. Train the model with model.fit()")
        else:
            dataX, _ = self.create_dataset()
            return self.predict(dataX).flatten()
        
    def load_weights_model(self, weights_path: str):
        """
        Loads the model weights from an .h5 file
        
        Args:
            weights_path (str): Path to the .h5 file containing the model weights.
        """
        
        # Recreate the model architecture 
        self.model = self.set_model()
        
        self.model.load_weights(weights_path)
        self.trained = True
        
    def summary_model(self):
        """
        Returns the summary of the trained model.
        """
        if not self.trained:
            raise ValueError("The model has not been trained. Train the model with model.fit()")
        
        y_pred_train = self.fittedvalues()
        
        # Calculate residuals
        residuals = self.endog[self.time_step_in:] - y_pred_train
        rss = np.sum(residuals ** 2)
        rse = np.sum((self.endog[self.time_step_in:] - y_pred_train.mean()) ** 2)
        
        n = len(self.endog[self.time_step_in:])
        k = self.time_step_out if self.exog is None else self.exog.shape[1] + self.time_step_out
        
        # Log-Likelihood, AIC, BIC, and HIC
        log_likelihood = -n / 2 * np.log(2 * np.pi * rss / n) - rss / (2 * rss / n)
        aic = 2 * k - 2 * log_likelihood
        bic = np.log(n) * k - 2 * log_likelihood
        hic = np.log(np.log(n)) * k - 2 * log_likelihood
        
        # Durbin-Watson statistic
        dw_stat = durbin_watson(residuals)
        
        # Ljung-Box test
        num_lags = min(10, len(residuals) // 2)
        ljungbox_results = acorr_ljungbox(residuals, lags=num_lags, return_df=True)
        ljungbox_stat = ljungbox_results['lb_stat'].iloc[-1]
        ljungbox_pvalue = ljungbox_results['lb_pvalue'].iloc[-1]
        
        # Other tests
        het = breakvar_heteroskedasticity_test(residuals)
        jb_test = jarque_bera(residuals)
        skewness = skew(residuals)
        kurt = kurtosis(residuals)
        
        data_fit = str(dt.strftime(self.date_fit, "%Y-%m-%d"))          
        
        # Display the summary
        summary_str = (
            f"           Deep Learning - GRU Results    \n"
            f"====================================================================================\n"
            f"Model:                 GRU                      Durbin-Watson:          {dw_stat:>8.2f}\n"
            f"Num step in:           {self.time_step_in:<25}Log-Likelihood:         {log_likelihood:>8.2f}\n"
            f"Num step outs:         {self.time_step_out:<25}AIC:                    {aic:>8.2f}\n"
            f"No. Observations:      {self.endog.shape[0]:<25}BIC:                    {bic:>8.2f}\n"
            f"Date:                  {data_fit:<25}HIC:                    {hic:>8.2f}\n"
            f"====================================================================================\n"
            f"Ljung-Box (Q):         {ljungbox_stat:>8.2f}                 Jarque-Bera (JB):       {jb_test[0]:>8.2f}\n"
            f"Prob(Q):               {ljungbox_pvalue:.2e}                 Prob(JB):              {jb_test[1]:>8.2f}\n"
            f"Heteroskedasticity(H): {het[0]:>8.2f}                 Skew:                  {skewness:>8.2f}\n"
            f"Prob(H) (two-side):    {het[1]:>8.2f}                 Kurtosis:              {kurt:>8.2f}\n"
            f"====================================================================================\n"
        )
        return summary_str