import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, 
    MultiHeadAttention, 
    LayerNormalization, 
    Dropout, 
    Dense, 
    GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping
        
from typing import Optional, Union

from statsmodels.stats.diagnostic import (
    acorr_ljungbox
)

from datetime import (
    datetime as dt
)

from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import breakvar_heteroskedasticity_test
from scipy.stats import jarque_bera, skew, kurtosis

class MultiHeadAttention_Regressor(Model):
    """
    Classe para construir e treinar um modelo MultiHeadAttention para regresssão de séries temporais,
    com suporte para variáveis exógenas e normalização opcional.
    """
    
    from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, 
    MultiHeadAttention, 
    LayerNormalization, 
    Dropout, 
    Dense, 
    GlobalAveragePooling1D
)

class MultiHeadAttention_Regressor(Model):
    def __init__(self, endog, exog=None, time_step_in=10, time_step_out=1, normalize=True, **kwargs):
        """
        Inicializa o modelo MultiHeadAttention_Regressor.
        
        Args:
            endog (np.array | pd.DataFrame | pd.Series): Série temporal endógena.
            exog (np.array | pd.DataFrame | pd.Series | None): Variáveis exógenas.
            time_step_in (int): Número de passos de tempo na entrada.
            time_step_out (int): Número de passos de tempo na saída.
            normalize (bool): Se True, normaliza os dados.
        """
        self.endog = endog
        self.exog = exog
        self.time_step_in = time_step_in
        self.time_step_out = time_step_out
        self.normalize = normalize

        # Definir a forma de entrada
        n = 1  # Variável endógena
        m = 0 if exog is None else exog.shape[1]  # Variáveis exógenas (se houver)
        input_shape = (time_step_in, n + m)

        # Definir as camadas diretamente no construtor
        inputs = Input(shape=input_shape)

        # MultiHeadAttention
        attention_output = MultiHeadAttention(num_heads=4, key_dim=16)(inputs, inputs)
        
        # Dropout para regularização
        dropout_output = Dropout(0.2)(attention_output)

        # GlobalAveragePooling1D para reduzir dimensionalidade
        pooled_output = GlobalAveragePooling1D()(dropout_output)

        # Camada densa final para previsão
        outputs = Dense(time_step_out, activation='linear')(pooled_output)

        # Inicializar o modelo base com inputs e outputs
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        # Compilar o modelo
        self.compile(optimizer='adam', loss='mean_squared_error')
        
        self.date_fit = None
        
    def get_input_shape_model(self):
        """
        Retorna a forma de entrada para a MultiHeadAttention
        
        Returns:
            tuple: A forma de entrada
        """
        n = 1
        m = 0 if self.exog is None else self.exog.shape[1]
        shape = n + m 
        return (self.time_step_in, shape)
        
    def create_dataset(self):
        """
        Cria janelas de tempo para a entrada MultiHeadAttention
        
        Returns:
            tuple: Arrays de entrada e saída para o modelo MultiHeadAttention.
        """
        
        dataX, dataY = [], []
        y = self.endog
        X = self.exog
        
        if self.normalize and self.exog is not None:
            X = self.scaler.fit_transform(X.copy())
        
        # Metodo de Divisao auto regressao sem variaveis exogenas
        if X is None:
            for i in range(len(y) - self.time_step_in):
                
                # Entrada Sequencia Entrada (Valores da Serie em um janela deslizante)
                dataX.append(np.array(y[i:self.time_step_in + i]))
                
                # Variavel Resposta Sequencia Saida (Valores da Serie em um janela deslizante)
                dataY.append(y[self.time_step_in + i: self.time_step_in + self.time_step_out + i])
                
            dataX = np.array(dataX).reshape(len(y) - self.time_step_in, self.time_step_in, 1)
            
        # Metodo de Divisao Auto Regressao com Variaveis Exogenas
        else:
            for i in range(len(y) - self.time_step_in):
                # Sequencia com variavel endog e variavel exogena começando no index (i = 0) 
                # até numero de passos definidos de tempo de entrada
                
                arrays = np.concatenate(np.array(y[i: self.time_step_in + 1].reshape(-1, 1)), np.array(X[i:self.time_step_in + 1]), axis = 1)
                dataX.append(arrays)
                
                # Sequencia com variavel endog a partir do numero de passos definidos de tempo de entrada 
                # até numero passos de saida
                dataY.append(y[self.time_step_in + i: self.time_step_in + self.time_step_out + i])
            
        return np.array(dataX), np.array(dataY)
    
    def fit(self, epochs = 100, batch_size = 16, patience = 10):
        """
        Treina o modelo MultiHeadAttention
        
        Args:
            epochs (int): Número de épocas para treinamento
            batch_size (int): Tamanho do lote para treinamento
            patience (int): Número de épocas sem melhoria para interromper o treinamento
        """
        # Caso a série temporal e variaveis exogenass sejam passadas
        
        dataX, dataY = self.create_dataset()
        early_stopping = EarlyStopping(monitor = 'loss', patience = patience)
        super().fit(dataX, dataY, epochs = epochs, batch_size = batch_size, verbose = 1, callbacks = [early_stopping])
        self.trained = True
        self.date_fit = dt.now()

    def get_forescating(self, exog: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None, steps: Optional[int] = None):
        """
        Método para realizar previsões com o modelo treinado dado
        
        exog (np.ndarray | pd.DataFrame | pd.Serie | None opctional): Variaveis Exogenas
        steps (int): Numero de passo para forescating a frente
        """
        if not self.trained:
            raise ValueError("O treinamento é requido. Faça o treinamento com model.fit()")
        
        else:
            # Numero de passos para projecao
            # Caso o modelo somente tenha a propria serie como parametro é necessario ajustar
            
            y_pred = []
            
            if self.exog is None:
                if steps is None:
                    raise ValueError("Defina o numero de steps para fazer o  forescating, steps = ")
                
                for step in range(steps):
                    if step == 0:
                        # Faz a proje;'ao do primeiro valor apos treinamento somente com variaveis endog
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
        Retorna a predicao dos valores do treino do modelo
        """
        if not self.trained:
            raise ValueError("O treinamento é requido. Faça o treinamento com model.fit()")
        else:
            dataX, _ = self.create_dataset()
            return self.predict(dataX).flatten()
        
    def load_weights_model(self, weights_paht: str):
        """
        Carrega os pesos do modelo a partir de um arquivo .h5
        
        Args:
            weights_path (str): Caminho para o arquivo .h5 contendo os pesos do modelo.
        """
        
        # Recria a arquitetura do modelo 
        
        self.model = self.set_mode()
        
        self.model.load_weights(weights_paht)
        self.trained = True
        
    def summary_model(self):
        """
        Retorna o sumário do modelo treinado.
        """
        if not self.trained:
            raise ValueError("O Modelo ainda não foi treinado. Faça o treinamento com model.fit()")
        
        y_pred_train = self.fittedvalues()
        
        # Calcula os resíduos
        residuals = self.endog[self.time_step_in:] - y_pred_train
        rss = np.sum((self.endog[self.time_step_in:] - y_pred_train) ** 2)
        rse = np.sum((self.endog[self.time_step_in:] - y_pred_train.mean()) ** 2)
        
        n = len(self.endog[self.time_step_in:])
        k = self.time_step_out if self.exog is None else self.exog.shape[1] + self.time_step_out
        
        # Calcular o R2
        r_square = 1 - (rss / rse)
        adj_r_square = 1 - ((1 - r_square) * (n - 1) / (n - k - 1))
        
        log_likelihood = -n / 2 * np.log(2 * np.pi * rss / n) - rss / (2 * rss / n)     
        aic = 2 * k - 2 * log_likelihood
        bic = np.log(n) * k - 2 * log_likelihood
        
        # Calcula o Durbin-Watson
        dw_stat = durbin_watson(residuals)
        
        # Teste de Ljung-Box
        num_lags = min(10, len(residuals) // 2)
        ljungbox_results = acorr_ljungbox(residuals, lags=num_lags, return_df=True)
        ljungbox_pvalue = ljungbox_results['lb_pvalue'].iloc[-1]
        
        # Outros testes
        het = breakvar_heteroskedasticity_test(residuals)
        jb_test = jarque_bera(residuals)
        skewness = skew(residuals)
        kurt = kurtosis(residuals)
        
        # Exibir o sumário
        summary_str = (
            f"                          Deep Learning MultiHeadAttention Regression Results                          \n"
            f"====================================================================================\n"
            f"Model:                 MultiHeadAttention                     R-squared:               {r_square:>8.2f}\n"
            f"Num step in:           {self.time_step_in:<25}Adj. R-squared:          {adj_r_square:>8.2f}\n"
            f"Num step outs:         {self.time_step_out:<25}Log-Likelihood:         {log_likelihood:>8.2f}\n"
            f"No. Observations:      {self.endog.shape[0]:<25}AIC:                    {aic:>8.2f}\n"
            f"Date:                  {data_fit:<25}BIC:                    {bic:>8.2f}\n"
            f"====================================================================================\n"
            f"Durbin-Watson:         {dw_stat:>8.2f}                 Jarque-Bera (JB):       {jb_test[0]:>8.2f}\n"
            f"Prob (Ljung-Box):      {ljungbox_pvalue:.2e}                 Prob(JB):              {jb_test[1]:>8.2f}\n"
            f"Heteroskedasticity(H): {het[0]:>8.2f}                 Skew:                  {skewness:>8.2f}\n"
            f"Prob(H) (two-side):    {het[1]:>8.2f}                 Kurtosis:              {kurt:>8.2f}\n"
            f"====================================================================================\n"
        )
        return summary_str