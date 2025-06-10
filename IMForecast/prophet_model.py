# Librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import itertools
from tqdm import tqdm
import os
import ast

# Statmodels para df de tiempo
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Prophet
from prophet import Prophet

# XGBOOST
import xgboost as xgb
from xgboost import plot_importance, plot_tree

# Optuna
import optuna
from optuna.samplers import TPESampler

# Métricas de error
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Librerías para matemáticas y estadística
import math
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import uniform

# Otras funcionalidades
from typing import Union
from tqdm import tqdm_notebook
from itertools import product
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import ParameterGrid

# Módulos propios
try:
    from IMForecast.ts_functions import ts_functions
except:
    from ts_functions import ts_functions


class ProphetIM:
    
    def __init__(self):
        self.ts_funcs = ts_functions()
        
    
    def prophet_training(self, 
                         training_data : pd.DataFrame, 
                         params        : dict
                         ):
        
        """
        Función que entrena un modelo Prophet
        Parámetros:
        -----------
        training_data : pd.DataFrame
            DataFrame con columna fecha 'ds' y variable objetivo 'y'
        params : dict
            Diccionario con los parámetros del modelo
        Retorna:
        --------
        model : Prophet
            Modelo Prophet ajustado
        """
        
        import logging

        # Configura el nivel de logging para desactivar los mensajes INFO de Prophet
        logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)

        # 1) Entrenamiento del modelo
        # ------------------------------------------------------------------------------------------------------
        if 'ds' not in training_data.columns and training_data.shape[1] == 1:
            training_data = training_data.reset_index()
            training_data.columns = ['ds', 'y']
        else:
            training_data.columns = ['ds', 'y']
        
        
        fit_model = Prophet(**params, interval_width = 0.95).fit(training_data)
        
        return fit_model


    # Función para calcular los pronósticos
    def prophet_forecast(self, 
                         training_data   : pd.DataFrame, 
                         model           : Prophet, 
                         fwd_periods     : int, 
                         freq_type       : str, 
                         do_simulations  : False, 
                         remove_outliers = False
                         ) -> pd.DataFrame: 
        
        """
        Función que entrena un modelo Prophet y realiza el pronóstico correspondiente
        Parámetros:
        -----------
        training_data : pd.DataFrame
            DataFrame con columna fecha 'ds' y variable objetivo 'y'
        model : Prophet
            Modelo Prophet ajustado
        fwd_periods : int
            Número de períodos hacia adelante a pronosticar
        freq_type : str
            Frecuencia de los datos (e.g., 'D', 'W', 'M', 'Q', 'Y')
        do_simulations : bool, opcional
            Indica si se realizan simulaciones de Monte Carlo
        remove_outliers : bool, opcional
            Indica si se eliminan los valores atípicos de las simulaciones
        Retorna:
        --------
        forecast : pd.DataFrame
            DataFrame con los pronósticos
        mc_results : pd.DataFrame
            DataFrame con las simulaciones
        model_fitted : pd.DataFrame
            DataFrame con los valores ajustados
        """
        if 'ds' not in training_data.columns and training_data.shape[1] == 1:
            training_data = training_data.reset_index()
            training_data.columns = ['ds', 'y']
            
        # Valores ajustados
        model_fitted = (model
                        .predict(training_data)[['ds', 'yhat']]
                        .rename(columns = {'ds': 'date', 'yhat': 'forecast'})
                        .set_index('date')['forecast']
                        )
        
        model_fitted = pd.DataFrame(model_fitted)
        model_fitted.index.name = 'date'
        model_fitted.columns = ['predicted']
        
        # Crear un dataframe con los datos de entrenamiento y los valores ajustados
        model_fitted['observed'] = training_data.tail(model_fitted.shape[0])['y'].values
        
        # Calcular residuales
        residuals = training_data['y'].values - model_fitted['predicted'].values  # Residuales
        residuals = pd.DataFrame(np.asarray(residuals), columns = ['ts'])
        residuals.index = training_data['ds']
        
        # 2) Pronóstico
        # ------------------------------------------------------------------------------------------------------
        # Crear futuro DataFrame para "steps" trimestres adelante
        future = model.make_future_dataframe(
            periods = fwd_periods,
            freq    = freq_type
        )

        # Objeto de pronóstico inicial
        forecast = (model.predict(future)
                    .tail(fwd_periods).copy()
                    .rename(columns = {'ds': 'date', 'yhat': 'forecast'})
                    .set_index('date')['forecast']
                    )

        forecast.index.name = 'date'
        
        # 2) Simulaciones
        # ------------------------------------------------------------------------------------------------------
        if do_simulations:
            # Fijar la semilla aleatoria
            np.random.seed(42)
            
            mc_results = ts_functions().simulate_from_residuals(
                residuals       = residuals,
                forecast        = forecast,
                repetitions     = 1000, 
                error           = "add", 
                random_state    = 42,
                remove_outliers = remove_outliers
            )
            
            simulations = pd.DataFrame(mc_results, index = forecast.index)
            
            # Validar si alguno de los tres elementos tiene NaN
            if forecast.isnull().any() or simulations.isnull().any().any():
                raise ValueError('El pronóstico o las simulaciones contienen valores NaN.')
            
            simulations.index.name = 'date'
        
            return (forecast, simulations, model_fitted)
        
        else:
            
            return (forecast, model_fitted)
        
    
    # Fución para la validación en ventana móvil
    def optimization_process_prophet(
        self,
        df                : pd.DataFrame,
        initial_train_date: str,
        val_size          : int,
        steps             : int,
        params            : dict,
        freq_type         : str,
        rolling_window    = False
    ):
        """
        Función que entrena un modelo Prophet en una ventana móvil
        Parámetros:
        -----------
        df : pd.DataFrame
            DataFrame con índice de tiempo 'date' y variable objetivo 'ts'
        initial_train_date : str
            Fecha de inicio para los datos de entrenamiento
        val_size : int
            Tamaño de la ventana de validación
        steps : int
            Número de pasos a pronosticar
        params : dict
            Diccionario con los parámetros del modelo Prophet
        freq_type : str
            Frecuencia de los datos (e.g., 'D', 'W', 'M', 'Q', 'Y')
        rolling_window : bool, opcional
            Indica si se utiliza una ventana móvil
        Retorna:
        --------
        mean_rmse : float
            Error promedio
        df_bt : pd.DataFrame
            DataFrame con los datos de bt
        df_fitted : pd.DataFrame
            DataFrame con los datos ajustados
        """
        
        # Resultados tras cada iteración
        errors = []
        bt = []
        fitted = []
        last_error = None
        
        # 1) Dividir los datos en folds
        # ------------------------------------------------------------------------------------------------------
        cv_data = self.ts_funcs.ts_split(df                 = df, 
                                         initial_train_date = initial_train_date, 
                                         val_size           = val_size,
                                         steps              = steps,
                                         rolling_window     = rolling_window
                                         )
        
        for i in range(len(cv_data[0])): # Número de folds
            try:
                import warnings
                warnings.filterwarnings('ignore')
                
                # 2) Datos de entrenamiento
                # ------------------------------------------------------------------------------------------------------
                training_data = cv_data[0]['fold_train_' + str(i)]
                
                # Ajuste dado el formato que exige Prophet
                training_data = training_data.reset_index()
                training_data.columns = ['ds', 'y']
                
                # 3) Datos de validación
                # ------------------------------------------------------------------------------------------------------
                val_data = cv_data[1]['fold_val_' + str(i)]
                
                if training_data['ds'].iloc[-1] >= val_data.index[0]:
                    raise ValueError(f'Los datos de entrenamiento y validación se superponen en el fold {i}.')
                    
                # 4) Ajuste del modelo y pronósticos
                # ------------------------------------------------------------------------------------------------------
                model = self.prophet_training(
                    training_data = training_data,
                    params        = params
                )
                
                forecast = self.prophet_forecast(
                    training_data  = training_data,
                    model          = model,
                    fwd_periods    = len(val_data),
                    freq_type      = freq_type,
                    do_simulations = False
                )
                
                # Se debe homogenizar el índice de los pronósticos
                forecast[0].index = val_data.index
                
                # Datos de bt
                bt.append(pd.DataFrame({
                    'observed'         : val_data['ts'],
                    'predicted'        : forecast[0],
                    'model'            : [params] * len(val_data),
                    'fold'             : [i] * len(val_data),
                    'date_range_train' : [f"{training_data['ds'].iloc[0]:%Y-%m-%d} - {training_data['ds'].iloc[-1]:%Y-%m-%d}"] * len(val_data),
                    'date_range_val'   : [f"{val_data.index[0]:%Y-%m-%d} - {val_data.index[-1]:%Y-%m-%d}"] * len(val_data)
                    }))
                
                # Datos ajustados
                mod_fitted = forecast[1].copy()
                mod_fitted['model'] = [params] * len(mod_fitted)
                mod_fitted['fold'] = [i] * len(mod_fitted)
                mod_fitted['date_range_train'] = [f"{training_data['ds'].iloc[0]:%Y-%m-%d} - {training_data['ds'].iloc[-1]:%Y-%m-%d}"] * len(mod_fitted)
                mod_fitted['date_range_val'] = [f"{val_data.index[0]:%Y-%m-%d} - {val_data.index[-1]:%Y-%m-%d}"] * len(mod_fitted)
                
                fitted.append(mod_fitted)

                # 5) Métrica de error
                # ------------------------------------------------------------------------------------------------------
                rmse_fold = self.ts_funcs.evaluate_model(y_test = val_data['ts'], 
                                                         y_pred = forecast[0]
                                                         )
                
                # Consolidar métricas de error
                errors.append(rmse_fold)
                
            except Exception as e:
                last_error = e
                pass
        
        # Calcular el error promedio
        if len(errors) == len(cv_data[0]):
            mean_rmse = np.mean(errors)
        else:
            raise ValueError(last_error) # Nos quedamos con el último error registrado
        
        
        # DataFrame bt consolidado
        if len(bt) > 0:
            df_bt = pd.concat(bt)
            
            df_fitted = pd.concat(fitted)
        else:
            raise ValueError('No se registraron datos para el backtest.')
        
        return [mean_rmse, df_bt, df_fitted]


    # Función para la hiperparametrización del modelo Prophet
    def optimal_regular_parameters_prophet(self,
                                           df                : pd.DataFrame,
                                           initial_train_date: str,
                                           val_size          : int,
                                           steps             : int,
                                           freq_type         : str,
                                           rolling_window    = False,
                                           verbose           = False
                                           ):
        """
        Realiza la hiperparametrización para el modelo Prophet
        Parámetros:
        -----------
        df : pd.DataFrame
            DataFrame con índice de tiempo 'date' y variable objetivo 'ts'
        initial_train_date : str
            Fecha de inicio para los datos de entrenamiento
        val_size : int
            Tamaño de la ventana de validación
        steps : int
            Número de pasos a pronosticar
        freq_type : str
            Frecuencia de los datos (e.g., 'D', 'W', 'M', 'Q', 'Y')
        rolling_window : bool, opcional
            Indica si se utiliza una ventana móvil
        verbose : bool, opcional
            Indica si se imprimen mensajes de progreso
        Retorna:
        --------    
        reg_res_prophet : pd.DataFrame
            DataFrame con los resultados de la hiperparametrización
        df_bt_prophet : pd.DataFrame
            DataFrame con los datos de bt
        u_fit_prophet : pd.DataFrame
            DataFrame con los datos ajustados
        """
        
        # 1) Grilla de hiperparámetros
        # ------------------------------------------------------------------------------------------------------
        param_grid_prophet = {
            'changepoint_prior_scale': [0.1, 0.2, 0.3, 0.4, 0.5],
            'seasonality_prior_scale': [0.01, 0.02, 0.03, 0.04, 0.05],
            'seasonality_mode': ['additive']
            }

        grid_prophet = list(ParameterGrid(param_grid_prophet))

        # 2) Iterar sobre la grilla de hiperparámetros
        # ------------------------------------------------------------------------------------------------------
        prophet_metrics = []
        bt_df_prophet = []
        fitted_df_prophet = []

        for i, params in enumerate(grid_prophet):
            
            try:                
                opt_prophet = self.optimization_process_prophet(
                    df                 = df,
                    initial_train_date = initial_train_date,
                    val_size           = val_size,
                    steps              = steps,
                    params             = params,
                    freq_type          = freq_type,
                    rolling_window     = rolling_window
                )
                
                prophet_metrics.append({
                    'model': params,
                    'error_metric': opt_prophet[0]
                })
                
                # Consolidar los datos de bt
                bt_df_prophet.append(opt_prophet[1])
                
                # Consolidar los datos ajustados
                fitted_df_prophet.append(opt_prophet[2])
                
                if verbose:
                    print('-' * 100)
                    print(f'¡Iteración Exitosa! {i} (MODELO PROPHET) con los parámetros {params}')
                    print('-' * 100)
                    
            except Exception as e:
                if verbose:
                    print('*' * 100)
                    print(f'¡ERROR EN LA ITERACIÓN! {i} (MODELO PROPHET)  con los parametros {params}: {e}')
                    print('*' * 100)              
                else:
                    pass
            
        # Union de los datos de métricas de error
        reg_res_prophet = pd.DataFrame(prophet_metrics).sort_values('error_metric').reset_index(drop = True) # NOTE: Importante ordenar por error

        # Unificar los datos de bt
        df_bt_prophet = pd.concat(bt_df_prophet)
        
        # Unificar los datos ajustados
        u_fit_prophet = pd.concat(fitted_df_prophet)
        
        return reg_res_prophet, df_bt_prophet, u_fit_prophet