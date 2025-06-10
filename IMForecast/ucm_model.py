# Importar el modelo de componentes no observadas
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
import calendar
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed
import multiprocessing as mp
from sklearn.model_selection import ParameterGrid

# Statmodels para df de tiempo
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents

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
from sklearn.model_selection import ParameterGrid

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

class UnobservedComponentsIM:
    
    def __init__(self):
        self.ts_funcs = ts_functions()
    
    
    # Función para entrenar el modelo UnobservedComponents
    def ucm_training(self,
                     training_data: pd.DataFrame, 
                     params       : dict) -> any:
        """
        Entrena el modelo UnobservedComponents.
        
        Parámetros:
        -----------
        training_data : pd.DataFrame
            DataFrame con índice de tiempo 'date' y variable objetivo 'ts'
        params : dict
            Diccionario con los parámetros del modelo (por ejemplo: trend, seasonal, cycle)
        
        Retorna:
        --------
        fit_model : Modelo ajustado
        """         
        
        model = UnobservedComponents(
            training_data['ts'],
            **params
        )
        
        fit_model = model.fit(disp = False)
        
        return fit_model
    
    
    # Función para realizar pronósticos con el modelo UnobservedComponents
    def ucm_forecast(self,
                     model           : any, 
                     fwd_periods     : int, 
                     do_simulations  : bool = False,
                     remove_outliers = False
                     ):
        """
        Realiza el pronóstico a partir de un modelo UnobservedComponents ajustado.
        
        Parámetros:
        -----------
        model : Modelo ajustado
        fwd_periods : int
            Número de pasos a pronosticar
        do_simulations : bool, opcional
            Si se desea realizar simulaciones para obtener intervalos o escenarios.
            
        Retorna:
        --------
        Si do_simulations es False:
            (forecast, fitted_values)
        Si do_simulations es True:
            (forecast, simulations, fitted_values)
        """
        # Pronóstico de pasos a futuro
        forecast = model.forecast(steps = fwd_periods).rename('forecast')
        forecast.index.name = 'date'
        
        # Calcular los valores ajusatdos del modelo
        # ------------------------------------------------------------------------------------------------------
        fitted   = pd.DataFrame(model.fittedvalues)
        fitted.columns = ['predicted']
        
        # Inclusión de los datos observados
        observed = pd.DataFrame(model.model.data.orig_endog)
        observed = observed[observed.index >= fitted.index[0]].copy()
        observed.columns = ['observed']

        # Unir los datos de pronóstico y los datos ajustados
        model_fitted = pd.concat([observed, fitted], axis = 1)
        
        # Modelado de los residuos
        residuals = model.resid
        residuals = pd.DataFrame(residuals, columns = ['ts'])
            
        # 2) Simulaciones
        if do_simulations:
            # Fijar la semilla aleatoria
            np.random.seed(42)
            
            mc_results = ts_functions().simulate_from_residuals(
                residuals       = residuals.tail(32), # NOTE: Se descartan los dos primeros años ya que tienen valores atípicos.
                forecast        = forecast,
                repetitions     = 1000, 
                error           = "add", 
                random_state    = 42,
                remove_outliers = remove_outliers
            )
            
            simulations = pd.DataFrame(mc_results, index = forecast.index)
            
            simulations.index.name = 'date'
            
            # Validar que no haya valores nulos en los datos de simulación
            if forecast.isnull().any() or simulations.isnull().any().any():
                raise ValueError('El pronóstico o las simulaciones contienen valores NaN.')
            
            return (forecast, simulations, model_fitted)
        
        else:
            
            return forecast, model_fitted
    
    
    # Función para validar el modelo en ventana móvil (cross-validation)
    def optimization_process_ucm(
        self,
        df                 : pd.DataFrame,
        initial_train_date : str,
        val_size           : int,
        steps              : int,
        params             : dict,
        rolling_window     : bool = False
        ):
        """
        Entrena el modelo UnobservedComponents en validación de ventana móvil.
        
        Parámetros:
        -----------
        df : pd.DataFrame
            DataFrame con índice de tiempo 'date' y variable objetivo 'ts'
        initial_train_date : str
            Fecha de inicio del entrenamiento (formato 'YYYY-MM-DD')
        val_size : int
            Tamaño de la ventana de validación
        steps : int
            Número de pasos a pronosticar
        params : dict
            Diccionario con los parámetros del modelo (por ejemplo: trend, seasonal, cycle)
        rolling_window : bool, opcional
            Si se usa una ventana móvil para el entrenamiento.
        Retorna:
        --------
        [mean_rmse, df_bt, df_fitted]
          - mean_rmse: Error promedio en validación
          - df_bt    : DataFrame con observados y pronosticados de cada fold
          - df_fitted: DataFrame con valores ajustados durante el entrenamiento de cada fold
        """
        errors = []
        bt = []
        fitted = []
        last_error = None
        
        # Dividir la serie en folds de entrenamiento/validación
        cv_data = self.ts_funcs.ts_split(df                 = df, 
                                         initial_train_date = initial_train_date,
                                         val_size           = val_size,
                                         steps              = steps, 
                                         rolling_window     = rolling_window
                                         )
        
        for i in range(len(cv_data[0])):  # Número de folds
            try:
                import warnings
                warnings.filterwarnings('ignore')
                
                # Datos de entrenamiento y validación
                training_data = cv_data[0]['fold_train_' + str(i)]
                
                val_data = cv_data[1]['fold_val_' + str(i)]
                
                if training_data.index[-1] >= val_data.index[0]:
                    raise ValueError(f'Los datos de entrenamiento y validación se superponen en el fold {i}.')
                
                # Entrenar el modelo
                fit_model = self.ucm_training(training_data = training_data, 
                                              params        = params
                                              )
                
                # Realizar el pronóstico
                forecast = self.ucm_forecast(model          = fit_model, 
                                             fwd_periods    = len(val_data), 
                                             do_simulations = False
                                             )
                
                # Ajustar índices de pronóstico y de valores ajustados
                forecast[0].index = val_data.index
                
                # Almacenar datos de validación y entrenamiento
                bt.append(pd.DataFrame({
                    'observed'         : val_data['ts'],
                    'predicted'        : forecast[0],
                    'model'            : [params] * len(val_data),
                    'fold'             : [i] * len(val_data),
                    'date_range_train': [f"{training_data.index[0]:%Y-%m-%d} - {training_data.index[-1]:%Y-%m-%d}"] * len(val_data),
                    'date_range_val'  : [f"{val_data.index[0]:%Y-%m-%d} - {val_data.index[-1]:%Y-%m-%d}"] * len(val_data)
                    }))
                
                # Datos ajustados
                mod_fitted = forecast[1].copy()
                mod_fitted['model'] = [params] * len(mod_fitted)
                mod_fitted['fold'] = [i] * len(mod_fitted)
                mod_fitted['date_range_train'] = [f"{training_data.index[0]:%Y-%m-%d} - {training_data.index[-1]:%Y-%m-%d}"] * len(mod_fitted)
                mod_fitted['date_range_val'] = [f"{val_data.index[0]:%Y-%m-%d} - {val_data.index[-1]:%Y-%m-%d}"] * len(mod_fitted)
                
                fitted.append(mod_fitted)
                         
                # Calcular métrica de error (por ejemplo, RMSE)
                rmse_fold = self.ts_funcs.evaluate_model(y_test = val_data['ts'], 
                                                         y_pred = forecast[0]
                                                         )
                
                errors.append(rmse_fold)
                
            except Exception as e:
                last_error = e
                pass
                
        # Calcular el error promedio
        if len(errors) == len(cv_data[0]):
            mean_rmse = np.mean(errors)
        else:
            raise ValueError(last_error)
        
        # DataFrame bt consolidado
        if len(bt) > 0:
            df_bt = pd.concat(bt)
            
            df_fitted = pd.concat(fitted)
        else:
            raise ValueError('No se registraron datos para el backtest.')
        
        return [mean_rmse, df_bt, df_fitted]
    
    
    # Función para optimizar los hiperparámetros del modelo UnobservedComponents
    def optimal_parameters_ucm(
            self,
            df                : pd.DataFrame,
            initial_train_date: str,
            val_size          : int,
            steps             : int,
            rolling_window    : bool = False,
            freq              : int  = 4,
            verbose           : bool = False
            ):
        """
        Búsqueda exhaustiva de hiper-parámetros UnobservedComponents en paralelo.

        Parámetros:
        -----------
        df : pd.DataFrame
            DataFrame con índice de tiempo 'date' y variable objetivo 'ts'
        initial_train_date : str
            Fecha de inicio del entrenamiento (formato 'YYYY-MM-DD')
        val_size : int
            Tamaño de la ventana de validación
        steps : int
            Número de pasos a pronosticar
        rolling_window : bool, opcional
            Si se usa una ventana móvil para el entrenamiento.
        freq : int, opcional
            Frecuencia de la serie temporal (por ejemplo, 4 para trimestral)
        verbose : bool, opcional
            Si se imprime información detallada sobre el proceso de optimización.
        n_jobs : int, opcional
            Número de núcleos a usar para el paralelismo. Si es None, usa todos menos uno.
        Retorna:
        --------
        df_ucm_metrics : pd.DataFrame
            DataFrame con los resultados de la evaluación de los modelos, ordenados por métrica de error
        bt_df_ucm : pd.DataFrame
            DataFrame con observados y pronosticados de cada fold
        fitted_df_ucm : pd.DataFrame
            DataFrame con valores ajustados durante el entrenamiento de cada fold
        """

        # 1) Definición de la grilla
        param_grid_ucm = {
            'trend'               : [True, False],
            'seasonal'            : [freq],
            'cycle'               : [True, False],
            'level'               : [True, False],
            'stochastic_cycle'    : [True, False],
            'stochastic_seasonal' : [True],
            'stochastic_trend'    : [True, False],
            'stochastic_level'    : [True, False],
            'irregular'           : [True, False],
            'autoregressive'      : [None, 1, 2, 3, 4]
        }
        
        grid_ucm = list(ParameterGrid(param_grid_ucm))

        # 2) Iterar sobre la grilla de hiperparámetros
        ucm_metrics = []
        bt_df_ucm = []
        fitted_df_ucm = []

        for i, params in enumerate(grid_ucm):
            
            try:                
                opt_ucm = self.optimization_process_ucm(
                    df                 = df,
                    initial_train_date = initial_train_date,
                    val_size           = val_size,
                    steps              = steps,
                    params             = params,
                    rolling_window     = rolling_window
                )
                
                # Almacenar los resultados
                ucm_metrics.append({
                    'model': params,
                    'error_metric': opt_ucm[0]
                })
                
                bt_df_ucm.append(opt_ucm[1])
                
                fitted_df_ucm.append(opt_ucm[2])
                
                if verbose:
                    print('-' * 100)
                    print(f'¡Iteración Exitosa! {i} (MODELO ucm) con los parámetros {params}')
                    print('-' * 100)
                    
            except Exception as e:
                if verbose:
                    print('*' * 100)
                    print(f'¡ERROR EN LA ITERACIÓN! {i} (MODELO ucm)  con los parametros {params}: {e}')
                    print('*' * 100)       
                pass
            

        # Unificar los datos de métricas de error y de resultados
        df_ucm_metrics = pd.DataFrame(ucm_metrics).sort_values('error_metric').reset_index(drop = True)
        
        # Unificar los datos de bt
        unified_bt_ucm = pd.concat(bt_df_ucm)
        
        # Unificar los datos de fitted
        u_fit_ucm = pd.concat(fitted_df_ucm)
        

        return df_ucm_metrics, unified_bt_ucm, u_fit_ucm
