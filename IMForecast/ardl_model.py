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
from statsmodels.tsa.ardl import ARDL

# Módulos propios
try:
    from IMForecast.ts_functions import ts_functions
except:
    from ts_functions import ts_functions


class ArdlIM:
    
    def __init__(self):
        self.ts_funcs = ts_functions()

    # Función para entrenar el modelo ARDL
    def ardl_training(self, 
                      training_data: pd.DataFrame, 
                      params       : dict
                      ) -> ARDL:
        """
        Función que entrena un modelo ARDL
        Parámetros:
        -----------
        training_data : pd.DataFrame
            DataFrame con índice de tiempo 'date' y variable objetivo 'ts'
        params : dict
            Diccionario con los parámetros del modelo
        Retorna:
        --------
        fit_model : ARDL
            Modelo ajustado
        """
        y = training_data['ts']
        
        model = ARDL(endog = y, 
                     exog = None,
                     **params
                     )
        
        fit_model = model.fit() 
        
        return fit_model


    # Función para calcular los pronósticos
    def ardl_forecast(self,
                      model           : ARDL, 
                      fwd_periods     : int, 
                      do_simulations  = False, 
                      remove_outliers = False
                      ) -> pd.Series:
        """
        Función que realiza el pronóstico con un modelo ARDL
        Parámetros:
        -----------
        model : ARDL
            Modelo ajustado
        fwd_periods : int
            Número de pasos a pronosticar
        do_simulations : bool, opcional
            Indica si se realizan simulaciones
        Retorna:
        --------
        forecast : lista
            Lista con el pronóstico y los valores ajustados
        """
        # 1) Pronóstico promedio
        forecast = model.forecast(fwd_periods).rename('forecast')
        forecast.index.name = 'date'
        
        # Calcular los valores ajustados del modelo
        fitted   = pd.DataFrame(model.fittedvalues)
        fitted.columns = ['predicted']
        
        # Añadir los valores observados
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
                residuals       = residuals,
                forecast        = forecast,
                repetitions     = 1000, 
                error           = "add", 
                random_state    = 42,
                remove_outliers = remove_outliers,
            )
            
            simulations = pd.DataFrame(mc_results, index = forecast.index)
            
            simulations.index.name = 'date'
            
            # Validar si alguno de los tres elementos tiene NaN
            if forecast.isnull().any() or simulations.isnull().any().any():
                raise ValueError('El pronóstico o las simulaciones contienen valores NaN.')
            
            return (forecast, simulations, model_fitted)
        
        else:
            
            return (forecast, model_fitted)
        
        
    # Función para la validación en ventana móvil
    def optimization_process_ardl(
        self,
        df                : pd.DataFrame,
        initial_train_date: str,
        val_size          : int,
        steps             : int,# Tamaño de la validación
        params            : dict,
        rolling_window    = False
        ):
        
        """
        Función que entrena un modelo ARDL en una ventana móvil
        Parámetros:
        -----------
        df : pd.DataFrame
            DataFrame con índice de tiempo 'date' y variable objetivo 'ts'
        initial_train_date : str
            Fecha de inicio de los datos de entrenamiento
        val_size : int
            Tamaño de la validación
        steps : int
            Tamaño del paso para la expansión de la ventana
        params : dict
            Diccionario con los parámetros del modelo ARDL
        rolling_window : bool, opcional
            Indica si se utiliza una ventana móvil
        Retorna:
        --------
        mean_rmse : float
            Error promedio de los modelos
        df_bt : pd.DataFrame
            DataFrame con los datos de bt de cada modelo
        df_fitted : pd.DataFrame
            DataFrame con los datos ajustados de cada modelo
        """
        
        # Resultados tras cada iteración
        errors = []
        bt = []
        fitted = []
        last_error = None
                
        # 1) Dividir los datos en folds
        cv_data = self.ts_funcs.ts_split(
            df                 = df, 
            initial_train_date = initial_train_date,
            val_size           = val_size, 
            steps              = steps, 
            rolling_window     = rolling_window
            )
        
        for i in range(len(cv_data[0])):  # Número de folds
            import warnings
            try:
                warnings.filterwarnings('ignore')
                
                # 2) Datos de entrenamiento
                training_data = cv_data[0]['fold_train_' + str(i)]
                
                # 3) Datos de validación
                val_data = cv_data[1]['fold_val_' + str(i)]
                
                if training_data.index[-1] >= val_data.index[0]:
                    raise ValueError(f'Los datos de entrenamiento y validación se superponen en el fold {i}.')

                # 4) Ajustar el modelo
                fit_model = self.ardl_training(training_data = training_data, 
                                               params        = params
                                               )  

                # 5) Pronóstico
                forecast = self.ardl_forecast(
                    model           = fit_model, 
                    fwd_periods     = len(val_data),
                    do_simulations  = False
                    )
                
                # Homogeneizar el índice de los pronósticos
                forecast[0].index = val_data.index
                
                # Backtesting
                bt.append(pd.DataFrame({
                    'observed'        : val_data['ts'],
                    'predicted'       : forecast[0],
                    'model'           : [params] * len(val_data),
                    'fold'            : [i] * len(val_data),
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

                # 6) Métrica de error
                rmse_fold = self.ts_funcs.evaluate_model(
                    y_test = val_data['ts'], 
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
            raise ValueError(last_error) # Se referencia el último error registrado.
        
        # DataFrame bt consolidado
        if len(bt) > 0:
            df_bt = pd.concat(bt)
            
            df_fitted = pd.concat(fitted)
        else:
            raise ValueError('No se registraron datos para el backtest.')
        
        return [mean_rmse, df_bt, df_fitted]


    # Aplicar la función predefinida y efectuar la hiperparametrización
    def optimal_parameters_ardl(self,
                                df                : pd.DataFrame, 
                                initial_train_date: str, 
                                val_size          : int,  # Tamaño de la validación
                                steps             : int,  # Tamaño del paso para la expansión de la ventana
                                rolling_window    = False,
                                freq              = 4,  # Trimestral
                                verbose           = False
                                ):
        """
        Realiza la hiperparametrización para el modelo ARDL
        Parámetros:
        -----------
        df : pd.DataFrame
            DataFrame con índice de tiempo 'date' y variable objetivo 'ts'
        initial_train_date : str
            Fecha de inicio de los datos de entrenamiento
        val_size : int
            Tamaño de la validación
        steps : int
            Tamaño del paso para la expansión de la ventana
        rolling_window : bool, opcional
            Indica si se utiliza una ventana móvil
        freq : int, opcional
            Frecuencia de los datos (por defecto 4 para trimestral)
        verbose : bool, opcional
            Indica si se imprime información detallada durante el proceso
        Retorna:
        --------
        df_ardl_metrics : pd.DataFrame
            DataFrame con las métricas de error de cada modelo
        unified_bt_ardl : pd.DataFrame
            DataFrame con los datos de bt de cada modelo
        u_fit_ardl : pd.DataFrame
            DataFrame con los datos ajustados de cada modelo
        """
        
        # 1) Grilla de hiperparámetros
        param_grid_ardl = {
            'lags'     : [1, 2, 3, 4, 5, 6, 7, 8],
            'trend'    : ['n', 'c', 't', 'ct'],
            'seasonal' : [True, False],
            'period'   : [freq]
            }
        

        grid_ardl = list(ParameterGrid(param_grid_ardl))

        # 2) Iterar sobre la grilla de hiperparámetros
        ardl_metrics = []
        bt_df_ardl = []
        fitted_df_ardl = []

        for i, params in enumerate(grid_ardl):
            
            try:                
                opt_ardl = self.optimization_process_ardl(
                    df                 = df,
                    initial_train_date = initial_train_date,
                    val_size           = val_size,
                    steps              = steps,
                    params             = params,
                    rolling_window     = rolling_window
                )
                
                # Almacenar los resultados
                ardl_metrics.append({
                    'model': params,
                    'error_metric': opt_ardl[0]
                })
                
                bt_df_ardl.append(opt_ardl[1])
                
                fitted_df_ardl.append(opt_ardl[2])
                
                if verbose:
                    print('-' * 100)
                    print(f'¡Iteración Exitosa! {i} (MODELO ARDL) con los parámetros {params}')
                    print('-' * 100)
                    
            except Exception as e:
                if verbose:
                    print('*' * 100)
                    print(f'¡ERROR EN LA ITERACIÓN! {i} (MODELO ARDL)  con los parametros {params}: {e}')
                    print('*' * 100)       
                pass
            

        # Unificar los datos de métricas de error y de resultados
        df_ardl_metrics = pd.DataFrame(ardl_metrics).sort_values('error_metric').reset_index(drop = True)
        
        # Unificar los datos de bt
        unified_bt_ardl = pd.concat(bt_df_ardl)
        
        # Unificar los datos de fitted
        u_fit_ardl = pd.concat(fitted_df_ardl)
        

        return df_ardl_metrics, unified_bt_ardl, u_fit_ardl