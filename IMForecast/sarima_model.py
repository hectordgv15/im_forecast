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
from joblib import Parallel, delayed
import multiprocessing as mp

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


class SarimaIM:
    
    def __init__(self):
        self.ts_funcs = ts_functions()
    
    # Función para la optimización del modelo SARIMA por AIC/BIC
    def optimize_SARIMA_AIC(self,
                            endog, 
                            params: dict
                            ) -> pd.DataFrame:
        """
        Función que optimiza un modelo SARIMA por AIC
        Parámetros:
        -----------
        endog : pd.DataFrame
            DataFrame con los datos de entrenamiento
        params : dict
            Lista de diccionarios con los parámetros del modelo SARIMA
        Retorna:
        --------
        result_df : pd.DataFrame
            DataFrame con los parámetros del modelo y su AIC
        """
    
        results = []
        
        for dict_params in params:
            try: 
                
                model = self.sarima_training(
                    training_data  = endog,
                    params         = dict_params
                )
                
            except Exception as e:
                print(f"Error al entrenar el modelo con los parámetros {dict_params}: {e}")
                continue
            
            results.append([dict_params, model.aic])
            
        result_df = pd.DataFrame(results)

        result_df.columns = ['model', 'AIC']

        
        return result_df
    
    
    def sarima_training(self,
                        training_data  : pd.DataFrame,
                        params         : dict
                        ) -> SARIMAX:
        """
        Función que entrena un modelo SARIMA
        Parámetros:
        -----------
        training_data : pd.DataFrame
            DataFrame con los datos de entrenamiento
        params : dict
            Diccionario con los parámetros del modelo SARIMA
        Retorna:
        --------
        fit_model : SARIMAX
            Modelo ajustado
        """
        
        import logging
        logging.getLogger('statsmodels').setLevel(logging.WARNING)
        
        # Ajustar el modelo SARIMA
        model = SARIMAX(
            training_data,
            order          = params['order'],
            seasonal_order = params['seasonal_order'],
            trend          = params['trend'],
        )

        fit_model = model.fit(disp = 0) 
        
        return(fit_model)


    # Función para calcular los pronósticos
    def sarima_forecast(self,
                        model                   : SARIMAX, 
                        fwd_periods             : int, 
                        do_simulations          = False,
                        alternative_simulations = False,
                        remove_outliers         = False
                        ) -> pd.Series:
        """
        Función que realiza el pronóstico con un modelo SARIMA
        Parámetros:
        -----------
        model : SARIMAX
            Modelo SARIMA ajustado
        fwd_periods : int
            Número de períodos a pronosticar
        do_simulations : bool, opcional
            Indica si se realizan simulaciones (por defecto es False)
        alternative_simulations : bool, opcional
            Indica si se realizan simulaciones alternativas (por defecto es False)
        remove_outliers : bool, opcional
            Indica si se eliminan los valores atípicos (por defecto es False)
        Retorna:
        --------
        forecast : pd.Series
            Serie con los pronósticos
        simulations : pd.DataFrame
            DataFrame con las simulaciones
        model_fitted : pd.Series
            Serie con los valores ajustados
        """
        import logging
        logging.getLogger('statsmodels').setLevel(logging.WARNING)
        
        # 1) Pronóstico promedio
        # ------------------------------------------------------------------------------------------------------
        forecast = model.get_forecast(steps = fwd_periods).predicted_mean.rename('forecast')
        forecast.index.name = 'date'
        
        # Calcular los valores ajusatdos del modelo
        # ------------------------------------------------------------------------------------------------------
        fitted   = pd.DataFrame(model.fittedvalues)
        fitted.columns = ['predicted']
        
        # Agregar los datos ajustados
        observed = pd.DataFrame(model.model.data.orig_endog)
        observed = observed[observed.index >= fitted.index[0]].copy()
        observed.columns = ['observed']

        # Unir los datos de pronóstico y los datos ajustados
        model_fitted = pd.concat([observed, fitted], axis = 1)
        
        
        # Si el último valor de los fitted values supera por 100 veces el último valor de los datos observados, se considera un error
        if model_fitted['predicted'].iloc[-1] > 100 * model_fitted['observed'].iloc[-1]:
            raise ValueError('El último valor de los fitted values supera por 100 veces el último valor de los datos observados. Esto puede indicar un problema con el modelo o los datos.')
        
        
        # Modelado de los residuos
        # ------------------------------------------------------------------------------------------------------
        residuals = model.resid
        residuals = pd.DataFrame(residuals, columns = ['ts'])
        
        # 2) Simulaciones
        # ------------------------------------------------------------------------------------------------------
        if do_simulations:
            
            if alternative_simulations == True:
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
                
            else:
                # Fijar la semilla aleatoria
                np.random.seed(42)

                # Simulaciones
                simulations = model.simulate(fwd_periods,
                                             repetitions  = 1000, 
                                             anchor       = len(model.data.endog),
                                             random_state = 42
                                             )
                
                simulations.columns = simulations.columns.get_level_values(1) # TODO: Tener en cuenta este cambio hecho
    
                simulations.index.name = 'date'

            # Validar si alguno de los tres elementos tiene NaN
            if forecast.isnull().any() or simulations.isnull().any().any():
                raise ValueError('El pronóstico o las simulaciones contienen valores NaN.')
            
            return(forecast, simulations, model_fitted)
        
        else:
            
            return(forecast, model_fitted)
        
        
        # Fución para la validación en ventana móvil
    def optimization_process_sarima(
        self,
        df                 : pd.DataFrame,
        initial_train_date : str,
        val_size           : int,
        steps              : int,
        params             : dict,
        rolling_window     = False
    ):
        """
        Función que entrena un modelo SARIMA en una ventana móvil
        Parámetros:
        -----------
        df : pd.DataFrame
            DataFrame con los datos de entrenamiento
        initial_train_date : str
            Fecha de inicio del entrenamiento
        val_size : int
            Tamaño de la ventana de validación
        steps : int
            Número de pasos a pronosticar
        params : dict
            Diccionario con los parámetros del modelo SARIMA
        rolling_window : bool, opcional
            Indica si se usa una ventana móvil (por defecto es False)
        Retorna:
        --------
        mean_rmse : float
            Error promedio de la validación cruzada
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
                
                
                # 3) Datos de validación
                # ------------------------------------------------------------------------------------------------------
                val_data = cv_data[1]['fold_val_' + str(i)]
                
                
                if training_data.index[-1] >= val_data.index[0]:
                    raise ValueError(f'Los datos de entrenamiento y validación se superponen en el fold {i}.')
                    
                # 4) Ajustar el modelo
                # ------------------------------------------------------------------------------------------------------
                fit_model = self.sarima_training(training_data  = training_data,
                                                 params         = params
                                                 )
                
                # 5) Pronóstico
                # ------------------------------------------------------------------------------------------------------
                forecast = self.sarima_forecast(model          = fit_model,
                                                fwd_periods    = len(val_data),
                                                do_simulations = False,
                                                )
                
                # Se debe homogenizar el índice de los pronósticos y los valores ajustados
                forecast[0].index = val_data.index
                
                # Almacenar datos en un dataframe consolidado
                bt.append(pd.DataFrame({
                    'observed'         : val_data['ts'],
                    'predicted'        : forecast[0],
                    'model'            : [params] * len(val_data),
                    'fold'             : [i] * len(val_data),
                    'date_range_train' : [f"{training_data.index[0]:%Y-%m-%d} - {training_data.index[-1]:%Y-%m-%d}"] * len(val_data),
                    'date_range_val'   : [f"{val_data.index[0]:%Y-%m-%d} - {val_data.index[-1]:%Y-%m-%d}"] * len(val_data)
                    }))
                
                # Datos ajustados
                mod_fitted = forecast[1].copy()
                mod_fitted['model'] = [params] * len(mod_fitted)
                mod_fitted['fold'] = [i] * len(mod_fitted)
                mod_fitted['date_range_train'] = [f"{training_data.index[0]:%Y-%m-%d} - {training_data.index[-1]:%Y-%m-%d}"] * len(mod_fitted)
                mod_fitted['date_range_val'] = [f"{val_data.index[0]:%Y-%m-%d} - {val_data.index[-1]:%Y-%m-%d}"] * len(mod_fitted)
                
                fitted.append(mod_fitted)
                    
                # 6) Métrica de error
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
            raise ValueError(last_error)
        
        # DataFrame bt consolidado
        if len(bt) > 0:
            df_bt = pd.concat(bt)
            
            df_fitted = pd.concat(fitted)
        
        else:
            raise ValueError('No se registraron datos para el backtest.')
        
        return [mean_rmse, df_bt, df_fitted]
    

    # Optimización regular
    # ---------------------------------------------------------------------
    def optimal_parameters_sarima(
            self,
            df                : pd.DataFrame,
            initial_train_date: str,
            val_size          : int,
            steps             : int,
            season_freq       : int,
            rolling_window    = False,
            verbose           = False,
            d_value           = None,
            s_d_value         = None
        ):
        """
        Búsqueda exhaustiva SARIMA paralelizada con joblib.

        Parámetros:
        -----------
        df : pd.DataFrame
            DataFrame con los datos de entrenamiento
        initial_train_date : str
            Fecha de inicio del entrenamiento
        val_size : int
            Tamaño de la ventana de validación
        steps : int
            Número de pasos a pronosticar
        season_freq : int
            Frecuencia estacional del modelo SARIMA
        rolling_window : bool, opcional
            Indica si se usa una ventana móvil (por defecto es False)
        verbose : bool, opcional
            Indica si se imprime información adicional (por defecto es False)
        d_value : int, opcional
            Valor de d para el modelo SARIMA (por defecto es None)
        s_d_value : int, opcional
            Valor de s_d para el modelo SARIMA (por defecto es None)
        n_jobs : int, opcional
            Número de trabajos paralelos a ejecutar (por defecto es None, lo que usa todos los núcleos menos uno)
        Retorna:
        --------
        res_sarima : pd.DataFrame
            DataFrame con los resultados de la optimización del modelo SARIMA 
        df_bt_sarima : pd.DataFrame
            DataFrame con los datos de backtest del modelo SARIMA
        u_fit_sarima : pd.DataFrame
            DataFrame con los valores ajustados del modelo SARIMA
        """

        p = range(0, 3) # NOTE: Revisar que esté el número correcto.
        q = range(0, 3)

        pdq  = list(itertools.product(p, [d_value], q))
        
        pdqs = [(x[0], x[1], x[2], season_freq)
                for x in itertools.product(p, [s_d_value], q)]

        trend_options = ['n', 'c', 't', 'ct', None]

        grid_sarima = [
            {'order': order, 'seasonal_order': seasonal_order, 'trend': trend}
            for order in pdq
            for seasonal_order in pdqs
            for trend in trend_options
        ]
        

        # 2) Iterar sobre la grilla de hiperparámetros
        sarima_metrics = []
        bt_df_sarima = []
        fitted_df_sarima = []

        for i, params in enumerate(grid_sarima):
            
            try:                
                opt_sarima = self.optimization_process_sarima(
                    df                 = df,
                    initial_train_date = initial_train_date,
                    val_size           = val_size,
                    steps              = steps,
                    params             = params,
                    rolling_window     = rolling_window
                )
                
                # Almacenar los resultados
                sarima_metrics.append({
                    'model': params,
                    'error_metric': opt_sarima[0]
                })
                
                bt_df_sarima.append(opt_sarima[1])
                
                fitted_df_sarima.append(opt_sarima[2])
                
                if verbose:
                    print('-' * 100)
                    print(f'¡Iteración Exitosa! {i} (MODELO sarima) con los parámetros {params}')
                    print('-' * 100)
                    
            except Exception as e:
                if verbose:
                    print('*' * 100)
                    print(f'¡ERROR EN LA ITERACIÓN! {i} (MODELO sarima)  con los parametros {params}: {e}')
                    print('*' * 100)       
                pass
            

        # Unificar los datos de métricas de error y de resultados
        df_sarima_metrics = pd.DataFrame(sarima_metrics).sort_values('error_metric').reset_index(drop = True)
        
        # Unificar los datos de bt
        unified_bt_sarima = pd.concat(bt_df_sarima)
        
        # Unificar los datos de fitted
        u_fit_sarima = pd.concat(fitted_df_sarima)
        
        # 6) Optimización por AIC
        opt_sarima_aic = self.optimize_SARIMA_AIC(
            endog = df,
            params = df_sarima_metrics['model'].tolist()
        )
        
        # Añadir AIC a los resultados
        df_sarima_metrics['AIC'] = opt_sarima_aic['AIC']
        
        return (
            df_sarima_metrics[['model', 'error_metric', 'AIC']],
            unified_bt_sarima,
            u_fit_sarima
        )