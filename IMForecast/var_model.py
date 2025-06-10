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
from statsmodels.tsa.vector_ar.var_model import VAR

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
from statsmodels.tsa.stattools import grangercausalitytests

import warnings

# Módulos propios
try:
    from IMForecast.ts_functions import ts_functions
except:
    from ts_functions import ts_functions


class varIM:
    
    def __init__(self):
        self.ts_funcs = ts_functions()
    
    
    def best_predictors(
        self,
        stationary_data : pd.DataFrame,
        var_name        : str,
        ) -> pd.DataFrame:
        """
        Función que determina los mejores regresores para una variable objetivo en un modelo VAR
        Parámetros:
        -----------
        stationary_data : pd.DataFrame
            DataFrame con los datos estacionarios
        var_name : str
            Nombre de la variable objetivo para la que se buscan los regresores
        Retorna:
        --------
        top_predictors : list
            Lista con los nombres de las mejores variables predictoras
        """

        # 1) Determinar los regresores óptimos
        # ------------------------------------------------------------------------------------------------------
        if var_name not in stationary_data.columns:
            raise ValueError(f'La variable {var_name} no está en el DataFrame.')
        
        corr_df = stationary_data.corr().abs()[var_name].sort_values(ascending = False)
        corr_df = pd.DataFrame(corr_df[1:])
        corr_df = corr_df[corr_df[var_name] > 0.3]
        
        # Si no hay regresores, reducir el umbral de correlación
        corr_threshold = 0.3
        
        while corr_df.empty:
            
            corr_df = stationary_data.corr().abs()[var_name].sort_values(ascending = False)
            corr_df = pd.DataFrame(corr_df[1:])
            corr_df = corr_df[corr_df[var_name] > (corr_threshold * 0.9)]
            
            corr_threshold *= 0.9

        maxlag = 4

        p_values_summary = {}
        
        warnings.filterwarnings('ignore')

        for predictor in corr_df.index:
            if predictor != var_name:
                test_result = grangercausalitytests(stationary_data[[var_name, predictor]], maxlag = maxlag, verbose = False)
                p_values = [test_result[i + 1][0]['ssr_ftest'][1] for i in range(maxlag)]
                min_p_value = np.min(p_values)
                p_values_summary[predictor] = min_p_value

        # Convertir a DataFrame para ordenarlo
        p_values_df = pd.DataFrame.from_dict(p_values_summary, orient = 'index', columns = ['min_p_value'])
        p_values_df = p_values_df.sort_values('min_p_value')

        # Seleccionar las 4 mejores variables predictoras
        top_predictors = p_values_df.head(5).index.tolist()

        # Añadir la variable objetivo
        top_predictors = list(set(top_predictors + [var_name]))
        
        # Pasar índices a datetime
        stationary_data.index = pd.to_datetime(stationary_data.index)
        
        return top_predictors


    def var_training(self,
                     stationary_data : pd.DataFrame,
                     top_predictors  : list,
                     params          : dict
                     ):
        """
        Función que entrena un modelo VAR con los datos estacionarios y los mejores regresores
        Parámetros:
        -----------
        stationary_data : pd.DataFrame
            DataFrame con los datos estacionarios
        top_predictors : list
            Lista con los nombres de las mejores variables predictoras
        params : dict
            Diccionario con los parámetros del modelo
        Retorna:
        --------
        fitted_model : VAR
            Modelo VAR ajustado
        """
        
        # 1) Preprocesar los datos
        # ------------------------------------------------------------------------------------------------------
        df_model = stationary_data[top_predictors].copy()
        
        # 2) Ajustar el modelo VAR
        # ------------------------------------------------------------------------------------------------------
        model = VAR(df_model)
        
        fitted_model = model.fit(maxlags = params['maxlags'], ic = 'aic') # NOTE: Revisar si se guarda el params['maxlags'] o el k_ar. Realmente los retardos usados son k_ar.
        
        # Si se forzó, refit con k_ar = 1
        if fitted_model.k_ar == 0:
            fitted_model = model.fit(1)
        
        return fitted_model


    # Función para calcular los pronósticos
    def var_forecast(self,
                     regular_data            : pd.DataFrame,
                     stationary_data         : pd.DataFrame,
                     fitted_model            : VAR, 
                     target_variable         : str,
                     top_predictors          : list,
                     fwd_periods             : int, 
                     alternative_simulations = False,
                     do_simulations          = False,
                     remove_outliers         : bool = False,
                     freq                    : int = 4,
                     d_var                   = 0,
                     D_var                   = 0
                     ) -> pd.DataFrame: 
        """
        Función que entrena un modelo var y realiza el pronóstico correspondiente
        Parámetros:
        -----------
        regular_data : pd.DataFrame
            DataFrame con los datos originales
        stationary_data : pd.DataFrame
            DataFrame con los datos estacionarios
        fitted_model : VAR
            Modelo VAR ajustado
        target_variable : str
            Nombre de la variable objetivo
        top_predictors : list
            Lista con los nombres de las mejores variables predictoras
        fwd_periods : int
            Número de pasos a pronosticar
        alternative_simulations : bool, opcional
            Indica si se utilizan simulaciones alternativas
        do_simulations : bool, opcional
            Indica si se realizan simulaciones
        remove_outliers : bool, opcional
            Indica si se eliminan los valores atípicos
        freq : int, opcional
            Frecuencia de los datos (por defecto 4 para datos trimestrales)
        d_var : int, opcional
            Orden de diferenciación estacional (por defecto 0)
        D_var : int, opcional
            Orden de diferenciación estacional (por defecto 0)
            
        Retorna:
        --------
        forecast : pd.DataFrame
            DataFrame con los pronósticos
        mc_results : pd.DataFrame
            DataFrame con las simulaciones
        model_fitted : pd.DataFrame
            DataFrame con los valores ajustados
        """
        # 1) Preprocesar los datos
        # ------------------------------------------------------------------------------------------------------
        df_model = stationary_data[top_predictors].copy().asfreq('Q')
        
        # 2) Ajustar el modelo VAR
        # ------------------------------------------------------------------------------------------------------
        objective_lags = fitted_model.k_ar
                
        # Pasos hacia delante
        mod_n_forecast  = fwd_periods + objective_lags
        last_obs        = df_model.values[-objective_lags:]
        point_forecast  = fitted_model.forecast(last_obs, steps = fwd_periods)

        # Índice futuro para alinear con el DataFrame original
        idx_future_point = pd.date_range(df_model.index[-1], periods = fwd_periods + 1, freq = df_model.index.freq)[1:]

        idx_future_sim = pd.date_range(df_model.index[-(objective_lags + 1)], periods = mod_n_forecast + 1, freq = df_model.index.freq)[1:]

        # DataFrame con la predicción puntual
        df_point_forecast = pd.DataFrame(point_forecast, 
                                         index   = idx_future_point, 
                                         columns = df_model.columns
                                         )
        
        
        # 3) Obtener pronósticos en niveles
        # ------------------------------------------------------------------------------------------------------
        # Pronóstico limpio
        forecast = df_point_forecast[target_variable]
        forecast.name = 'forecast'
        
        # Residuales
        residuals = fitted_model.resid[target_variable]
        
        # Valores ajustados
        fitted_values = fitted_model.fittedvalues
        fitted_values = pd.DataFrame(fitted_values[target_variable])
        fitted_values.columns = ['predicted']
        fitted_values['observed'] = regular_data[regular_data.index.isin(fitted_values.index)][target_variable]
        
        
        # 3) Convertir los datos ajustados a través de la inversa
        # ------------------------------------------------------------------------------------------------------
        col_pre = fitted_values.columns.get_loc('predicted')
        col_obs = fitted_values.columns.get_loc('observed')
        
        
        if d_var == 1 and D_var == 0:
            for i in range(fitted_values.shape[0]):
                if i == 0:
                    fitted_values.iloc[i, col_pre] = np.nan
                else:
                    fitted_values.iloc[i, col_pre] = fitted_values.iloc[i, col_pre] + fitted_values.iloc[i - 1, col_obs]
        
        elif d_var == 0 and D_var == 1:
            for i in range(fitted_values.shape[0]):
                if i < freq + 1:
                    fitted_values.iloc[i, col_pre] = np.nan
                else:
                    fitted_values.iloc[i, col_pre] = fitted_values.iloc[i, col_pre] + fitted_values.iloc[i - (freq), col_obs]
        
        elif d_var == 1 and D_var == 1:
            for i in range(fitted_values.shape[0]):
                if i < freq + 2:
                    fitted_values.iloc[i, col_pre] = np.nan
                else:                                        
                    fitted_values.iloc[i, col_pre] = (fitted_values.iloc[i, col_pre] + 
                                                      (fitted_values.iloc[i - 1, col_obs] - fitted_values.iloc[(i - 1) - (freq), col_obs]) + 
                                                      fitted_values.iloc[i - (freq), col_obs]
                                                      )
        
        elif d_var == 0 and D_var == 0:
            pass
        
        else:
            raise ValueError('Los valores de d_var y D_var deben ser 0 o 1.')
        
        # Eliminar NaN
        fitted_values = fitted_values.dropna()
        
        # 4) Simulaciones
        # ------------------------------------------------------------------------------------------------------
        # Generar dataframe con 1000 simulaciones
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
                n_simulations = 1000
                
                simulations = fitted_model.simulate_var(steps          = mod_n_forecast, 
                                                        offset         = None, 
                                                        seed           = 42, 
                                                        initial_values = df_model.values[-objective_lags:], 
                                                        nsimulations   = n_simulations
                                                        ) 

                simulations_dfs = {}

                # Recorrer las 5 variables
                for i, col in enumerate(df_model.columns):
                    # Extraer todas las simulaciones de la variable i
                    sim_data = simulations[:, :, i].T  # Transponer para que (fwd_periods, 1000)

                    # Crear el DataFrame
                    simulations_dfs[col] = pd.DataFrame(sim_data, index = idx_future_sim, columns = [j for j in range(0, n_simulations)])
                    
                    # Tomar los últimos n_forecast datos
                    simulations_dfs[col] = simulations_dfs[col].iloc[-(mod_n_forecast - objective_lags):]
                
                simulations = simulations_dfs[target_variable]
            
            simulations.index.name = 'date'
            
            if forecast.isnull().any() or simulations.isnull().any().any():
                raise ValueError('El pronóstico o las simulaciones contienen valores NaN.')

            # Transformar forecast y simulaciones
            hist_original_data = regular_data[regular_data.index < forecast.index[0]]
            
            forecast = self.ts_funcs.invert_diff_forecast(forecast,
                                                          original_series = hist_original_data[target_variable],
                                                          d = d_var,
                                                          D = D_var,
                                                          m = freq
                                                          )
            
            forecast.name = 'forecast'
            forecast.index.name = 'date'
            
            simulations = simulations.apply(
                lambda x: self.ts_funcs.invert_diff_forecast(
                    x,
                    original_series = hist_original_data[target_variable],
                    d = d_var,
                    D = D_var,
                    m = freq
                )
            )
            
            simulations.index.name = 'date'
            
            
            return forecast, simulations, fitted_values
        
        else:
            hist_original_data = regular_data[regular_data.index < forecast.index[0]]
            
            forecast = self.ts_funcs.invert_diff_forecast(forecast,
                                                          original_series = hist_original_data[target_variable],
                                                          d = d_var,
                                                          D = D_var,
                                                          m = freq
                                                          )
            forecast.name = 'forecast'
            forecast.index.name = 'date'
            
            return forecast, fitted_values
        
        
        
    # Fución para la validación en ventana móvil
    def optimization_process_var(
        self,
        regular_data     : pd.DataFrame,
        stationary_data   : pd.DataFrame,
        target_variable   : str,
        top_predictors    : list,
        initial_train_date: str,
        val_size          : int,
        steps             : int,
        freq              : int,
        d_var             : int,
        D_var             : int,
        params            : dict,
        rolling_window    = False
    ):
        """
        Función que entrena un modelo var en una ventana móvil
        Parámetros:
        -----------
        regular_data : pd.DataFrame
            DataFrame con los datos originales
        stationary_data : pd.DataFrame
            DataFrame con los datos estacionarios
        target_variable : str
            Nombre de la variable objetivo
        top_predictors : list
            Lista con los nombres de las mejores variables predictoras
        initial_train_date : str
            Fecha de inicio de los datos de entrenamiento
        val_size : int
            Tamaño de los datos de validación
        steps : int
            Número de pasos a pronosticar
        freq : int
            Frecuencia de los datos (por defecto 4 para datos trimestrales)
        d_var : int
            Orden de diferenciación estacional (por defecto 0)
        D_var : int
            Orden de diferenciación estacional (por defecto 0)
        params : dict
            Diccionario con los parámetros del modelo
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
        cv_stationary_data = self.ts_funcs.ts_split(df                 = stationary_data,
                                                    initial_train_date = initial_train_date,
                                                    val_size           = val_size, 
                                                    steps              = steps, 
                                                    rolling_window     = rolling_window
                                                    )
        
        for i in range(len(cv_stationary_data[0])): # Número de folds
            try:
                import warnings
                warnings.filterwarnings('ignore')
                
                # 2) Datos de entrenamiento
                # ------------------------------------------------------------------------------------------------------
                tmp_stationary_data = cv_stationary_data[0]['fold_train_' + str(i)]
                
                
                # 3) Datos de validación
                # ------------------------------------------------------------------------------------------------------
                val_data = cv_stationary_data[1]['fold_val_' + str(i)][target_variable]
                val_data.name = 'ts'
                val_data = pd.DataFrame(val_data)
                
                if tmp_stationary_data.index[-1] >= val_data.index[0]:
                    raise ValueError(f'Los datos de entrenamiento y validación se superponen en el fold {i}.')
                    
                # 4) Ajuste del modelo y pronósticos
                # ------------------------------------------------------------------------------------------------------
                fit_model = self.var_training(
                    stationary_data = tmp_stationary_data,
                    top_predictors  = top_predictors,
                    params          = params
                )
                
                forecast = self.var_forecast(
                    regular_data    = regular_data[regular_data.index.isin(tmp_stationary_data.index)],
                    stationary_data = tmp_stationary_data,
                    fitted_model    = fit_model,
                    target_variable = target_variable,
                    top_predictors  = top_predictors,
                    fwd_periods     = len(val_data),
                    freq            = freq,
                    d_var           = d_var,
                    D_var           = D_var,
                    do_simulations  = False
                )
                
                # Se debe homogenizar el índice de los pronósticos y de los valores ajustados
                forecast[0].index = val_data.index
                
                # Almacenar datos en un dataframe consolidado
                bt.append(pd.DataFrame({
                    'observed'        : regular_data[regular_data.index.isin(val_data.index)][target_variable],
                    'predicted'       : forecast[0],
                    'model'           : [params] * len(val_data),
                    'fold'            : [i] * len(val_data),
                    'date_range_train': [f"{tmp_stationary_data.index[0]:%Y-%m-%d} - {tmp_stationary_data.index[-1]:%Y-%m-%d}"] * len(val_data),
                    'date_range_val'  : [f"{val_data.index[0]:%Y-%m-%d} - {val_data.index[-1]:%Y-%m-%d}"] * len(val_data)
                    }))
                
                # Datos ajustados
                mod_fitted = forecast[1].copy()
                mod_fitted['model'] = [params] * len(mod_fitted)
                mod_fitted['fold'] = [i] * len(mod_fitted)
                mod_fitted['date_range_train'] = [f"{tmp_stationary_data.index[0]:%Y-%m-%d} - {tmp_stationary_data.index[-1]:%Y-%m-%d}"] * len(mod_fitted)
                mod_fitted['date_range_val'] = [f"{val_data.index[0]:%Y-%m-%d} - {val_data.index[-1]:%Y-%m-%d}"] * len(mod_fitted)
                
                fitted.append(mod_fitted)
                    
                # 6) Métrica de error
                # ------------------------------------------------------------------------------------------------------
                rmse_fold = self.ts_funcs.evaluate_model(y_test = regular_data[regular_data.index.isin(val_data.index)][target_variable], # NOTE: Importante
                                                         y_pred = forecast[0]
                                                         )
                
                # Consolidar métricas de error
                errors.append(rmse_fold)
                
            except Exception as e:
                last_error = e
                pass
                
        
        # Calcular el error promedio
        if len(errors) == len(cv_stationary_data[0]):
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


    # Función para la hiperparametrización del modelo var
    def optimal_parameters_var(self, 
                               regular_data       : pd.DataFrame,
                               stationary_data    : pd.DataFrame, 
                               target_variable    : str,
                               top_predictors     : list,
                               initial_train_date : str, 
                               val_size           : int,
                               steps              : int,
                               rolling_window     = False,
                               freq               = 4,
                               d_var              = 0,
                               D_var              = 0, 
                               verbose            = False
                               ) -> tuple:
        """
        Realiza la hiperparametrización para el modelo var
        Parámetros:
        -----------
        regular_data : pd.DataFrame
            DataFrame con los datos originales
        stationary_data : pd.DataFrame
            DataFrame con los datos estacionarios
        target_variable : str
            Nombre de la variable objetivo
        top_predictors : list
            Lista con los nombres de las mejores variables predictoras
        initial_train_date : str
            Fecha de inicio de los datos de entrenamiento
        val_size : int
            Tamaño de los datos de validación
        steps : int
            Número de pasos a pronosticar
        rolling_window : bool, opcional
            Indica si se utiliza una ventana móvil
        freq : int, opcional
            Frecuencia de los datos (por defecto 4 para datos trimestrales)
        d_var : int, opcional
            Orden de diferenciación estacional (por defecto 0)
        D_var : int, opcional
            Orden de diferenciación estacional (por defecto 0)
        verbose : bool, opcional
            Indica si se imprime información adicional durante el proceso
            
        Retorna:
        --------    
        reg_res_var : pd.DataFrame
            DataFrame con los resultados de la hiperparametrización
        df_bt_var : pd.DataFrame
            DataFrame con los datos de bt
        u_fit_var : pd.DataFrame
            DataFrame con los datos ajustados
        """
        
        # 1) Grilla de hiperparámetros
        # ------------------------------------------------------------------------------------------------------
        param_grid_var = {
            'maxlags': [1, 2, 3, 4]
            }

        grid_var = list(ParameterGrid(param_grid_var))

        # 2) Iterar sobre la grilla de hiperparámetros
        # ------------------------------------------------------------------------------------------------------
        var_metrics = []
        bt_df_var = []
        fitted_df_var = []

        for i, params in enumerate(grid_var):
            try:          
                opt_var = self.optimization_process_var(
                    regular_data       = regular_data,
                    stationary_data    = stationary_data, 
                    target_variable    = target_variable,
                    top_predictors     = top_predictors,
                    initial_train_date = initial_train_date,
                    val_size           = val_size,
                    steps              = steps,
                    freq               = freq,
                    d_var              = d_var,
                    D_var              = D_var,
                    params             = params,
                    rolling_window     = rolling_window
                )
                var_metrics.append({
                    'model': params,
                    'error_metric': opt_var[0]
                })        
                    
                # Consolidar los datos de bt
                bt_df_var.append(opt_var[1])
                
                # Consolidar los datos ajustados
                fitted_df_var.append(opt_var[2])
                    
                if verbose:
                    print('-' * 100)
                    print(f'¡Iteración Exitosa! {i} (MODELO VAR) con los parámetros {params}')
                    print('-' * 100)
            except Exception as e:
                if verbose:
                    print('*' * 100)
                    print(f'¡ERROR EN LA ITERACIÓN! {i} (MODELO VAR)  con los parametros {params}: {e}')
                    print('*' * 100)              
                else:
                    pass
                
            
        # Union de los datos de métricas de error
        reg_res_var = pd.DataFrame(var_metrics).dropna().sort_values('error_metric').reset_index(drop = True)

        # Unificar los datos de bt
        df_bt_var = pd.concat(bt_df_var)
        
        # Unificar los datos ajustados
        u_fit_var = pd.concat(fitted_df_var)
        u_fit_var.index.name = 'date'
        
        return reg_res_var, df_bt_var, u_fit_var