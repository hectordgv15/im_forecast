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
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading
from joblib import Parallel, delayed

# Statmodels para df de tiempo
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Prophet
from prophet import Prophet

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
from IMForecast.ts_functions import ts_functions
from IMForecast.hw_model import HoltWintersIM
from IMForecast.sarima_model import SarimaIM
from IMForecast.prophet_model import ProphetIM
from IMForecast.ucm_model import UnobservedComponentsIM
from IMForecast.ardl_model import ArdlIM
from IMForecast.var_model import varIM
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import grangercausalitytests



# Leer información por cada variable
def executive_report(var_name            : str, 
                     ts_df               : pd.DataFrame,
                     macro_vars_df       : pd.DataFrame,
                     replace_outliers    = False,
                     deseasonalize       = False,
                     rolling_window      = False,
                     final_date          = None,
                     val_size            = None,
                     steps               = None,
                     season_freq         = None,
                     freq_type           = 'Q',
                     ts_type             = 'abs',
                     initial_date        = '2014-01-01',
                     end_forecast_period = None,
                     ): # Tener en cuenta el bool en los outliers
    
    """
    Función para ejecutar el informe ejecutivo de una variable macroeconómica.
    Parámetros:
    - var_name: Nombre de la variable macroeconómica.
    - ts_df: DataFrame con la serie temporal de la variable.
    - macro_vars_df: DataFrame con las variables macroeconómicas adicionales.
    - replace_outliers: Booleano para indicar si se deben reemplazar los valores atípicos.
    - deseasonalize: Booleano para indicar si se debe desestacionalizar la serie temporal.
    - rolling_window: Booleano para indicar si se debe usar una ventana deslizante.
    - final_date: Fecha final para el análisis.
    - val_size: Tamaño del conjunto de validación.
    - steps: Número de pasos para el pronóstico y reentrenamiento.
    - season_freq: Frecuencia estacional de la serie temporal.
    - freq_type: Tipo de frecuencia de la serie temporal (por defecto trimestral).
    - ts_type: Tipo de serie temporal ('abs' para absoluta, 'pct' para porcentual).
    - initial_date: Fecha inicial para el análisis.
    - end_forecast_period: Fecha final del periodo de pronóstico.
    """
    
    # =============================================================================
    # CONFIGURACIÓN DE PARÁMETROS
    # =============================================================================    
    # Desestacionalización
    if deseasonalize:
        s_d = seasonal_decompose(ts_df['ts'].copy(), model = 'additive', period = season_freq)  # Periodo 4 porque los datos son trimestrales
        ts_df['ts'] = ts_df['ts'] - s_d.seasonal
    
    # Períodos para entrenamiento y validación
    date_init_train = (datetime.strptime(final_date, '%Y-%m-%d') - relativedelta(years = 4)).strftime('%Y-%m-%d')
    date_init_val = final_date

    # Calcular la diferencia en trimestres
    periods_ahead = (datetime.strptime(end_forecast_period, '%Y-%m-%d').year - datetime.strptime(final_date, '%Y-%m-%d').year) * 4 + (datetime.strptime(end_forecast_period, '%Y-%m-%d').month - datetime.strptime(final_date, '%Y-%m-%d').month) // 3

    print('=' * 110)
    print(f'Periods ahead: {periods_ahead}')
    print('=' * 110)
    
    # Crear un dataframe con todos los parametros especificados
    params_df = pd.DataFrame({
        'season_freq': [season_freq],
        'freq_type': [freq_type],
        'initial_date': [initial_date],
        'final_date': [final_date],
        'date_init_train': [date_init_train],
        'date_init_val': [date_init_val],
        'rolling_window': [rolling_window],
        'steps': [steps],
        'periods_ahead': [periods_ahead]
    })

    # =============================================================================
    # REEMPLAZO DE DATOS ATÍPICOS
    # =============================================================================
    # Dataframe definitivo
    clean_ts = ts_df.copy()

    clean_ts = clean_ts.loc[(clean_ts.index >= initial_date) & (clean_ts.index <= final_date)].copy()
    
    if replace_outliers:
        clean_ts = ts_functions().clean_outliers(
            df     = clean_ts.copy(),
            method = 'iqr'
            )
    else:
        print('No se han reemplazado los valores atípicos')

    clean_ts.index.name = 'date'
    
    # =============================================================================
    # DATAFRAMES EXPORTABLES PARA GRÁFICOS
    # =============================================================================
    last_full_year = clean_ts.index.max().year - (1 if clean_ts.index.max().quarter < 4 else 0)

    if ts_type == 'abs':
        df_abs = clean_ts.copy()
        df_abs_y = clean_ts.loc[lambda x: x.index.year <= last_full_year].groupby(clean_ts.loc[lambda x: x.index.year <= last_full_year].index.year).last()
        df_pct = np.log(clean_ts['ts']).diff().dropna() 
        df_interannual = np.log(clean_ts['ts']).diff(season_freq).dropna().rename('ts')
        df_avg_year = df_interannual.loc[lambda x: x.index.year <= last_full_year].groupby(df_interannual.loc[lambda x: x.index.year <= last_full_year].index.year).mean()
        df_last_year = df_interannual.loc[lambda x: x.index.year <= last_full_year].groupby(df_interannual.loc[lambda x: x.index.year <= last_full_year].index.year).last()
    elif ts_type == 'pct':
        df_abs = clean_ts.copy()
        df_abs_y = clean_ts.loc[lambda x: x.index.year <= last_full_year].groupby(clean_ts.loc[lambda x: x.index.year <= last_full_year].index.year).last()
        df_pct = clean_ts.copy()
        df_interannual = clean_ts['ts'].rolling(window = season_freq).sum().dropna()
        df_avg_year = df_interannual.loc[lambda x: x.index.year <= last_full_year].groupby(df_interannual.loc[lambda x: x.index.year <= last_full_year].index.year).mean()
        df_last_year = df_interannual.loc[lambda x: x.index.year <= last_full_year].groupby(df_interannual.loc[lambda x: x.index.year <= last_full_year].index.year).last()
    else:
        df_abs = clean_ts.copy()
        df_abs_y = clean_ts.loc[lambda x: x.index.year <= last_full_year].groupby(clean_ts.loc[lambda x: x.index.year <= last_full_year].index.year).last()
        df_pct = clean_ts.copy()
        df_interannual = clean_ts.copy()
        df_avg_year = df_interannual.loc[lambda x: x.index.year <= last_full_year].groupby(df_interannual.loc[lambda x: x.index.year <= last_full_year].index.year).last()
        df_last_year = df_interannual.loc[lambda x: x.index.year <= last_full_year].groupby(df_interannual.loc[lambda x: x.index.year <= last_full_year].index.year).last()
        
    # =============================================================================
    # FILTRAR LOS DATOS PARA LAS FECHAS DE INTERÉS
    # =============================================================================
    # Dataframe modelos multivariantes
    macro_vars_df = macro_vars_df.dropna()
    macro_vars_df.index.name = 'date'
    
    if var_name not in macro_vars_df.columns:
        macro_vars_df = pd.merge(ts_df, macro_vars_df, left_index = True, right_index = True, how = 'left')
        macro_vars_df = macro_vars_df.rename(columns = {'ts': var_name})
        macro_vars_df = macro_vars_df.bfill()
        macro_vars_df = macro_vars_df.dropna()
    
    # =============================================================================
    # CONVERTIR LA SERIE EN ESTACIONARIA
    # =============================================================================
    d_var, D_var = ts_functions().determine_diff_orders(ts_df['ts'], 
                                                        freq          = season_freq, 
                                                        return_series = False,
                                                        initial_date  = initial_date, # Para evitar problemas con el test de Dickey-Fuller,
                                                        end_date      = final_date
                                                        )
        
    print(f'Diferenciación estacional: {D_var}, Diferenciación no estacional: {d_var}')
    
    # Añadir al fichero parmas
    params_df['d_var'] = d_var
    params_df['D_var'] = D_var
    
    # Aplicar la transformación a todo el dataframe
    stationary_macro_df = macro_vars_df.copy().apply(lambda x: ts_functions().determine_diff_orders(x, freq = season_freq, return_series = True, initial_date = initial_date, end_date = final_date)[2], axis = 0).dropna()
    
    # NOTE: Los ordenes de diferenciación de la variable individual no pueden ser diferentes a los del dataframe macroeconómico
    
    # Filtrar dataframe. Solo el dataframe estacionario
    stationary_macro_df = stationary_macro_df.loc[(stationary_macro_df.index >= initial_date) & (stationary_macro_df.index <= final_date)].copy()
    
    # =============================================================================
    # PIPELINE FEATURE SELECTION
    # =============================================================================  
    top_predictors = varIM().best_predictors(
        stationary_data = stationary_macro_df,
        var_name        = var_name
    )

    # =============================================================================
    # PIPELINE HIPERPARAMETRIZACIÓN
    # =============================================================================
    # Holt Winters
    df_hw_metrics, unified_backtest_hw, unified_fitted_hw = HoltWintersIM().optimal_parameters_hw(
        df                 = clean_ts[:date_init_val],
        initial_train_date = date_init_train,
        val_size           = val_size,
        steps              = steps,
        rolling_window     = rolling_window,
        freq               = season_freq,
        verbose            = True
    )

    # SARIMA
    df_sarima_metrics, unified_backtest_sarima, unified_fitted_sarima = SarimaIM().optimal_parameters_sarima(
        df                 = clean_ts[:date_init_val],
        initial_train_date = date_init_train,
        val_size           = val_size,
        steps              = steps,
        season_freq        = season_freq,
        rolling_window     = rolling_window,
        verbose            = True,
        d_value            = d_var,
        s_d_value          = D_var
    )


    # Prophet
    df_prophet_metrics, unified_backtest_prophet, unified_fitted_prophet = ProphetIM().optimal_regular_parameters_prophet(
        df                 = clean_ts[:date_init_val],
        initial_train_date = date_init_train,
        val_size           = val_size,
        steps              = steps,
        freq_type          = freq_type,
        rolling_window     = rolling_window,
        verbose            = True
    )

    
    # Unobserved components
    df_ucm_metrics, unified_backtest_ucm, unified_fitted_ucm = UnobservedComponentsIM().optimal_parameters_ucm(
        df                 = clean_ts[:date_init_val],
        initial_train_date = date_init_train,
        val_size           = val_size,
        steps              = steps,
        rolling_window     = rolling_window,
        freq               = season_freq,
        verbose            = True
    )

    
    # ARDL (AutoRegressive Distributed Lag)
    df_ardl_metrics, unified_backtest_ardl, unified_fitted_ardl = ArdlIM().optimal_parameters_ardl(
        df                 = clean_ts[:date_init_val],
        initial_train_date = date_init_train,
        val_size           = val_size,
        steps              = steps,
        rolling_window     = rolling_window,
        freq               = season_freq, 
        verbose            = True
    )

    
    # VAR (Vector AutoRegressive)
    df_var_metrics, unified_backtest_var, unified_fitted_var = varIM().optimal_parameters_var(
        regular_data       = macro_vars_df[:date_init_val],
        stationary_data    = stationary_macro_df[:date_init_val],
        target_variable    = var_name,
        top_predictors     = top_predictors,
        initial_train_date = date_init_train,
        val_size           = val_size,
        steps              = steps,
        rolling_window     = rolling_window,
        freq               = season_freq,
        d_var              = d_var,
        D_var              = D_var,
        verbose            = True
    )
    
    # Número esperado de datos en el backtest
    expected_bt = pd.date_range(start = date_init_train, end = date_init_val, freq = 'Q').size - 1
    
    # =============================================================================
    # PROCESO HOLT-WINTERS
    # =============================================================================
    # 1. Mejores parámetros
    # -----------------------------------------------------------------------------    
    hw_candidate_models = df_hw_metrics['model'].tolist()
        
    
    for candidate_model in hw_candidate_models:
        
        try:
            best_params_hw = candidate_model # Iterar sobre los modelos
            

            backtest_hw = HoltWintersIM().optimization_process_hw(df                 = clean_ts[:date_init_val],
                                                                initial_train_date = date_init_train,
                                                                val_size           = val_size,
                                                                steps              = steps,
                                                                params             = best_params_hw,
                                                                rolling_window     = rolling_window
                                                                )[1] # Importante
            
            # Levantar un error pero no detener el ciclo si no hay datos de backtest
            if backtest_hw.shape[0] < expected_bt:
                raise Exception(f"El modelo Holt-Winters ({var_name}) no tiene datos suficientes de Backtest: {backtest_hw.shape[0]} en lugar de {expected_bt}")
            
            # 2. Backtesting
            # -----------------------------------------------------------------------------
            if pd.to_datetime(date_init_val) < clean_ts.tail(1).index[0]:
                os_backtest_hw = HoltWintersIM().optimization_process_hw(df                 = clean_ts, # Dataframe completo      
                                                                        initial_train_date = date_init_val, # Tamaño del conjunto inicial de entrenamiento
                                                                        val_size           = clean_ts[date_init_val:].shape[0] - 1,
                                                                        steps              = season_freq,
                                                                        params             = best_params_hw,
                                                                        rolling_window     = rolling_window,
                                                                        )[1] # Importante

            else:
                os_backtest_hw = pd.DataFrame()
            
            
            
            # 3. Modelo definitivo y pronóstico final
            # -----------------------------------------------------------------------------
            if rolling_window:
                training_data_hw = clean_ts.tail(clean_ts[:date_init_train].shape[0])
            else:
                training_data_hw = clean_ts.copy()
            
            
            definitive_hw_model = HoltWintersIM().hw_training(training_data = training_data_hw, # NOTE: Dataframe completo
                                                              params        = best_params_hw
                                                              )


            # Pronóstico final y simulaciones
            el_sim = [(False, False), 
                      (True, False), 
                      (True, True)]

            for alt_sims, remove_out in el_sim:
                forecast_hw, simulations_hw, fitted_hw = HoltWintersIM().hw_forecast(model                   = definitive_hw_model, 
                                                                                     fwd_periods             = periods_ahead, 
                                                                                     do_simulations          = True, 
                                                                                     alternative_simulations = alt_sims, # NOTE: Se configura en False como estado inicial.
                                                                                     remove_outliers         = remove_out
                                                                                     )
                
                # Si el tipo de la serie es diferente a "abs", parar el ciclo en la primera iteración
                if ts_type != 'abs':
                    break
                
                if not (simulations_hw < 0).any().any():
                    break
            
            params_df['alt_simulations_hw'] = alt_sims
            params_df['remove_outliers_hw'] = remove_out
            
            # Si aún así persisten los valores negativos, se reemplazan por el valor anterior
            if (simulations_hw < 0).any().any() and ts_type == 'abs':
                simulations_hw = (simulations_hw.mask(simulations_hw < 0).ffill().bfill().fillna(0))
                params_df['replace_neg_sims_hw'] = True
            else:
                pass
                
            
            # Aplicar función para el cálculo de los percentiles            
            hw_abs_q, hw_abs_y, hw_pct_q, hw_interannual, hw_interannual_avg, hw_interannual_last = ts_functions().percentiles_compute(history   = clean_ts, 
                                                                                                                                        forecast  = forecast_hw, 
                                                                                                                                        sims      = simulations_hw.copy(), 
                                                                                                                                        data_type = ts_type,
                                                                                                                                        freq      = season_freq
                                                                                                                                        )        
        

                
                
            # Si se encuentran resultados correctos se para el bucle
            break
            
        except Exception as e:
            
            print('*' * 110)
            print(f"Error en el modelo Holt-Winters ({var_name}) con parámetros ({candidate_model}): {e}")
            print('*' * 110)
            pass
            
    params_df['best_model_hw'] = str(f"{best_params_hw}")
    
    # Índice del df metrics al cual pertenece el modelo
    params_df['top_hw'] = df_hw_metrics.loc[df_hw_metrics['model'] == best_params_hw].index
    
    print(f'Los parámetros óptimos que funcionaron para el modelo Holt-Winters ({var_name}) corresponden a: {best_params_hw}')
    
    
    # =============================================================================
    # PROCESO SARIMA
    # =============================================================================
    # 1. Mejores parámetros
    # -----------------------------------------------------------------------------    
    sarima_candidate_models = df_sarima_metrics['model'].tolist()
        
    for candidate_model in sarima_candidate_models:
        
        try:
            best_params_sarima = candidate_model # NOTE: Se debe agregar el ast.literal_eval
            

            backtest_sarima = SarimaIM().optimization_process_sarima(df                 = clean_ts[:date_init_val],
                                                                     initial_train_date = date_init_train,
                                                                     val_size           = val_size,
                                                                     steps              = steps,
                                                                     params             = best_params_sarima,
                                                                     rolling_window     = rolling_window,
                                                                     )[1] # Importante
            
            # Levantar un error pero no detener el ciclo si no hay datos de backtest
            if backtest_sarima.shape[0] < expected_bt:
                raise Exception(f"El modelo SARIMA ({var_name}) no tiene datos suficientes de Backtest {backtest_sarima.shape[0]} en vez de {expected_bt}")
            
            # 2. Backtesting
            # -----------------------------------------------------------------------------
            if pd.to_datetime(date_init_val) < clean_ts.tail(1).index[0]:
                os_backtest_sarima = SarimaIM().optimization_process_sarima(df                 = clean_ts, # Dataframe completo
                                                                            initial_train_date = date_init_val,
                                                                            val_size           = clean_ts[date_init_val:].shape[0] - 1,
                                                                            steps              = season_freq,
                                                                            params             = best_params_sarima, # Parámetros óptimos
                                                                            rolling_window     = rolling_window
                                                                            )[1] # Importante

            else:
                os_backtest_sarima = pd.DataFrame()
            
            
            
            # 3. Modelo definitivo y pronóstico final
            # -----------------------------------------------------------------------------
            if rolling_window:
                training_data_sarima = clean_ts.tail(clean_ts[:date_init_train].shape[0])
            else:
                training_data_sarima = clean_ts.copy()
                
            
            definitive_sarima_model = SarimaIM().sarima_training(training_data = training_data_sarima, # NOTE: Dataframe completo
                                                                 params        = best_params_sarima
                                                                 )

            # Pronóstico final y simulaciones
            el_sim = [(False, False), 
                      (True, False), 
                      (True, True)]

            for alt_sims, remove_out in el_sim:
                forecast_sarima, simulations_sarima, fitted_sarima = SarimaIM().sarima_forecast(model                   = definitive_sarima_model, 
                                                                                                fwd_periods             = periods_ahead, 
                                                                                                do_simulations          = True, 
                                                                                                alternative_simulations = alt_sims, # NOTE: Se configura en False como estado inicial. 
                                                                                                remove_outliers         = remove_out
                                                                                                )
                
                # Si el tipo de la serie es diferente a "abs", parar el ciclo en la primera iteración
                if ts_type != 'abs':
                    break
                
                if not (simulations_sarima < 0).any().any():
                    break
            
            params_df['alt_simulations_sarima'] = alt_sims
            params_df['remove_outliers_sarima'] = remove_out
            
            # Si aún así persisten los valores negativos, se reemplazan por el valor anterior
            if (simulations_sarima < 0).any().any() and ts_type == 'abs':
                simulations_sarima = (simulations_sarima.mask(simulations_sarima < 0).ffill().bfill().fillna(0))
                params_df['replace_neg_sims_sarima'] = True
            else:
                pass
            

            
            # Aplicar función para el cálculo de los percentiles
            sarima_abs_q, sarima_abs_y, sarima_pct_q, sarima_interannual, sarima_interannual_avg, sarima_interannual_last = ts_functions().percentiles_compute(history   = clean_ts, 
                                                                                                                                                               forecast  = forecast_sarima, 
                                                                                                                                                               sims      = simulations_sarima.copy(),
                                                                                                                                                               data_type = ts_type,
                                                                                                                                                               freq      = season_freq
                                                                                                                                                               )
            
            # Si se encuentran resultados correctos se para el bucle
            break
        
        except Exception as e:
            
            print('*' * 110)
            print(f"Error en el modelo SARIMA con parámetros ({candidate_model}): {e}")
            print('*' * 110)
            pass
            
    params_df['best_model_sarima'] = str(f"{best_params_sarima}")
    
    # Índice del df metrics al cual pertenece el modelo
    params_df['top_sarima'] = df_sarima_metrics.loc[df_sarima_metrics['model'] == best_params_sarima].index
    
    print(f'Los parámetros óptimos que funcionaron para el modelo SARIMA ({var_name}) corresponden a: {best_params_sarima}')
    
    
    # =============================================================================
    # PROCESO PROPHET
    # =============================================================================
    prophet_candidate_models = df_prophet_metrics['model'].tolist()
    
    for candidate_model in prophet_candidate_models:
        
        try:
            best_params_prophet = candidate_model
            

            backtest_prophet = ProphetIM().optimization_process_prophet(df                 = clean_ts[:date_init_val],
                                                                        initial_train_date = date_init_train,
                                                                        val_size           = val_size,
                                                                        steps              = steps,
                                                                        params             = best_params_prophet, # Parámetros óptimos
                                                                        freq_type          = freq_type,
                                                                        rolling_window     = rolling_window,
                                                                        )[1] # Importante
            
            # Levantar un error pero no detener el ciclo si no hay datos de backtest
            if backtest_prophet.shape[0] < expected_bt:
                raise Exception(f"El modelo prophet ({var_name}) no tiene datos suficientes de Backtest {backtest_prophet.shape[0]} en vez de {expected_bt}")
            
            # 2. Backtesting
            # -----------------------------------------------------------------------------
            if pd.to_datetime(date_init_val) < clean_ts.tail(1).index[0]:
                os_backtest_prophet = ProphetIM().optimization_process_prophet(df                 = clean_ts, # Dataframe completo
                                                                               initial_train_date = date_init_val,
                                                                               val_size           = clean_ts[date_init_val:].shape[0] - 1,
                                                                               steps              = season_freq,
                                                                               params             = best_params_prophet, # Parámetros óptimos
                                                                               freq_type          = freq_type,
                                                                               rolling_window     = rolling_window
                                                                               )[1] # Importante

            else:
                os_backtest_prophet = pd.DataFrame()
            
            
            
            # 3. Modelo definitivo y pronóstico final
            # -----------------------------------------------------------------------------
            if rolling_window:
                training_data_prophet = clean_ts.tail(clean_ts[:date_init_train].shape[0])
            else:
                training_data_prophet = clean_ts.copy()
                
            
            definitive_prophet_model = ProphetIM().prophet_training(training_data = training_data_prophet, # NOTE: Dataframe completo
                                                                    params        = best_params_prophet
                                                                    )


            # Pronóstico final y simulaciones
            el_sim = [False, True] # NOTE: Importante que primero sea False

            for remove_out in el_sim:
                forecast_prophet, simulations_prophet, fitted_prophet = ProphetIM().prophet_forecast(training_data           = training_data_prophet, 
                                                                                                     model                   = definitive_prophet_model,
                                                                                                     fwd_periods             = periods_ahead,
                                                                                                     freq_type               = freq_type,
                                                                                                     do_simulations          = True,
                                                                                                     remove_outliers         = remove_out
                                                                                                     )
                
                # Si el tipo de la serie es diferente a "abs", parar el ciclo en la primera iteración
                if ts_type != 'abs':
                    break
                
                if not (simulations_prophet < 0).any().any():
                    break
            
            params_df['remove_outliers_prophet'] = remove_out
            
            # Si aún así persisten los valores negativos, se reemplazan por el valor anterior
            if (simulations_prophet < 0).any().any() and ts_type == 'abs':
                simulations_prophet = (simulations_prophet.mask(simulations_prophet < 0).ffill().bfill().fillna(0))
                params_df['replace_neg_sims_prophet'] = True
            else:
                pass
            

            
            # Aplicar función para el cálculo de los percentiles
            prophet_abs_q, prophet_abs_y, prophet_pct_q, prophet_interannual, prophet_interannual_avg, prophet_interannual_last = ts_functions().percentiles_compute(history   = clean_ts, 
                                                                                                                                                                     forecast  = forecast_prophet, 
                                                                                                                                                                     sims      = simulations_prophet.copy(),
                                                                                                                                                                     data_type = ts_type,
                                                                                                                                                                     freq      = season_freq
                                                                                                                                                                     )
            
            # Si se encuentran resultados correctos se para el bucle
            break
        
        except Exception as e:
            
            print('*' * 110)
            print(f"Error en el modelo Prophet con parámetros ({candidate_model}): {e}")
            print('*' * 110)
            pass
            
    params_df['best_model_prophet'] = str(f"{best_params_prophet}")
    
    # Índice del df metrics al cual pertenece el modelo
    params_df['top_prophet'] = df_prophet_metrics.loc[df_prophet_metrics['model'] == best_params_prophet].index
    
    print(f'Los parámetros óptimos que funcionaron para el modelo Prophet ({var_name}) corresponden a: {best_params_prophet}')
    
    
    # =============================================================================
    # PROCESO UNOBSERVED COMPONENTS
    # =============================================================================
    # 1. Mejores parámetros
    # -----------------------------------------------------------------------------
    ucm_candidate_models = df_ucm_metrics['model'].tolist()
    
    for candidate_model in ucm_candidate_models:
        
        try:
            best_params_ucm = candidate_model # Iterar sobre los modelos
            

            backtest_ucm = UnobservedComponentsIM().optimization_process_ucm(df                 = clean_ts[:date_init_val],
                                                                             initial_train_date = date_init_train,
                                                                             val_size           = val_size,
                                                                             steps              = steps,
                                                                             params             = best_params_ucm, # Parámetros óptimos
                                                                             rolling_window     = rolling_window
                                                                             )[1] # Importante
            
            # Levantar un error pero no detener el ciclo si no hay datos de backtest
            if backtest_ucm.shape[0] < expected_bt:
                raise Exception(f"El modelo UCM ({var_name}) no tiene datos suficientes de Backtest {backtest_ucm.shape[0]} en vez de {expected_bt}")
            
            # 2. Backtesting
            # -----------------------------------------------------------------------------
            if pd.to_datetime(date_init_val) < clean_ts.tail(1).index[0]:
                os_backtest_ucm = UnobservedComponentsIM().optimization_process_ucm(df                 = clean_ts, # Dataframe completo
                                                                                    initial_train_date = date_init_val, # Tamaño del conjunto inicial de entrenamiento
                                                                                    val_size           = clean_ts[date_init_val:].shape[0] - 1,
                                                                                    steps              = season_freq,
                                                                                    params             = best_params_ucm, # Parámetros óptimos
                                                                                    rolling_window     = rolling_window
                                                                                    )[1] # Importante

            else:
                os_backtest_ucm = pd.DataFrame()
            
            
            
            # 3. Modelo definitivo y pronóstico final
            # -----------------------------------------------------------------------------
            if rolling_window:
                training_data_ucm = clean_ts.tail(clean_ts[:date_init_train].shape[0])
            else:
                training_data_ucm = clean_ts.copy()
                
            
            definitive_ucm_model = UnobservedComponentsIM().ucm_training(training_data = training_data_ucm, # NOTE: Dataframe completo
                                                                         params        = best_params_ucm
                                                                         )


            # Pronóstico final y simulaciones
            el_sim = [False, True] # NOTE: Importante que primero sea False

            for remove_out in el_sim:
                forecast_ucm, simulations_ucm, fitted_ucm = UnobservedComponentsIM().ucm_forecast(model                   = definitive_ucm_model, 
                                                                                                  fwd_periods             = periods_ahead, 
                                                                                                  do_simulations          = True, 
                                                                                                  remove_outliers         = remove_out, # NOTE: Se configura en False como estado inicial.
                                                                                                  )
                
                # Si el tipo de la serie es diferente a "abs", parar el ciclo en la primera iteración
                if ts_type != 'abs':
                    break
                
                if not (simulations_ucm < 0).any().any():
                    break
            
            params_df['remove_outliers_ucm'] = remove_out
            
            # Si aún así persisten los valores negativos, se reemplazan por el valor anterior
            if (simulations_ucm < 0).any().any() and ts_type == 'abs':
                simulations_ucm = (simulations_ucm.mask(simulations_ucm < 0).ffill().bfill().fillna(0))
                params_df['replace_neg_sims_ucm'] = True
            else:
                pass

            
            # Aplicar función para el cálculo de los percentiles
            ucm_abs_q, ucm_abs_y, ucm_pct_q, ucm_interannual, ucm_interannual_avg, ucm_interannual_last = ts_functions().percentiles_compute(history   = clean_ts, 
                                                                                                                                             forecast  = forecast_ucm, 
                                                                                                                                             sims      = simulations_ucm.copy(),
                                                                                                                                             data_type = ts_type,
                                                                                                                                             freq      = season_freq
                                                                                                                                             )
                

            # Si se encuentran resultados correctos se para el bucle
            break
            
        except Exception as e:
            
            print('*' * 110)
            print(f"Error en el modelo UCM ({var_name}) con parámetros ({candidate_model}): {e}")
            print('*' * 110)
            pass
            
    params_df['best_model_ucm'] = str(f"{best_params_ucm}")
    
    # Índice del df metrics al cual pertenece el modelo
    params_df['top_ucm'] = df_ucm_metrics.loc[df_ucm_metrics['model'] == best_params_ucm].index
    
    print(f'Los parámetros óptimos que funcionaron para el modelo UCM ({var_name}) corresponden a: {best_params_ucm}')
    
    
    # =============================================================================
    # PROCESO ARDL
    # =============================================================================
    ardl_candidate_models = df_ardl_metrics['model'].tolist()
    
    for candidate_model in ardl_candidate_models:
        
        try:
            best_params_ardl = candidate_model # Iterar sobre los modelos
            

            backtest_ardl = ArdlIM().optimization_process_ardl(df                 = clean_ts[:date_init_val],
                                                               initial_train_date = date_init_train,
                                                               val_size           = val_size,
                                                               steps              = steps,
                                                               params             = best_params_ardl, # Parámetros óptimos
                                                               rolling_window     = rolling_window
                                                               )[1] # Importante
            
            # Levantar un error pero no detener el ciclo si no hay datos de backtest
            if backtest_ardl.shape[0] < expected_bt:
                raise Exception(f"El modelo ARDL ({var_name}) no tiene datos suficientes de Backtest {backtest_ardl.shape[0]} en vez de {expected_bt}")
            
            # 2. Backtesting
            # -----------------------------------------------------------------------------
            if pd.to_datetime(date_init_val) < clean_ts.tail(1).index[0]:
                os_backtest_ardl = ArdlIM().optimization_process_ardl(df                 = clean_ts, # Dataframe completo
                                                                      initial_train_date = date_init_val,
                                                                      val_size           = clean_ts[date_init_val:].shape[0] - 1,
                                                                      steps              = season_freq,
                                                                      params             = best_params_ardl, # Parámetros óptimos
                                                                      rolling_window     = rolling_window
                                                                      )[1] # Importante

            else:
                os_backtest_ardl = pd.DataFrame()
            
            
            
            # 3. Modelo definitivo y pronóstico final
            # -----------------------------------------------------------------------------
            if rolling_window:
                training_data_ardl = clean_ts.tail(clean_ts[:date_init_train].shape[0])
            else:
                training_data_ardl = clean_ts.copy()
                
            
            definitive_ardl_model = ArdlIM().ardl_training(training_data = training_data_ardl, 
                                                           params        = best_params_ardl
                                                           )


            # Pronóstico final y simulaciones
            el_sim = [False, True] # NOTE: Importante que primero sea False

            for remove_out in el_sim:
                forecast_ardl, simulations_ardl, fitted_ardl = ArdlIM().ardl_forecast(model                   = definitive_ardl_model, 
                                                                                      fwd_periods             = periods_ahead, 
                                                                                      do_simulations          = True, 
                                                                                      remove_outliers         = remove_out, # NOTE: Se configura en False como estado inicial.
                                                                                      )
                
                # Si el tipo de la serie es diferente a "abs", parar el ciclo en la primera iteración
                if ts_type != 'abs':
                    break
                
                if not (simulations_ardl < 0).any().any():
                    break
            
            params_df['remove_outliers_ardl'] = remove_out
            
            # Si aún así persisten los valores negativos, se reemplazan por el valor anterior
            if (simulations_ardl < 0).any().any() and ts_type == 'abs':
                simulations_ardl = (simulations_ardl.mask(simulations_ardl < 0).ffill().bfill().fillna(0))
                params_df['replace_neg_sims_ardl'] = True
            else:
                pass
                

                
            # Aplicar función para el cálculo de los percentiles
            ardl_abs_q, ardl_abs_y, ardl_pct_q, ardl_interannual, ardl_interannual_avg, ardl_interannual_last = ts_functions().percentiles_compute(history   = clean_ts, 
                                                                                                                                                   forecast  = forecast_ardl, 
                                                                                                                                                   sims      = simulations_ardl.copy(),
                                                                                                                                                   data_type = ts_type,
                                                                                                                                                   freq      = season_freq
                                                                                                                                                   )
                
                
            # Si se encuentran resultados correctos se para el bucle
            break
            
        except Exception as e:
            
            print('*' * 110)
            print(f"Error en el modelo ARDL ({var_name}) con parámetros ({candidate_model}): {e}")
            print('*' * 110)
            pass
            
    params_df['best_model_ardl'] = str(f"{best_params_ardl}")
    
    # Índice del df metrics al cual pertenece el modelo
    params_df['top_ardl'] = df_ardl_metrics.loc[df_ardl_metrics['model'] == best_params_ardl].index
    
    print(f'Los parámetros óptimos que funcionaron para el modelo ARDL ({var_name}) corresponden a: {best_params_ardl}')
        
    
    # =============================================================================
    # PROCESO VAR
    # =============================================================================
    var_candidate_models = df_var_metrics['model'].tolist()
    
    for candidate_model in var_candidate_models:
        
        try:
            best_params_var = candidate_model
            

            backtest_var = varIM().optimization_process_var(regular_data       = macro_vars_df[:date_init_val],
                                                            stationary_data    = stationary_macro_df[:date_init_val],
                                                            target_variable    = var_name,
                                                            top_predictors     = top_predictors,
                                                            initial_train_date = date_init_train,
                                                            val_size           = val_size,
                                                            steps              = steps,
                                                            freq               = season_freq,
                                                            d_var              = d_var,
                                                            D_var              = D_var,
                                                            params             = best_params_var,
                                                            rolling_window     = rolling_window
                                                            )[1] # Importante
        
            
            # Levantar un error pero no detener el ciclo si no hay datos de backtest
            if backtest_var.shape[0] < expected_bt:
                raise Exception(f"El modelo VAR ({var_name}) no tiene datos suficientes de Backtest {backtest_var.shape[0]} en vez de {expected_bt}")
            
            # 2. Backtesting
            # -----------------------------------------------------------------------------
            if pd.to_datetime(date_init_val) < clean_ts.tail(1).index[0]:
                os_backtest_var = varIM().optimization_process_var(regular_data       = macro_vars_df,
                                                                   stationary_data    = stationary_macro_df,
                                                                   target_variable    = var_name,
                                                                   top_predictors     = top_predictors,
                                                                   initial_train_date = date_init_val,
                                                                   val_size           = clean_ts[date_init_val:].shape[0] - 1, # Tamaño del conjunto de validación
                                                                   steps              = season_freq, # Número de pasos para el pronóstico y reentrenamiento
                                                                   freq               = season_freq,
                                                                   d_var              = d_var,
                                                                   D_var              = D_var,
                                                                   params             = best_params_var,
                                                                   rolling_window     = rolling_window
                                                                   )[1]

            else:
                os_backtest_var = pd.DataFrame()
                
            # 3. Modelo definitivo y pronóstico final
            # -----------------------------------------------------------------------------
            if rolling_window:
                training_var_stationary = stationary_macro_df.tail(clean_ts[:date_init_train].shape[0])
            else:
                training_var_stationary = stationary_macro_df.copy()
                
                
            
            definitive_var_model = varIM().var_training(stationary_data = training_var_stationary,
                                                        top_predictors  = top_predictors,
                                                        params          = best_params_var
                                                        )

            # Predicción final y simulaciones
            # NOTE: Se debe configurar el parámetro alternative_simulations en False para evitar problemas con los datos de entrenamiento.
            el_sim = [(False, False),
                      (True, False), 
                      (True, True)]
            
            for alt_sims, remove_out in el_sim:
            
                forecast_var, simulations_var, fitted_var = varIM().var_forecast(regular_data           = macro_vars_df,
                                                                                stationary_data         = training_var_stationary,
                                                                                fitted_model            = definitive_var_model,
                                                                                target_variable         = var_name,
                                                                                top_predictors          = top_predictors,
                                                                                fwd_periods             = periods_ahead,
                                                                                alternative_simulations = alt_sims,
                                                                                do_simulations          = True,
                                                                                remove_outliers         = remove_out,
                                                                                freq                    = season_freq,
                                                                                d_var                   = d_var,
                                                                                D_var                   = D_var
                                                                                )
                
                # Si el tipo de la serie es diferente a "abs", parar el ciclo en la primera iteración
                if ts_type != 'abs':
                    break
                
                if not (simulations_var < 0).any().any():
                    break
                
            params_df['alt_simulations_var'] = alt_sims
            params_df['remove_outliers_var'] = remove_out
            
            # Si aún así persisten los valores negativos, se reemplazan por el valor anterior
            if (simulations_var < 0).any().any() and ts_type == 'abs':
                simulations_var = (simulations_var.mask(simulations_var < 0).ffill().bfill().fillna(0))
                params_df['replace_neg_sims_var'] = True
            else:
                pass
            
            
            # Aplicar función para el cálculo de los eprcentiles
            var_abs_q, var_abs_y, var_pct_q, var_interannual, var_interannual_avg, var_interannual_last = ts_functions().percentiles_compute(history   = clean_ts, 
                                                                                                                                             forecast  = forecast_var, 
                                                                                                                                             sims      = simulations_var.copy(),
                                                                                                                                             data_type = ts_type,
                                                                                                                                             freq      = season_freq
                                                                                                                                             )
            
            
            # Si se encuentran resultados correctos se para el bucle
            break
        
        except Exception as e:
            
            print('*' * 110)
            print(f"Error en el modelo VAR con parámetros ({candidate_model}): {e}")
            print('*' * 110)
            pass
            
    params_df['best_model_var'] = str(f"{best_params_var}")
    
    params_df['top_predictors'] = ','.join(top_predictors)
    
    # Índice del df metrics al cual pertenece el modelo
    params_df['top_var'] = df_var_metrics.loc[df_var_metrics['model'] == best_params_var].index
    
    print(f'Los parámetros óptimos que funcionaron para el modelo VAR ({var_name}) corresponden a: {best_params_var}')
    
    
    # =============================================================================
    # RESUMEN MÉTRICAS DE ERROR
    # =============================================================================
    # Lista de modelos y sus backtests    
    if pd.to_datetime(date_init_val) < clean_ts.tail(1).index[0]:
        models = [
            {"name": "Holt-Winters", "validation": backtest_hw, "test": os_backtest_hw},
            {"name": "SARIMA", "validation": backtest_sarima, "test": os_backtest_sarima},
            {"name": "Prophet", "validation": backtest_prophet, "test": os_backtest_prophet},
            {"name": "Unobserved Components", "validation": backtest_ucm, "test": os_backtest_ucm},
            {"name": "ARDL", "validation": backtest_ardl, "test": os_backtest_ardl},
            {"name": "VAR", "validation": backtest_var, "test": os_backtest_var}
        ]

        # Consolidar errores en dataframes
        error_metrics = []
        
        for model in models:
            error_metrics.append(ts_functions().calculate_error_metrics(model["validation"], model["name"], "validation"))
            error_metrics.append(ts_functions().calculate_error_metrics(model["test"], model["name"], "test"))

        # Combinar todos los resultados en un único dataframe
        error_metrics_df = pd.concat(error_metrics, ignore_index = True)
    
    else:
        models = [
            {"name": "Holt-Winters", "validation": backtest_hw},
            {"name": "SARIMA", "validation": backtest_sarima},
            {"name": "Prophet", "validation": backtest_prophet},
            {"name": "Unobserved Components", "validation": backtest_ucm},
            {"name": "ARDL", "validation": backtest_ardl},
            {"name": "VAR", "validation": backtest_var}
        ]

        # Consolidar errores en dataframes
        error_metrics = []
        
        for model in models:
            error_metrics.append(ts_functions().calculate_error_metrics(model["validation"], model["name"], "validation"))

        # Combinar todos los resultados en un único dataframe
        error_metrics_df = pd.concat(error_metrics, ignore_index = True)
    
    
    # # =============================================================================
    # # EXPORTAR RESULTADOS
    # # =============================================================================
    # NOTE: No se exporta el unified_backtest, ni el unnified fitted ya que no es necesario para el reporte ejecutivo y su peso es enorme.
    
    outputs_list = [clean_ts,
                    ts_df,
                    df_abs,
                    df_abs_y,
                    df_pct,
                    df_interannual,
                    df_avg_year,
                    df_last_year,
                    params_df,
                    error_metrics_df,
                    df_hw_metrics,
                    backtest_hw,
                    os_backtest_hw,
                    forecast_hw,
                    simulations_hw,
                    fitted_hw,
                    hw_abs_q,
                    hw_abs_y,
                    hw_pct_q,
                    hw_interannual,
                    hw_interannual_avg,
                    hw_interannual_last,
                    df_sarima_metrics,
                    backtest_sarima,
                    os_backtest_sarima,
                    forecast_sarima,
                    simulations_sarima,
                    fitted_sarima,
                    sarima_abs_q,
                    sarima_abs_y,
                    sarima_pct_q,
                    sarima_interannual,
                    sarima_interannual_avg,
                    sarima_interannual_last,
                    df_prophet_metrics,
                    backtest_prophet,
                    os_backtest_prophet,
                    forecast_prophet,
                    simulations_prophet,
                    fitted_prophet,
                    prophet_abs_q,
                    prophet_abs_y,
                    prophet_pct_q,
                    prophet_interannual,
                    prophet_interannual_avg,
                    prophet_interannual_last,
                    df_ucm_metrics,
                    backtest_ucm,
                    os_backtest_ucm,
                    forecast_ucm,
                    simulations_ucm,
                    fitted_ucm,
                    ucm_abs_q,
                    ucm_abs_y,
                    ucm_pct_q,
                    ucm_interannual,
                    ucm_interannual_avg,
                    ucm_interannual_last,
                    df_ardl_metrics,
                    backtest_ardl,
                    os_backtest_ardl,
                    forecast_ardl,
                    simulations_ardl,
                    fitted_ardl,
                    ardl_abs_q,
                    ardl_abs_y,
                    ardl_pct_q,
                    ardl_interannual,
                    ardl_interannual_avg,
                    ardl_interannual_last,
                    df_var_metrics,
                    backtest_var,
                    os_backtest_var,
                    forecast_var,
                    simulations_var,
                    fitted_var,
                    var_abs_q,
                    var_abs_y,
                    var_pct_q,
                    var_interannual,
                    var_interannual_avg,
                    var_interannual_last
                    ]
    
    sheet_names = [
        "clean_ts",
        "ts_df",
        "df_abs",
        "df_abs_y",
        "df_pct",
        "df_interannual",
        "df_avg_year",
        "df_last_year",
        "params_df",
        "error_metrics_df",
        "df_hw_metrics",
        "backtest_hw",
        "os_backtest_hw",
        "forecast_hw",
        "simulations_hw",
        "fitted_hw",
        "hw_abs_q",
        "hw_abs_y",
        "hw_pct_q",
        "hw_interannual",
        "hw_interannual_avg",
        "hw_interannual_last",
        "df_sarima_metrics",
        "backtest_sarima",
        "os_backtest_sarima",
        "forecast_sarima",
        "simulations_sarima",
        "fitted_sarima",
        "sarima_abs_q",
        "sarima_abs_y",
        "sarima_pct_q",
        "sarima_interannual",
        "sarima_interannual_avg",
        "sarima_interannual_last",
        "df_prophet_metrics",
        "backtest_prophet",
        "os_backtest_prophet",
        "forecast_prophet",
        "simulations_prophet",
        "fitted_prophet",
        "prophet_abs_q",
        "prophet_abs_y",
        "prophet_pct_q",
        "prophet_interannual",
        "prophet_interannual_avg",
        "prophet_interannual_last",
        "df_ucm_metrics",
        "backtest_ucm",
        "os_backtest_ucm",
        "forecast_ucm",
        "simulations_ucm",
        "fitted_ucm",
        "ucm_abs_q",
        "ucm_abs_y",
        "ucm_pct_q",
        "ucm_interannual",
        "ucm_interannual_avg",
        "ucm_interannual_last",
        "df_ardl_metrics",
        "backtest_ardl",
        "os_backtest_ardl",
        "forecast_ardl",
        "simulations_ardl",
        "fitted_ardl",
        "ardl_abs_q",
        "ardl_abs_y",
        "ardl_pct_q",
        "ardl_interannual",
        "ardl_interannual_avg",
        "ardl_interannual_last",
        "df_var_metrics",
        "backtest_var",
        "os_backtest_var",
        "forecast_var",
        "simulations_var",
        "fitted_var",
        "var_abs_q",
        "var_abs_y",
        "var_pct_q",
        "var_interannual",
        "var_interannual_avg",
        "var_interannual_last"
        ]


    # Exportal
    excel_path = f"evidence_macro/evidence_{var_name}.xlsx"
        
    with pd.ExcelWriter(excel_path, engine = "openpyxl") as writer:
        for sheet_name, df in zip(sheet_names, outputs_list):
            try:
                df.to_excel(writer, sheet_name = sheet_name)
            except Exception as e:
                print(f"Error escribiendo la hoja {sheet_name}: {e}")
    
    # Mensaje del proceso terminado
    print('=' * 110)
    print('*' * 110)
    print(f"¡EL PROCESO HA FINALIZADO! Los resultados se encuentran en el archivo {excel_path}")
    print('*' * 110)
    print('=' * 110)
    
    return None # outputs_list


# Aplicar función
# ----------------------------------------------------------------------------------------------------------
list_error_vars = []

def run_executive_report(*args):
    try:
        return executive_report(*args)
    except Exception as e:
        print(f"Error en la ejecución del reporte: {e}")

if __name__ == '__main__':
    # Cargar datos
    dict_data = pd.read_excel('dict_variables.xlsx')
           
    macro_dict_data = dict_data[(dict_data['mark'] == 5) & (dict_data['methodology'] == 'macro')].reset_index(drop = True)
    
    list_variables = macro_dict_data['var_name'].values
    list_end_date = macro_dict_data['end_date'].dt.strftime('%Y-%m-%d').tolist()
    lista_ts_type = macro_dict_data['ts_type'].values
    
    # Dataframe múltiple para modelo VAR
    multiple_macro_df = pd.read_excel('data/processed/variables_cleaned.xlsx', sheet_name = 'multiple_macro', index_col = 0)
    
    tasks = []
    
    for i, var_name in enumerate(list_variables):
        if i < 200:  # Limitar para pruebas
            try:
                ts_df = pd.read_excel('data/processed/variables_cleaned.xlsx', sheet_name = var_name, index_col = 0)
                
                try:
                    monthly_df = pd.read_excel('data/processed/variables_cleaned.xlsx', sheet_name = var_name + ' M', index_col = 0)
                except:
                    monthly_df = pd.DataFrame()
                    
                params = [
                    (var_name, 
                     ts_df, 
                     multiple_macro_df, 
                     False, False, False, 
                     list_end_date[i],
                     4, # Número de pasos para el pronóstico 
                     1, # Número de pasos para el reentrenamiento
                     4, # Frecuencia de la serie temporal (Q = 4)
                     'Q',
                     lista_ts_type[i],
                     '2014-01-01', 
                     '2028-03-31'
                     ), # NOTE: Cuidado con los steps.
                ]
    
                tasks.extend(params)
    
            except Exception as e:
                print(f"Error en la variable {var_name}: {e}")
        else:
            pass

    results = Parallel(n_jobs = os.cpu_count())(delayed(run_executive_report)(*p) for p in tasks)
    

# Ruta donde se almacenan los archivos de resultados
results_directory = 'evidence_macro/'

# Obtener la lista de archivos que comienzan con 'evidence_'
result_files = [filename for filename in os.listdir(results_directory) if filename.startswith('evidence_')]

# Extraer las variables contenidas en los nombres de los archivos
extracted_vars = [filename.replace('evidence_', '') for filename in result_files]
extracted_vars = [filename.replace('.xlsx', '') for filename in extracted_vars]

# Cargar el diccionario de variables y filtrar por las relevantes para macro
variables_catalog = pd.read_excel('dict_variables.xlsx')
macro_variables = variables_catalog[
    (variables_catalog['mark'] == 1) & 
    (variables_catalog['methodology'] == 'macro')
]['var_name'].reset_index(drop=True).to_list()

not_executed_vars = [var for var in macro_variables if var not in extracted_vars]

print("Variables que no se ejecutaron:", not_executed_vars)
