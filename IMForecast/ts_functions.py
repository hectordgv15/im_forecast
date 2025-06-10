# Librearías
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import skew
import math
from statsmodels.tsa.stattools import adfuller
from scipy.interpolate import PchipInterpolator


class ts_functions:
    """
    Clase con funciones para el análisis de series de tiempo y modelos de predicción
    """
    
    def __init__(self):
        pass
    
    # Función para la creación de folds (CV)
    def ts_split(self, 
                 df: pd.DataFrame,
                 initial_train_date,
                 val_size: int,
                 steps: int = 1,
                 rolling_window: bool = False
                 ):
        """
        Divide un DataFrame temporal en folds de entrenamiento / validación.

        Parámetros
        ----------
        df : pd.DataFrame
            Debe tener un índice tipo fecha y la(s) columna(s) que quieras modelar.
        initial_train_date : str | pd.Timestamp
            Fecha límite (inclusive) del primer bloque de entrenamiento.
        val_size : int
            Número de observaciones en cada bloque de validación (horizonte).
        steps : int, default 1
            Cuántos periodos avanza el corte entre folds consecutivos.
        rolling_window : bool, default False
            True  → ventana deslizante (tamaño fijo).  
            False → ventana expansiva (crece con cada fold).

        Devuelve
        --------
        (dict_train, dict_val) : tuple[dict, dict]
            Dos diccionarios con los folds.
            · dict_train['fold_train_i'] → DataFrame de entrenamiento del fold i  
            · dict_val['fold_val_i']   → DataFrame de validación del fold i
        """
        # Tamaño inicial de entrenamiento
        initial_train_size = df[df.index <= initial_train_date].shape[0]
        n_total = len(df)

        # Índices de inicio de cada ventana de validación
        start_idxs = range(
            initial_train_size,                # comienza justo después del primer entrenamiento
            n_total - val_size + 1,            # último inicio posible que quepa val_size
            steps                              # tamaño del salto entre folds
        )

        dict_train, dict_val = {}, {}

        for i, start in enumerate(start_idxs):
            end = start + val_size
            if end > n_total:      # por seguridad; se ignora ventana incompleta
                break

            # Selección de datos de entrenamiento
            if rolling_window:
                train_start = start - initial_train_size
                train_df = df.iloc[train_start:start]      # ventana fija
            else:
                train_df = df.iloc[:start]                 # ventana expansiva

            # Datos de validación
            val_df = df.iloc[start:end]
            
            if train_df.index[-1] >= val_df.index[0]:
                raise ValueError(f"El índice de entrenamiento ({train_df.index[-1]}) no debe ser mayor o igual que el índice de validación ({val_df.index[0]}).")

            dict_train[f'fold_train_{i}'] = train_df.copy()
            dict_val[f'fold_val_{i}'] = val_df.copy()

        return dict_train, dict_val
        
    
    # Métricas de error
    def evaluate_model(self, y_test, y_pred):
        """
        Evalúa el modelo con métricas clásicas.
        Parámetros:
        -----------
        model : modelo entrenado
        y_test : target de prueba
        y_pred : predicciones
        Retorna:
        --------
        rmse : float
        """
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        if np.isnan(rmse) or np.isinf(rmse):
            raise ValueError(f"RMSE inválido: {rmse}")
        else:    
            return rmse
    
    
    # Métricas de error consolidadas
    # NOTE: En el proceso de optimización se calcula la métrica de error RSME por separado, es decir, primero una ventana
    # luego la otra y, porteriormente, se promedia el error. Cuando calculamos las métricas consolidadas que finalmente
    # se presentarán al usuario, se calcula el RMSE promediando los errores (RMSE) de forma agregada, es decir, no se computa
    # por ventanas, con lo cual, tienda a haber una pequeña diferencia respecto a la métrcia de optimización (solo para el caso del RMSE).
    def calculate_error_metrics(self, df, model_name, type_backtest = 'validation'):
        observed = df['observed'].values
        predicted = df['predicted'].values

        mae = mean_absolute_error(observed, predicted)
        mse = mean_squared_error(observed, predicted)
        rmse = np.sqrt(mse)  # RMSE directo del MSE
        mape = np.mean(np.abs((observed - predicted) / observed)) * 100  # MAPE en porcentaje

        # Consolidar métricas en un dataframe
        return pd.DataFrame([{
            "MODELO": model_name,
            "MAE"   : mae,
            "MSE"   : mse,
            "RMSE"  : rmse,
            "MAPE"  : mape,
            "TYPE"  : type_backtest
        }])
            
    
    def simulate_from_residuals(
        self,
        residuals,
        forecast,
        repetitions     = None,
        error           = "add",
        random_errors   = None,
        random_state    = None,
        remove_outliers = False,
    ):
        """
        Genera caminos de simulación que giran en torno a un forecast dado,
        utilizando la variabilidad estimada a partir de los residuos del modelo.
        
        Cada camino simulado se construye aplicando, para cada paso futuro,
        un error aleatorio al valor del forecast:
        - Para error aditivo: sim[t] = forecast[t] + error
        - Para error multiplicativo: sim[t] = forecast[t] * (1 + error)
        
        Parámetros
        ----------
        residuals : array-like
            Residuos del modelo, a partir de los cuales se estima la desviación estándar.
        forecast : array-like
            Vector de forecast (valores esperados) para cada paso futuro.
        repetitions : int, opcional
            Número de caminos simulados a generar. Por defecto es 1.
        error : {"add", "mul", "additive", "multiplicative"}, opcional
            Tipo de error a aplicar:
            - "add" o "additive": se suma el error al forecast.
            - "mul" o "multiplicative": se multiplica el forecast por (1 + error).
        random_errors : array-like, opcional
            Si se proporciona, debe tener forma (n_steps, repetitions), donde n_steps es la
            longitud del vector de forecast. Se usarán estos valores en lugar de generar
            errores aleatorios.
        random_state : int o None, opcional
            Semilla para el generador de números aleatorios.
        
        Retorna
        -------
        sim : np.ndarray
            Array de simulaciones de forma (n_steps, repetitions), donde cada columna es un camino simulado.
        """
        # Convertir a arrays de NumPy
        residuals = np.asarray(residuals)
        forecast = np.asarray(forecast)
        n_steps = forecast.shape[0]
        
        if remove_outliers:
            # Filtrar outliers usando el criterio del IQR (rango intercuartílico)
            q1 = np.percentile(residuals, 25)
            q3 = np.percentile(residuals, 75)
            iqr = q3 - q1

            # Definir límites para filtrar (por ejemplo, 1.5 veces el IQR)
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Aplicar el filtro
            residuals = residuals[(residuals >= lower_bound) & (residuals <= upper_bound)]
        
            # Estimar la desviación estándar de los residuos
            sigma = np.std(residuals, ddof = 1)
        else:
            # Estimar la desviación estándar de los residuos
            sigma = np.std(residuals, ddof = 1)
        
        # Configurar el generador aleatorio
        rng = np.random.default_rng(random_state)
        
        # Obtener la matriz de errores aleatorios o usar los proporcionados
        if random_errors is not None:
            eps = np.asarray(random_errors)
            if eps.shape != (n_steps, repetitions):
                raise ValueError(
                    "Si random_errors es un ndarray, debe tener forma (n_steps, repetitions)"
                )
        else:
            eps = rng.normal(loc = 0, scale = sigma, size = (n_steps, repetitions))
        
        # Aplicar el error al forecast para cada paso
        if error in ["add", "additive"]:
            # Se suma el error al forecast de cada periodo
            sim = forecast[:, None] + eps
        elif error in ["mul", "multiplicative"]:
            # Se multiplica el forecast por (1 + error)
            sim = forecast[:, None] * (1 + eps)
        else:
            raise ValueError("El parámetro error debe ser 'add' o 'mul' (o 'additive'/'multiplicative').")
        
        return sim


    # Función para el cálculo de percentiles
    def percentiles_compute(
        self,
        forecast  : pd.Series,
        history   : pd.DataFrame,
        sims      : pd.DataFrame, 
        freq      = 4,
        data_type = None
        ):
        """
        Calcula percentiles de un forecast y sus simulaciones, tanto en valores absolutos como en variación porcentual.
        Parámetros:
        -----------
        forecast : pd.Series
            Serie de pronósticos con índice temporal.
        history : pd.DataFrame
            DataFrame con datos históricos.
        sims : pd.DataFrame
            DataFrame con simulaciones, debe tener el mismo índice que forecast.
        freq : int, opcional
            Frecuencia de los datos (por defecto 4, para datos trimestrales).
        """        
        if not isinstance(forecast, pd.Series):
            raise ValueError("El parámetro 'forecast' debe ser una Serie de Pandas.")
        if not isinstance(history, pd.DataFrame):
            raise ValueError("El parámetro 'history' debe ser un DataFrame de Pandas.")
        if not isinstance(sims, pd.DataFrame):
            raise ValueError("El parámetro 'sims' debe ser un DataFrame de Pandas.")
        
        # Las fechas iniciales y finales de forecast y sims deben ser las mismas
        if forecast.index[0] != sims.index[0] or forecast.index[-1] != sims.index[-1]:
            raise ValueError("Las fechas iniciales y finales de 'forecast' y 'sims' deben ser las mismas.")
        
        # La última fecha de history debe ser anterior a la primera fecha de forecast
        if history.index[-1] >= forecast.index[0] or history.index[-1] >= sims.index[0]:
            raise ValueError("La última fecha de 'history' debe ser anterior a la primera fecha de 'forecast'. Estas son las fechas:\n"
                             f"Última fecha de history: {history.index[-1]}\n Primera fecha de forecast: {forecast.index[0]}")
        
        # Tamaño del pronóstico
        len_forecast = len(forecast)
        
        # Lista de percentiles a calcular
        pct_list = [1, 5] + list(range(10, 50, 5)) + list(range(55, 100, 5)) + [99]       

        # Añadir valores históricos al forecast
        add_forecast = history.iloc[-freq:, 0].rename('forecast')
        
        all_forecast = pd.concat([add_forecast, forecast.copy()], axis = 0)
        
        if all_forecast.index.duplicated().any():
            raise ValueError("El índice de all_forecast tiene duplicados. Por favor, revise los datos.")
        
        # Añadir valores históricos a las simulaciones
        add_sims = pd.concat([history.iloc[-freq:, :]] * sims.shape[1], axis = 1)
        add_sims.columns = sims.columns
        all_sims = pd.concat([add_sims, sims], axis = 0)
        
        if all_sims.index.duplicated().any():
            raise ValueError("El índice de all_sims tiene duplicados. Por favor, revise los datos.")

        # Cálculo de percentiles en valores absolutos
        # ----------------------------------------------------------------------------------------------
        # NOTE: Se usan las simulaciones y el forecast normal
        # 1)
        perc_abs_q = sims.apply(
            lambda row: np.percentile(row, pct_list),
            axis = 1, 
            result_type = 'expand'
        )
        
        perc_abs_q.columns = [f"P{p}" for p in pct_list]
        
        # 2)
        perc_abs_q['forecast'] = forecast.values

        # 3)
        perc_abs_y = sims.groupby(sims.index.year).last().apply(
            lambda row: np.percentile(row, pct_list),
            axis = 1, 
            result_type = 'expand'
        )
        
        perc_abs_y.columns = [f"P{p}" for p in pct_list]
        
        perc_abs_y['forecast'] = forecast.groupby(forecast.index.year).last().values
        
        # Cálculo de percentiles en variación porcentual
        # ----------------------------------------------------------------------------------------------
        if data_type == 'abs':
            pct_sims = all_sims.apply(lambda x: np.log(x / x.shift(1)).dropna(), axis = 0)
            pct_forecast = np.log(all_forecast / all_forecast.shift(1)).dropna()  
        elif data_type == 'pct':
            pct_sims = all_sims.copy()
            pct_forecast = all_forecast.copy()
        
        elif data_type == 'rate':
            pct_sims = sims.copy()
            pct_forecast = forecast.copy()
        
        else:
            raise ValueError("data_type debe ser 'abs', 'pct' o 'rate'.")
        
        
        if data_type == 'abs' or data_type == 'pct':    
            # A)
            interannual_sims = pct_sims.rolling(window = 4).sum().dropna()            
            interannual_forecast = pct_forecast.rolling(window = 4).sum().dropna()
            
            # B)
            annual_avg_sims = interannual_sims.groupby(interannual_sims.index.year).mean()
            annual_avg_forecast = interannual_forecast.groupby(interannual_forecast.index.year).mean()
            
            # C)
            annual_last_sims = interannual_sims.groupby(interannual_sims.index.year).last()
            annual_last_forecast = interannual_forecast.groupby(interannual_forecast.index.year).last()
        elif data_type == 'rate':
            # A)
            interannual_sims = pct_sims.copy()
            interannual_forecast = pct_forecast.copy()
            
            # B)
            annual_avg_sims = interannual_sims.groupby(interannual_sims.index.year).last()
            annual_avg_forecast = interannual_forecast.groupby(interannual_forecast.index.year).last()
            
            # C)
            annual_last_sims = interannual_sims.groupby(interannual_sims.index.year).last()
            annual_last_forecast = interannual_forecast.groupby(interannual_forecast.index.year).last()
        else:
            raise ValueError("data_type debe ser 'abs', 'pct' o 'rate'.")
            
            
        # ----------------------------------------------------------------------------------------------
        perc_pct_q = pct_sims.apply(
            lambda row: np.percentile(row, pct_list),
            axis = 1, 
            result_type = 'expand'
        ).tail(len_forecast)
        
        perc_pct_q.columns = [f"P{p}" for p in pct_list]
        
        perc_pct_q['forecast'] = pct_forecast.tail(len_forecast).values
        
        # ----------------------------------------------------------------------------------------------
        perc_interannual = interannual_sims.apply(
            lambda row: np.percentile(row, pct_list),
            axis = 1, 
            result_type = 'expand'
            ).tail(len_forecast)
        
        perc_interannual.columns = [f"P{p}" for p in pct_list]
        
        perc_interannual['forecast'] = interannual_forecast.tail(len_forecast).values
        
        # ----------------------------------------------------------------------------------------------    
        perc_annual_avg = annual_avg_sims.apply(
            lambda row: np.percentile(row, pct_list),
            axis = 1,
            result_type = 'expand'
        )
        
        perc_annual_avg.columns = [f"P{p}" for p in pct_list]
        
        perc_annual_avg['forecast'] = annual_avg_forecast.values
        
        # ----------------------------------------------------------------------------------------------
        perc_annual_last = annual_last_sims.apply(
            lambda row: np.percentile(row, pct_list),
            axis = 1,
            result_type = 'expand'
        )
        
        perc_annual_last.columns = [f"P{p}" for p in pct_list]
        
        perc_annual_last['forecast'] = annual_last_forecast.values
            
        
        return perc_abs_q, perc_abs_y, perc_pct_q, perc_interannual, perc_annual_avg, perc_annual_last
      
      
    # Función para limpiar outliers
    def clean_outliers(self, 
                       df, 
                       col    = 'ts', 
                       method = 'z', 
                       thresh = 3, 
                       k      = 1.5
                       ):
        """
        Limpia outliers en df[col] usando 'z' (Z-score) o 'iqr' (IQR) y luego interpola.
        
        Parámetros:
            df (pd.DataFrame): DataFrame con índice temporal.
            col (str): Nombre de la columna a limpiar (por defecto 'ts').
            method (str): Método de detección ('z' o 'iqr').
            thresh (float): Umbral de Z-score para método 'z'.
            k (float): Multiplicador de IQR para método 'iqr'.
        
        Imprime los outliers detectados y sus valores antes y después de la interpolación.
        """
        df = df.copy()
        # Calcular la máscara de outliers
        if method == 'z':
            m, s = df[col].mean(), df[col].std()
            z = (df[col] - m) / s
            mask = z.abs() > thresh
        elif method == 'iqr':
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            mask = (df[col] < q1 - k * iqr) | (df[col] > q3 + k * iqr)
        else:
            raise ValueError("method must be 'z' or 'iqr'")

        # Guardar valores originales de los outliers
        outliers_original = df.loc[mask, col].copy()

        # Reemplazar outliers por NaN y luego interpolar
        df.loc[mask, col] = np.nan
        df[col] = df[col].interpolate(method = 'time', limit_direction = 'both')

        # Obtener los nuevos valores tras la interpolación
        outliers_replaced = df.loc[outliers_original.index, col]

        # Imprimir resultados
        if not outliers_original.empty:
            print(f"Outliers detectados en la columna '{col}':")
            for idx, val in outliers_original.items():
                print(f"  Índice {idx}: valor original = {val}")
            print("\nValores tras la interpolación:")
            for idx, val in outliers_replaced.items():
                print(f"  Índice {idx}: nuevo valor = {val}")
        else:
            print(f"No se detectaron outliers en la columna '{col}' con el método '{method}'.")

        return df
    

    def determine_diff_orders(self, 
                              series        : pd.Series, 
                              freq          = 4,
                              alpha         = 0.01, # NOTE: Parámetro importante
                              return_series = False,
                              initial_date  = None,
                              end_date      = None  # Fecha de corte para la serie
                              ):
        """
        Determina los órdenes de diferenciación d y D para una serie temporal.
        Parámetros
        ----------
        series : pd.Series
            Serie temporal con índice datetime y .freq definido.
        freq : int, opcional
            Periodo de estacionalidad (por defecto 4, para datos trimestrales).
        alpha : float, opcional
            Nivel de significancia para la prueba ADF (por defecto 0.05).
        return_series : bool, opcional
            Si es True, retorna la serie transformada además de los órdenes d y D (por defecto False).
        Retorna
        -------
        tuple
            (d, D, series_transformed) si return_series es True,
            o (d, D) si return_series es False.
        """
        
        # 1) Test para la serie original
        # filtrar la serie después de una fecha dada
        filt_series = series[(series.index >= pd.Timestamp(initial_date)) & (series.index <= pd.Timestamp(end_date))].copy()
        
        pd_value_ts_n = adfuller(filt_series)[1]
        
        if pd_value_ts_n < alpha:
            D, d = 0, 0
        else:
            # 2) Test para la serie diferenciada
            ts_d = filt_series.diff(periods = 1).dropna()
            pvalue_d = adfuller(ts_d)[1]
            if pvalue_d < alpha:
                D, d = 0, 1

            else:
                # 3) Test para la serie diferenciada estacionalmente
                ts_D = filt_series.diff(periods = freq).dropna()
                pvalue_D = adfuller(ts_D)[1]

                if pvalue_D < alpha:
                    D, d = 1, 0
                else:
                    # 4) Test para la serie diferenciada estacionalmente y luego diferenciada regular
                    ts_Dd = filt_series.diff(periods = freq).diff(periods = 1).dropna()
                    pdvalue_Dd = adfuller(ts_Dd)[1]
                    if pdvalue_Dd < alpha:
                        D, d = 1, 1
                    else:
                        # Si ninguna de las pruebas anteriores es significativa, se deja la serie original
                        print(f"La serie {filt_series.name} no es estacionaria ni con diferenciación regular ni estacional.")
                        D, d = 0, 0
        
        # --- 3) Serie transformada final --- NOTE: Serie original.
        series_transformed = series.copy()
        
        if D > 0:
            series_transformed = series_transformed.diff(freq).dropna()
        if d > 0:
            series_transformed = series_transformed.diff(d).dropna()
            
        # series_transformed = series_transformed.dropna()
        
        print(f"Determinados órdenes de diferenciación: d = {d}, D = {D} para la variable {series.name}")

        return (d, D, series_transformed) if return_series else (d, D)


    def invert_diff_forecast(self,
                             forecast_diff: pd.Series,
                             original_series: pd.Series,
                             d: int,
                             D: int,
                             m: int
                             ) -> pd.Series:
            """
            Invierte las diferencias regulares (d) y estacionales (D) de un vector
            de pronósticos para devolverlos a la escala de la serie original.

            Parámetros:
            forecast_diff : pd.Series
                Serie de pronósticos diferenciados.
            original_series : pd.Series
                Serie original de la que se han obtenido las diferencias.
            d : int
                Orden de diferenciación regular (diferencias simples).
            D : int
                Orden de diferenciación estacional (diferencias estacionales).
            m : int
                Periodo de estacionalidad (número de observaciones por ciclo estacional).
            Retorna:
            --------
            pd.Series
                Serie de pronósticos invertidos a la escala de la serie original.
            """
            # La última fecha de la serie original debe ser menor que la primera fecha de forecast_diff
            if original_series.index[-1] >= forecast_diff.index[0]:
                raise ValueError("La última fecha de original_series debe ser anterior a la primera fecha de forecast_diff. Estas son las fechas:\n"
                                 f"Última fecha de original_series: {original_series.index[-1]}\n Primera fecha de forecast_diff: {forecast_diff.index[0]}")
                
            if D > 0 and len(original_series) < (m + 1):
                raise ValueError("La serie original debe tener al menos m + 1 observaciones para poder invertir las diferencias. "
                                 f"Tamaño de la serie original: {len(original_series)}, m + 1 = {m + 1}.")
                
            if d > 1 or D > 1:
                raise ValueError("Los órdenes de diferenciación d y D deben ser 0 o 1. Valores recibidos: d = {}, D = {}".format(d, D))
            
            if d == 0 and D == 0:
                return pd.Series(forecast_diff)
                        
            # Guardar el índice original de la serie
            forecast_index = forecast_diff.index
            
            # Tamaño de la serie
            h = len(forecast_diff)
            
            if d == 1 and D == 0:
                # Vector inicial
                s1 = original_series.copy()
                
                inverted_vector = []
                
                for i in range(h):
                    if i == 0:
                        inverted_vector.append(s1.iloc[-1] + forecast_diff.iloc[i])
                    else:
                        inverted_vector.append(inverted_vector[i - 1] + forecast_diff.iloc[i])
                
                return pd.Series(inverted_vector, index = forecast_index)      
             
            elif D == 1 and d == 0:
                # Vector inicial
                s1 = original_series.copy()
                
                # Si sólo se ha hecho una diferenciación estacional
                inverted_vector = []
                
                for i in range(h):
                    if i < m:
                        # para los primeros m pasos usamos el último valor histórico
                        inverted_vector.append(s1.iloc[-m + i] + forecast_diff.iloc[i])
                    else:
                        # luego ya referenciamos a pronósticos previos
                        inverted_vector.append(inverted_vector[i - m] + forecast_diff.iloc[i])
                
                return pd.Series(inverted_vector, index = forecast_index)   
            
            
            elif d == 1 and D == 1:
                # Vector inicial
                s1 = original_series.copy()

                inverted_vector = []
                
                for i in range(h):
                    
                    m_mod = m + 1
                    
                    if i == 0:
                        inverted_vector.append((s1.iloc[-1] - s1.iloc[-m_mod]) + forecast_diff.iloc[i] + s1.iloc[-m_mod + 1])  
                    elif i > 0 and i < (m_mod - 1):
                        inverted_vector.append((inverted_vector[i - 1] - s1.iloc[-m_mod + i]) + forecast_diff.iloc[i] + s1.iloc[(-m_mod + i) + 1])
                    elif i == (m_mod - 1):
                        inverted_vector.append((inverted_vector[-1] - s1.iloc[-1]) + forecast_diff.iloc[i] + inverted_vector[-i])
                    else: 
                        inverted_vector.append((inverted_vector[-1] - inverted_vector[-m_mod]) + forecast_diff.iloc[i] + inverted_vector[-m_mod + 1])
                
                return pd.Series(inverted_vector, index = forecast_index)
            
            else:
                raise ValueError("Los órdenes de diferenciación d y D deben ser 0 o 1. Valores recibidos: d = {}, D = {}".format(d, D))