# Librerías
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import FuncFormatter
import math
from joblib import Parallel, delayed
import os

from sklearn.preprocessing import StandardScaler

# Plotly
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
import plotly.graph_objs as go

from scipy.optimize import minimize

# Módulos propios
from IMForecast.ts_functions import ts_functions

# Módulo para el manejo de fechas
import holidays

# Optuna
import optuna
from optuna.samplers import TPESampler

# Modelo principal
from IMForecast.vasicek_model import VasicekIM

# =============================================================================
# LECTURA DE DATOS
# =============================================================================    
# Diccionario datos
dict_data = pd.read_excel('dict_variables.xlsx')
dict_data = dict_data[(dict_data['methodology'] == 'ir') & (dict_data['mark'] == 1)].reset_index(drop = True)

print("Variables a pronosticar:")
print(dict_data['var_name'].values)

dict_curves = {}

for i in range(len(dict_data)):
    df = pd.read_excel(dict_data['end_path'][i], sheet_name = dict_data['var_name'][i], index_col = 0)
    dict_curves[dict_data['var_name'][i]] = df
    
# =============================================================================
# FUNCIONES PARA LA GENERACIÓN DE FECHAS
# =============================================================================  
# Ajustar a día hábil
def adjust_to_business_day(date, next_day = False, country = 'ES'):
    """
    Ajusta una fecha al siguiente día hábil si cae en fin de semana o festivo.
    
    Parámetros:
    date (str o datetime): Fecha a ajustar.
    country (str): Código del país para considerar días festivos (por defecto España 'ES').
    
    Retorna:
    datetime.date: Fecha ajustada al siguiente día hábil.
    """
    # Convertir la fecha a datetime si es necesario
    date = pd.to_datetime(date).date()
    
    # Obtener la lista de festivos del país
    country_holidays = holidays.country_holidays(country)
    
    # Mientras la fecha caiga en fin de semana o sea festivo, avanzar un día
    while date.weekday() in [5, 6] or date in country_holidays:
        if next_day:
            date += pd.DateOffset(days = 1)
        else:
            date -= pd.DateOffset(days = 1)
            
        date = date.date()  # Convertir nuevamente a formato date
    
    return date


# Función para generar fechas en un rango específico
def generate_business_days(start_date, end_date, freq, country = "ES"):
    """
    Genera un rango de fechas hábiles excluyendo fines de semana y días festivos solo si la frecuencia es diaria.
    
    Args:
        start_date (str): Fecha de inicio en formato 'YYYY-MM-DD'.
        end_date (str): Fecha de finalización en formato 'YYYY-MM-DD'.
        freq (str): Frecuencia de la serie de fechas ('D' para diario, 'M' para mensual, '3M' para trimestral, etc.).
        country (str): Código del país para días festivos (por defecto, España 'ES').
    
    Returns:
        pd.DatetimeIndex: Fechas generadas según la frecuencia.
    """
    # Obtener días festivos del país seleccionado
    country_holidays = holidays.country_holidays(country)
    
    if freq == 'D':
        # Crear rango de fechas diarias
        date_range = pd.date_range(start = start_date, end = end_date, freq = freq)
        # Filtrar fechas que no sean fines de semana ni festivos
        business_days = date_range[
            (date_range.weekday < 5) & (~date_range.isin(country_holidays))
        ]
        return business_days

    elif freq in ['M', '3M']:
        # Generar los últimos días de cada mes o trimestre sin ajustes adicionales
        return pd.date_range(start = start_date, end = end_date, freq = freq)

    else:
        
        return pd.date_range(start = start_date, end = end_date, freq = freq)
    

# =============================================================================
# PARAMETROS INICIALES Y GENERACION DE FECHAS
# =============================================================================
final_date = adjust_to_business_day('2025-05-06')
end_date_forecast = adjust_to_business_day("2028-03-31")


start_date_forecast = adjust_to_business_day(str(final_date + pd.DateOffset(days = 1)).replace(' 00:00:00', ''), next_day = True)

print(f'Fecha final: {final_date}')
print(f'Fecha inicial forecast: {start_date_forecast}')

# Gerenerar secuencia de fechas para el pronóstico
business_days = generate_business_days(start_date_forecast, end_date_forecast, freq = 'D', country = "ES")

print("Fechas hábiles en el rango:")
print(business_days)
print("-" * 70)
print(f"Número de días a pronósticar: {len(business_days)}")


# =============================================================================
# CALIBRACION DE LOS PARAMETROS DEL MODELO DE VASICEK
# =============================================================================
# Función para el pronóstico individual
def curves_forecast(data,
                    business_days_forecast = None,
                    freq                   = 'D',
                    init_date              = None,
                    volatility_smoothing   = None
                    ):
    
    # Parámetros iniciales
    # -------------------------------------------------------------------------
    # Factor de cambio en el tiempo
    if freq == 'D':
        dt_factor = 1 / 252
    elif freq == 'M':
        dt_factor = 1 / 12
    elif freq == '3M':
        dt_factor = 1 / 4
        
    # Calibración de los parámetros del modelo de Vasicek
    # -------------------------------------------------------------------------
    # Semilla para reproducibilidad
    np.random.seed(42)

    # Filtrar dataframe
    curve_data = data[data.index >= pd.to_datetime(init_date)].copy()

    # Parámetros teóricos
    calibrate_data = curve_data.copy()
    
    if volatility_smoothing > 0:
        calibrate_data = calibrate_data.rolling(window = volatility_smoothing).mean().dropna()
    
    theorical_kappa, theorical_theta, theorical_sigma, r0 = VasicekIM().calibrate_vasicek_parameters(calibrate_data, dt = dt_factor)
    
    parameters = pd.DataFrame({'kappa':[theorical_kappa], 'theta': [theorical_theta], 'sigma': [theorical_sigma], 'r0': [r0]})
    
    print('=' * 100)
    print('Parámetros teóricos')
    print(parameters)
        
    parameters = parameters.copy()
    
    sim_arr = VasicekIM().multiple_vacisek_sim(M     = 1000, 
                                               N     = len(business_days_forecast), 
                                               r0    = curve_data.iloc[-1], 
                                               kappa = parameters.loc[0, 'kappa'], 
                                               theta = parameters.loc[0, 'theta'], 
                                               sigma = parameters.loc[0, 'sigma'], 
                                               dt    = dt_factor
                                               )

    # Calcular la media de las simulaciones
    forecast = np.percentile(sim_arr, 0.5, axis = 1)
    forecast = pd.Series(forecast, index = business_days_forecast)
    
    # Calcular percentiles a través de sim_arr
    # -------------------------------------------------------------------------
    percentile_df = pd.DataFrame()

    for i in [1, 99]:
        percentile_df[f'P{i}'] = np.percentile(sim_arr, i / 100, axis = 1)
    
    percentile_df.index = business_days_forecast
    percentile_df['forecast'] = forecast
    
    # Agrupar resultados por trimestre
    # -------------------------------------------------------------------------
    percentile_df = percentile_df.resample('Q').last()
    
    # Amplicar la distancia entre percentiles y forecast
    if volatility_smoothing == 0:
        percentile_df['P1'] = percentile_df['forecast'] - (percentile_df['forecast'] - percentile_df['P1']) * 3
        percentile_df['P99'] = percentile_df['forecast'] + (percentile_df['P99'] - percentile_df['forecast']) * 6
    elif volatility_smoothing == 5:
        percentile_df['P1'] = percentile_df['forecast'] - (percentile_df['forecast'] - percentile_df['P1']) * 3.5
        percentile_df['P99'] = percentile_df['forecast'] + (percentile_df['P99'] - percentile_df['forecast']) * 6.5
    elif volatility_smoothing == 10:
        percentile_df['P1'] = percentile_df['forecast'] - (percentile_df['forecast'] - percentile_df['P1']) * 4
        percentile_df['P99'] = percentile_df['forecast'] + (percentile_df['P99'] - percentile_df['forecast']) * 7
    elif volatility_smoothing == 15:
        percentile_df['P1'] = percentile_df['forecast'] - (percentile_df['forecast'] - percentile_df['P1']) * 4.5
        percentile_df['P99'] = percentile_df['forecast'] + (percentile_df['P99'] - percentile_df['forecast']) * 7.5
    elif volatility_smoothing == 20:
        percentile_df['P1'] = percentile_df['forecast'] - (percentile_df['forecast'] - percentile_df['P1']) * 5
        percentile_df['P99'] = percentile_df['forecast'] + (percentile_df['P99'] - percentile_df['forecast']) * 8
    elif volatility_smoothing == 25:
        percentile_df['P1'] = percentile_df['forecast'] - (percentile_df['forecast'] - percentile_df['P1']) * 5.5
        percentile_df['P99'] = percentile_df['forecast'] + (percentile_df['P99'] - percentile_df['forecast']) * 8.5
    elif volatility_smoothing == 30:
        percentile_df['P1'] = percentile_df['forecast'] - (percentile_df['forecast'] - percentile_df['P1']) * 6
        percentile_df['P99'] = percentile_df['forecast'] + (percentile_df['P99'] - percentile_df['forecast']) * 9
    elif volatility_smoothing == 35:
        percentile_df['P1'] = percentile_df['forecast'] - (percentile_df['forecast'] - percentile_df['P1']) * 6.5
        percentile_df['P99'] = percentile_df['forecast'] + (percentile_df['P99'] - percentile_df['forecast']) * 9.5
    elif volatility_smoothing == 40:
        percentile_df['P1'] = percentile_df['forecast'] - (percentile_df['forecast'] - percentile_df['P1']) * 7
        percentile_df['P99'] = percentile_df['forecast'] + (percentile_df['P99'] - percentile_df['forecast']) * 10
    else:
        print("Error: Volatility smoothing no válido")
    
    # Retornar resultados y parámetros
    return percentile_df


# =============================================================================
# APLICACION DEL MODELO A CADA CURVA
# =============================================================================
# Función que procesa cada combinación de curva, año y smoothing
def process_forecast_task(name, df, year, vol):
    forecast = curves_forecast(
        df['ts'],
        business_days_forecast = business_days,
        freq                   = 'D',
        init_date              = year,
        volatility_smoothing   = vol
    )
    
    print(f"Procesando: {name} - Año: {year} - Volatility Smoothing: {vol}")
    
    if forecast.empty:
        return None
    else:
        last_value = forecast['forecast'].iloc[-1]
        # Agregar columnas al dataframe forecast
        forecast['var_name'] = name
        forecast['smooth_volatility'] = vol
        forecast['date_ref'] = year
        
        # Imprimir el resultado
        print(f"Variable: {name} - Valor final: {last_value:.4f} - Volatility Smoothing: {vol}")
        
        # Preparar métricas para esta tarea
        metrics = {
            'date_ref': year,
            'final_value': last_value,
            'var_name': name,
            'smooth_volatility': vol
        }
        return forecast, metrics

# Generar lista de tareas: para cada curva, para cada año y para cada valor de volatilidad
tasks = []
years = [f"{year}-01-01" for year in range(2000, 2025)]
for name, df in dict_curves.items():
    for year in years:
        for vol in [0, 5, 10, 15, 20, 25, 30, 35, 40]:
            tasks.append((name, df, year, vol))

# Ejecutar en paralelo usando todos los cores disponibles
results = Parallel(n_jobs = os.cpu_count())(
    delayed(process_forecast_task)(name, df, year, vol) for name, df, year, vol in tasks
)

# Separar los dataframes de forecast y las métricas, descartando tareas sin resultados
forecast_list = []
metrics_list = []
for res in results:
    if res is not None:
        forecast, metrics = res
        forecast_list.append(forecast)
        metrics_list.append(metrics)

# Concatenar todos los dataframes de forecast y métricas
df_results = pd.concat(forecast_list)
df_results.index.name = 'date'
df_metrics = pd.DataFrame(metrics_list)
df_metrics = df_metrics.sort_values('final_value', ascending=False).reset_index(drop=True)

# =============================================================================
# EXPORTAR RESULTADOS
# =============================================================================
df_results.to_csv('evidence_curves/results_vasicek.csv', index = True)
df_metrics.to_csv('evidence_curves/metrics_vasicek.csv', index = False)