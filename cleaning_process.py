# Librerías
import pandas as pd
import numpy as np
import ast
import time
import matplotlib.pyplot as plt
from CleanDataIM.clean_data import ProcessDataIM
import pmdarima as pm  # para auto_arima
from statsmodels.tsa.arima.model import ARIMA

# Diccionario datos
dict_data = pd.read_excel('dict_variables.xlsx')

# Filtrar variables a procesar
dict_data = dict_data[dict_data['alternative_clean'] == False].reset_index(drop = True)

# =============================================================================
# PRIMERA ETAPA: Limpieza regular
# =============================================================================
# Temporalidad y ventana de entrenamiento
# ---------------------------------------------------------------------------------------------------------------------------
output_excel_path = 'data/processed/variables_cleaned.xlsx'

with pd.ExcelWriter(output_excel_path, engine = "openpyxl") as writer:
    
    for i in range(len(dict_data)):
        
        var_number = i
            
        print('=' * 140)
        print('=' * 140)
        print(f'Procesando variable: {dict_data.iloc[var_number]["var_name"]}')
        
        init_data = pd.read_excel(f"{dict_data.iloc[var_number]['source_path']}", sheet_name = dict_data.iloc[var_number]['sheet_name'])

        df_clean = ProcessDataIM().clean_data(init_data        = init_data,
                                              start_idx_row    = dict_data.iloc[var_number]['start_idx_row'], 
                                              end_idx_row      = dict_data.iloc[var_number]['end_idx_row'], 
                                              cols_num         = ast.literal_eval(dict_data.iloc[var_number]['cols_num']),
                                              col_names        = ast.literal_eval(dict_data.iloc[var_number]['col_names']),
                                              time_agg         = dict_data.iloc[var_number]['time_agg'], # Temporalidad trimestral
                                              start_date       = '1995-01-01', 
                                              end_date         = dict_data.iloc[var_number]['end_date']
                                              )
        
        # Obtener el nombre de la hoja a partir del nombre de la variable
        sheet_name = dict_data.iloc[var_number]["var_name"]
        
        # Escribir el DataFrame en la hoja correspondiente
        df_clean.to_excel(writer, sheet_name = sheet_name[:31], index = True)  # Limitar nombre de hoja a 31 caracteres
        
        print(f'Datos de {sheet_name} escritos en hoja de Excel. En la ruta: {output_excel_path}')

    print('=' * 140)
    print('=' * 140)
    

# Esperar 5 segundos antes de continuar
time.sleep(5)

# =============================================================================
# SEGUNDA ETAPA: Variables con procesamiento alternativo
# =============================================================================
def process_fac_dep(input_file, sheet_name, skip_rows, date_format, merge_date_range, output_file, output_sheet, noise_std=0.01):
    """
    Procesa la serie de 'Facilidad depósito':
      - Lee y limpia los datos (resampleo mensual).
      - Interpola los valores a partir de un rango de fechas dado.
      - Añade ruido aleatorio controlado.
      - Grafica y exporta la serie resultante.
    """
    # Lectura y limpieza
    df = pd.read_excel(input_file, sheet_name=sheet_name).iloc[skip_rows:, [0, 1]].reset_index(drop=True)
    df.columns = ['date', 'ts']
    df['ts'] = pd.to_numeric(df['ts'], errors='coerce')
    df['date'] = pd.to_datetime(df['date'], format=date_format)
    df.set_index('date', inplace=True)
    df = df.resample('M').last()

    # Interpolación y adición de ruido
    df_interp = pd.DataFrame({"date": merge_date_range})
    df_interp = df_interp.merge(df.reset_index(), on="date", how="left")
    df_interp["ts"] = df_interp["ts"].interpolate(method="linear")
    np.random.seed(42)
    df_interp["ts"] += np.random.normal(0, noise_std, size=len(df_interp))
    df_interp.dropna(inplace=True)
    df_interp.set_index("date", inplace=True)

    # Graficar
    plt.style.use('classic')
    plt.figure(figsize=(12, 5))
    plt.plot(df_interp["ts"], label="Serie interpolada", linewidth=1.5)
    plt.xlabel("Fecha")
    plt.ylabel("Tasa")
    plt.title("Evolución de la serie interpolada con baja volatilidad")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Exportar la serie a Excel
    with pd.ExcelWriter(output_file, engine="openpyxl", mode='a', if_sheet_exists='replace') as writer:
        df_interp.to_excel(writer, sheet_name=output_sheet)


# Ruta de salida para exportar las series procesadas
output_excel_path = 'data/processed/variables_cleaned.xlsx'

# === Procesar "Facilidad depósito" ===
# Se utiliza como rango de fechas el índice de la hoja 'EURIBOR 1M'
merge_date_range = pd.read_excel(output_excel_path, sheet_name='EURIBOR_1M', index_col=0).index
process_fac_dep(
    input_file='data/raw/Fichero 5.xlsx',
    sheet_name='Facilidad depósito',
    skip_rows=7,
    date_format='%d/%m/%Y',
    merge_date_range=merge_date_range,
    output_file=output_excel_path,
    output_sheet="FACILIDAD_DEPOSITO"
)

time.sleep(5)

# =============================================================================
# TERCERA ETAPA: Dataframe múltiple para modelos multifactoriales
# =============================================================================
# Cargar diccionario
dict_data = pd.read_excel('dict_variables.xlsx')
vars_to_estimate = dict_data.loc[dict_data['alt_estimation'] == 1, 'var_name']

final_date = pd.to_datetime('2025-03-31')  # 1T 2025
df_multiple_list = []

for var in vars_to_estimate:
    print(f"Procesando variable: {var}")
    
    # 1) cargo datos y recorto
    df = pd.read_excel('data/processed/variables_cleaned.xlsx',
                       sheet_name=var, index_col=0, parse_dates=True)
    df = df[df.index <= final_date].copy()
    
    # 2) forzar frecuencia trimestral
    df = df.asfreq('Q')
    last_date = df.index.max()
    
    if last_date < final_date:
        # 3) cuántos trimestres faltan
        # convierte años en trimestres y suma la diferencia de trimestres
        n_periods = ((final_date.year - last_date.year) * 4 +
                     (final_date.quarter - last_date.quarter))
        
        # 4) ajusto auto_arima
        stepwise = pm.auto_arima(df['ts'],
                                 start_p=1, start_q=1,
                                 max_p=3, max_q=3,
                                 seasonal=False,
                                 stepwise=True,
                                 suppress_warnings=True)
        fc = stepwise.predict(n_periods=n_periods)
        
        # 5) índice de fechas futuras (QuarterEnd por defecto)
        future_idx = pd.date_range(start=last_date,
                                   periods=n_periods + 1,
                                   freq='Q')[1:]
        
        # 6) concatenar pronóstico
        df_fc = pd.DataFrame({'ts': fc}, index=future_idx)
        df = pd.concat([df, df_fc])
        
        print(f"Pronosticados {n_periods} trimestres con ARIMA para '{var}'")
    
    df_multiple_list.append(df)


    
# Concantar los DataFrames de formato largo
df_multiple_list[-1]
multiple_data = pd.concat(df_multiple_list, axis = 1)
multiple_data.columns = vars_to_estimate

# Exportar en el mismo archivo excel
with pd.ExcelWriter(output_excel_path, engine = "openpyxl", mode = 'a', if_sheet_exists = 'replace') as writer:
    multiple_data.to_excel(writer, sheet_name = "multiple_macro")