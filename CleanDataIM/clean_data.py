# Librerías
import pandas as pd
import numpy as np

# Función para leer datos
class ProcessDataIM:
    """
    Clase para leer datos de diferentes fuentes y formatos
    """
    def __init__(self):
        pass

    # Función para leer datos de diferentes fuentes
    def clean_data(self,
                   init_data         : pd.DataFrame,
                   start_idx_row     : int, 
                   end_idx_row       : int, 
                   cols_num          : list,
                   col_names         : list,
                   time_agg          : str,
                   start_date        : None,  
                   end_date          : None,
                   ): 
        
        """
        Función que limpia los datos de un archivo excel y los prepara para su análisis
        
        Parámetros:
        -----------
        start_idx_row : int
            Índice de la primera fila de los datos
        end_idx_row : int   
            Índice de la última fila de los datos
        cols_num : list 
            Lista con los índices de las columnas a seleccionar
        time_agg : str  
            Periodo de agregación de los datos (D, M, 3M, 6M)
        start_date : str
            Fecha inicial de los datos
        end_date : str  
            Fecha final de los datos
        outliers_list : list, opcional
            Lista con las fechas de los datos anómalos
        replace_outliers : bool, opcional
            Indica si se reemplazan los datos anómalos por valores nulos
            
        Retorna:
        --------
        init_data : pd.DataFrame
            DataFrame con indice de tiempo 'date' y variable objetivo 'ts'
        """
        # Limpieza
        cols_num = [int(i) for i in cols_num]
        end_idx_row = int(end_idx_row)
        start_idx_row = int(start_idx_row)
        
        # Delimitación de los datos
        init_data = init_data.iloc[start_idx_row : end_idx_row, cols_num]
        init_data.columns = col_names
        init_data = init_data.reset_index(drop = True)
        
        # Valores nulos en la variable fecha
        if init_data['date'].isnull().sum() > 0:
            print(f'Se han detectado {init_data["date"].isnull().sum()} valores nulos en la variable de fechas. Los ínidices de las filas son: {init_data[init_data["date"].isnull()].index}')  
            
            # Quitar filas con valores nulos en la variable de fecha
            init_data = init_data.dropna(subset = ['date'])
            
        for i in init_data.columns[1:]:
            
            def detect_str(x):
                if type(x) == str:
                    return np.nan
                else:
                    return x
            
            init_data[i] = init_data[i].apply(lambda x: detect_str(x))
            
            if init_data[i].isnull().sum() > 0:
                print(f'Se han detectado {init_data[i].isnull().sum()} valores nulos en la variable {i}. Los ínidices de las filas son: {init_data[init_data[i].isnull()].index}')
            
            # Quitar filas con valores nulos en las variables numéricas
            init_data[i].dropna(inplace = True)
        
        # Adaptar diferentes formatos de fecha
        def date_format_sep(x):
            x = x[:4] + '-' + x[4:6] + '-' + x[6:8]
            return x

        def quarter_id(x):
            if 'T1' in x:
                return x.replace('T1', '03')
            elif 'T2' in x:
                return x.replace('T2', '06')
            elif 'T3' in x:
                return x.replace('T3', '09')
            else:
                return x.replace('T4', '12')
            
        
        if 'M' in str(init_data.loc[0, 'date']):
            init_data['date'] = init_data['date'].str.strip()
            init_data['date'] = init_data['date'].str.replace('M', '')
            init_data['date'] = init_data['date'] + '01'
            init_data['date'] = init_data['date'].apply(lambda x: date_format_sep(x))
        
        
        if 'T' in str(init_data.loc[0, 'date']):
            init_data['date'] = init_data['date'].str.strip()
            init_data['date'] = init_data['date'].apply(lambda x: quarter_id(x))
            init_data['date'] = init_data['date'] + '01'
            init_data['date'] = init_data['date'].apply(lambda x: date_format_sep(x))
        
        # Formatos
        init_data['date'] = pd.to_datetime(init_data['date'], errors = 'raise')
        for column in init_data.columns[1: ]:
            init_data[column] = init_data[column].astype('float64', errors = 'raise')
        
        # Imprimir el horizonte temporal
        print(f'Horizonte temporal: {init_data["date"].min()} - {init_data["date"].max()}')
        
        if init_data['date'].max() < pd.to_datetime(end_date):
            print('*' * 140)
            print('La fecha final es mayor que la fecha máxima de los datos')
            print('*' * 140)
        
        # Ordenar datos
        init_data = init_data.sort_values('date')
        
        # Detectar si los los datos están en formato diario, mensual o trimestral
        date_diff = init_data['date'].diff().dropna()
        date_diff = date_diff.value_counts().idxmax()

        if date_diff.days < 5:
            print('los datos iniciales son diarios')
        elif date_diff.days > 25 and date_diff.days < 35:
            print('los datos iniciales son mensuales')
        elif date_diff.days > 85 and date_diff.days < 95:
            print('los datos iniciales son trimestrales')
        elif date_diff.days > 170 and date_diff.days < 190:
            print('los datos iniciales son semestrales')
        elif date_diff.days > 350 and date_diff.days < 370:
            print('los datos iniciales son anuales') 
        else:
            print('No se ha podido detectar el periodo temporal')

        # Garantizar que la primera fecha sea dicimebre o junio y filtrar el dataframe
        if init_data['date'].iloc[0].month < 12: # TODO: Tener en cuenta este ajuste
            init_data = init_data[init_data['date'] >= pd.to_datetime(f"{init_data['date'].iloc[0].year}-12-01")]
        
        # Agregación de los datos
        if time_agg == 'D':
            print('No se requiere agregación de los datos')
        elif time_agg == 'M':
            init_data = init_data.resample('M', on = 'date').last().reset_index()
            print('datos agregados a nivel mensual (Agg: Last Value)')
        elif time_agg == '3M':
            init_data = init_data.resample('3M', on = 'date').last().reset_index()
            print('datos agregados a nivel trimestral (Agg: Last Value)')
        elif time_agg == '6M':
            init_data = init_data.resample('6M', on = 'date').last().reset_index()
            print('datos agregados a nivel semestral (Agg: Last Value)')
        elif time_agg not in ['D', 'M', '3M', '6M']:
            print('Datos transformados sin agregación')
        else:
            print('No se ha podido realizar la agregación de los datos')
            
        # Valores nulos posterior a su agregación
        if init_data['date'].isnull().sum() > 0:
            print('Se han detectado valores nulos (posterior a su agregación) en la variable date')
            return None
        
        # Reemplazar valores nulos y atípicos
        for column in init_data.columns[1:]:
            # Verificar valores nulos
            if init_data[column].isnull().sum() > 0:
                print('*' * 140)
                print(f'¡¡¡Se han detectado {init_data[column].isnull().sum()} valores nulos en la variable {column}. Método de relleno: Linear Interpolation!!!')
                init_data[column] = init_data[column].interpolate(method='linear')
                print('*' * 140)
        
        # Filtrar los datos por fecha
        init_data['date'] = pd.to_datetime(init_data['date'])
        
        if start_date != None:
            init_data = init_data[init_data['date'] >= start_date]
            
        if end_date != None:
            init_data = init_data[init_data['date'] <= end_date]
        
        # Pasar la variable de fecha como índice
        init_data.index = init_data['date']
        init_data = init_data.drop(columns = ['date'])
        
        
        return(init_data)
