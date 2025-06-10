# Manejo y procesamiento de datos
import pandas as pd
import numpy as np

# Visualización de datos
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Optimización y utilidades
import itertools
from tqdm import tqdm

# Desarrollo web y UI
import streamlit as st
from streamlit_option_menu import option_menu

import io

# Configuración de la página
st.set_page_config(page_title = "Forecast de Series Temporales", layout = "wide")

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html = True
)


# CSS personalizado
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    :root {
        --primary-color: #1F4E79;
        --secondary-color: #E6F2FF;
        --background-color: #F0F8FF;
        --accent-color: #4A90E2;
    }
    body {
        font-family: 'Roboto', sans-serif;
    }
    /* Barra superior con gradiente y animación */
    .top-bar {
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
        width: 100%;
        height: 60px;
        font-size: 54px;
        display: flex;
        align-items: center;
        padding-left: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: background 0.3s ease;
    }
    .top-bar:hover {
        background: var(--primary-color);
    }
    .top-bar h2 {
        color: #fff;
        margin: 0;
        letter-spacing: 2px;
    }
    /* Fondo general */
    .reportview-container {
        background: var(--background-color);
    }
    /* Personalización de la barra lateral */
    .css-1d391kg { 
        background-color: var(--secondary-color);
        border-right: 2px solid var(--primary-color);
    }
    /* Títulos y subtítulos */
    h1, h2, h3, h4, h5, h6 {
        color: var(--primary-color);
        letter-spacing: 1px;
    }
    /* Estilo para el texto centrado */
    .centered-title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: var(--primary-color);
        margin-top: 20px;
        margin-bottom: 20px;
    }
    /* Separador de secciones con gradiente */
    .section-divider {
        border: none;
        height: 2px;
        background: linear-gradient(to right, var(--primary-color), var(--accent-color));
        margin: 30px 0;
    }
    /* Estilo para tarjetas (cards) para resaltar información */
    .card {
        background: #fff;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        padding: 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.15);
    }
    /* Pie de página (footer) */
    .footer {
        text-align: center;
        font-size: 14px;
        color: #777;
        margin-top: 60px;
        padding-top: 20px;
        border-top: 1px solid #ccc;
    }
    </style>
    """, unsafe_allow_html = True)

# Barra superior (top bar)
st.markdown("""
    <div class="top-bar">
        <h2>MARKET DATA FORECASTING</h2>
    </div>
    """, unsafe_allow_html = True
    )

# Barra lateral
try:
    with st.sidebar:
        st.image("images/logoim.png", use_container_width = True)
        st.markdown('<div class="centered-title">INTERMONEY CONSULTING</div>', unsafe_allow_html = True)
        st.markdown("---")
except:
    with st.sidebar:
        st.image("images/logoim.png", use_column_width = True)
        st.markdown('<div class="centered-title">INTERMONEY CONSULTING</div>', unsafe_allow_html = True)
        st.markdown("---")
    

# Opciones del menú
menu_options = ["Macroeconomical Variables", "Interest Rate Curves", "Communities", "Delivery"]
icons = ["globe", "graph-up", "list-task", "download"]


# CSS personalizado para mejorar el estilo de la barra lateral
st.markdown(
    """
    <style>
    /* Estilo general del sidebar */
    .css-1d391kg {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
    }
    /* Títulos y textos dentro de la sidebar */
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg h4, .css-1d391kg h5, .css-1d391kg h6,
    .css-1d391kg, .css-1uccc91 {
        color: #333333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* Separador personalizado */
    .sidebar-separator {
        margin: 15px 0;
        border-top: 1px solid #cccccc;
    }
    </style>
    """, unsafe_allow_html = True
)

with st.sidebar:
    selected = option_menu("Menu", menu_options, icons = icons, menu_icon = "cast", default_index = 0)
    
    if selected == "Macroeconomical Variables":
        dict_data = pd.read_excel('dict_variables.xlsx')
        list_variables = dict_data[(dict_data['mark'] == 1) & (dict_data['methodology'] == 'macro')].reset_index(drop = True)['var_name'].values
        
        st.markdown('<div class="sidebar-separator"></div>', unsafe_allow_html = True)
        select_variable = st.selectbox("Macroeconomic Variable:", list_variables)
        
        st.markdown('<div class="sidebar-separator"></div>', unsafe_allow_html = True)
        
    elif selected == "Interest Rate Curves":
        dict_data = pd.read_excel('dict_variables.xlsx')
        list_variables_curves = dict_data[(dict_data['mark'] == 1) & (dict_data['methodology'] == 'ir')].reset_index(drop = True)['var_name'].values
            
        st.markdown('<div class="sidebar-separator"></div>', unsafe_allow_html = True)
        select_variable_curves = st.selectbox("Interest Rate Curve:", list_variables_curves)
    
    elif selected == "Communities":
        st.markdown('<div class="sidebar-separator"></div>', unsafe_allow_html = True)
        select_variable_com = st.selectbox("Community Variable:", ['PIB', 'OCUP', 'PACT', 'PARO', 'VIV_VENT'])
        
        st.markdown('<div class="sidebar-separator"></div>', unsafe_allow_html = True)
    
    else:
        pass
        
        
# Ajuste de margen para evitar solapamiento con la barra superior
st.markdown("<br><br>", unsafe_allow_html = True)

# =============================================================================
# LECTURA DE DATOS
# ============================================================================= 
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

list_files = {}

if selected == "Macroeconomical Variables":
    var_name = select_variable
else:
    pass

# =============================================================================
# GRÁFICO TIME SERIES SPLIT
# =============================================================================  
def plot_time_series_split(df                 : pd.DataFrame, 
                           var_name           : str, 
                           initial_train_date : str, 
                           initial_val_date   : str
                           ):
    
    """
    Representa la evolución de la serie temporal, indicando la fecha 
    a partir de la cual se establece el conjunto de validación.

    Parámetros:
    ----------- 
    df : DataFrame
        DataFrame con la serie temporal, debe tener un índice de tipo fecha y una columna 'ts'.
    var_name : str
        Nombre de la variable para el título del gráfico.
    initial_train_date : str
        Fecha (YYYY-MM-DD) donde comienza el conjunto de entrenamiento.
    initial_val_date : str
        Fecha (YYYY-MM-DD) donde comienza el conjunto de validación.
    initial_test_date : str
        Fecha (YYYY-MM-DD) donde comienza el conjunto de prueba (opcional, no usado en este gráfico).
        
    Retorna:
    --------
    fig : plotly.graph_objects.Figure
    """
    # Calcular los índices de inicio de validación y prueba
    initial_train_size_plot = df[df.index <= initial_train_date].shape[0]
    initial_val_size_plot = df[df.index <= initial_val_date].shape[0]
    
    # Mensajes en consola para información adicional
    print('=' * 100)
    print(f'La fecha de inicio del conjunto de validación es {initial_train_date}')
    if initial_val_size_plot < df.shape[0]:
        print(f'La fecha de inicio del conjunto de prueba es {initial_val_date}')
    print('=' * 100)
    
    # Fechas clave para las líneas verticales
    highlight_date = df.iloc[initial_train_size_plot - 1].name
    highlight_date_val = df.iloc[initial_val_size_plot - 1].name
    
    fig = go.Figure()

    # Serie temporal
    fig.add_trace(go.Scatter(
        x    = df.index,
        y    = df['ts'],
        mode = 'lines',
        name = 'Time Series',
        line = dict(color = '#1f77b4', width = 2.5)
    ))

    # Línea vertical para inicio de validación
    fig.add_trace(go.Scatter(
        x    = [highlight_date, highlight_date],
        y    = [df['ts'].min(), df['ts'].max()],
        mode = 'lines',
        name = 'Validation beginning',
        line = dict(color = '#006400', dash = 'dash', width = 1.5)
    ))

    # Línea vertical para inicio de prueba
    fig.add_trace(go.Scatter(
        x    = [highlight_date_val, highlight_date_val],
        y    = [df['ts'].min(), df['ts'].max()],
        mode = 'lines',
        name = 'Test beginning',
        line = dict(color = '#8B0000', dash = 'dash', width = 1.5)
    ))

    # Áreas sombreadas
    fig.add_trace(go.Scatter(
        x         = [highlight_date, highlight_date_val, highlight_date_val, highlight_date],
        y         = [df['ts'].min(), df['ts'].min(), df['ts'].max(), df['ts'].max()],
        fill      = 'toself',
        fillcolor = 'rgba(184, 250, 198, 0.3)',
        line      = dict(width = 0),
        name      = 'Validation Set'
    ))

    fig.add_trace(go.Scatter(
        x         = [highlight_date_val, df.index[-1], df.index[-1], highlight_date_val],
        y         = [df['ts'].min(), df['ts'].min(), df['ts'].max(), df['ts'].max()],
        fill      = 'toself',
        fillcolor = 'rgba(255, 204, 203, 0.3)',
        line      = dict(width = 0),
        name      = 'Test Set'
    ))

    # Configuración del gráfico
    fig.update_layout(
        title       = f'{var_name}',
        title_font  = dict(size=20, color='black'),
        xaxis_title = '',
        yaxis_title = '',
        legend      = dict(
            orientation = "h",  # Leyenda horizontal
            yanchor     = "bottom",
            y           = 1.1,
            xanchor     = "center",
            x           = 0.5,
            font        = dict(size = 12)
        ),
        template = 'plotly_white',
        margin   = dict(l = 40, r = 40, t = 80, b = 40)
    )
    
    return fig


# =============================================================================
# GRÁFICO BACKTESTING
# ============================================================================= 
def plot_backtesting_plotly(
        df_fitted      : pd.DataFrame,
        df_validation  : pd.DataFrame,
        df_test        : pd.DataFrame,
        date_init_train: str,
        date_init_val  : str,
        show_legend: bool = True
    ):
    """
    Dibuja las series de entrenamiento, validación y prueba en una sola figura Plotly.

    Parameters
    ----------
    df_fitted : DataFrame
        Debe contener columnas 'observed' y 'predicted' del tramo de entrenamiento.
    df_validation : DataFrame
        Idem, para el tramo de validación.
    df_test : DataFrame
        Idem, para el tramo de prueba (puede estar vacío o tener un solo punto).
    date_init_train : str
        Fecha (YYYY-MM-DD) donde comienza el entrenamiento.
    date_init_val : str
        Fecha (YYYY-MM-DD) donde comienza la validación.
    show_legend : bool, default True
        Mostrar u ocultar las entradas de leyenda.
    """
    markersize = 8
    fig = go.Figure()
    
    # Si los dataframes tienen duplicados en la fecha, quedarse con el último valor
    # NOTE: Cambio cuando en el proceso de optimización hay solapamiento
    df_fitted = df_fitted[~df_fitted.index.duplicated(keep = 'last')]
    df_validation = df_validation[~df_validation.index.duplicated(keep = 'last')]
    df_test = df_test[~df_test.index.duplicated(keep = 'last')]
    
    # Historico­­­­
    fig.add_trace(go.Scatter(
        x=df_fitted.index,
        y=df_fitted['observed'],
        mode='lines+markers',
        name='Historical',
        line=dict(color='#1f77b4', width=2.5),
        marker=dict(symbol='circle', size=markersize),
        showlegend=show_legend
    ))

    fig.add_trace(go.Scatter(
        x=df_fitted.index,
        y=df_fitted['predicted'],
        mode='lines+markers',
        name='Adjusted',
        line=dict(color='#aec7e8', width=2, dash='dash'),
        marker=dict(symbol='square', size=markersize),
        showlegend=show_legend
    ))

    # Línea de corte entrenamiento
    fig.add_vline(x=pd.to_datetime(date_init_train),
                  line=dict(color='#006400', width=1.5, dash='dash'))

    # Validación
    fig.add_trace(go.Scatter(
        x=df_validation.index,
        y=df_validation['observed'],
        mode='lines+markers',
        name='',
        line=dict(color='#1f77b4', width=2.5),
        marker=dict(symbol='circle', size=markersize),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=df_validation.index,
        y=df_validation['predicted'],
        mode='lines+markers',
        name='Validation Forecasts',
        line=dict(color='#228B22', width=2, dash='dash'),
        marker=dict(symbol='square', size=markersize),
        showlegend=show_legend
    ))

    fig.add_vline(x=pd.to_datetime(date_init_val),
                  line=dict(color='#8B0000', width=1.5, dash='dash'))

    # Área sombreada validación
    fig.add_trace(go.Scatter(
        x=list(df_validation.index) + list(df_validation.index)[::-1],
        y=list(df_validation['observed']) + list(df_validation['predicted'])[::-1],
        fill='toself',
        fillcolor='#b8fac6',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ))

    # Prueba
    if df_test.shape[0] > 1:
        fig.add_trace(go.Scatter(
            x=df_test.index,
            y=df_test['observed'],
            mode='lines+markers',
            name='',
            line=dict(color='#1f77b4', width=2.5),
            marker=dict(symbol='circle', size=markersize),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=df_test.index,
            y=df_test['predicted'],
            mode='lines+markers',
            name='Test Forecasts',
            line=dict(color='#B22222', width=2, dash='dash'),
            marker=dict(symbol='square', size=markersize),
            showlegend=show_legend
        ))

        # Área sombreada prueba
        fig.add_trace(go.Scatter(
            x=list(df_test.index) + list(df_test.index)[::-1],
            y=list(df_test['observed']) + list(df_test['predicted'])[::-1],
            fill='toself',
            fillcolor='#fac9c9',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ))

    # Ajustes globales
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(automargin=True)
    fig.update_layout(
        hovermode='x unified',
        legend=dict(
            orientation = "h",
            x = 0, 
            y = 1.02,
            xanchor = "left",
            yanchor = "bottom"
        ),
        margin=dict(l=40, r=20, t=40, b=60)
    )
    
    return fig



# =============================================================================
# GRÁFICO FANCHART PLOTLY
# =============================================================================  
def plot_fanchart_plotly(data               : pd.DataFrame, 
                         percentile_df      : pd.DataFrame,
                         fitted_values      : pd.Series, 
                         len_hist_plot      : int,
                         percentile_list    = ['P1', 'P20', 'P40', 'P60', 'P80', 'P99'],
                         title              = "Fanchart",
                         last_value_text    = False,
                         color_theme        = "blue",
                         final_plot         = False
                         ):
    """
    Dibuja un gráfico de tipo fanchart utilizando Plotly, mostrando la serie temporal histórica,
    los valores ajustados, el pronóstico y las bandas de percentiles.
    
    Parámetros:
    ----------
    data : pd.DataFrame
        DataFrame que contiene la serie temporal histórica. Debe tener un índice de tipo fecha y una columna 'ts'.
    percentile_df : pd.DataFrame
        DataFrame que contiene los percentiles del pronóstico. Debe tener un índice de tipo fecha y una columna 'forecast'.
    fitted_values : pd.Series
        Serie de valores ajustados. Debe tener un índice de tipo fecha.
    len_hist_plot : int
        Número de puntos históricos a mostrar en el gráfico.
    percentile_list : list, opcional
        Lista de nombres de percentiles a mostrar en las bandas. Por defecto es ['P1', 'P20', 'P40', 'P60', 'P80', 'P99'].
    title : str, opcional
        Título del gráfico. Por defecto es "Fanchart".
    last_value_text : bool, opcional
        Si es True, muestra el valor del último pronóstico como anotación en el gráfico. Por defecto es False.
    color_theme : str, opcional
        Tema de color para el gráfico. Puede ser "purple" o "blue". Por defecto es "blue".
    final_plot : bool, opcional
        Si es True, no muestra las bandas P20-P80 y P40-P60. Por defecto es False.
    """

    # Definir la paleta de colores según el tema elegido
    if color_theme == "purple":
        hist_line_color      = '#430d4d'
        fitted_line_color    = '#6a51a3'
        forecast_line_color  = '#6a51a3'
        annotation_font_color= '#6a51a3'
        band1_fill           = 'rgba(136,86,167, 0.3)'
        band2_fill           = 'rgba(140,150,198, 0.5)'
        band3_fill           = 'rgba(140,107,177, 0.7)'
    else:
        hist_line_color      = '#1f77b4'
        fitted_line_color    = '#006400'
        forecast_line_color  = '#006400'
        annotation_font_color= '#006400'
        band1_fill           = 'rgba(199, 212, 229, 0.3)'
        band2_fill           = 'rgba(166,189,219, 0.5)'
        band3_fill           = 'rgba(116,169,207, 0.7)'

    fig = go.Figure()
    
    # Datos Históricos
    hist_data = data.tail(len_hist_plot)
    fig.add_trace(go.Scatter(
        x    = hist_data.index,
        y    = hist_data['ts'],
        mode = 'lines',
        name = 'Historical',
        legendgroup = 'historicos',
        showlegend = True,
        line = dict(color = hist_line_color, width = 1.8)
    ))
    
    # Datos Ajustados
    if fitted_values is not None:
        fitted = fitted_values.tail(len_hist_plot)
        fig.add_trace(go.Scatter(
            x    = fitted.index,
            y    = fitted,
            mode = 'lines',
            name = 'Adjusted',
            legendgroup = 'ajustados',
            showlegend = True,
            line = dict(color = fitted_line_color, width = 1, dash = 'dash')
        ))
    
    # Pronóstico
    fig.add_trace(go.Scatter(
        x        = percentile_df.index,
        y        = percentile_df['forecast'],
        mode     = 'lines+markers',
        name     = 'Forecast',
        legendgroup = 'pronostico',
        showlegend = True,
        line     = dict(color = forecast_line_color, width = 2, dash = 'dash'),
        marker   = dict(size = 5)
    ))
    
    # Anotación para el último valor del pronóstico
    if last_value_text:
        last_forecast_index = percentile_df.index[-1]
        last_forecast_value = percentile_df['forecast'].iloc[-1]
        fig.add_annotation(
            x         = last_forecast_index,
            y         = last_forecast_value,
            text      = f"{last_forecast_value:.2f}",
            showarrow = True,
            arrowhead = 1,
            ax        = 0,
            ay        = -20,
            font      = dict(color = annotation_font_color, size = 9),
            bgcolor   = "lightgray"
        )
    
    # Banda P1-P99
    fig.add_trace(go.Scatter(
        x          = list(percentile_df.index) + list(percentile_df.index)[::-1],
        y          = list(percentile_df[percentile_list[0]]) + list(percentile_df[percentile_list[-1]])[::-1],
        fill       = 'toself',
        fillcolor  = band1_fill,
        line       = dict(color = 'rgba(255,255,255,0)'),
        hoverinfo  = "skip",
        showlegend = True,
        name       = f"{percentile_list[0]}-{percentile_list[-1]}",
        legendgroup = 'p1p99'
    ))
    
    if not final_plot:
        # Banda P20-P80
        fig.add_trace(go.Scatter(
            x          = list(percentile_df.index) + list(percentile_df.index)[::-1],
            y          = list(percentile_df[percentile_list[1]]) + list(percentile_df[percentile_list[-2]])[::-1],
            fill       = 'toself',
            fillcolor  = band2_fill,
            line       = dict(color = 'rgba(255,255,255,0)'),
            hoverinfo  = "skip",
            showlegend = True,
            name       = f"{percentile_list[1]}-{percentile_list[-2]}",
            legendgroup = 'p20p80'
        ))
        
        # Banda P40-P60
        fig.add_trace(go.Scatter(
            x          = list(percentile_df.index) + list(percentile_df.index)[::-1],
            y          = list(percentile_df[percentile_list[2]]) + list(percentile_df[percentile_list[-3]])[::-1],
            fill       = 'toself',
            fillcolor  = band3_fill,
            line       = dict(color = 'rgba(255,255,255,0)'),
            hoverinfo  = "skip",
            showlegend = True,
            name       = f"{percentile_list[2]}-{percentile_list[-3]}",
            legendgroup = 'p40p60'
        ))
    
    # Configuración del layout (leyenda horizontal)
    fig.update_layout(
        title = title,
        xaxis = dict(
            title     = '',
            tickangle = 45,
            showgrid  = True,
            gridcolor = 'gray'
        ),
        yaxis = dict(
            title =''
        ), 
        legend = dict(
            orientation = "h",
            x = 0, 
            y = 1.02,
            xanchor = "left",
            yanchor = "bottom"
        ),
        template = 'plotly_white'
    )
    
    return fig


# =============================================================================
# GRÁFICO DE BARRAS
# =============================================================================  
def bar_plot_y(df, model_name, color="blue", number_format="percent", decimals=2):
    """
    Crea un gráfico de barras para visualizar los valores de una serie temporal.

    Parámetros:
    ----------
    df : pd.DataFrame
        DataFrame que contiene la serie temporal. Debe tener un índice de tipo fecha y una columna 'ts'.
    model_name : str
        Nombre del modelo para el título del gráfico.
    color : str, opcional
        Color del gráfico. Puede ser "blue" o "purple". Por defecto es "blue".
    number_format : str, opcional
        Formato de los valores: "percent" para porcentaje, "decimal" para número normal. Por defecto es "percent".
    decimals : int, opcional
        Número de decimales a mostrar. Por defecto es 2.

    Retorna:
    -------
    fig : plotly.graph_objects.Figure
        Figura de Plotly con el gráfico de barras.
    """
    # Definir colores y estilos
    if color == "blue":
        bar_colors = ['#1f77b4'] * (len(df) - 4) + ['#87CEFA'] * 4
    else:
        bar_colors = ['#7b3f87'] * (len(df) - 4) + ['#eab5f4'] * 4

    # Seleccionar función de formateo
    if number_format == "percent":
        fmt = lambda v: f"{v:.{decimals}%}"
        yaxis_title = "Cambio porcentual"
    else:
        fmt = lambda v: f"{v:.{decimals}f}"
        yaxis_title = "Valor"

    # Crear gráfico de barras
    fig = go.Figure()
    for i, (date, value) in enumerate(zip(df.index, df['ts'])):
        trace_props = dict(
            x=[date],
            y=[value],
            name=model_name,
            text=[fmt(value)],
            textposition='outside'
        )
        if i >= len(df) - 4: # NOTE: Número de años a destacar.
            trace_props['marker'] = dict(color=bar_colors[i], line=dict(color='black', width=1.5))
        else:
            trace_props['marker'] = dict(color=bar_colors[i])

        fig.add_trace(go.Bar(**trace_props))

    # Mejorar estética
    fig.update_layout(
        title=f"Forecast {model_name}",
        xaxis_title="Año",
        yaxis_title=yaxis_title,
        template="plotly_white",
        showlegend=False
    )

    return fig

# =============================================================================
# GRÁFICO DE BARRAS CON INTERVALOS DE CONFIANZA
# =============================================================================  
def bar_plot_y_conf(df, model_name, color="blue", number_format="percent", decimals=2):
    """
    Crea un gráfico de barras para visualizar los valores de una serie temporal con intervalos de confianza.

    Parámetros:
    ----------
    df : pd.DataFrame
        DataFrame que contiene la serie temporal. Debe tener un índice de tipo fecha y una columna 'ts'.
    model_name : str
        Nombre del modelo para el título del gráfico.
    color : str, opcional
        Color del gráfico. Puede ser "blue" o "purple". Por defecto es "blue".
    number_format : str, opcional
        Formato de los valores: "percent" para porcentaje, "decimal" para número normal. Por defecto es "percent".
    decimals : int, opcional
        Número de decimales a mostrar. Por defecto es 2.

    Retorna:
    -------
    fig : plotly.graph_objects.Figure
        Figura de Plotly con el gráfico de barras y puntos.
    """
    # Definir colores base
    if color == "blue":
        base_colors = ['#1f77b4'] * (len(df) - 4) + ['#87CEFA'] * 4
    else:
        base_colors = ['#7b3f87'] * (len(df) - 4) + ['#eab5f4'] * 4

    # Formato de números
    if number_format == "percent":
        fmt = lambda v: f"{v:.{decimals}%}"
        yaxis_title = "Cambio porcentual"
    else:
        fmt = lambda v: f"{v:.{decimals}f}"
        yaxis_title = "Valor"

    fig = go.Figure()
    grouped = df.groupby(df.index)['ts'].apply(list)

    for i, (date, values) in enumerate(grouped.items()):
        main_value = values[0]
        # Color de barra
        bar_color = base_colors[i % len(base_colors)]
        # Si es la tonalidad clara de púrpura, siempre borde negro
        if bar_color == '#eab5f4':
            line_opts = dict(color='black', width=1.5)
        else:
            line_opts = None

        # Agregar la barra principal
        fig.add_trace(go.Bar(
            x=[date], y=[main_value], name=model_name,
            marker=dict(color=bar_color,
                        line=line_opts) if line_opts else dict(color=bar_color),
            text=[fmt(main_value)], textposition='outside'
        ))

        # Puntos de intervalo de confianza
        if len(values) > 1:
            scatter_x = [date] * (len(values) - 1)
            scatter_y = values[1:]
            fig.add_trace(go.Scatter(
                x=scatter_x, y=scatter_y,
                mode='markers',
                marker=dict(color='rgba(0,0,0,0.3)', size=8),
                line=dict(color='black', width=1.5),
                showlegend=False
            ))

    # Layout
    fig.update_layout(
        title=f"Forecast Anual: {model_name}",
        xaxis_title="Año",
        yaxis_title=yaxis_title,
        template="plotly_white",
        showlegend=False
    )

    return fig



# =============================================================================
# APLICACIÓN DE LOS GRÁFICOS
# =============================================================================  
try:

    for sheet in sheet_names:
        list_files[sheet] = pd.read_excel(f'evidence_macro/evidence_{var_name}.xlsx', sheet_name = sheet, index_col = 0)


    # =============================================================================
    # GRÁFICOS TIME SERIES SPLIT
    # =============================================================================    
    fig_init_ts = plot_time_series_split(df                 = list_files['df_abs'], 
                                        var_name           = '', 
                                        initial_train_date = list_files['params_df'].loc[0, 'date_init_train'], 
                                        initial_val_date   = list_files['params_df'].loc[0, 'date_init_val']
                                        )
        
        
    # =============================================================================
    # GRÁFICOS BACKTESTING
    # =============================================================================
    models_info_backtest = [
        (list_files['fitted_hw'],    list_files['backtest_hw'],    list_files['os_backtest_hw'],    "Holt-Winters"),
        (list_files['fitted_sarima'], list_files['backtest_sarima'], list_files['os_backtest_sarima'], "SARIMA"),
        (list_files['fitted_prophet'],list_files['backtest_prophet'],list_files['os_backtest_prophet'], "Prophet"),
        (list_files['fitted_ucm'],    list_files['backtest_ucm'],    list_files['os_backtest_ucm'],    "UCM"),
        (list_files['fitted_ardl'],   list_files['backtest_ardl'],   list_files['os_backtest_ardl'],   "ARDL"),
        (list_files['fitted_var'],    list_files['backtest_var'],    list_files['os_backtest_var'],    "VAR")
    ]

    backtest_list = []

    for fitted_df, validation_df, test_df, model_name in models_info_backtest:
        # Crear gráfico de backtesting
        fig = plot_backtesting_plotly(
            df_fitted       = fitted_df[12:],
            df_validation   = validation_df,
            df_test         = test_df,
            date_init_train = list_files['params_df'].loc[0, 'date_init_train'],
            date_init_val   = list_files['params_df'].loc[0, 'date_init_val'],
            show_legend     = True
        )
        
        # Añadir a la lista
        backtest_list.append(fig)

    # =============================================================================
    # MÉTRICAS DE ERROR
    # =============================================================================      
    # Variables definidas
    colors = ['#4472C4', '#A9D8F5']
    hatches = ['/', '']
    background_color = '#ffffff'
    type_mapping = {'validation': 'Validation', 'test': 'Test'}
    bar_width = 0.35
    metrics = ['MAE', 'MSE', 'RMSE', 'MAPE']

    # Obtener los nombres de modelos y asegurar un orden consistente
    df = list_files['error_metrics_df']
    modelos = df['MODELO'].unique()
    x = list(modelos)

    fig_error_metrics = make_subplots(
        rows = 2, cols = 2,
        subplot_titles = [f"{metric} {var_name}" for metric in metrics]
    )

    tipos = df['TYPE'].unique()

    # Para controlar que la leyenda se muestre solo una vez por cada tipo
    legend_added = set()

    for i, metric in enumerate(metrics):
        row = i // 2 + 1
        col = i % 2 + 1

        for j, model_type in enumerate(tipos):
            # Filtrar el DataFrame según el tipo
            subset = df[df['TYPE'] == model_type]
            
            # Crear un diccionario para mapear cada modelo con su valor para la métrica
            data = {modelo: val for modelo, val in zip(subset['MODELO'], subset[metric])}
            
            # Asegurar el mismo orden de modelos, dejando None si falta algún dato
            y = [data.get(modelo, None) for modelo in x]
            
            # Controlar la visualización de la leyenda: sólo la primera traza de cada tipo
            show_legend = False if model_type in legend_added else True
            legend_added.add(model_type)
            
            # Ajustar el ancho del borde para el color azul
            line_width = 0.5 if colors[j % len(colors)] == '#4472C4' else 1.0
            
            fig_error_metrics.add_trace(
                go.Bar(
                    x = x,
                    y = y,
                    name = type_mapping.get(model_type, model_type),
                    showlegend = show_legend,
                    marker = dict(
                        color = colors[j % len(colors)],
                        pattern = dict(
                            shape = hatches[j % len(hatches)]
                        ),
                        line = dict(width = line_width)
                    ),
                    opacity = 0.9
                ),
                row = row, col = col
            )
        
        # Configurar etiquetas y cuadrícula para cada subgráfico
        fig_error_metrics.update_xaxes(
            tickangle = 45,
            tickfont  = dict(size = 12),
            row       = row, 
            col       = col
        )
        
        fig_error_metrics.update_yaxes(
            tickfont  = dict(size = 12),
            gridcolor = 'lightgrey', gridwidth = 0.5,
            row       = row, 
            col       = col
        )

    fig_error_metrics.update_layout(
        barmode       = 'group',
        plot_bgcolor  = background_color,
        paper_bgcolor = background_color,
        legend        = dict(font = dict(size = 12)),
        margin        = dict(t = 50, b = 50, l = 50, r = 50)
    )


    # =============================================================================
    # GRÁFICOS PRONÓSTICOS EN VALOR ABSOLUTO
    # =============================================================================
    models_info_forecast = [
        (list_files['hw_abs_q'],     "Holt-Winters"),
        (list_files['sarima_abs_q'], "SARIMA"),
        (list_files['prophet_abs_q'],"Prophet"),
        (list_files['ucm_abs_q'],    "UCM"),
        (list_files['ardl_abs_q'],   "ARDL"),
        (list_files['var_abs_q'],    "VAR")
    ]

    abs_forecast_list = []

    for percentile_df, model_name in models_info_forecast:
        # Generamos el fanchart en figura temporal
        fig = plot_fanchart_plotly(
            data            = list_files['df_abs'],
            percentile_df   = percentile_df,
            fitted_values   = None,
            len_hist_plot   = 30,
            title           = f"Forecast ({model_name}) - {var_name}",
            last_value_text = False
        )

        # Añadimos a la lista
        abs_forecast_list.append(fig)


    # =============================================================================
    # GRÁFICOS PRONÓSTICOS EN VARIACIÓN PORCENTUAL
    # =============================================================================
    models_info_forecast_pct = [
        (list_files["hw_pct_q"],   "Holt-Winters"),
        (list_files["sarima_pct_q"],"SARIMA"),
        (list_files["prophet_pct_q"],"Prophet"),
        (list_files["ucm_pct_q"],  "UCM"),
        (list_files["ardl_pct_q"], "ARDL"),
        (list_files["var_pct_q"],  "VAR")
    ]

    pct_forecast_list = []

    for percentile_df, model_name in models_info_forecast_pct:
        # Generamos el fanchart en figura temporal
        fig = plot_fanchart_plotly(
            data            = list_files['df_pct'],
            percentile_df   = percentile_df,
            fitted_values   = None,
            len_hist_plot   = 30,
            title           = f"Forecast ({model_name}) - {var_name}",
            last_value_text = False
        )

        # Añadimos a la lista
        pct_forecast_list.append(fig)
        

    # =============================================================================
    # GRÁFICOS PRONÓSTICOS EN VARIACIÓN INTERANUAL TRIMESTRAL
    # =============================================================================
    models_info_forecast_pct_y = [
        (list_files["hw_interannual"],      "Holt-Winters"),
        (list_files["sarima_interannual"],  "SARIMA"),
        (list_files["prophet_interannual"], "Prophet"),
        (list_files["ucm_interannual"],     "UCM"),
        (list_files["ardl_interannual"],    "ARDL"),
        (list_files["var_interannual"],     "VAR")
    ]   

    interyear_forecast_list = []

    for percentile_df, model_name in models_info_forecast_pct_y:
        # Generamos el fanchart en figura temporal
        fig = plot_fanchart_plotly(
            data            = list_files['df_interannual'],
            percentile_df   = percentile_df,
            fitted_values   = None,
            len_hist_plot   = 30,
            title           = f"Forecast ({model_name}) - {var_name}",
            last_value_text = False,
            color_theme     = "purple",
        )

        # Añadimos a la lista
        interyear_forecast_list.append(fig)
        

    # =============================================================================
    # GRÁFICOS DE BARRAS PARA PRONÓSTICOS ANUALES
    # =============================================================================
    # Configuración de los modelos
    models = ["hw", "sarima", "prophet", "ucm", "ardl", "var"]

    titles = {
        "hw": "Holt-Winters",
        "sarima": "SARIMA",
        "prophet": "Prophet",
        "ucm": "UCM",
        "ardl": "ARDL",
        "var": "VAR",
    }
    # Para el segundo bloque, todos usan color "Greens"
    colors = {m: "Greens" for m in models}

    # Generar y graficar
    bar_plots_abs = {}
    bar_plots_pct_y = {}
    bar_plots_pct_y_avg = {}
    
    
    for m in models:
        # 0) Pronóstico anual (%): concat base_pct + forecast del modelo
        df_abs_y = pd.concat([
            list_files["df_abs_y"].tail(8),
            list_files[f"{m}_abs_y"][["forecast"]]
            .rename(columns = {"forecast": "ts"})
        ], axis=0)
        
        bar_plots_abs[m] = bar_plot_y(df_abs_y.copy(), titles[m], number_format = 'absolute', decimals = 2)
        
        # 2) Pronóstico anual con promedio interanual: concat base_interyear + forecast
        df_pct_y_last = pd.concat([
            list_files["df_last_year"].tail(8),
            list_files[f"{m}_interannual_last"][["forecast"]]
            .rename(columns = {"forecast": "ts"})
        ], axis=0)
        
        
        if list_files[f"{m}_interannual_last"][["forecast"]].max().values[0] > 1 or list_files[f"{m}_interannual_last"][["forecast"]].min().values[0] < -1:
            bar_plots_pct_y[m] = bar_plot_y(df_pct_y_last.copy(), titles[m], number_format = 'absolute', decimals = 2)
        else:
            bar_plots_pct_y[m] = bar_plot_y(df_pct_y_last.copy(), titles[m], number_format = 'percent', decimals = 2)

        # 1) Pronóstico anual (%): concat base_pct + forecast del modelo
        df_pct_y_avg = pd.concat([
            list_files["df_avg_year"].tail(8),
            list_files[f"{m}_interannual_avg"][["forecast"]]
            .rename(columns = {"forecast": "ts"})
        ], axis=0)
        
        if list_files[f"{m}_interannual_avg"][["forecast"]].max().values[0] > 1 or list_files[f"{m}_interannual_avg"][["forecast"]].min().values[0] < -1:
            bar_plots_pct_y_avg[m] = bar_plot_y_conf(df_pct_y_avg.copy(), titles[m], color = colors[m], number_format = 'absolute', decimals = 2)
        else:
            bar_plots_pct_y_avg[m] = bar_plot_y_conf(df_pct_y_avg.copy(), titles[m], color = colors[m], number_format = 'percent', decimals = 2)

        
except Exception as e:
    print(f"Error: {e}")


# =============================================================================
# INTERFAZ DE USUARIO
# =============================================================================
# Función para dibujar un divisor de sección
def section_divider():
    st.markdown("""
        <style>
        .section-divider {
            border-top: 4px solid #3399FF; /* Azul más suave */
            margin: 20px 0;
            box-shadow: 0px 3px 6px rgba(51, 153, 255, 0.2); /* Sombra azul suave */
        }
        </style>
        <div class="section-divider"></div>
    """, unsafe_allow_html = True)


def macro():
    
    # =============================================================================
    # DESCRIPCIÓN DE LA SERIE TEMPORAL
    # =============================================================================  
    st.markdown("<h1 style='font-size: 40px;'>MACROECONOMICAL VARIABLES</h1>", unsafe_allow_html = True)
    st.markdown("### TIME SERIES SPLIT")
    section_divider()
    
    st.markdown(
        """
        The dataset split offers three approaches:

        - **With Test:** Includes a training set, a validation set (2 years) to tune the model’s optimal hyperparameters, and a test set (1 year) to evaluate the stability of predictions.
        - **Short Validation:** Does not include a test set. Only a validation set (1 year) is used to tune the model’s optimal hyperparameters.
        - **Long Validation:** Extends the validation period to 2 years, again without using a test set.

        For all three approaches, training is performed in steps of 4 periods (1 year), and for **Long Validation**, an additional model with **8 steps (2 years)** is included.
        """,
        unsafe_allow_html=True
    )

    # Expander para mostrar el DataFrame solo si se despliega
    with st.expander("Show parameters (technical view)"):
        # Convertir todas las columnas booleanas a string
        params_df_str = list_files['params_df'].copy()
        for col in params_df_str.select_dtypes(include=["bool"]).columns:
            params_df_str[col] = params_df_str[col].astype(str)
        st.dataframe(params_df_str)


    # Crear dos columnas
    col1, col2 = st.columns([4, 1])  # Ajusta los tamaños según prefieras

    # Mostrar el gráfico en la primera columna
    with col1:
        st.plotly_chart(fig_init_ts, use_container_width = True)

    # Mostrar la tabla en la segunda columna
    with col2:
        st.dataframe(
            list_files["df_abs"].style.background_gradient(cmap = "cividis")
        )
    
    # =============================================================================
    # BACKTESTING
    # ============================================================================= 
    st.markdown("### BACKTESTING")
    section_divider()
    
    st.markdown(
        """
        <div style="text-align: justify;">
        Various metrics are used to evaluate the model and measure prediction errors:

        - **MAE (Mean Absolute Error):** Represents the average of the absolute differences between actual and predicted values.
        - **MSE (Mean Squared Error):** Calculates the average of squared errors, penalizing larger mistakes and highlighting significant deviations.
        - **RMSE (Root Mean Squared Error):** The square root of the MSE, allowing the error to be interpreted in the same units as the original data. It is used as the **target metric** for model optimization due to its sensitivity to large errors.
        - **MAPE (Mean Absolute Percentage Error):** Expresses the error as a percentage, making it easier to compare across different data scales.
        """,
        unsafe_allow_html = True
    )
    
    models = [
        ("Holt-Winters", 0),
        ("SARIMA",       1),
        ("Prophet",      2),
        ("UCM",          3),
        ("ARDL",         4),
        ("VAR",          5),
    ]

    for i in range(0, len(models), 2):
        cols = st.columns(2)
        for col, (title, idx) in zip(cols, models[i : i + 2]):
            with col:
                st.markdown(f"<div class='card'><h4>{title}</h4></div>", unsafe_allow_html=True)
                st.plotly_chart(backtest_list[idx], use_container_width=True)

    
    st.markdown("#### ERROR METRICS")
    
    st.plotly_chart(fig_error_metrics, use_container_width = True)
    

    # =============================================================================
    # RESULTS: FORECAST ABSOLUTE VALUE
    # =============================================================================  
    st.markdown("### FORECAST: ABSOLUTE VALUE")
    section_divider()
    
    st.markdown(
        """
        <div style="text-align: justify;">
            Below are the absolute value forecasts generated by the models, along with their corresponding confidence intervals, 
            which are computed across a <strong>wide range of percentiles</strong> from 1 to 99 in increments of 5. 
            This extensive coverage enables users to not only observe the central trend of the forecasts but also to gain insight 
            into the full range of potential outcomes, taking into account the uncertainty inherent in the predictions.
        </div>
        """,
        unsafe_allow_html=True
    )

    models = [
        ("Holt-Winters", "hw",      0),
        ("SARIMA",       "sarima",  1),
        ("Prophet",      "prophet", 2),
        ("UCM",          "ucm",     3),
        ("ARDL",         "ardl",    4),
        ("VAR",          "var",     5),
    ]

    # Gráfico de fanchart absoluto
    for i in range(0, len(models), 2):
        cols = st.columns(2)
        for col, (title, key, idx) in zip(cols, models[i : i + 2]):
            with col:
                st.markdown(f"<div class='card'><h4>{title}</h4></div>", unsafe_allow_html=True)
                st.plotly_chart(abs_forecast_list[idx], use_container_width=True)

    # Tablas de percentiles absolutos
    for i in range(0, len(models), 2):
        cols = st.columns(2)
        for col, (title, key, _ ) in zip(cols, models[i : i + 2]):
            with col:
                st.markdown(f"<div class='card'><h4>{title}</h4></div>", unsafe_allow_html=True)
                df = list_files[f"{key}_abs_q"]
                st.dataframe(
                    df.style
                    .format({c: "{:.4f}" for c in df.columns})
                    .background_gradient(cmap="Blues")
                )
    
    
    st.markdown("### FORECAST: ANNUAL NOMINAL VALUE")
    section_divider()
    
    # Texto descriptivo sin recuadro
    st.markdown(
        """
        The following results represent the **annual percentage change** of the time series.
        The changes in the series are calculated using the logarithmic approach, given its additive properties.
        Confidence intervals are also included.
        
        The formula used for the calculation is shown below:
        """
    )

    # Bloque para la fórmula centrada
    st.latex(r"""
    r_{\text{anual}} = \ln\left(\frac{P_y}{P_{y-1}}\right) \quad
    """)

    models = [
        ("Holt-Winters", "hw"),
        ("SARIMA",       "sarima"),
        ("Prophet",      "prophet"),
        ("UCM",          "ucm"),
        ("ARDL",         "ardl"),
        ("VAR",          "var"),
    ]

    # Comparación variación anual
    for i in range(0, len(models), 2):
        cols = st.columns(2)
        for col, (title, key) in zip(cols, models[i : i + 2]):
            with col:
                st.markdown(f"<div class='card'><h4>{title}</h4></div>", unsafe_allow_html=True)
                st.plotly_chart(bar_plots_abs[key], use_container_width=True)

    # Tablas de percentiles
    for i in range(0, len(models), 2):
        cols = st.columns(2)
        for col, (title, key) in zip(cols, models[i : i + 2]):
            with col:
                st.markdown(f"<div class='card'><h4>{title}</h4></div>", unsafe_allow_html=True)
                df = list_files[f"{key}_abs_y"]
                st.dataframe(
                    df.style
                    .format({c: "{:.4f}" for c in df.columns})
                    .background_gradient(cmap="Blues")
                )


    # =============================================================================
    # RESULTS: FORECAST PERCENTAGE VARIATION
    # =============================================================================  
    st.markdown("### FORECAST: QUARTERLY VARIATION")
    section_divider()

    # Texto descriptivo sin recuadro
    st.markdown(
        """
        The results below show the **quarterly percentage variation** of the time series, where changes are calculated using the logarithmic 
        approach due to its additive properties. Confidence intervals are also included to provide a sense of the uncertainty around the estimates. 

        The formula used for this calculation is shown below:
        """,
        unsafe_allow_html=True
    )


    # Bloque para la fórmula centrada, con indicación de que t representa trimestres
    st.latex(r"""
    r_q = \ln\left(\frac{P_q}{P_{q-1}}\right) \quad
    """)
            
    models = [
        ("Holt-Winters", "hw",      0),
        ("SARIMA",       "sarima",  1),
        ("Prophet",      "prophet", 2),
        ("UCM",          "ucm",     3),
        ("ARDL",         "ardl",    4),
        ("VAR",          "var",     5),
    ]

    # Gráfico interanual de porcentaje
    for i in range(0, len(models), 2):
        cols = st.columns(2)
        for col, (title, key, idx) in zip(cols, models[i : i + 2]):
            with col:
                st.markdown(f"<div class='card'><h4>{title}</h4></div>", unsafe_allow_html=True)
                st.plotly_chart(pct_forecast_list[idx], use_container_width=True, key = f"pct_forecast_{key}")

    # Tablas de percentiles por trimestre
    for i in range(0, len(models), 2):
        cols = st.columns(2)
        for col, (title, key, _ ) in zip(cols, models[i : i + 2]):
            with col:
                st.markdown(f"<div class='card'><h4>{title}</h4></div>", unsafe_allow_html=True)
                df = list_files[f"{key}_pct_q"]
                st.dataframe(
                    df.style
                    .format({c: "{:.4f}" for c in df.columns})
                    .background_gradient(cmap="Blues")
                )

        
    # =============================================================================
    # RESULTS: ANNUAL PERCENTAGE VARIATION
    # =============================================================================  
    st.markdown("### FORECAST: YEAR-ON-YEAR QUARTERLY VARIATION")
    section_divider()
    
    # Texto descriptivo sin recuadro
    st.markdown(
        """
        The following results reflect the **year-over-year percentage change by quarter** of the time series.
        The changes in the series are calculated using the logarithmic approach, given its additive properties.
        Confidence intervals are also included.
        
        The formula used for the calculation is shown below:
        """
    )


    # Bloque para la fórmula centrada, con indicación de que q representa trimestres
    st.latex(r"""
    r_q^{\text{YoY}} = \ln\left(\frac{P_q}{P_{q-4}}\right) \quad
    """)

    # Gráfico
    models = [
        ("Holt-Winters", "hw",      0),
        ("SARIMA",       "sarima",  1),
        ("Prophet",      "prophet", 2),
        ("UCM",          "ucm",     3),
        ("ARDL",         "ardl",    4),
        ("VAR",          "var",     5),
    ]

    # Gráfico interanual
    for i in range(0, len(models), 2):
        cols = st.columns(2)
        for col, (title, key, idx) in zip(cols, models[i : i + 2]):
            with col:
                st.markdown(f"<div class='card'><h4>{title}</h4></div>", unsafe_allow_html=True)
                st.plotly_chart(interyear_forecast_list[idx], use_container_width=True)

    # Tablas Regular interanual por trimestre
    for i in range(0, len(models), 2):
        cols = st.columns(2)
        for col, (title, key, _ ) in zip(cols, models[i : i + 2]):
            with col:
                st.markdown(f"<div class='card'><h4>{title}</h4></div>", unsafe_allow_html=True)
                df = list_files[f"{key}_interannual"]
                st.dataframe(
                    df.style
                    .format({c: "{:.4f}" for c in df.columns})
                    .background_gradient(cmap="BuPu")
                )

    
    # =============================================================================
    # RESULTS: ANNUAL PERCENTAGE VARIATION
    # =============================================================================  
    st.markdown("### FORECAST: ANNUAL VARIATION")
    section_divider()
    
    # Texto descriptivo sin recuadro
    st.markdown(
        """
        The following results represent the **annual percentage change** of the time series.
        The changes in the series are calculated using the logarithmic approach, given its additive properties.
        Confidence intervals are also included.
        
        The formula used for the calculation is shown below:
        """
    )


    # Bloque para la fórmula centrada
    st.latex(r"""
    r_{\text{anual}} = \ln\left(\frac{P_y}{P_{y-1}}\right) \quad
    """)

    models = [
        ("Holt-Winters", "hw"),
        ("SARIMA",       "sarima"),
        ("Prophet",      "prophet"),
        ("UCM",          "ucm"),
        ("ARDL",         "ardl"),
        ("VAR",          "var"),
    ]

    # Comparación variación anual
    for i in range(0, len(models), 2):
        cols = st.columns(2)
        for col, (title, key) in zip(cols, models[i : i + 2]):
            with col:
                st.markdown(f"<div class='card'><h4>{title}</h4></div>", unsafe_allow_html=True)
                st.plotly_chart(bar_plots_pct_y[key], use_container_width=True, key = f"bar_plot_y_{key}")

    # Tablas de percentiles
    for i in range(0, len(models), 2):
        cols = st.columns(2)
        for col, (title, key) in zip(cols, models[i : i + 2]):
            with col:
                st.markdown(f"<div class='card'><h4>{title}</h4></div>", unsafe_allow_html=True)
                df = list_files[f"{key}_interannual_last"]
                st.dataframe(
                    df.style
                    .format({c: "{:.4f}" for c in df.columns})
                    .background_gradient(cmap="Blues")
                )
                
    
    # =============================================================================
    # RESULTS: ANNUAL BARPLOT VARIATION WITH INTERANNUAL COMPONENT
    # =============================================================================  
    st.markdown("### FORECAST: AVERAGE YEAR-ON-YEAR QUARTERLY VARIATION")
    section_divider()

    # Texto descriptivo sin recuadro
    st.markdown(
        """
        The following results represent the **average quarterly year-over-year change per year** of the time series.
        This reflects the average change of a value compared to the same period of the previous year within each year.
        
        The formula used for the calculation is shown below:
        """
    )


    # Bloque para la fórmula centrada
    st.latex(r"""
    \bar{r}_{\text{YoY, year}} = \frac{1}{4} \sum_{q=1}^{4} \ln\left(\frac{P_q}{P_{q-4}}\right)
    """)

    models = [
        ("Holt-Winters", "hw"),
        ("SARIMA",       "sarima"),
        ("Prophet",      "prophet"),
        ("UCM",          "ucm"),
        ("ARDL",         "ardl"),
        ("VAR",          "var"),
    ]

    # Comparación variación anual
    for i in range(0, len(models), 2):
        cols = st.columns(2)
        for col, (title, key) in zip(cols, models[i : i + 2]):
            with col:
                st.markdown(f"<div class='card'><h4>{title}</h4></div>", unsafe_allow_html=True)
                st.plotly_chart(bar_plots_pct_y_avg[key], use_container_width=True, key=f"plotly_{key}")

    # Regular
    for i in range(0, len(models), 2):
        cols = st.columns(2)
        for col, (title, key) in zip(cols, models[i : i + 2]):
            with col:
                st.markdown(f"<div class='card'><h4>{title}</h4></div>", unsafe_allow_html=True)
                df = list_files[f"{key}_interannual_avg"]
                st.dataframe(
                    df.style
                    .format({c: "{:.4f}" for c in df.columns})
                    .background_gradient(cmap="BuPu")
                )

        
# =============================================================================
# HOJA CURVAS
# =============================================================================
def curves():
    # =============================================================================
    # DESCRIPCIÓN DE LA SERIE TEMPORAL
    # =============================================================================  
    st.markdown("<h1 style='font-size: 40px;'>INTEREST RATE CURVES</h1>", unsafe_allow_html = True)
    st.markdown("### AGGREGRATED RESULTS")
    section_divider()

    st.markdown( 
        """
        <div style="text-align: justify">

        This section presents the results obtained from the estimation and simulation of interest rate curves. The analysis was carried out by applying the following methodology:

        **Vasicek Model**: This is a short-rate interest rate model used to describe the evolution of interest rates over time. The model assumes that interest rates follow a mean-reverting stochastic process, which helps to capture realistic dynamics in financial markets.

        </div>
        
        $$ 
        dr_t = a(b - r_t)dt + \\sigma dW_t 
        $$

        <div style="text-align: justify">
        
        Where:  
        - $r_t$ is the short-term interest rate at time $t$,  
        - $a$ is the **speed of mean reversion**,  
        - $b$ is the **long-term mean level**,  
        - $\\sigma$ is the **volatility** of the interest rate,  
        - $W_t$ is a Wiener process (Brownian motion).  

        This model allows for negative interest rates and captures the tendency of interest rates to revert to a long-term mean over time.

        In order to capture a wider variety of behaviors and improve the robustness of the model, multiple iterations have been performed, varying both the start date of the training period and the degree of **volatility smoothing**. This thorough analysis enables the selection of the most appropriate model specification, prioritizing consistency with the observed and projected macroeconomic environment.

        </div>
        """,
        unsafe_allow_html=True
    )

    # Cargar datos
    curves_metrics = pd.read_csv('evidence_curves/metrics_vasicek.csv')
    curves_metrics.columns = ['Reference Date', 'Final Forecast', 'Variable Name', 'Smooth Volatility']
    curves_metrics = curves_metrics[curves_metrics['Variable Name'] == select_variable_curves]
    
    curves_results = pd.read_csv('evidence_curves/results_vasicek.csv')
    curves_results.columns = ['Date', 'P1', 'P99', 'Forecast', 'Variable Name', 'Smooth Volatility', 'Reference Date']
    curves_results = curves_results[curves_results['Variable Name'] == select_variable_curves]

    # Mostrar el DataFrame de métricas
    curves_metrics = curves_metrics.sort_values(by = 'Final Forecast', ascending = False).reset_index(drop = True)
    
    st.dataframe(curves_metrics)

    # Inyecta CSS personalizado para dar un toque estético a los selectboxes
    st.markdown("### MODEL SELECTION")
    
    section_divider()
    
    st.markdown(
        """
        Based on the metrics table presented in the previous section, the following fields must be completed:
        1. **Reference date** to determine the analysis period.
        2. **Inclusion of the smoothed volatility** to refine the curve projection.

        Once these steps are completed, the filtered results and the corresponding chart for the selected configuration will be displayed.
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <style>
        /* Aplica estilos a todos los selectboxes */
        div[data-testid="stSelectbox"] {
            border: 1px solid #ddd;
            border-radius: 8px;
            background-color: #fff;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
            padding: 5px 10px;
        }
        </style>
        """, unsafe_allow_html=True
    )

    # Distribuir los selectboxes en tres columnas para mayor organización
    col1, col2 = st.columns(2)

    with col1:
        selected_date_ref = st.selectbox(
            "Reference date",
            sorted(curves_metrics["Reference Date"].unique())
        )

    with col2:
        selected_smooth_volatility = st.selectbox(
            "Smooth volatility",
            curves_metrics["Smooth Volatility"].unique()
        )


    # Filtrar curves_results según la selección
    df2_filtered = curves_results[
        (curves_results["Reference Date"] == selected_date_ref) &
        (curves_results["Smooth Volatility"] == selected_smooth_volatility)
    ]


    # Preparar el DataFrame de percentiles y establecer 'date' como índice
    df_curve_percentile = df2_filtered[['Date', 'P1', 'P99', 'Forecast']].copy().set_index('Date')

    # Mostrar el DataFrame filtrado
    df2_filtered = df2_filtered.style.format(
        {"P1": "{:.4f}", "P99": "{:.4f}", "Forecast": "{:.4f}"}
    ).background_gradient(
        cmap   = "Blues", 
        subset = ["P1", "P99", "Forecast"]
    )
    
    st.dataframe(df2_filtered)

    # Cargar y procesar el DataFrame histórico desde Excel
    hist = pd.read_excel('data/processed/variables_cleaned.xlsx', sheet_name = select_variable_curves, index_col=0)
    hist = hist.resample('Q').last()


    # Calcular la fecha límite: primer registro - 3 meses + 2 días
    day_filt_hist = pd.to_datetime(df_curve_percentile.index[0]) + pd.DateOffset(months=-3, days=2)
    hist = hist[hist.index <= day_filt_hist]

    # Generar el gráfico utilizando plot_fanchart_plotly
    temp_fig = plot_fanchart_plotly(
        data            = hist,
        percentile_df   = df_curve_percentile.rename(columns={'Forecast': 'forecast'}),
        fitted_values   = None,
        len_hist_plot   = 30,
        title           = f"Forecast {select_variable_curves}",
        percentile_list = ['P1', 'P99'],
        final_plot      = True,
        last_value_text = False
    )

    st.plotly_chart(temp_fig, use_container_width = True)


# =============================================================================
# COMUNIDADES
# =============================================================================
def comunidades():
    st.markdown("<h1 style='font-size: 40px;'>AUTONOMOUS COMMUNITIES</h1>", unsafe_allow_html = True)

    general_var_name = select_variable_com
        
    # Variables que se suman en lugar de ponderarse    
    list_variables = pd.read_excel('weights.xlsx', sheet_name = general_var_name).dropna(subset = ['Comunidades'])['Comunidades'].tolist()
    weighted_var_name = f"W_{general_var_name}"
    
    # Añadir la variable a la lista de variables
    list_variables.append(weighted_var_name)
    
    sheet_names = [
    "df_abs",
    "df_abs_y",
    "df_pct",
    "df_interannual",
    "df_avg_year",
    "df_last_year",
    "hw_abs_q",
    "hw_abs_y",
    "hw_pct_q",
    "hw_interannual",
    "hw_interannual_avg",
    "hw_interannual_last",
    "sarima_abs_q",
    "sarima_abs_y",
    "sarima_pct_q",
    "sarima_interannual",
    "sarima_interannual_avg",
    "sarima_interannual_last",
    "prophet_abs_q",
    "prophet_abs_y",
    "prophet_pct_q",
    "prophet_interannual",
    "prophet_interannual_avg",
    "prophet_interannual_last",
    "ucm_abs_q",
    "ucm_abs_y",
    "ucm_pct_q",
    "ucm_interannual",
    "ucm_interannual_avg",
    "ucm_interannual_last",
    "ardl_abs_q",
    "ardl_abs_y",
    "ardl_pct_q",
    "ardl_interannual",
    "ardl_interannual_avg",
    "ardl_interannual_last",
    "var_abs_q",
    "var_abs_y",
    "var_pct_q",
    "var_interannual",
    "var_interannual_avg",
    "var_interannual_last"
    ]
    
    for i in range(len(list_variables)):
        
        # Agregar un título con el nombre de la comunidad
        st.markdown(f"<h2 style='font-size: 30px;'>{list_variables[i]}</h2>", unsafe_allow_html = True)
        section_divider()
        
        var_name = list_variables[i]
        
        # Lectura de la información del modelo
        list_files = {}  
        
        for sheet in sheet_names:
            list_files[sheet] = pd.read_excel(f'evidence_macro/evidence_{var_name}.xlsx', sheet_name = sheet, index_col = 0)
            

        # =============================================================================
        # GRÁFICOS PRONÓSTICOS EN VALOR ABSOLUTO
        # =============================================================================
        models_info_forecast = [
            (list_files['hw_abs_q'],     "Holt-Winters"),
            (list_files['sarima_abs_q'], "SARIMA"),
            (list_files['prophet_abs_q'],"Prophet"),
            (list_files['ucm_abs_q'],    "UCM"),
            (list_files['ardl_abs_q'],   "ARDL"),
            (list_files['var_abs_q'],    "VAR")
        ]

        abs_forecast_list = []

        if var_name == weighted_var_name:
            color_theme = 'purple'
            color_theme_2 = 'BuPu'
        else:
            color_theme = 'blue'
            color_theme_2 = 'Blues'

        for percentile_df, model_name in models_info_forecast:
            # Generamos el fanchart en figura temporal
            fig = plot_fanchart_plotly(
                data            = list_files['df_abs'],
                percentile_df   = percentile_df,
                fitted_values   = None,
                len_hist_plot   = 30,
                title           = f"Forecast ({model_name}) - {var_name}",
                last_value_text = False,
                color_theme     = color_theme,
            )

            # Añadimos a la lista
            abs_forecast_list.append(fig)
            
            
        # =============================================================================
        # GRÁFICOS DE BARRAS PARA PRONÓSTICOS ANUALES
        # =============================================================================
        # Configuración de los modelos
        models = ["hw", "sarima", "prophet", "ucm", "ardl", "var"]

        titles = {
            "hw": "Holt-Winters",
            "sarima": "SARIMA",
            "prophet": "Prophet",
            "ucm": "UCM",
            "ardl": "ARDL",
            "var": "VAR",
        }

        # Generar y graficar
        bar_plots_abs = {}
        bar_plots_pct_y = {}
        bar_plots_pct_y_avg = {}
        
        
        for m in models:
            # 0) Pronóstico anual (%): concat base_pct + forecast del modelo
            df_abs_y = pd.concat([
                list_files["df_abs_y"].tail(8),
                list_files[f"{m}_abs_y"][["forecast"]]
                .rename(columns = {"forecast": "ts"})
            ], axis=0)
            
            bar_plots_abs[m] = bar_plot_y(df_abs_y.copy(), titles[m], color = color_theme)
            
            
            # 2) Pronóstico anual con promedio interanual: concat base_interyear + forecast
            df_pct_y_last = pd.concat([
                list_files["df_last_year"].tail(8),
                list_files[f"{m}_interannual_last"][["forecast"]]
                .rename(columns = {"forecast": "ts"})
            ], axis=0)
            
            
            if df_pct_y_last['ts'].max() > 1 or df_pct_y_last['ts'].min() < -1:
                bar_plots_pct_y[m] = bar_plot_y(df_pct_y_last.copy(), titles[m], color = color_theme, number_format = 'absolute', decimals = 2)
            else:
                bar_plots_pct_y[m] = bar_plot_y(df_pct_y_last.copy(), titles[m], color = color_theme, number_format = 'percent', decimals = 2)

            # 1) Pronóstico anual (%): concat base_pct + forecast del modelo
            df_pct_y_avg = pd.concat([
                list_files["df_avg_year"].tail(8),
                list_files[f"{m}_interannual_avg"][["forecast"]] # NOTE: Promedio interanual
                .rename(columns = {"forecast": "ts"})
            ], axis=0)
            

            if df_pct_y_avg['ts'].max() > 1 or df_pct_y_avg['ts'].min() < -1:
                bar_plots_pct_y_avg[m] = bar_plot_y(df_pct_y_avg.copy(), titles[m], color = color_theme, number_format = 'absolute', decimals = 2)
            else:
                bar_plots_pct_y_avg[m] = bar_plot_y(df_pct_y_avg.copy(), titles[m], color = color_theme, number_format = 'percent', decimals = 2)
            

            
        # Mostrar los gráficos: Absolute value
        # -------------------------------------------------------------------------------------------
        models = [
        ("Holt-Winters", "hw",      0),
        ("SARIMA",       "sarima",  1),
        ("Prophet",      "prophet", 2),
        ("UCM",          "ucm",     3),
        ("ARDL",         "ardl",    4),
        ("VAR",          "var",     5),
        ]

        # Gráfico de fanchart absoluto
        st.markdown("### ABSOLUTE VALUE FORECAST")
        
        for i in range(0, len(models), 2):
            cols = st.columns(2)
            for col, (title, key, idx) in zip(cols, models[i : i + 2]):
                with col:
                    st.markdown(f"<div class='card'><h4>{title}</h4></div>", unsafe_allow_html = True)
                    st.plotly_chart(abs_forecast_list[idx], use_container_width = True)

            
        # Mostrar graficos con datos anuales
        # -------------------------------------------------------------------------------------------
        models = [
            ("Holt-Winters", "hw"),
            ("SARIMA",       "sarima"),
            ("Prophet",      "prophet"),
            ("UCM",          "ucm"),
            ("ARDL",         "ardl"),
            ("VAR",          "var"),
        ]

        # Comparación variación anual
        st.markdown("### INTERANNUAL AVG FORECAST")
        
        for i in range(0, len(models), 2):
            cols = st.columns(2)
            for col, (title, key) in zip(cols, models[i : i + 2]):
                with col:
                    st.markdown(f"<div class='card'><h4>{title}</h4></div>", unsafe_allow_html=True)
                    st.plotly_chart(bar_plots_pct_y_avg[key], use_container_width=True)
                    
        # Tablas de percentiles
        st.markdown("### INTERANNUAL LAST FORECAST")
        for i in range(0, len(models), 2):
            cols = st.columns(2)
            for col, (title, key) in zip(cols, models[i : i + 2]):
                with col:
                    st.markdown(f"<div class='card'><h4>{title}</h4></div>", unsafe_allow_html=True)
                    df = list_files[f"{key}_interannual_last"]
                    st.dataframe(
                        df.style
                        .format({c: "{:.4f}" for c in df.columns})
                        .background_gradient(cmap=color_theme_2)
                    )
    
# =============================================================================
# HOJA DELIVERY
# =============================================================================
def delivery():
    st.markdown("<h1 style='font-size: 40px;'>FINAL RESULTS</h1>", unsafe_allow_html = True)
    st.markdown("### WINNING MODELS")
    section_divider()
    
    # Espacio para cargar el fihcero en la herramienta
    uploaded_file = st.file_uploader("Upload final_models.xlsx", type = ["xlsx"])
    
    if uploaded_file is not None:
        final_models_info = pd.read_excel(uploaded_file, sheet_name = 'MACRO')
    else:
        st.warning("Please upload the 'final_models.xlsx' file.")
        return
    
    # Lectura de los datos
    var_deliery_name_list = final_models_info['MAIN NAME'].values
    var_name_list = final_models_info['TOOL NAME'].values
    
    models_1_list = final_models_info['MODEL A'].values
    models_2_list = final_models_info['MODEL B'].values
    
    delivery_type = final_models_info['DELIVERY'].values
    
    esc_1 = final_models_info['DOWN'].values
    esc_2 = final_models_info['UP'].values
    
    # Selección de los mejores modelos
    # =============================================================================
    monthly_forecast_list = []
    quarter_forecast_list = []
    annual_forecast_list = []
    
    for i in range(len(var_name_list)):
        st.markdown(f"### {var_deliery_name_list[i]}")
        section_divider()
        
        # Parametros ajustables
        var_name = var_name_list[i]
        models = [models_1_list[i], models_2_list[i]] if models_2_list[i] != 0 else [models_1_list[i]]
        percentiles = [esc_1[i], esc_2[i]]
        
        # Se importa este primer fichero solo con el fin deobtener el histórico de la serie    
        hist_df_abs = pd.read_excel(f'evidence_macro/evidence_{var_name}.xlsx', sheet_name = 'df_abs', index_col = 0)
        hist_df_abs_y = pd.read_excel(f'evidence_macro/evidence_{var_name}.xlsx', sheet_name = 'df_abs_y', index_col = 0)
        hist_df_pct = pd.read_excel(f'evidence_macro/evidence_{var_name}.xlsx', sheet_name = 'df_pct', index_col = 0)
        hist_df_interannual = pd.read_excel(f'evidence_macro/evidence_{var_name}.xlsx', sheet_name = 'df_interannual', index_col = 0)
        hist_df_avg_year = pd.read_excel(f'evidence_macro/evidence_{var_name}.xlsx', sheet_name = 'df_avg_year', index_col = 0)
        hist_df_last_year = pd.read_excel(f'evidence_macro/evidence_{var_name}.xlsx', sheet_name = 'df_last_year', index_col = 0)

        if len(models) == 1:
            # Títulos para los gráficos
            title_plot_1 = f"Forecast Trimestral: {models[0].upper()}"
            title_plot_2 = f"{models[0].upper()}"
            
            
            # Cargar los datos de los percentiles y la previsión
            if var_name == 'IPC' or var_name == 'IPCA' or var_name == 'IPCS':
                forecast_month = pd.read_excel(f'evidence_macro/evidence_{var_name}.xlsx', sheet_name = f"{models[0]}_interannual_monthly", index_col = 0)[[percentiles[0], percentiles[1], 'forecast']]
            
            
            df_abs_q = pd.read_excel(f'evidence_macro/evidence_{var_name}.xlsx', sheet_name = f"{models[0]}_abs_q", index_col = 0)[[percentiles[0], percentiles[1], 'forecast']]
            df_abs_y = pd.read_excel(f'evidence_macro/evidence_{var_name}.xlsx', sheet_name = f"{models[0]}_abs_y", index_col = 0)[[percentiles[0], percentiles[1], 'forecast']]
            df_pct_q = pd.read_excel(f'evidence_macro/evidence_{var_name}.xlsx', sheet_name = f"{models[0]}_pct_q", index_col = 0)[[percentiles[0], percentiles[1], 'forecast']]
            df_interannual = pd.read_excel(f'evidence_macro/evidence_{var_name}.xlsx', sheet_name = f"{models[0]}_interannual", index_col = 0)[[percentiles[0], percentiles[1], 'forecast']]
            df_interannual_avg = pd.read_excel(f'evidence_macro/evidence_{var_name}.xlsx', sheet_name = f"{models[0]}_interannual_avg", index_col = 0)[[percentiles[0], percentiles[1], 'forecast']]
            df_interannual_last = pd.read_excel(f'evidence_macro/evidence_{var_name}.xlsx', sheet_name = f"{models[0]}_interannual_last", index_col = 0)[[percentiles[0], percentiles[1], 'forecast']]
            
            if delivery_type[i] == 'Nominal':
                hist_q_df = hist_df_abs.copy()
                hist_y_df = hist_df_abs_y.copy()
                forecas_q_df = df_abs_q.copy()
                forecas_y_df = df_abs_y.copy()
            elif delivery_type[i] == 'Last Variation':
                hist_q_df = hist_df_interannual.copy()
                hist_y_df = hist_df_last_year.copy()
                forecas_q_df = df_interannual.copy()
                forecas_y_df = df_interannual_last.copy()
            elif delivery_type[i] == 'Avg Variation':
                hist_q_df = hist_df_interannual.copy()
                hist_y_df = hist_df_avg_year.copy()
                forecas_q_df = df_interannual.copy()
                forecas_y_df = df_interannual_avg.copy()
            else:
                st.warning("Please select a valid delivery type: 'Nominal', 'Last Variation', or 'Avg Variation'.")
            
        elif len(models) == 2:
            # Títulos para los gráficos
            title_plot_1 = f"Quarterly Forecast: {models[0].upper()} y {models[1].upper()}"
            title_plot_2 = f"{models[0].upper()} y {models[1].upper()}"
            
            if var_name == 'IPC' or var_name == 'IPCA' or var_name == 'IPCS':
                forecast_month_1 = pd.read_excel(f'evidence_macro/evidence_{var_name}.xlsx', sheet_name = f"{models[0]}_interannual_monthly", index_col = 0)[[percentiles[0], percentiles[1], 'forecast']]
                forecast_month_2 = pd.read_excel(f'evidence_macro/evidence_{var_name}.xlsx', sheet_name = f"{models[1]}_interannual_monthly", index_col = 0)[[percentiles[0], percentiles[1], 'forecast']]
                forecast_month = (forecast_month_1 + forecast_month_2) / 2            
            
            df_abs_q_1 = pd.read_excel(f'evidence_macro/evidence_{var_name}.xlsx', sheet_name = f"{models[0]}_abs_q", index_col = 0)[[percentiles[0], percentiles[1], 'forecast']]
            df_abs_q_2 = pd.read_excel(f'evidence_macro/evidence_{var_name}.xlsx', sheet_name = f"{models[1]}_abs_q", index_col = 0)[[percentiles[0], percentiles[1], 'forecast']]
            df_abs_q = (df_abs_q_1 + df_abs_q_2) / 2
            
            df_abs_y_1 = pd.read_excel(f'evidence_macro/evidence_{var_name}.xlsx', sheet_name = f"{models[0]}_abs_y", index_col = 0)[[percentiles[0], percentiles[1], 'forecast']]
            df_abs_y_2 = pd.read_excel(f'evidence_macro/evidence_{var_name}.xlsx', sheet_name = f"{models[1]}_abs_y", index_col = 0)[[percentiles[0], percentiles[1], 'forecast']]
            df_abs_y = (df_abs_y_1 + df_abs_y_2) / 2
            
            df_pct_q_1 = pd.read_excel(f'evidence_macro/evidence_{var_name}.xlsx', sheet_name = f"{models[0]}_pct_q", index_col = 0)[[percentiles[0], percentiles[1], 'forecast']]
            df_pct_q_2 = pd.read_excel(f'evidence_macro/evidence_{var_name}.xlsx', sheet_name = f"{models[1]}_pct_q", index_col = 0)[[percentiles[0], percentiles[1], 'forecast']]
            df_pct_q = (df_pct_q_1 + df_pct_q_2) / 2
            
            df_interannual_1 = pd.read_excel(f'evidence_macro/evidence_{var_name}.xlsx', sheet_name = f"{models[0]}_interannual", index_col = 0)[[percentiles[0], percentiles[1], 'forecast']]
            df_interannual_2 = pd.read_excel(f'evidence_macro/evidence_{var_name}.xlsx', sheet_name = f"{models[1]}_interannual", index_col = 0)[[percentiles[0], percentiles[1], 'forecast']]
            df_interannual = (df_interannual_1 + df_interannual_2) / 2
            
            df_interannual_avg_1 = pd.read_excel(f'evidence_macro/evidence_{var_name}.xlsx', sheet_name = f"{models[0]}_interannual_avg", index_col = 0)[[percentiles[0], percentiles[1], 'forecast']]
            df_interannual_avg_2 = pd.read_excel(f'evidence_macro/evidence_{var_name}.xlsx', sheet_name = f"{models[1]}_interannual_avg", index_col = 0)[[percentiles[0], percentiles[1], 'forecast']]
            df_interannual_avg = (df_interannual_avg_1 + df_interannual_avg_2) / 2
            
            df_interannual_last_1 = pd.read_excel(f'evidence_macro/evidence_{var_name}.xlsx', sheet_name = f"{models[0]}_interannual_last", index_col = 0)[[percentiles[0], percentiles[1], 'forecast']]
            df_interannual_last_2 = pd.read_excel(f'evidence_macro/evidence_{var_name}.xlsx', sheet_name = f"{models[1]}_interannual_last", index_col = 0)[[percentiles[0], percentiles[1], 'forecast']]
            df_interannual_last = (df_interannual_last_1 + df_interannual_last_2) / 2
            
            if delivery_type[i] == 'Nominal':
                hist_q_df = hist_df_abs.copy()
                hist_y_df = hist_df_abs_y.copy()
                forecas_q_df = df_abs_q.copy()
                forecas_y_df = df_abs_y.copy()
                
                quarter_forecast_list.append(forecas_q_df)
                annual_forecast_list.append(forecas_y_df)
            elif delivery_type[i] == 'Last Variation':
                hist_q_df = hist_df_interannual.copy()
                hist_y_df = hist_df_last_year.copy()
                forecas_q_df = df_interannual.copy()
                forecas_y_df = df_interannual_last.copy()
            elif delivery_type[i] == 'Avg Variation':
                hist_q_df = hist_df_interannual.copy()
                hist_y_df = hist_df_avg_year.copy()
                forecas_q_df = df_interannual.copy()
                forecas_y_df = df_interannual_avg.copy()
        else:
            st.warning("Please select at least one model for the forecast.")
        
        # Gráfico fanchart con intervalos de confianza
        if i < 10:
            fanchart_fig = plot_fanchart_plotly(
                data            = hist_q_df,
                percentile_df   = forecas_q_df,
                fitted_values   = None,
                len_hist_plot   = 30,
                percentile_list = percentiles,
                title           = title_plot_1,
                last_value_text = False,
                color_theme     = "purple",
                final_plot      = True
            )
                
            st.plotly_chart(fanchart_fig, use_container_width = True)
            
            df_barplot = pd.concat([hist_y_df, 
                                    forecas_y_df[['forecast']].rename(columns = {'forecast': 'ts'}),
                                    forecas_y_df[[percentiles[0]]].rename(columns = {percentiles[0]: 'ts'}),
                                    forecas_y_df[[percentiles[1]]].rename(columns = {percentiles[1]: 'ts'})],
                                    axis = 0
                                    ).tail(16)
                        
            if df_barplot["ts"].max() > 1 or df_barplot["ts"].min() < -1:
                bar_plot_fig = bar_plot_y_conf(df_barplot, title_plot_2, color = 'purple', number_format = 'absolute', decimals = 2)
            else:
                bar_plot_fig = bar_plot_y_conf(df_barplot, title_plot_2, color = 'purple', number_format = 'percent', decimals = 2)
            
            
            st.plotly_chart(bar_plot_fig, use_container_width = True)
                
        # Dataframe consolidado trimestral
        forecas_q_df = pd.DataFrame({'Escenario Adverso': forecas_q_df[percentiles[0]], 'Escenario Optimista': forecas_q_df[percentiles[1]], 'Pronóstico': forecas_q_df['forecast'], 'Variable': var_deliery_name_list[i]})
        forecas_y_df = pd.DataFrame({'Escenario Adverso': forecas_y_df[percentiles[0]], 'Escenario Optimista': forecas_y_df[percentiles[1]], 'Pronóstico': forecas_y_df['forecast'], 'Variable': var_deliery_name_list[i]})

        if var_name == 'IPC' or var_name == 'IPCA' or var_name == 'IPCS':
            forecast_month_df = pd.DataFrame({'Escenario Adverso': forecast_month[percentiles[0]], 'Escenario Optimista': forecast_month[percentiles[1]], 'Pronóstico': forecast_month['forecast'], 'Variable': var_deliery_name_list[i]})
            forecast_month_df.iloc[:, 0:3] = forecast_month_df.iloc[:, 0:3].copy() * 100  # Ajuste de los datos mensuales al porcentaje
            monthly_forecast_list.append(forecast_month_df)  # Ajuste de los datos mensuales al porcentaje


        # Ajuste de los datos según el tipo de entrega
        if delivery_type[i] == 'Last Variation' or delivery_type[i] == 'Avg Variation':
            forecas_q_df.iloc[:, 0:3] = forecas_q_df.iloc[:, 0:3].copy() * 100
            forecas_y_df.iloc[:, 0:3] = forecas_y_df.iloc[:, 0:3].copy() * 100


        quarter_forecast_list.append(forecas_q_df)
        annual_forecast_list.append(forecas_y_df)
    
    # Cargar las hojas disponibles
    xls = pd.ExcelFile(uploaded_file)

    # Verificar si la hoja 'IR' existe
    if 'IR' in xls.sheet_names:
        ir_models_info = pd.read_excel(uploaded_file, sheet_name='IR')
        print("La hoja 'IR' fue cargada correctamente.")
    
        st.markdown(f"### CURVES")
        ir_models_info = pd.read_excel(uploaded_file, sheet_name = 'IR')
        
        # Lectura de los datos
        var_deliery_name_list = ir_models_info['MAIN NAME'].values
        curve_name_list = ir_models_info['TOOL NAME'].values
        reference_date_list = ir_models_info['REFERENCE DATE'].values
        smooth_volatility_list = ir_models_info['SMOOTH VOLATILITY'].values
        
        list_forecast_curves = []
        
        for i in range(len(curve_name_list)):
            df_curve_result = pd.read_csv(f'evidence_curves/results_vasicek.csv', index_col = 0)
            df_curve_result['date_ref'] = pd.to_datetime(df_curve_result['date_ref'])
            
            df_curve_result = df_curve_result[(df_curve_result['var_name'] == curve_name_list[i]) & (df_curve_result['date_ref'] == pd.to_datetime(reference_date_list[i])) & (df_curve_result['smooth_volatility'] == smooth_volatility_list[i])]
            
            df_curve_result.index = pd.to_datetime(df_curve_result.index)
            list_forecast_curves.append(df_curve_result[['P1', 'P99', 'forecast']])
        
        # Crear lista de DataFrames de datos históricos
        list_hist_curves = []
        
        for curve_name in curve_name_list:
            df_curve_hist = pd.read_excel('data/processed/variables_cleaned.xlsx', sheet_name = curve_name, index_col = 0)
            df_curve_hist = df_curve_hist.resample('Q').last()
            
            # Calcular la fecha límite: primer registro - 3 meses + 2 días
            day_filt_hist = pd.to_datetime(list_forecast_curves[0].index[0]) + pd.DateOffset(months = -3, days = 2)
            df_curve_hist = df_curve_hist[df_curve_hist.index <= day_filt_hist]
            
            list_hist_curves.append(df_curve_hist)
            
        # Crear lista con los pronósticos de las curvas
        for hist_df, forecast_df, curve_name in zip(list_hist_curves, list_forecast_curves, var_deliery_name_list):
            
            fig_curve = plot_fanchart_plotly(
                data            = hist_df,
                percentile_df   = forecast_df,
                fitted_values   = None,
                len_hist_plot   = 30,
                percentile_list = ['P1', 'P99'],
                title           = f"Forecast Trimestral: {curve_name}",
                last_value_text = False,
                color_theme     = "purple",
                final_plot      = True
            )
            
            # Resample para obtener los datos anuales
            year_curves = forecast_df.resample('Y').last()
            
            new_percentiles = ['P1', 'P99']
            
            # Gráfico fanchart con intervalos de confianza
            st.plotly_chart(fig_curve, use_container_width = True)

        
            col30, col40 = st.columns(2)
            with col30:
                st.dataframe(
                    forecast_df.style.format({col: "{:.5f}" for col in forecast_df.columns}).background_gradient(cmap="BuPu")
                )
            with col40:
                st.dataframe(
                    forecast_df.resample('Y').last().style.format({col: "{:.5f}" for col in forecast_df.columns}).background_gradient(cmap="BuPu")
                )

            quarter_df = pd.DataFrame({'Escenario Adverso': forecast_df[new_percentiles[0]], 'Escenario Optimista': forecast_df[new_percentiles[1]], 'Pronóstico': forecast_df['forecast'], 'Variable': curve_name})
            year_df = pd.DataFrame({'Escenario Adverso': year_curves[new_percentiles[0]], 'Escenario Optimista': year_curves[new_percentiles[1]], 'Pronóstico': year_curves['forecast'], 'Variable': curve_name})
            
            quarter_df.index.name = 'date'
            year_df.index.name = 'date'
            year_df.index = year_df.index.year # Convertir el índice a tipo datetime
            
            # Añadir los DataFrames a las listas correspondientes
            quarter_forecast_list.append(quarter_df)
            annual_forecast_list.append(year_df)
                
  
    final_quarterly = pd.concat(quarter_forecast_list)
    final_yearly = pd.concat(annual_forecast_list)
    
    if "IPC" in var_name_list or "IPCA" in var_name_list or "IPCS" in var_name_list:
        final_monthly = pd.concat(monthly_forecast_list)
    

    # 2) Hacemos un melt para pasar las columnas de escenarios a una sola columna 'Escenario'
    df_melted = final_quarterly.reset_index().melt(
        id_vars=['date', 'Variable'],  # columnas que se mantienen fijas
        value_vars=['Escenario Adverso', 'Pronóstico', 'Escenario Optimista'],
        var_name='Escenario',
        value_name='Valor'
    )

    # 3) Pivotamos para que cada valor de 'Variable' sea una columna
    df_pivot = df_melted.pivot_table(
        index=['date', 'Escenario'],
        columns='Variable',
        values='Valor',
        # En caso de duplicados, pivot_table por defecto hace un promedio;
        # puedes cambiar aggfunc si lo deseas (por ejemplo, aggfunc='first')
        aggfunc='mean'
    ).reset_index()

    # 4) Ordenamos la columna 'Escenario' en el orden deseado
    orden_escenarios = ['Escenario Adverso', 'Pronóstico', 'Escenario Optimista']
    df_pivot['Escenario'] = pd.Categorical(df_pivot['Escenario'], categories=orden_escenarios, ordered=True)
    df_pivot = df_pivot.sort_values(['date', 'Escenario']).reset_index(drop=True)

    # 5) Renombramos la columna 'date' a 'fecha'
    df_pivot.rename(columns={'date': 'fecha'}, inplace=True)
    
    # 5) Definimos el orden de los escenarios: Adverso -> Optimista -> Pronóstico
    orden_escenarios = ['Escenario Adverso', 'Pronóstico', 'Escenario Optimista']
    df_pivot['Escenario'] = pd.Categorical(
        df_pivot['Escenario'], 
        categories=orden_escenarios, 
        ordered=True
    )

    # 6) Ordenamos primero por escenario y luego por fecha
    df_pivot = df_pivot.sort_values(['Escenario', 'fecha']).reset_index(drop=True)

    # Fecha en el índice
    df_pivot.set_index('fecha', inplace=True)
    
    df_pivot.index.name = 'date'

    # Calcular diferenciales de curvas si estas están en la lista
    if 'BONO_10Y_ESP' in df_pivot.columns and 'BONO_10Y_ALE' in df_pivot.columns and 'BONO_10Y_ITA' in df_pivot.columns and 'BONO_10Y_POR' in df_pivot.columns:
        # Calcular los diferenciales de las curvas
        df_pivot['DIFF_ESP_ALE'] = df_pivot['BONO_10Y_ESP'] - df_pivot['BONO_10Y_ALE']
        df_pivot['DIFF_ITA_ALE'] = df_pivot['BONO_10Y_ITA'] - df_pivot['BONO_10Y_ALE']
        df_pivot['DIFF_POR_ALE'] = df_pivot['BONO_10Y_POR'] - df_pivot['BONO_10Y_ALE']
        
    # Para final_yearly
    # 1) Realizamos el melt para pasar las columnas de escenarios a una sola columna 'Escenario'
    df_melted_yearly = final_yearly.reset_index().melt(
        id_vars=['date', 'Variable'],  # columnas fijas
        value_vars=['Escenario Adverso', 'Pronóstico', 'Escenario Optimista'],
        var_name='Escenario',
        value_name='Valor'
    )

    # 2) Pivotamos para que cada valor de 'Variable' se convierta en una columna
    df_pivot_yearly = df_melted_yearly.pivot_table(
        index=['date', 'Escenario'],
        columns='Variable',
        values='Valor',
        aggfunc='mean'
    ).reset_index()

    # 3) Definimos el orden deseado de escenarios y ordenamos
    df_pivot_yearly['Escenario'] = pd.Categorical(df_pivot_yearly['Escenario'], categories=orden_escenarios, ordered=True)
    df_pivot_yearly = df_pivot_yearly.sort_values(['date', 'Escenario']).reset_index(drop=True)

    # 4) Renombramos la columna 'date' a 'fecha'
    df_pivot_yearly.rename(columns={'date': 'fecha'}, inplace=True)

    # 5) (Opcional) Reafirmamos el orden de los escenarios y ordenamos por 'Escenario' y 'fecha'
    df_pivot_yearly['Escenario'] = pd.Categorical(df_pivot_yearly['Escenario'], categories=orden_escenarios, ordered=True)
    df_pivot_yearly = df_pivot_yearly.sort_values(['Escenario', 'fecha']).reset_index(drop=True)
    
    df_pivot_yearly.set_index('fecha', inplace=True)
    df_pivot_yearly.index.name = 'date'
    
    if 'IPC' in var_name_list or 'IPCA' in var_name_list or 'IPCS' in var_name_list:
        # Procesamiento para final_monthly

        # 1) Realizamos el melt para pasar las columnas de escenarios a una sola columna 'Escenario'
        df_melted_monthly = final_monthly.reset_index().melt(
            id_vars=['date', 'Variable'],  # columnas fijas
            value_vars=['Escenario Adverso', 'Pronóstico', 'Escenario Optimista'],
            var_name='Escenario',
            value_name='Valor'
        )

        # 2) Pivotamos para que cada valor de 'Variable' se convierta en una columna
        df_pivot_monthly = df_melted_monthly.pivot_table(
            index=['date', 'Escenario'],
            columns='Variable',
            values='Valor',
            aggfunc='mean'
        ).reset_index()

        # 3) Definimos el orden deseado de escenarios y ordenamos
        df_pivot_monthly['Escenario'] = pd.Categorical(df_pivot_monthly['Escenario'], categories=orden_escenarios, ordered=True)
        df_pivot_monthly = df_pivot_monthly.sort_values(['date', 'Escenario']).reset_index(drop=True)

        # 4) Renombramos la columna 'date' a 'fecha'
        df_pivot_monthly.rename(columns={'date': 'fecha'}, inplace=True)

        # 5) Reafirmamos el orden de los escenarios y ordenamos por 'Escenario' y 'fecha'
        df_pivot_monthly['Escenario'] = pd.Categorical(df_pivot_monthly['Escenario'], categories=orden_escenarios, ordered=True)
        df_pivot_monthly = df_pivot_monthly.sort_values(['Escenario', 'fecha']).reset_index(drop=True)

        # 6) Colocamos la fecha como índice
        df_pivot_monthly.set_index('fecha', inplace=True)
        df_pivot_monthly.index.name = 'date'

    
    # Función para exportar un DataFrame a Excel con formato personalizado
    def to_excel_with_format(df, sheet_name, sep_miles, sep_decimales):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=True, sheet_name=sheet_name)
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]
            # Si el usuario seleccionó separadores diferentes a los por defecto
            if (sep_miles, sep_decimales) != (",", "."):
                # Ejemplo: si se elige "." para miles y "," para decimales, se usa el formato "#.##0,00"
                if sep_miles == "." and sep_decimales == ",":
                    custom_format_str = "#.##0,00"
                else:
                    # En otros casos, se puede ajustar el formato o mantener el por defecto
                    custom_format_str = "#,##0.00"
                num_format = workbook.add_format({'num_format': custom_format_str})
                # Se aplica el formato a todas las columnas de tipo numérico
                for idx, col in enumerate(df.columns):
                    if pd.api.types.is_numeric_dtype(df[col]):
                        worksheet.set_column(idx, idx, None, num_format)
        output.seek(0)
        return output    

    st.markdown("## REPORTS")
    # Aquí se puede definir o llamar a la función section_divider() que se usa para separar secciones
    section_divider()
    
        
    # Seleccionar separadores para miles y decimales
    st.markdown("Select thousands and decimal separators:")
    sep_miles = st.selectbox("Thousands separator", options=[",", "."])
    sep_decimales = st.selectbox("Decimal separator", options=[".", ","])  

    st.markdown("#### QUARTERLY")
    section_divider()
    df_report = df_pivot.dropna()
    st.dataframe(df_report)
    buffer = to_excel_with_format(df_report, "Trimestral", sep_miles, sep_decimales)
    st.download_button(
        label="Download Quarterly (.xlsx)",
        data=buffer,
        file_name="reporte_trimestral.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.markdown("#### YEARLY")
    section_divider()
    df_report = df_pivot_yearly.dropna()
    st.dataframe(df_report)
    buffer = to_excel_with_format(df_report, "Anual", sep_miles, sep_decimales)
    st.download_button(
        label="Download Annual (.xlsx)",
        data=buffer,
        file_name="reporte_anual.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    if 'IPC' in var_name_list or 'IPCA' in var_name_list or 'IPCS' in var_name_list:
        st.markdown("#### MONTHLY")
        section_divider()
        df_report = df_pivot_monthly.dropna()
        st.dataframe(df_report)
        buffer = to_excel_with_format(df_report, "Mensual", sep_miles, sep_decimales)
        st.download_button(
            label="Download Monthly (.xlsx)",
            data=buffer,
            file_name="reporte_mensual.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


# =============================================================================
# EJECUTAR APP
# =============================================================================  
def footer():
    st.markdown('<div class="footer">© 2025 Intermoney Consultoría. All rights reserved.</div>', unsafe_allow_html = True)

if selected == 'Macroeconomical Variables':
    macro()
elif selected == 'Interest Rate Curves':
    curves()
elif selected == 'Communities':
    comunidades()
elif selected == 'Delivery':
    delivery()
else:
    st.error("Please select a valid option.")
    
footer()