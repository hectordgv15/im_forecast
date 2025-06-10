# Librerías
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator

# Parámetros
NON_WEIGHTED = {'OCUP', 'PACT'}
COM_VARS = ['PIB', 'OCUP', 'PACT', 'PARO', 'VIV_VENT']
IND_VARS = ['PIB_REAL', 'OCUP_EPA', 'POB_ACT_EPA', 'TASA_PARO', 'P_VIV_VENTA']
HIST_SHEETS = [
    "clean_ts", "ts_df", "df_abs", "df_abs_y", "df_pct",
    "df_interannual", "df_avg_year", "df_last_year"
]
METHODS = ['hw', 'sarima', 'prophet', 'ucm', 'ardl', 'var']
PERIODS = {
    'abs_q': 'Abs trimestral',
    'abs_y': 'Abs anual',
    'pct_q': 'Pct trimestral',
    'interannual': 'Interanual',
    'interannual_avg': 'Interanual avg',
    'interannual_last': 'Interanual last'
}


# Para cada variable común ↔ índice
for com, ind in zip(COM_VARS, IND_VARS):
    
    print(f"Procesando {com} para {ind} ...")
    
    # cargar comunidades y weights_values
    df_w = pd.read_excel('weights.xlsx', sheet_name=com).dropna(subset=['Comunidades', 'W'])
    comunidades = df_w['Comunidades'].tolist()
    weights_values = df_w['W'].tolist()
    
    # cargar historiales globales
    global_info = {
        sheet: pd.read_excel(f'evidence_macro/evidence_{ind}.xlsx', sheet_name=sheet, index_col=0)
        for sheet in HIST_SHEETS
    }
    
    # función helper para leer todos los métodos/periodos de una comunidad
    def load_sheets(prefix):
        return {
            f"{m}_{suffix}": pd.read_excel(
                f'evidence_macro/evidence_{prefix}.xlsx',
                sheet_name=f"{m}_{suffix}", index_col=0
            )
            for m in METHODS
            for suffix in PERIODS
        }

    # cargar datos por comunidad
    data_por_com = [load_sheets(comu) for comu in comunidades]
    
    # helper de agregado (suma o suma ponderada)
    def aggregate(key):
        dfs = [d[key] for d in data_por_com]
        if com in NON_WEIGHTED and 'abs' in key:
            return sum(dfs)
        return sum(w * df for w, df in zip(weights_values, dfs))
    
    # construir dict de resultados
    final = {**global_info}
    for m in METHODS:
        for suf in PERIODS:
            key = f"{m}_{suf}"
            final[key] = aggregate(key)
    
    # exportar a Excel
    out_path = f"evidence_macro/evidence_W_{com}.xlsx" # NOTE: Revisar qué nombre es el más adecuado.
    with pd.ExcelWriter(out_path, engine = 'openpyxl') as writer:
        for sheet, df in final.items():
            df.to_excel(writer, sheet_name = sheet)



from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# ──────────────────────────────
#  Parámetros generales
# ──────────────────────────────
BASE_DIR = Path("evidence_macro")
FILES = {
    "di":  "evidence_DEMANDA_INTERNA.xlsx",
    "de":  "evidence_DEMANDA_EXTERNA.xlsx",
    "pib": "evidence_W_PIB.xlsx",
}

MODELS: List[str]   = ["hw", "sarima", "prophet", "ucm", "ardl", "var"]
METRICS: List[str]  = [
    "abs_q", "abs_y",
    "pct_q",
    "interannual", "interannual_avg", "interannual_last",
]

# Métrica → sufijo de la hoja PIB que le corresponde
PIB_MAP: Dict[str, str] = {
    "abs_q"            : "interannual",
    "abs_y"            : "interannual_last",
    "pct_q"            : "interannual",
    "interannual"      : "interannual",
    "interannual_avg"  : "interannual_last",
    "interannual_last" : "interannual_last",
}


# ──────────────────────────────
#  Funciones auxiliares
# ──────────────────────────────
def load_workbook(path: Path) -> Dict[str, pd.DataFrame]:
    """Lee todas las hojas de un workbook a un diccionario."""
    return pd.read_excel(path, sheet_name=None, index_col=0)

# 1. Ajuste proporcional (ratio-scaling)
# -------------------------------------
def reconcile_weighted_a(df_int: pd.DataFrame,
                         df_ext: pd.DataFrame,
                         df_gdp: pd.DataFrame
                         ):
    """
    Devuelve (df_int_adj, df_ext_adj).

    • Multiplica DI y DE por un factor λ = PIB / (DI+DE) en cada fecha-percentil.
    • eps evita divisiones por cero.
    """
    
    df_gdp = df_gdp.copy() * 100
    
    total = df_int + df_ext
    factor = df_gdp / total.replace(0, np.nan)       # NaN donde total=0
    factor = factor.fillna(0)                        # si total=0 ⇒ ajuste a 0
    df_int_adj = df_int * factor
    df_ext_adj = df_ext * factor
    return df_int_adj, df_ext_adj


# 2. Ajuste residual (uno fijo, otro corrige)
# -------------------------------------------
def reconcile_weighted_b(df_int: pd.DataFrame,
                         df_ext: pd.DataFrame,
                         df_gdp: pd.DataFrame,
                         adjust: str = "external"
                         ):
    """
    Devuelve (df_int_adj, df_ext_adj).

    • Si adjust='internal'  → corrige Demanda Interna y deja DE fija.
    • Si adjust='external'  → corrige Demanda Externa y deja DI fija (por defecto).
    """
    
    df_gdp = df_gdp.copy() * 100
    
    if adjust not in ("internal", "external"):
        raise ValueError("adjust debe ser 'internal' o 'external'.")

    if adjust == "internal":
        df_int_adj = df_gdp - df_ext
        df_ext_adj = df_ext.copy()
    else:  # adjust == 'external'
        df_int_adj = df_int.copy()
        df_ext_adj = df_gdp - df_int

    return df_int_adj, df_ext_adj


# 3. Mínimos cuadrados condicionados (CLS)
# ----------------------------------------
def reconcile_weighted_c(df_int: pd.DataFrame,
                         df_ext: pd.DataFrame,
                         df_gdp: pd.DataFrame,
                         weight_int = 1.0,
                         weight_ext = 1.0
                         ):
    """
    Devuelve (df_int_adj, df_ext_adj).

    Minimiza  w_I·ΔI² + w_E·ΔE²  sujeto a  (I+ΔI)+(E+ΔE)=PIB.
    Los pesos pueden ser escalares o DataFrames con la misma forma.
    """
    
    df_gdp = df_gdp.copy() * 100    

    # Asegura broadcast correcto de los pesos
    w_I = weight_int if isinstance(weight_int, pd.DataFrame) else pd.DataFrame(weight_int,
                                                                               index=df_int.index,
                                                                               columns=df_int.columns)
    w_E = weight_ext if isinstance(weight_ext, pd.DataFrame) else pd.DataFrame(weight_ext,
                                                                               index=df_int.index,
                                                                               columns=df_int.columns)

    delta = df_gdp - (df_int + df_ext)
    denom = w_I + w_E
    # Evita división por cero (si ambos pesos = 0)
    denom = denom.replace(0, np.nan)

    df_int_adj = df_int + delta * (w_E / denom)
    df_ext_adj = df_ext + delta * (w_I / denom)

    return df_int_adj, df_ext_adj


# ──────────────────────────────
#  Carga de libros
# ──────────────────────────────
dict_di  = load_workbook(BASE_DIR / FILES["di"])
dict_de  = load_workbook(BASE_DIR / FILES["de"])
dict_pib = load_workbook(BASE_DIR / FILES["pib"])

# ──────────────────────────────
#  Conciliación en bloque
# ──────────────────────────────
di_adj: Dict[str, pd.DataFrame] = {}
de_adj: Dict[str, pd.DataFrame] = {}

for model in MODELS:
    for metric in METRICS:
        sheet = f"{model}_{metric}"
        # Comprobamos que existan las hojas en los tres libros
        if (
            sheet in dict_di
            and sheet in dict_de
            and f"{model}_{PIB_MAP[metric]}" in dict_pib
        ):
            di_adj[sheet], de_adj[sheet] = reconcile_weighted_c(
                dict_di[sheet],
                dict_de[sheet],
                dict_pib[f"{model}_{PIB_MAP[metric]}"],
            )

# Actualizamos los diccionarios originales con los ajustados
dict_di.update(di_adj)
dict_de.update(de_adj)

# Exportar cada diccionario en hojas en un solo archivo
def export_to_excel(data: Dict[str, pd.DataFrame], filename: str):
    """Exporta un diccionario de DataFrames a un archivo Excel."""
    with pd.ExcelWriter(BASE_DIR / filename, engine='openpyxl') as writer:
        for sheet_name, df in data.items():
            df.to_excel(writer, sheet_name=sheet_name)

export_to_excel(dict_di, "evidence_ADJ_DEMANDA_INTERNA.xlsx")
export_to_excel(dict_de, "evidence_ADJ_DEMANDA_EXTERNA.xlsx")


# Mensualizar datos del IPC
# -------------------------------
def quarterly_to_monthly(
    df_quarterly: pd.DataFrame,
    *,
    ruido_scale = None,
    clip_min = None,
    random_state = 42
) -> pd.DataFrame:
    """
    Convierte un DataFrame con frecuencia trimestral a mensual mediante
    interpolación PCHIP (Piecewise Cubic Hermite).

    Parámetros
    ----------
    df_quarterly : pd.DataFrame
        Índice: fechas trimestrales (DatetimeIndex).
        Columnas: percentiles o cualquier serie numérica.

    ruido_scale : float, opcional
        Desviación típica del ruido gaussiano que se añadirá
        a las series mensualizadas.  Si es None (por defecto),
        no se añade ruido.

    clip_min : float, opcional
        Límite inferior para recortar los valores tras añadir ruido.
        Póngalo a None para no recortar.

    random_state : int, opcional
        Semilla para reproducibilidad del ruido.

    Devuelve
    --------
    pd.DataFrame
        DataFrame mensual (fin de mes) con las mismas columnas que
        ``df_quarterly``.
    """
    if not isinstance(df_quarterly.index, pd.DatetimeIndex):
        raise TypeError("El índice debe ser un DatetimeIndex.")

    # Índice mensual de fin de mes
    idx_m = pd.date_range(df_quarterly.index.min(),
                        df_quarterly.index.max(),
                        freq="M")

    # Interpolación PCHIP columna por columna
    def _interp(col: pd.Series) -> pd.Series:
        x = col.index.view(np.int64)          # nanosegundos desde época
        f = PchipInterpolator(x, col.values)  # ajusta el interpolador
        y = f(idx_m.view(np.int64))           # evalúa en mensual
        return pd.Series(y, index=idx_m)

    df_m = df_quarterly.apply(_interp)
    
    # Opcional: añadir ruido gaussiano
    if ruido_scale is not None and ruido_scale > 0:
        rng = np.random.default_rng(seed=random_state)
        ruido = rng.normal(0.0, ruido_scale, size=df_m.shape)
        
        mask = ~df_m.index.is_quarter_end
        df_m = df_m + ruido * mask[:, None]

        if clip_min is not None:
            df_m = df_m.clip(lower=clip_min)

    return df_m

sheet_names = [
    "hw_interannual",
    "sarima_interannual",
    "prophet_interannual",
    "ucm_interannual",
    "ardl_interannual",
    "var_interannual",
]

# ------------------------------------------------------------------
# 1) Cargamos todas las hojas, mensualizamos y las guardamos en memoria
# ------------------------------------------------------------------
FILE_IPC = BASE_DIR / "evidence_IPC.xlsx"
FILE_IPCA = BASE_DIR / "evidence_IPCA.xlsx"
FILE_IPCS = BASE_DIR / "evidence_IPCS.xlsx"


def monthly_process(FILE_PATH: Path):
    mensuales = {}                     # dict {nombre_hoja: DataFrame}
    for hoja in sheet_names:
        df_q = pd.read_excel(FILE_PATH, sheet_name=hoja, index_col=0, parse_dates=True)
        df_m = quarterly_to_monthly(df_q, ruido_scale=0.0006)  # ajusta args a tu gusto
        df_m.index.name = "date"
        mensuales[f"{hoja}_monthly"] = df_m

    # ------------------------------------------------------------------
    # 2) Escribimos las nuevas hojas en el MISMO fichero
    #    - mode="a"   → abrir en modo append
    #    - if_sheet_exists="replace"  → sobrescribir si ya existía
    #    - engine="openpyxl"          → necesario para modo append
    # ------------------------------------------------------------------
    with pd.ExcelWriter(FILE_PATH, engine="openpyxl", mode="a", if_sheet_exists="replace",
                        datetime_format="yyyy-mm-dd") as writer:
        for hoja, df_m in mensuales.items():
            df_m.to_excel(writer, sheet_name=hoja)

    print("Mensualización completada y guardada en el mismo archivo")
    
monthly_process(FILE_IPC)
monthly_process(FILE_IPCA)
monthly_process(FILE_IPCS)