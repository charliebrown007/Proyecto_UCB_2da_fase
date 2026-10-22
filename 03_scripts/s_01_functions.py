# =========================================================================================================
# 01. LIBRERÍAS

# Medición de tiempo
from glob import glob
from matplotlib.pyplot import viridis

# Gestión de directorios
from pathlib import Path
from typing import Literal

# Gráficos
import matplotlib.pyplot as plt
import numpy as np

# Manipulación de datos y derivación de gráficos
import polars as pl
import seaborn as sns
from matplotlib import ticker
from pyprojroot import here
import glob

# =========================================================================================================
# 02. CARGA DE DATOS

# Función de carga de archivo de texto

def load_data_own(file_folder_param: str) -> pl.LazyFrame | None:

    print(f'Iniciando carga de datos locales desde: {file_folder_param}')

    try:
        df = (
            pl.scan_csv(
                source = file_folder_param,
                has_header =  True,
                separator = ',',
                null_values = ['',' ','NA','na','NULL','null','None','NONE','Na'],
                encoding = 'utf8',
                infer_schema = True,
                schema_overrides = {'bottle_volume_ml': pl.String, 'store_no': pl.String}
            )
        )
        
        return df

    except (FileNotFoundError, pl.exceptions.ComputeError):
        print(f"❌ Error: No se encontró la ruta o archivos en: ({file_folder_param}).")
        return None
    except pl.exceptions.PolarsError as e:
        print(f"❌ Error de Polars al procesar el CSV: {e}")
        return None
    except Exception as e:  # noqa: BLE001
        print(f"❌ Error inesperado: {e}")
        return None

# =========================================================================================================
# 03. LIMPIEZA DE DATOS
# Función de limpieza de datos

def limpiar_texto_own(*, column_param: pl.Expr) -> pl.Expr:

    return (
        column_param
        
        # 1. Convertir a mayúsculas
        .str.to_uppercase()

        # 2. Normalizar para separar los
        # acentos de las letras (Ej: 'á' -> 'a' + '´')
        .str.normalize('NFD')

        # 3. Borramos acentos, pero NO la línea de la ñ (\u0303)
        .str.replace_all('[\\u0300-\\u0302\\u0304-\\u036f]', '')

        # 4. Recomponer caracteres a estándar Unicode (e.g. 'N' + '~' -> 'Ñ')
        .str.normalize('NFC')

        # 5. Reemplazar múltiples espacios por uno solo
        .str.replace_all('\\s+', ' ')
        
        # 6. Quitar espacios al inicio y al final
        .str.strip_chars()
    )

# =========================================================================================================
# 04. ESTADÍSTICOS DESCRIPPTIVOS

def describe_extended(*, df_param: pl.DataFrame, col_name_param: str, by_cat_param: str) -> pl.DataFrame:

    if not by_cat_param:
        col = pl.col(col_name_param)
        stats = (
            df_param
            .select(
                count    = col.len().cast(pl.Float64),
                nulls    = col.null_count().cast(pl.Float64),
                mean     = col.mean(),
                std      = col.std(),
                cv       = (col.std() / col.mean()),
                min      = col.min().cast(pl.Float64),
                p25      = col.quantile(0.25),
                median   = col.median(),
                p75      = col.quantile(0.75),
                p95      = col.quantile(0.95),
                max      = col.max().cast(pl.Float64),
                skew     = col.skew(),
                kurtosis = col.kurtosis()) )
        return stats.unpivot(variable_name = 'statistic', value_name = col_name_param)
    else:
        col = pl.col(col_name_param)
        stats = (
            df_param
            .group_by(by_cat_param)
            .select(
                count    = col.len().cast(pl.Float64),
                nulls    = col.null_count().cast(pl.Float64),
                mean     = col.mean(),
                std      = col.std(),
                cv       = (col.std() / col.mean()),
                min      = col.min().cast(pl.Float64),
                p25      = col.quantile(0.25),
                median   = col.median(),
                p75      = col.quantile(0.75),
                p95      = col.quantile(0.95),
                max      = col.max().cast(pl.Float64),
                skew     = col.skew(),
                kurtosis = col.kurtosis()) )
        return stats.unpivot(variable_name = 'statistic', value_name = col_name_param)


# =========================================================================================================
# 05. VISUALIZACIÓN DE GRÁFICOS

# Gráfico de barras para una variable categórica
def plot_cat_function_01(
    *,
    df_param: pl.DataFrame,
    column_param: str,
    calculate_param: Literal['count_', 'sum_bottles'] = 'count_',
    top_param: int | None = None,
    suptitle_param: str,
    text_param: str,
    file_param: str,
    show_param: bool = True ) -> Path:
    
    # 1. Tabla de frecuencias

    match calculate_param:
        case 'count_':
            aux_analysis = ( df_param
            .get_column(column_param)
            .value_counts(name = 'count_0', sort = True)
            .with_columns( (pl.col('count_0') / pl.sum('count_0')).alias('ptc_0')) )
        
        case 'sum_bottles':
            aux_analysis = ( df_param
            .group_by(column_param)
            .agg(pl.sum('sales_bottles').alias('count_0'))
            .with_columns((pl.col('count_0') / pl.sum('count_0')).alias('ptc_0'))
            .sort( by = ['count_0'], descending = [True]) )
        case _:
            raise ValueError("Criterio no válido")
    

    if top_param is not None:
        aux_analysis = aux_analysis.head(top_param)

    
    # 2. Tema general con Seaborn
    sns.set_theme( context = 'paper', style = 'white',
    rc = {'axes.grid': True, 'grid.linestyle': '--', 'grid.alpha': 0.4})

    # 3. Gráfico de barras
    g = sns.catplot(
        data = aux_analysis,
        x = 'count_0',
        y = column_param,
        kind = 'bar',
        color = '#DCDCDC',
        edgecolor = '#777777',
        alpha = 0.6,
        height = 5, # Altura en pulgadas
        aspect = 1.2, # Ancho = 5 * 1.2 = 6 pulgadas
    )

    # 4. Ajustes directos en el objeto FacetGrid de Seaborn
    g.despine(left = True, bottom = False)
    g.set_axis_labels( x_var = 'Frecuencia', y_var = '' )
    g.figure.subplots_adjust(top = 0.90)

    g.figure.suptitle(
        suptitle_param,
        x = 0.08,
        y = 1.00,
        ha = 'left',
        weight = 'bold',
        size = 11
    )

    g.figure.text(
        x = 0.08,
        y = 0.94,
        s = text_param,
        ha = 'left',
        fontsize = 10,
        color = '#666666'
    )

    # 5. Formato al eje X y etiquetas de barras

    etiquetas_barras = [
        f"{cnt:,.0f} ({pct:.1%})"
        for cnt, pct in zip(aux_analysis['count_0'], aux_analysis['ptc_0'])
    ]

    for ax in g.axes.flat:
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        ax.tick_params(axis = 'x', bottom = True, length = 4, color = '#777777')
        ax.spines['bottom'].set_color('#777777')
        ax.bar_label(ax.containers[0], labels = etiquetas_barras, padding = 5, fontsize = 8)

    # Exportar gráfico
    file_destiny = here() / '05_graphs' / file_param

    g.figure.savefig(
        file_destiny,
        dpi = 300,
        bbox_inches = 'tight',
        facecolor = 'white'
    )

    if show_param:
        plt.show()
    else:
        plt.close(g.figure)
        
    print(f'Guardado como: "{file_param}" en 05_graphs')
    return file_destiny


# Histograma para una variable numérica en escala logarítmica para el eje x

def plot_dist_function_02(
    *,
    df_param: pl.DataFrame,
    df_var: str,
    bins_param: int,
    suptitle_param: str,
    text_param: str,
    file_param: str,
    show_param: bool = True ) -> Path:

    sns.set_theme( context = 'paper',
        style = 'white',
        rc = {'axes.grid': True, 'grid.linestyle': '--', 'grid.alpha': 0.3})

    g = sns.displot(
        data = df_param,
        x = df_var,
        kind = 'hist',
        bins = bins_param,
        log_scale = (True, False),       # Escala logarítmica en base 10: eje "x" y eje "y"
        color = '#DCDCDC',
        edgecolor = '#777777',
        alpha = 0.5,
        height = 5,
        aspect = 1.2 )

    g.despine(left = True, bottom = False)
    g.set_axis_labels( x_var = 'Escala logarítmica (base 10)', y_var = 'Frecuencia' )
    g.figure.subplots_adjust(top = 0.90)

    g.figure.suptitle(
            suptitle_param,
            x = 0.08,
            y = 1.00,
            ha = 'left',
            weight = 'bold',
            size = 11 )

    g.figure.text(
        x = 0.08,
        y = 0.94,
        s = text_param,
        ha = 'left',
        fontsize = 10,
        color = '#666666' )

    for ax in g.axes.flat:
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        ax.tick_params(axis = 'x', bottom = True, length = 4, color = '#777777')
        ax.spines['bottom'].set_color('#777777')

    # Exportar gráfico
    file_destiny = here() / '05_graphs' / file_param

    g.figure.savefig(
        file_destiny,
        dpi = 300,
        bbox_inches = 'tight',
        facecolor = 'white'
    )

    if show_param:
        plt.show()
    else:
        plt.close(g.figure)
        
    print(f'Guardado como: "{file_param}" en 05_graphs')
    return file_destiny

# Diagrama de dispersión (scatterplot)

def plot_rel_function_03(
    *,
    df_param: pl.DataFrame,
    x_param: str,
    x_title_param: str,
    y_title_param: str,
    y_param: str,
    hue_param: str,
    hue_tag_param: str,
    suptitle_param: str,
    text_param: str,
    file_param: str,
    show_param: bool = True ) -> Path:
    
    
    # 1. Tema general con Seaborn
    sns.set_theme( context = 'paper', style = 'white',
    rc = {'axes.grid': True, 'grid.linestyle': '--', 'grid.alpha': 0.4})

    # 2. Gráfico de barras
    g = sns.relplot(
        data = df_param,
        x = x_param,
        y = y_param,
        kind = 'scatter',
        # color = '#DCDCDC',
        # edgecolor = '#777777',
        # col = col_param,
        # col_wrap = 2,
        edgecolor = None,
        hue = hue_param,
        palette = 'viridis_r',
        alpha = 0.6,
        height = 5, # Altura en pulgadas
        aspect = 1.2, # Ancho = 5 * 1.2 = 6 pulgadas
    )

    # 3. Ajustes directos en el objeto FacetGrid de Seaborn
    g.despine(left = False, bottom = False)
    g.set_axis_labels( x_var = x_title_param, y_var = y_title_param )
    g.figure.subplots_adjust(top = 0.90)

    g.figure.suptitle(
        suptitle_param,
        x = 0.08,
        y = 1.00,
        ha = 'left',
        weight = 'bold',
        size = 11 )

    g.figure.text(
        x = 0.08,
        y = 0.94,
        s = text_param,
        ha = 'left',
        fontsize = 10,
        color = '#666666' )

    for ax in g.axes.flat:
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        ax.tick_params(axis = 'both', bottom = True, left = True, length = 4, color = '#777777')
        ax.spines['bottom'].set_color('#777777')
        ax.spines['left'].set_color('#777777')

    sns.move_legend(
        g,
        loc = 'upper right',
        title = hue_tag_param )

    # Exportar gráfico
    file_destiny = here() / '05_graphs' / file_param

    g.figure.savefig(
        file_destiny,
        dpi = 300,
        bbox_inches = 'tight',
        facecolor = 'white'
    )

    if show_param:
        plt.show()
    else:
        plt.close(g.figure)
        
    print(f'Guardado como: "{file_param}" en 05_graphs')
    return file_destiny



# =========================================================================================================
# 06. MODELO DE LA CANTIDAD ECONÓMICA DEL PEDIDO (BASELINE EOQ).

def eoq_rop_baseline(
    df_demanda_diaria: pl.DataFrame, 
    precio_u_param: float,
    pack_param: int,
    costo_pedido_S: float = 50.0,
    tasa_mantenimiento_i: float = 0.20,
    dias_year: int = 365,
    lead_time_dias: int = 10,        # Días que tarda el camión en llegar
    nivel_confianza_z: float = 1.65  # Valor Z para 95% de nivel de servicio
) -> tuple[int, int]:
    '''
    Calcula la Cantidad Económica de Pedido (EOQ) en lotes y el 
    Punto de Reorden (ROP) estadístico para el modelo base (baseline).
    '''
    
    # ---------------------------------------------------------
    # 1. CÁLCULO DEL EOQ (Cantidad a pedir)
    # ---------------------------------------------------------
    demanda_total_periodo = df_demanda_diaria['sales_bottles'].sum()
    dias_simulacion = df_demanda_diaria.height

    demanda_anual_D = (demanda_total_periodo / dias_simulacion) * dias_year
    costo_mantenimiento_H = precio_u_param * tasa_mantenimiento_i

    eoq_botellas = np.sqrt((2 * demanda_anual_D * costo_pedido_S) / costo_mantenimiento_H)
    eoq_packs = int(np.ceil(eoq_botellas / pack_param))

    # ---------------------------------------------------------
    # 2. CÁLCULO DEL ROP (Punto de Reorden con Stock de Seguridad)
    # ---------------------------------------------------------
    demanda_diaria_promedio = df_demanda_diaria['sales_bottles'].mean()
    desviacion_diaria = df_demanda_diaria['sales_bottles'].std()

    # Stock de Seguridad = Z * Desviación_Estándar * sqrt(Lead_Time)
    stock_seguridad = nivel_confianza_z * desviacion_diaria * np.sqrt(lead_time_dias)
    
    # ROP = (Demanda Diaria * Lead Time) + Stock de Seguridad
    rop_botellas = (demanda_diaria_promedio * lead_time_dias) + stock_seguridad
    rop_botellas_redondeado = int(np.ceil(rop_botellas))

    # ---------------------------------------------------------
    # 3. CONSOLA Y RETORNO
    # ---------------------------------------------------------
    print('-' * 62)
    print('RESULTADOS DEL MODELO TEÓRICO (EOQ + ROP)')
    print('-' * 62)
    print(f'EOQ Ajustado a Lotes (Q*) : {eoq_packs:,.0f} packs')
    print(f'Punto de Reorden (ROP)    : {rop_botellas_redondeado:,.0f} botellas')
    print(f'  ↳ Demanda durante viaje : {(demanda_diaria_promedio * lead_time_dias):,.0f} botellas')
    print(f'  ↳ Stock de Seguridad    : {stock_seguridad:,.0f} botellas')
    print('-' * 62)
    
    # Devuelve la tupla con ambos valores
    return eoq_packs, rop_botellas_redondeado