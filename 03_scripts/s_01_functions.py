# =========================================================================================================
# 01. LIBRERÍAS

# Medición de tiempo
import time

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

# =========================================================================================================
# 02. CARGA DE DATOS

# Función de carga de archivo de texto

def load_data_own(file_path_param: str) -> pl.DataFrame | None:

    print(f'Iniciando carga de datos locales desde: {file_path_param}')
    inicio_tiempo = time.time()

    try:
        df = (
            pl.read_csv(
                source = here()/'00_raw_data'/file_path_param,
                has_header =  True,
                separator = ',',
                null_values = ['',' ','NA','na','NULL','null','None','NONE','Na'],
                encoding = 'utf8',
                infer_schema = True,
                schema_overrides = {'bottle_volume_ml': pl.String, 'store_no': pl.String}
            )
            .pipe(lambda dataframe: dataframe.select(sorted(dataframe.columns)))
        )
        
        tiempo_carga = time.time() - inicio_tiempo
        print(f'Carga completada: {df.height} registros en {tiempo_carga:.2f} segundos.')
        
        return df

    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo en la ruta especificada ({file_path_param}).")
        print("Asegurar de que el archivo CSV esté en la carpeta '00_raw_data'.")
        return None
    except pl.exceptions.PolarsError as e:
        print(f"❌ Error de Polars al procesar el CSV: {e}")
        return None
    except Exception as e:  # noqa: BLE001 (No quality assurance - blind exception)
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
# 04. VISUALIZACIÓN DE GRÁFICOS

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


# Histograma para una varaible numérica en escala logarítmica para el eje x

def plot_cat_function_02(
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


# =========================================================================================================
# 04. MODELO DE LA CANTIDAD ECONÓMICA DEL PEDIDO (BASELINE EOQ).

def eoq_baseline(
    df_demanda: pl.DataFrame, 
    df_precio: pl.DataFrame, 
    dias_year: int = 365, 
    costo_pedido_S: float = 50.0, 
    tasa_mantenimiento_i: float = 0.20 ) -> float:
    '''
    Calcula la Cantidad Económica de Pedido (EOQ) ajustada a lotes (cajas) 
    para establecer el modelo base (baseline) de la simulación.
    
    Parámetros:
    - df_demanda (pl.DataFrame): DataFrame con la demanda histórica diaria.
    - df_precio (pl.DataFrame): DataFrame con el precio medio y el tamaño del pack.
    - dias_ano (int): Días base para la anualización.
    - costo_pedido_S (float): Costo fijo por emitir una orden ($).
    - tasa_mantenimiento_i (float): Porcentaje anual del valor del producto para costo de inventario.
    
    Retorna:
    - float: Cantidad de cajas a pedir sugerida por el modelo EOQ.
    '''
    
    # 1. Extraer datos reales del producto desde los DataFrames
    demanda_total_periodo = df_demanda['demanda_diaria'].sum()
    precio_unitario_C = df_precio['precio_medio'][0]
    
    # Extraer el tamaño del pack (manejando el caso en el que venga dentro de una lista de Polars)
    size_pack = df_precio['pack'][0]
    if isinstance(size_pack, (list, pl.Series)):
        size_pack = size_pack[0]
        
    # 2. Calcular los componentes del EOQ

    # dias_simulacion (int): Días totales del horizonte de datos.
    dias_simulacion = df_demanda.height
    
    demanda_anual_D = (demanda_total_periodo / dias_simulacion) * dias_year
    costo_mantenimiento_H = precio_unitario_C * tasa_mantenimiento_i

    # 3. Aplicar la fórmula matemática
    eoq_botellas = np.sqrt((2 * demanda_anual_D * costo_pedido_S) / costo_mantenimiento_H)

    # 4. Convertir a la acción real del entorno (Cajas/Packs a pedir)
    # No se pueden pedir fracciones de packs, entonces se
    # redondeará al inmmediato superior
    eoq_cajas = np.ceil(eoq_botellas / size_pack)

    # 5. Imprimir resultados formateados
    print('-' * 55)
    print('RESULTADOS DEL MODELO BASE (EOQ)')
    print('-' * 55)
    print(f'Demanda Anual Estimada (D)       : {demanda_anual_D:,.0f} botellas')
    print(f'Costo de Mantenimiento Anual (H) : ${costo_mantenimiento_H:,.2f} por botella')
    print(f'Costo Fijo por Pedido (S)        : ${costo_pedido_S:,.2f}')
    print(f'EOQ Óptimo Teórico               : {eoq_botellas:,.0f} botellas')
    print(f'EOQ Ajustado a Lotes (Acción)    : {eoq_cajas:,.0f} cajas (packs de {size_pack})')
    print('-' * 55)
    
    return eoq_cajas