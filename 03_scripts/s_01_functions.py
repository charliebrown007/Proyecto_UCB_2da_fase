# =========================================================================================================
# 01. LIBRERÍAS

# Medición de tiempo
import time

# Gestión de directorios
from pathlib import Path

import numpy as np

# Manipulación de datos
import polars as pl
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
                source = Path(here(),'00_raw_data',file_path_param),
                has_header =  True,
                separator = ',',
                null_values = ['',' ','NA','na','NULL','null','None','NONE','Na'],
                encoding = 'utf8',
                infer_schema = True
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

# Variables categóricas

def plot_cat_function():

    aux_analysis = (
    df_02_cat
    .get_column('category_name')
    .value_counts(name = 'count_0', sort = True)
    .with_columns( (pl.col('count_0') / pl.sum('count_0')).alias('ptc_0') )
    .head(10)
)


    return print('Hola mundo')




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