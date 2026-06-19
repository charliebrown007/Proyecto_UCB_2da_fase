# =========================================================================================================
# LIBRERÍAS
# =========================================================================================================

# Crga de archivos
import polars as pl

# Medición de tiempo
import time

# Gestión de ubicaciones y directorios
from pyprojroot import here
from pathlib import Path

# =========================================================================================================
# CARGA DE DATOS
# Función de carga de archivo de texto


def load_data_own(file_path_param):

    print(f'📦 Iniciando carga de datos locales desde: {file_path_param}')
    inicio_tiempo = time.time()
    
    try:

        df = pl.read_csv(
            source = Path(here(),'00_raw_data', file_path_param),
            has_header =  True,
            separator = ',',
            null_values = ['',' ','NA','na','NULL','null','None','NONE','Na'],
            encoding = 'utf8',
            infer_schema = True
            )
        
        tiempo_carga = time.time() - inicio_tiempo
        print(f"✅ Carga completada: {df.height} registros en {tiempo_carga:.2f} segundos.")
        
        return df

    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo en la ruta especificada ({file_path_param}).")
        print("Asegúrate de que el archivo CSV esté en tu carpeta '00_raw_data'.")
        return None
    except pl.exceptions.PolarsError as e:
        print(f"❌ Error de Polars al procesar el CSV: {e}")
        return None
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return None

