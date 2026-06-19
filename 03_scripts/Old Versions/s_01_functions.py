# =========================================================================================================
# LIBRERÍAS
# =========================================================================================================

import requests
import polars as pl
import time

# =========================================================================================================
# EXTRACCIÓN DE DATOS A TRAVÉS DE https://data.cms.gov/
# =========================================================================================================

def extraer_datos_cms(api_url, limite = None):
    '''
    Se conecta con una API y descarga los datos en formato JSON
    manejando la paginación automáticamente.
    
    Args:
        api_url (str): La URL proporcionada por la ventana modal de CMS.
        limite (int, opcional): Cantidad de registros a extraer. Si es None, descarga TODOS los registros.
    '''
    if limite is None:
        print('Iniciando descarga MASIVA desde la API...')
    else:
        print(f'Iniciando conexión a la API (Modo Muestra: {limite} registros)...')
        
    print(f'URL: {api_url}')
    
    all_data = []
    offset = 0
    chunk_size = 5000
    inicio_tiempo = time.time()
    
    try:
        while True:

            # Si hay un límite, pedimos solo lo que falta para no pasarnos
            if (limite is not None) and ((offset + chunk_size) > limite):
                current_size = limite - offset
            else:
                current_size = chunk_size
                
            parametros = { 'size': current_size, 'offset': offset }
            
            respuesta = requests.get( url = api_url, params = parametros)
            respuesta.raise_for_status()
            
            datos_chunk = respuesta.json()
            
            # Condición de parada: Si la API devuelve un paquete vacío, terminamos
            if not datos_chunk:
                break
                
            all_data.extend(datos_chunk)
            offset += len(datos_chunk)
            
            if limite is None:
                print(f"⏳ Descargando... {offset} registros extraídos.")
            
            # Condición de parada por llegar al límite o al final de la base de datos
            if (limite is not None and offset >= limite) or len(datos_chunk) < current_size:
                break
                
        tiempo_descarga = time.time() - inicio_tiempo
        tipo_descarga = 'TOTAL' if limite is None else 'MUESTRA'
        print(f'✅ Descarga {tipo_descarga} completada: {len(all_data)} registros en {tiempo_descarga:.2f} segundos.')
        
        return pl.DataFrame(all_data)

    except requests.exceptions.RequestException as e:
        print(f'❌ Error al conectar con la API: {e}')
        return None
    except pl.exceptions.PolarsError as e:
        print(f'❌ Error de Polars al procesar el JSON: {e}')
        return None

