# =========================================================================================================
# LIBRERÍAS

# Manipulación de datos y cálculo numérico
import polars as pl
import numpy as np

# Medición de tiempo
import time

# Gestión de ubicaciones y directorios
from pyprojroot import here
from pathlib import Path

# Tipado de datos estático (Type Hinting)
from typing import Optional

# =========================================================================================================
# 01. CARGA DE DATOS
# Función de carga de archivo de texto


def load_data_own(file_path_param: str) -> Optional[pl.DataFrame]:

    print(f'📦 Iniciando carga de datos locales desde: {file_path_param}')
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


# =========================================================================================================
# 02. LIMPIEZA DE DATOS
# Función de limpieza de datos

def limpiar_texto_own(column_param) -> pl.Expr:

    return (
        column_param
        
        # 1. Convertir a mayúsculas
        .str.to_uppercase()

        # 2. Normalizar para separar los
        # acentos de las letras (Ej: 'á' -> 'a' + '´')
        .str.normalize('NFD')

        # 3. Borramos acentos, pero NO la colita de la ñ (\u0303)
        .str.replace_all(r"[\u0300-\u0302\u0304-\u036f]", "")
        
        # 4. Eliminar todo lo que NO sea letras, números o espacios
        #.str.replace_all(r"[^a-z0-9\s]", "")
        
        # 5. Reemplazar múltiples espacios por uno solo
        .str.replace_all(r'\s+', ' ')
        
        # 6. Quitar espacios al inicio y al final
        .str.strip_chars()
    )


# =========================================================================================================
# 03. LÍNEA BASE DE ANÁLISIS PARA EL AGENTE

class AgenteMinMaxBolivia:
    def __init__(self, punto_reorden: int, nivel_maximo: int, unidades_por_caja: int):
        self.s = punto_reorden
        self.S = nivel_maximo
        self.pack = unidades_por_caja
        
    def predecir(self, observacion: np.ndarray) -> int:
        inventario_fisico = observacion[0]
        inventario_en_transito = observacion[1]
        
        # El agente humano suma lo que tiene y lo que viene en camino
        posicion_inventario = inventario_fisico + inventario_en_transito
        
        if posicion_inventario <= self.s:
            botellas_faltantes = self.S - posicion_inventario
            cajas_a_pedir = int(np.ceil(botellas_faltantes / self.pack))
            return min(cajas_a_pedir, 50) # No podemos superar la capacidad del camión
        else:
            return 0


def evaluar_agente(env, agente, num_episodios=1):
    """
    Corre la simulación completa para evaluar el rendimiento financiero de un agente.
    """
    beneficio_total_historico = []
    
    for episodio in range(num_episodios):
        obs, _ = env.reset()
        terminado = False
        recompensa_acumulada = 0
        dias_con_stockout = 0
        
        while not terminado:
            # El agente toma una decisión basada en su inventario actual
            accion = agente.predecir(obs)
            
            # El entorno procesa la decisión
            obs, recompensa, terminado, truncado, info = env.step(accion)
            
            recompensa_acumulada += recompensa
            if info['stockout'] > 0:
                dias_con_stockout += 1
                
        beneficio_total_historico.append(recompensa_acumulada)
        
        print(f"📊 RESULTADOS DEL EPISODIO {episodio + 1}")
        print(f"💰 Beneficio Total Generado: ${recompensa_acumulada:,.2f}")
        print(f"🚨 Días con Quiebre de Stock: {dias_con_stockout} días")
        print("-" * 50)
        
    return beneficio_total_historico



# =========================================================================================================
# 03. LÍNEA BASE DE ANÁLISIS PARA EL AGENTE

class AgenteMinMaxIdeal:
    """Agente heurístico para el entorno ideal (solo observa inventario físico)"""
    def __init__(self, punto_reorden: int, nivel_maximo: int, unidades_por_caja: int):
        self.s = punto_reorden
        self.S = nivel_maximo
        self.pack = unidades_por_caja
        
    def predecir(self, observacion: np.ndarray) -> int:
        # El entorno ideal solo devuelve [inventario_fisico]
        inventario_fisico = observacion[0]
        
        if inventario_fisico <= self.s:
            botellas_faltantes = self.S - inventario_fisico
            cajas_a_pedir = int(np.ceil(botellas_faltantes / self.pack))
            return min(cajas_a_pedir, 50) # Límite de capacidad del camión
        else:
            return 0
