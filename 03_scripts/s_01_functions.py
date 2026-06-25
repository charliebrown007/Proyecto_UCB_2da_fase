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

# Ambiente de pruebas
import gymnasium as gym
from gymnasium import spaces

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

def limpiar_texto_own(column_param: str) -> pl.Expr:

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
# 03. CONFIGURACIÓN DEL AMBIENTE DE PRUEBAS EN GYMNASIUM
# Simulador Logístico Tropicalizado (Bolivia).
# Incluye Costos de Capital (BCB) y Riesgo Estocástico de Bloqueos (Cadenas de Markov).

class EntornoBolivia(gym.Env):

    # Constructor
    def __init__(self, 
                df_episodio: pl.DataFrame,
                capacidad_max_cajas: int = 300):

        # Llamar el entorno "padre" de la clase gym
        super(EntornoBolivia, self).__init__()

        # Atributo: Demanda diaria en un array de numpy
        self.demanda_historica = df_episodio['demanda_diaria'].to_numpy()

        # Atributo: Cantidad de días a través de los cuales se desenvuelve la demanda
        self.dias_totales = len(self.demanda_historica)

        # Atributo: Precio de venta al público
        self.precio_venta = 2.76 

        # Atributo: Unidades (botellas) disponibles por caja
        self.unidades_por_caja = 24

        # Atributo: Capacidad máxima por flota de transporte
        self.capacidad_max = capacidad_max_cajas * self.unidades_por_caja
        
        # Atributo: Costo de oportunidad
        tasa_anual_bcb = 0.06

        # Atributo: Costo financiero 
        self.costo_mantenimiento = (self.precio_venta * tasa_anual_bcb) / 365

        # C
        self.costo_stockout = 3.50 

        # --- 2. LOGÍSTICA DE MARKOV ---
        self.lead_time_normal = 2 # Días estándar de viaje
        
        # Matriz de Transición de Markov
        # P[estado_actual] = [probabilidad_de_ir_a_0, probabilidad_de_ir_a_1]
        self.matriz_transicion = {
            0: [0.80, 0.20], # Si hoy está libre: 80% libre mañana, 20% estalla bloqueo
            1: [0.20, 0.80]  # Si hoy está bloqueado: 20% se levanta, 80% sigue bloqueado
        }

        self.max_cajas_pedido = 50 
        self.action_space = spaces.Discrete(self.max_cajas_pedido + 1)

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]), 
            high=np.array([self.capacidad_max, self.capacidad_max * 5, 1]), 
            dtype=np.float32
        )

        self.dia_actual = 0
        self.inventario_actual = 0
        self.pedidos_en_transito = []
        self.estado_bloqueo = 0 # 0 = Libre, 1 = Bloqueado

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.dia_actual = 0
        self.inventario_actual = self.capacidad_max // 2 
        self.pedidos_en_transito = []
        self.estado_bloqueo = 0
        
        obs = np.array([self.inventario_actual, 0, self.estado_bloqueo], dtype=np.float32)
        return obs, {}

    def step(self, action):
        # 1. TRAMITAR NUEVO PEDIDO
        cajas_pedidas = int(action)
        if cajas_pedidas > 0:
            self.pedidos_en_transito.append({
                "botellas": cajas_pedidas * self.unidades_por_caja,
                "dias_restantes": self.lead_time_normal
            })

        # 2. MOTOR DE CONFLICTOS SOCIALES (Cadena de Markov)
        # Extraemos las probabilidades basándonos en cómo estaba la carretera ayer
        probabilidades_hoy = self.matriz_transicion[self.estado_bloqueo]
        # Transición al nuevo estado
        self.estado_bloqueo = np.random.choice([0, 1], p=probabilidades_hoy)

        # 3. AVANCE DE CAMIONES
        botellas_llegaron_hoy = 0
        if self.estado_bloqueo == 0:
            for pedido in self.pedidos_en_transito:
                pedido["dias_restantes"] -= 1
            
            llegaron = [p for p in self.pedidos_en_transito if p["dias_restantes"] <= 0]
            self.pedidos_en_transito = [p for p in self.pedidos_en_transito if p["dias_restantes"] > 0]
            
            botellas_llegaron_hoy = sum(p["botellas"] for p in llegaron)
            self.inventario_actual += botellas_llegaron_hoy

        self.inventario_actual = min(self.inventario_actual, self.capacidad_max)

        # 4. ENFRENTAR AL MERCADO
        demanda_hoy = self.demanda_historica[self.dia_actual]
        botellas_vendidas = min(self.inventario_actual, demanda_hoy)
        botellas_faltantes = max(0, demanda_hoy - self.inventario_actual)
        self.inventario_actual -= botellas_vendidas

        # 5. FINANZAS
        ingreso = botellas_vendidas * self.precio_venta
        penalizacion_almacenaje = self.inventario_actual * self.costo_mantenimiento
        penalizacion_stockout = botellas_faltantes * self.costo_stockout
        reward = ingreso - penalizacion_almacenaje - penalizacion_stockout

        self.dia_actual += 1
        terminado = bool(self.dia_actual >= self.dias_totales)

        # 6. ACTUALIZAR PANTALLA
        total_en_transito = sum(p["botellas"] for p in self.pedidos_en_transito)
        obs = np.array([self.inventario_actual, total_en_transito, self.estado_bloqueo], dtype=np.float32)
        
        info = {'stockout': botellas_faltantes, 'estado_bloqueo': self.estado_bloqueo}
        return obs, float(reward), terminado, False, info





























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
