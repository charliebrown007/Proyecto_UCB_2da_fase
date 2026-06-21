import gymnasium as gym
from gymnasium import spaces
import numpy as np
import polars as pl

class IowaLiquorEnv(gym.Env):
    """
    Entorno de Simulación Logística (Single-Item) para Deep Reinforcement Learning.
    Basado en los datos reales del producto "MCCORMICK 80PRF VODKA PET".
    """
    
    # Constructor del entorno: Aquí definimos los parámetros, espacios de acción/observación y cargamos los datos históricos.
    def __init__(self, df_episodio: pl.DataFrame, capacidad_max_cajas: int = 50):

        # 1. Llamar al constructor "padre" de gym.Env para inicializar la infraestructura base
        super(IowaLiquorEnv, self).__init__()
        
        # 2. Atributos: Demanda histórica
        self.demanda_historica = df_episodio['demanda_diaria'].to_numpy().astype(int)
        self.dias_totales = len(self.demanda_historica)
        
        # 3. Atributos: Parámetros Constantes del Entorno
        self.precio_venta = 2.76                                            # Precio de venta al público (ingreso bruto) [USD por botella]
        self.unidades_por_caja = 24                                         # Cantidad de botellas en un 'pack'
        self.capacidad_max = capacidad_max_cajas * self.unidades_por_caja   # Límite físico del almacén en botellas
        
        # 4. Atributos: Costos Logísticos (Ajustables para calibrar al agente)
        self.costo_mantenimiento = 0.05                                     # USD por botella/día inmovilizada en almacén
        self.costo_stockout = 3.50                                          # USD de penalización por botella no entregada al cliente (consumidor final) (Venta perdida + prestigio)
        
        # 5. Atributos: Espacios de Acción y Observación (Gymnasium)
    
        # ACTION SPACE: El agente decide cuántas cajas pedir (ej. de 0 a 10 cajas)
        self.max_cajas_pedido = 10
        self.action_space = spaces.Discrete(self.max_cajas_pedido + 1)
        
        # OBSERVATION SPACE: Lo que el agente "ve" para tomar decisiones.

        # Por ahora, solo ve el nivel de inventario actual (en botellas).
        self.observation_space = spaces.Box(
            low   = np.array([0]), 
            high  = np.array([self.capacidad_max]), 
            dtype = np.float32
        )
        
        # Variables de estado interno
        self.dia_actual = 0
        self.inventario_actual = 0

    def reset(self, seed = None, options = None):
        """
        Reinicia la simulación al Día 0. Se ejecuta al inicio de cada episodio de entrenamiento.
        """
        super().reset(seed = seed)
        
        self.dia_actual = 0

        # Iniciamos con el almacén a media capacidad para no castigar al agente de inmediato
        self.inventario_actual = self.capacidad_max // 2 
        
        # La observación debe ser un array de NumPy (Regla de Gymnasium)
        observacion = np.array([self.inventario_actual], dtype=np.float32)
        info = {} # Diccionario opcional para logs extras
        
        return observacion, info

    def step(self, action):
        """
        El motor físico del entorno. Ejecuta un día completo de simulación.
        action: El número de CAJAS que el agente decidió pedir hoy.
        """
        # 1. LOGÍSTICA DE ENTRADA: Recibimos el pedido del agente
        botellas_pedidas = int(action) * self.unidades_por_caja
        self.inventario_actual += botellas_pedidas
        
        # Restricción física: El inventario no puede superar el tamaño del almacén
        self.inventario_actual = min(self.inventario_actual, self.capacidad_max)
        
        # 2. LOGÍSTICA DE SALIDA: Nos enfrentamos a la demanda real del día
        demanda_hoy = self.demanda_historica[self.dia_actual]
        
        botellas_vendidas = min(self.inventario_actual, demanda_hoy)
        botellas_faltantes = max(0, demanda_hoy - self.inventario_actual)
        
        # Actualizamos el inventario tras despachar a los clientes
        self.inventario_actual -= botellas_vendidas
        
        # 3. FUNCIÓN DE RECOMPENSA (Reward): ¿Qué tan bien lo hizo el agente hoy?
        ingreso = botellas_vendidas * self.precio_venta
        penalizacion_almacenaje = self.inventario_actual * self.costo_mantenimiento
        penalizacion_stockout = botellas_faltantes * self.costo_stockout
        
        # Recompensa Neta del Día
        reward = ingreso - penalizacion_almacenaje - penalizacion_stockout
        
        # 4. AVANCE DEL TIEMPO
        self.dia_actual += 1
        
        # ¿Terminó el episodio? (Llegamos al final de la historia de datos)
        terminado = bool(self.dia_actual >= self.dias_totales)
        truncado = False # No aplicamos límite de tiempo artificial aquí
        
        # Preparamos la observación para el día de mañana
        observacion = np.array([self.inventario_actual], dtype = np.float32)
        
        # Info adicional para depurar y graficar más adelante
        info = {
            'demanda': demanda_hoy,
            'vendido': botellas_vendidas,
            'stockout': botellas_faltantes,
            'reward': reward
        }
        
        return observacion, float(reward), terminado, truncado, info