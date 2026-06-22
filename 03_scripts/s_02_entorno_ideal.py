import gymnasium as gym
from gymnasium import spaces
import numpy as np
import polars as pl

class IowaLiquorEnvIdeal(gym.Env):
    """
    Simulador Logístico en Condiciones Ideales (Línea Base).
    Sin tiempos de tránsito estocásticos ni bloqueos en carretera.
    """
    def __init__(self, df_episodio: pl.DataFrame, capacidad_max_cajas: int = 300):
        super(IowaLiquorEnvIdeal, self).__init__()
        self.demanda_historica = df_episodio['demanda_diaria'].to_numpy().astype(int)
        self.dias_totales = len(self.demanda_historica)

        # Parámetros comerciales base
        self.precio_venta = 2.76
        self.unidades_por_caja = 24
        self.capacidad_max = capacidad_max_cajas * self.unidades_por_caja
        
        # Costos estándar de la simulación inicial
        self.costo_mantenimiento = 0.01  
        self.costo_stockout = 3.50 

        # Espacio de acciones: pedir entre 0 y 50 cajas
        self.max_cajas_pedido = 50 
        self.action_space = spaces.Discrete(self.max_cajas_pedido + 1)

        # El estado solo observa el inventario físico disponible en bodega
        self.observation_space = spaces.Box(
            low=np.array([0]), 
            high=np.array([self.capacidad_max]), 
            dtype=np.float32
        )

        self.dia_actual = 0
        self.inventario_actual = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.dia_actual = 0
        # Iniciamos con la bodega a media capacidad
        self.inventario_actual = self.capacidad_max // 2 
        
        obs = np.array([self.inventario_actual], dtype=np.float32)
        return obs, {}

    def step(self, action):
        # En el entorno ideal el pedido se procesa e ingresa instantáneamente
        cajas_pedidas = int(action)
        self.inventario_actual += cajas_pedidas * self.unidades_por_caja
        self.inventario_actual = min(self.inventario_actual, self.capacidad_max)

        # Enfrentar la demanda del día
        demanda_hoy = self.demanda_historica[self.dia_actual]
        botellas_vendidas = min(self.inventario_actual, demanda_hoy)
        botellas_faltantes = max(0, demanda_hoy - self.inventario_actual)
        self.inventario_actual -= botellas_vendidas

        # Balance financiero del paso
        ingreso = botellas_vendidas * self.precio_venta
        penalizacion_almacenaje = self.inventario_actual * self.costo_mantenimiento
        penalizacion_stockout = botellas_faltantes * self.costo_stockout
        reward = ingreso - penalizacion_almacenaje - penalizacion_stockout

        self.dia_actual += 1
        terminado = bool(self.dia_actual >= self.dias_totales)

        obs = np.array([self.inventario_actual], dtype=np.float32)
        info = {'stockout': botellas_faltantes}
        return obs, float(reward), terminado, False, info