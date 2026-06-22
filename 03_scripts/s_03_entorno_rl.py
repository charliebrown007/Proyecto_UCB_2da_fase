import gymnasium as gym
from gymnasium import spaces
import numpy as np
import polars as pl

class EntornoBolivia(gym.Env):
    """
    Simulador Logístico Tropicalizado (Bolivia).
    Incluye Costos de Capital (BCB) y Riesgo Estocástico de Bloqueos en Carretera.
    """
    def __init__(self, df_episodio: pl.DataFrame, capacidad_max_cajas: int = 300):
        super(EntornoBolivia, self).__init__()
        self.demanda_historica = df_episodio['demanda_diaria'].to_numpy().astype(int)
        self.dias_totales = len(self.demanda_historica)

        # --- 1. TROPICALIZACIÓN FINANCIERA ---
        self.precio_venta = 2.76
        self.unidades_por_caja = 24
        self.capacidad_max = capacidad_max_cajas * self.unidades_por_caja
        
        tasa_anual_bcb = 0.06 # 6% Costo de oportunidad (TRE / DPF)
        self.costo_mantenimiento = (self.precio_venta * tasa_anual_bcb) / 365
        self.costo_stockout = 3.50 

        # --- 2. TROPICALIZACIÓN LOGÍSTICA (Bloqueos) ---
        self.lead_time_normal = 2 # Días estándar en llegar el pedido
        self.probabilidad_bloqueo = 15 / 30 # 50% de probabilidad diaria

        self.max_cajas_pedido = 50 # Subimos el límite del camión
        self.action_space = spaces.Discrete(self.max_cajas_pedido + 1)

        # NUEVO OBSERVATION SPACE: La IA ahora necesita ver 3 cosas:
        # [Inventario en Bodega, Inventario Congelado en Carretera, ¿Hay Bloqueo Hoy?]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]), 
            high=np.array([self.capacidad_max, self.capacidad_max * 5, 1]), 
            dtype=np.float32
        )

        self.dia_actual = 0
        self.inventario_actual = 0
        self.pedidos_en_transito = []
        self.bloqueo_hoy = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.dia_actual = 0
        self.inventario_actual = self.capacidad_max // 2 
        self.pedidos_en_transito = []
        self.bloqueo_hoy = 0
        
        obs = np.array([self.inventario_actual, 0, self.bloqueo_hoy], dtype=np.float32)
        return obs, {}

    def step(self, action):
        # 1. TRAMITAR NUEVO PEDIDO
        cajas_pedidas = int(action)
        if cajas_pedidas > 0:
            self.pedidos_en_transito.append({
                "botellas": cajas_pedidas * self.unidades_por_caja,
                "dias_restantes": self.lead_time_normal
            })

        # 2. MOTOR DE CONFLICTOS SOCIALES (Lanzar el dado)
        self.bloqueo_hoy = 1 if np.random.rand() < self.probabilidad_bloqueo else 0

        # 3. AVANCE DE CAMIONES
        botellas_llegaron_hoy = 0
        if self.bloqueo_hoy == 0:
            # Si la vía está libre, los camiones descuentan 1 día de viaje
            for pedido in self.pedidos_en_transito:
                pedido["dias_restantes"] -= 1
            
            # Descargamos los camiones que llegaron a 0 días
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
        obs = np.array([self.inventario_actual, total_en_transito, self.bloqueo_hoy], dtype=np.float32)
        
        info = {'stockout': botellas_faltantes}
        return obs, float(reward), terminado, False, info