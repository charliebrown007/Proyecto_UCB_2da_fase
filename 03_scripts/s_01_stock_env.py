# =========================================================================================================
# 01. LIBRERÍAS

import gymnasium as gym
import numpy as np
import polars as pl
from gymnasium import spaces

# =========================================================================================================
# 02. ENTORNO PERSONALIZADO (v4.0 - Contexto Temporal)

class EntornoInventario(gym.Env):
    def __init__(self,
                 df_demanda: pl.DataFrame,
                 precio_unitario: float,
                 tamano_pack: int,
                 max_cajas_pedido: int = 100,
                 ventana_historica: int = 7,
                 max_capacidad_almacen: int = 3000):
        super().__init__()
        
        # 1. Carga de Datos y Constantes
        self.demanda_diaria = df_demanda['demanda_diaria'].to_numpy()
        # NUEVO: Cargamos el arreglo de días de la semana
        self.dia_semana = df_demanda['dia_semana'].to_numpy() 
        self.dias_totales = len(self.demanda_diaria)
        
        self.ventana_historica = ventana_historica
        self.max_capacidad = max_capacidad_almacen 
        
        self.p = precio_unitario        
        self.pack = tamano_pack         
        self.S = 50.0                   
        self.h = 0.20                   
        self.k = 2.0                    
        self.L = 3                      
        
        # 2. Espacio de Acciones
        self.action_space = spaces.Discrete(max_cajas_pedido + 1)
        
        # 3. Espacio de Observación
        # NUEVO: Añadimos +1 al espacio para incluir el día de la semana actual
        self.observation_space = spaces.Box(
            low = 0.0,
            high = np.inf,
            shape = (1 + self.ventana_historica + 1,), 
            dtype = np.float32 )
        
        self.dia_actual = 0
        self.inventario = 0
        self.pedidos_en_transito = []
        self.historial_demanda = np.zeros(self.ventana_historica, dtype=np.float32)

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)
        self.dia_actual = 0
        self.inventario = 0  
        self.pedidos_en_transito = []
        self.historial_demanda = np.zeros(self.ventana_historica, dtype=np.float32)
        
        # NUEVO: El estado inicial ahora incluye el día de la semana en la última posición
        estado_inicial = np.concatenate((
            [self.inventario], 
            self.historial_demanda, 
            [self.dia_semana[self.dia_actual]]
        )).astype(np.float32)
        
        return estado_inicial, {}
        
    def step(self, action):
        botellas_recibidas = self._procesar_pedidos_en_transito()
        self.inventario += botellas_recibidas
        
        botellas_pedidas = action * self.pack
        if botellas_pedidas > 0:
            self.pedidos_en_transito.append({'dias_faltantes': self.L, 'cantidad': botellas_pedidas})
            costo_orden = self.S
        else:
            costo_orden = 0.0
            
        demanda_hoy = self.demanda_diaria[self.dia_actual]
        ventas_hoy = min(self.inventario, demanda_hoy)
        demanda_insatisfecha = demanda_hoy - ventas_hoy
        self.inventario -= ventas_hoy
        
        # Restricción Física
        exceso_inventario = max(0, self.inventario - self.max_capacidad)
        if exceso_inventario > 0:
            costo_exceso = exceso_inventario * (self.p * 0.5) 
            self.inventario = self.max_capacidad 
        else:
            costo_exceso = 0.0

        self.historial_demanda = np.roll(self.historial_demanda, -1)
        self.historial_demanda[-1] = demanda_hoy
        
        # Reward Shaping
        ingresos = ventas_hoy * self.p
        costo_mantenimiento = (self.inventario ** 1.15) * self.h 
        costo_quiebre = demanda_insatisfecha * self.k
        
        recompensa = ingresos - costo_orden - costo_mantenimiento - costo_quiebre - costo_exceso
        
        self.dia_actual += 1
        terminado = bool(self.dia_actual >= self.dias_totales)
        truncado = False
        
        # NUEVO: Extraemos el día de la semana para el estado de mañana
        dia_siguiente = self.dia_semana[self.dia_actual] if not terminado else 0
        
        siguiente_estado = np.concatenate((
            [self.inventario], 
            self.historial_demanda, 
            [dia_siguiente]
        )).astype(np.float32)
        
        info = {
            'ventas': ventas_hoy,
            'quiebre': demanda_insatisfecha,
            'recompensa_bruta': recompensa
        }
        
        return siguiente_estado, recompensa, terminado, truncado, info

    def _procesar_pedidos_en_transito(self):
        recibidos_hoy = 0
        pedidos_actualizados = []
        for pedido in self.pedidos_en_transito:
            pedido['dias_faltantes'] -= 1
            if pedido['dias_faltantes'] <= 0:
                recibidos_hoy += pedido['cantidad']
            else:
                pedidos_actualizados.append(pedido)
        self.pedidos_en_transito = pedidos_actualizados
        return recibidos_hoy