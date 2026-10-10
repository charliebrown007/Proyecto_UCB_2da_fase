# =========================================================================================================
# 01. LIBRERÍAS

import gymnasium as gym
import numpy as np
import polars as pl
from gymnasium import spaces

# =========================================================================================================
# 02. ENTORNO PERSONALIZADO (v6.0 - Escenario Crisis y Estado de Excepción Bolivia)

class EntornoInventario(gym.Env):
    def __init__(self,
                 df_demanda: pl.DataFrame,
                 precio_unitario: float,
                 tamano_pack: int,
                 max_cajas_pedido: int = 1500, # Aumentado para permitir llenado rápido en ventanas de calma
                 ventana_historica: int = 7,
                 max_capacidad_almacen: int = 42000): # Capacidad blindada a 30 días de cobertura
        super().__init__()
        
        # 1. Carga de Datos y Constantes
        self.demanda_diaria = df_demanda['demanda_diaria'].to_numpy()
        self.dia_semana = df_demanda['dia_semana'].to_numpy() 
        self.dias_totales = len(self.demanda_diaria)
        
        self.ventana_historica = ventana_historica
        self.max_capacidad = max_capacidad_almacen 
        
        self.p = precio_unitario        
        self.pack = tamano_pack         
        self.S = 50.0                   
        self.h = 0.20 
        
        # Penalización por demanda insatisfecha (Aumentada drásticamente para reflejar pérdidas millonarias)                  
        self.k = 5.0  
        
        # 2. Espacios de Acción y Observación
        self.action_space = spaces.Discrete(max_cajas_pedido + 1)
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
        
        estado_inicial = np.concatenate((
            [self.inventario], 
            self.historial_demanda, 
            [self.dia_semana[self.dia_actual]]
        )).astype(np.float32)
        
        return estado_inicial, {}
        
    def step(self, action):
        botellas_recibidas = self._procesar_pedidos_en_transito()
        self.inventario += botellas_recibidas
        
        # --- NUEVA REGLA: LEAD TIME ESTOCÁSTICO EXTREMO ---
        botellas_pedidas = action * self.pack
        if botellas_pedidas > 0:
            # Distribución logística que refleja crisis de mayo/junio 2026:
            # 75% Tránsito normal o expedito (3 días)
            # 15% Desvíos por conflictos menores (7 días)
            #  8% Bloqueo severo en ruta troncal (21 días)
            #  2% Cerco y Estado de Excepción nacional (53 días)
            tiempo_entrega_real = np.random.choice([3, 7, 21, 53], p=[0.75, 0.15, 0.08, 0.02])
            
            self.pedidos_en_transito.append({'dias_faltantes': tiempo_entrega_real, 'cantidad': botellas_pedidas})
            costo_orden = self.S
        else:
            costo_orden = 0.0
            
        # Ventas y Quiebres
        demanda_hoy = self.demanda_diaria[self.dia_actual]
        ventas_hoy = min(self.inventario, demanda_hoy)
        demanda_insatisfecha = demanda_hoy - ventas_hoy
        self.inventario -= ventas_hoy
        
        # Restricción Física CEDI
        exceso_inventario = max(0, self.inventario - self.max_capacidad)
        if exceso_inventario > 0:
            costo_exceso = exceso_inventario * (self.p * 0.5) 
            self.inventario = self.max_capacidad 
        else:
            costo_exceso = 0.0

        self.historial_demanda = np.roll(self.historial_demanda, -1)
        self.historial_demanda[-1] = demanda_hoy
        
        # Reward Shaping: Sistema de recompensas ajustado a economía de escala
        ingresos = ventas_hoy * self.p
        
        # Ajuste de exponente de mantenimiento (1.05 en lugar de 1.15) para equilibrar gran volumen
        costo_mantenimiento = (self.inventario ** 1.05) * self.h 
        costo_quiebre = demanda_insatisfecha * self.k
        
        recompensa = ingresos - costo_orden - costo_mantenimiento - costo_quiebre - costo_exceso
        
        self.dia_actual += 1
        terminado = bool(self.dia_actual >= self.dias_totales)
        truncado = False
        
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