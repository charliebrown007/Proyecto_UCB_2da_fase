# =========================================================================================================
# 01. LIBRERÍAS

import gymnasium as gym
import numpy as np
import polars as pl
from gymnasium import spaces

# =========================================================================================================
# 02. ENTORNO PERSONALIZADO

class EntornoInventario(gym.Env):
    '''
    Entorno de simulación logística para Aprendizaje por Refuerzo.
    Representa un Proceso de Decisión de Markov (MDP) diario.
    '''
    
    def __init__(self,
                 df_demanda: pl.DataFrame,
                 precio_unitario: float,
                 tamano_pack: int,
                 max_cajas_pedido: int = 100,
                 ventana_historica: int = 7):
        super().__init__()
        
        # 1. Carga de Datos y Constantes
        self.demanda_diaria = df_demanda['demanda_diaria'].to_numpy()
        self.dias_totales = len(self.demanda_diaria)
        self.ventana_historica = ventana_historica
        
        self.p = precio_unitario        # Precio de venta por botella
        self.pack = tamano_pack         # Botellas por caja
        self.S = 50.0                   # Costo fijo por pedido (Ordering Cost)
        self.h = 0.01                   # Costo de mantenimiento diario por botella (Holding Cost)
        self.k = 2.0                    # Penalización por botella no vendida (Stockout Cost)
        self.L = 3                      # Lead time (Días en que tarda en llegar el pedido)
        
        # 2. Espacio de Acciones (Action Space)
        self.action_space = spaces.Discrete(max_cajas_pedido + 1)
        
        # 3. Espacio de Observación (Observation Space)
        # Tamaño: 1 (inventario) + ventana_historica (ej. 7 días de demanda) = 8 variables continuas
        self.observation_space = spaces.Box(
            low = 0.0,
            high = np.inf,
            shape = (1 + self.ventana_historica,),
            dtype = np.float32 )
        
        # Variables de estado interno
        self.dia_actual = 0
        self.inventario = 0
        self.pedidos_en_transito = []
        self.historial_demanda = np.zeros(self.ventana_historica, dtype=np.float32)

    def reset(self, seed = None, options = None):
        '''
        Reinicia el entorno al día 0. Se ejecuta al inicio de cada episodio.
        '''
        super().reset(seed = seed)
        
        self.dia_actual = 0
        self.inventario = 0  
        self.pedidos_en_transito = []
        
        # Reiniciar el historial de demanda con ceros
        self.historial_demanda = np.zeros(self.ventana_historica, dtype=np.float32)
        
        # El estado inicial es el inventario concatenado con el historial de demanda
        estado_inicial = np.concatenate(([self.inventario], self.historial_demanda)).astype(np.float32)
        
        return estado_inicial, {}
        
    def step(self, action):
        '''
        Ejecuta la acción del agente, calcula ventas, costos y avanza al siguiente día.
        '''
        # 1. Recibir pedidos que llegaron hoy
        botellas_recibidas = self._procesar_pedidos_en_transito()
        self.inventario += botellas_recibidas
        
        # 2. Registrar el nuevo pedido emitido hoy por el agente
        botellas_pedidas = action * self.pack
        if botellas_pedidas > 0:
            self.pedidos_en_transito.append({'dias_faltantes': self.L, 'cantidad': botellas_pedidas})
            costo_orden = self.S
        else:
            costo_orden = 0.0
            
        # 3. Observar la demanda del día y calcular ventas
        demanda_hoy = self.demanda_diaria[self.dia_actual]
        ventas_hoy = min(self.inventario, demanda_hoy)
        demanda_insatisfecha = demanda_hoy - ventas_hoy
        
        # Actualizar inventario tras las ventas
        self.inventario -= ventas_hoy
        
        # --- ACTUALIZACIÓN DE LA VENTANA HISTÓRICA ---
        # np.roll desplaza todos los elementos a la izquierda y el último espacio recibe la demanda de hoy
        self.historial_demanda = np.roll(self.historial_demanda, -1)
        self.historial_demanda[-1] = demanda_hoy
        
        # 4. Calcular la Recompensa (Rentabilidad Neta)
        ingresos = ventas_hoy * self.p
        costo_mantenimiento = self.inventario * self.h
        costo_quiebre = demanda_insatisfecha * self.k
        
        recompensa = ingresos - costo_orden - costo_mantenimiento - costo_quiebre
        
        # 5. Avanzar el tiempo y verificar si el episodio terminó
        self.dia_actual += 1
        terminado = bool(self.dia_actual >= self.dias_totales)
        truncado = False
        
        # --- SIGUIENTE ESTADO ---
        siguiente_estado = np.concatenate(([self.inventario], self.historial_demanda)).astype(np.float32)
        
        info = {
            'ventas': ventas_hoy,
            'quiebre': demanda_insatisfecha,
            'recompensa_bruta': recompensa
        }
        
        return siguiente_estado, recompensa, terminado, truncado, info

    def _procesar_pedidos_en_transito(self):
        '''Método auxiliar interno para gestionar el Lead Time.'''
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