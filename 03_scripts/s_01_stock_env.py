# =========================================================================================================
# 01. LIBRERÍAS

import gymnasium as gym
from gymnasium import spaces
import numpy as np


# =========================================================================================================
# 02. ENTORNO PERSONALIZADO

class EntornoInventario(gym.Env):
    """
    Entorno de simulación logística para Aprendizaje por Refuerzo.
    Representa un Proceso de Decisión de Markov (MDP) diario.
    """
    
    def __init__(self, df_demanda, precio_unitario, tamano_pack, max_cajas_pedido = 100):
        super(EntornoInventario, self).__init__()
        
        # 1. Carga de Datos y Constantes (Placeholders financieros)
        self.demanda_diaria = df_demanda['demanda_diaria'].to_numpy()
        self.dias_totales = len(self.demanda_diaria)
        
        self.p = precio_unitario        # Precio de venta por botella
        self.pack = tamano_pack         # Botellas por caja
        self.S = 50.0                   # Costo fijo por pedido (Ordering Cost)
        self.h = 0.01                   # Costo de mantenimiento diario por botella (Holding Cost)
        self.k = 2.0                    # Penalización por botella no vendida (Stockout Cost)
        self.L = 3                      # Lead time (Días en que tarda en llegar el pedido)
        
        # 2. Espacio de Acciones (Action Space)
        # El agente decide cuántas cajas pedir: 0, 1, 2, ..., max_cajas_pedido
        self.action_space = spaces.Discrete(max_cajas_pedido + 1)
        
        # 3. Espacio de Observación (Observation Space)
        # Definimos los límites [mínimo, máximo] de las variables que "ve" el agente
        # Por ahora: [Nivel de inventario actual]
        # (Se ampliará en el futuro con más variables como días de la semana, etc.)
        self.observation_space = spaces.Box(
            low=np.array([0], dtype=np.float32),
            high=np.array([np.inf], dtype=np.float32),
            dtype=np.float32
        )
        
        # Variables de estado interno
        self.dia_actual = 0
        self.inventario = 0
        self.pedidos_en_transito = []

    def reset(self, seed=None, options=None):
        """
        Reinicia el entorno al día 0. Se ejecuta al inicio de cada episodio.
        """
        super().reset(seed=seed)
        
        self.dia_actual = 0
        self.inventario = 0  # Se puede iniciar con un inventario inicial de seguridad
        self.pedidos_en_transito = []
        
        # Devolver el estado inicial y un diccionario vacío de información
        estado_inicial = np.array([self.inventario], dtype=np.float32)
        return estado_inicial, {}

    def step(self, action):
        """
        Ejecuta la acción del agente, calcula ventas, costos y avanza al siguiente día.
        """
        # 1. Recibir pedidos que llegaron hoy (según el Lead Time)
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
        
        # 4. Calcular la Recompensa (Rentabilidad Neta)
        ingresos = ventas_hoy * self.p
        costo_mantenimiento = self.inventario * self.h
        costo_quiebre = demanda_insatisfecha * self.k
        
        recompensa = ingresos - costo_orden - costo_mantenimiento - costo_quiebre
        
        # 5. Avanzar el tiempo y verificar si el episodio terminó
        self.dia_actual += 1
        terminado = bool(self.dia_actual >= self.dias_totales)
        truncado = False # Se usa si hay límites de tiempo artificiales
        
        # Siguiente estado
        siguiente_estado = np.array([self.inventario], dtype=np.float32)
        
        # Diccionario para guardar métricas adicionales útiles para tus gráficos
        info = {
            'ventas': ventas_hoy,
            'quiebre': demanda_insatisfecha,
            'recompensa_bruta': recompensa
        }
        
        return siguiente_estado, recompensa, terminado, truncado, info

    def _procesar_pedidos_en_transito(self):
        """Método auxiliar interno para gestionar el Lead Time."""
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