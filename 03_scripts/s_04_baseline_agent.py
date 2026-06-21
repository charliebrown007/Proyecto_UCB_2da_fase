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