# Optimizacion Dinamica de Politicas de Inventario mediante Aprendizaje por Refuerzo Profundo (PPO vs. EOQ)

Proyecto de investigacion aplicada para la Maestria en Ciencia de Datos de la Universidad Catolica Boliviana "San Pablo" (UCB).

* **Autor:** Carlos Andres Perez Guzman
* **Fecha:** Octubre 2026
* **Area:** Investigacion de Operaciones, Cadenas de Suministro y Aprendizaje por Refuerzo Profundo (DRL)

---

## 1. Descripcion General

El proyecto evalua el desempeno financiero y operativo de un agente autonomo de Aprendizaje por Refuerzo Profundo basado en el algoritmo **Proximal Policy Optimization (PPO)** frente al modelo analitico clasico de **Cantidad Economica de Pedido y Punto de Reorden (EOQ + ROP)**.

La experimentacion utiliza datos reales de transacciones mayoristas B2B (*Iowa Liquor Sales 2026*) con mas de 1.6 millones de registros. Se construyo un entorno de simulacion estocastico en Python (Gymnasium) calibrado con retrasos multimodales de transporte (desde 3 hasta 53 dias) para evaluar la resiliencia de ambas politicas bajo condiciones de crisis e incertidumbre severa.

---

## 2. Resultados y Logros Principales

La comparacion de politicas se efectuo mediante 100 simulaciones de Montecarlo pareadas por semilla en el conjunto de prueba (62 dias no observados):

| Metrica / Indicador | Modelo Clasico (EOQ + ROP) | Agente DRL (PPO con VecNormalize) | Diferencia Relativa |
| :--- | :--- | :--- | :--- |
| **Beneficio Neto Medio** | $354,467.39 USD | $441,914.84 USD | **+24.67% de incremento** |
| **Volatilidad Financiera (Desv. Est.)** | $225,744.95 USD | $36,227.07 USD | **-83.95% de reduccion de riesgo** |
| **Piso Minimo de Rentabilidad** | -$200,000.00 USD (perdidas) | +$360,000.00 USD | **Eliminacion de quiebras operativas** |
| **Prueba de Hipotesis (T de Welch)** | t = 2.6774 | p = 0.00993 | **Significativo al 99% (p < 0.01)** |

### Logro Tecnico Destacado
* **Estabilizacion de Gradientes:** Se implemento la normalizacion vectorial dinamica (`VecNormalize`) de observaciones y recompensas monetarias, resolviendo la dispersion inicial de perdidas y logrando una varianza explicada del 95% (`explained_variance = 0.95`) tras 1,000,000 de pasos de entrenamiento.

---

## 3. Entorno de Simulacion (Proceso de Decision de Markov)

El simulador implementa un Proceso de Decision de Markov (MDP) de un solo escalon para el producto representativo *Black Velvet* (cajas de 6 botellas de 1,750 ml):

* **Espacio de Estados ($S_t$):** Vector continuo de 9 dimensiones compuesto por el inventario fisico disponible, el historial de demanda diaria de los ultimos 7 dias y el dia de la semana siguiente.
* **Espacio de Acciones ($A_t$):** Espacio discreto entre 0 y 1,500 cajas diarias ($Q_t = 6 \cdot a_t$ botellas).
* **Tiempos de Entrega Estocasticos ($L_t$):**
  * 3 dias: 75% de probabilidad (flujo normal).
  * 7 dias: 15% de probabilidad (desvios menores).
  * 21 dias: 8% de probabilidad (bloqueo de ruta troncal).
  * 53 dias: 2% de probabilidad (cierre prolongado).
* **Funcion de Recompensa Diaria ($R_t$):**
  $$R_t = (V_t \cdot p) - (S \cdot \mathbb{I}_{\{a_t > 0\}}) - (h \cdot I_t^{1.05}) - (k \cdot U_t) - (c_{\text{exceso}} \cdot E_t)$$
  * $V_t$: Botellas vendidas a precio unitario $p = \$17.25\text{ USD}$.
  * $S$: Costo administrativo fijo por pedido (\$50.00 USD).
  * $h$: Tasa de mantenimiento con penalizacion exponencial ($0.20 \cdot I_t^{1.05}$).
  * $k$: Costo de quiebre de stock (\$5.00 USD por botella faltante).
  * $E_t$: Penalizacion por superar la capacidad fisica de 42,000 botellas.

---

## 4. Estructura del Repositorio

```text
├── 00_raw_data/             # Datos transaccionales originales (CSV)
├── 01_cleaned_data/         # Datos depurados y procesados en Parquet
├── 02_environments/         # Entorno virtual y dependencias
├── 03_scripts/              # Codigo fuente modular y notebooks de ejecucion
│   ├── s_00_directories.py  # Generacion de directorios del proyecto
│   ├── s_01_functions.py    # Funciones auxiliares y calculo del EOQ baseline
│   ├── s_02_stock_env.py    # Clase Gymnasium del entorno de simulacion
│   ├── s_03_Fase_01.ipynb   # Ingenieria de datos y depuracion
│   ├── s_04_Fase_02.ipynb   # Analisis exploratorio y seleccion del SKU
│   └── s_04_Fase_03.ipynb   # Entrenamiento PPO y contrastes de hipotesis
├── 04_tables_models/        # Checkpoints del modelo PPO, logs y estadisticas
├── 05_graphs/               # Graficos de convergencia y diagramas de caja
├── 06_quarto/               # Reportes y documentacion tecnica
└── README.md
```

---

## 5. Requisitos y Ejecucion

### Dependencias Principales
* Python >= 3.10
* Polars
* Gymnasium
* Stable-Baselines3
* PyTorch
* SciPy
* Matplotlib / Seaborn

### Flujo de Ejecucion
1. Generar la estructura de carpetas:
   ```bash
   python 03_scripts/s_00_directories.py
   ```
2. Ejecutar la limpieza y preparacion de datos:
   * Correr `03_scripts/s_03_Fase_01.ipynb`.
3. Ejecutar el analisis descriptivo:
   * Correr `03_scripts/s_04_Fase_02.ipynb`.
4. Entrenar el agente y ejecutar la evaluacion de Montecarlo:
   * Correr `03_scripts/s_04_Fase_03.ipynb`.