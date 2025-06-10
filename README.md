# Pronóstico de variables macroeconómicas

Este repositorio contiene la metodología y código utilizados para la proyección de variables macroeconómicas y de mercado, basada en modelos econométricos, de series temporales y de machine learning.

## 1. Introducción
El presente repositorio contiene la estimación y proyección de variables clave en el análisis macroeconómico y financiero. Se han desarrollado modelos que combinan técnicas estadísticas tradicionales con enfoques modernos de aprendizaje automático para mejorar la precisión y robustez de las predicciones.

## 2. Modelos utilizados
Se han aplicado distintos modelos según la naturaleza de las variables a proyectar:

### 2.1 Modelos para variables macroeconómicas
1. **SARIMA (Seasonal ARIMA)**: Modelo autoregresivo integrado de medias móviles con componentes estacionales, adecuado para series con patrones recurrentes.
2. **Holt-Winters (Suavizado Exponencial)**: Modelo que descompone la serie en nivel, tendencia y estacionalidad.
3. **Prophet**: Algoritmo de series temporales desarrollado por Facebook (META) que combina tendencias lineales y no lineales con efectos estacionales.
4. **XGBoost**: Modelo de machine learning basado en boosting de árboles de decisión, utilizado para capturar relaciones no lineales complejas.
5. **Análisis de Componentes Principales (PCA)**: Reducción de dimensionalidad aplicada a variables macroeconómicas desagregadas por regiones.

### 2.2 Modelos para variables de mercado
1. **Vasicek**: Modelo estocástico para la dinámica de tasas de interés basado en procesos de reversión a la media.
2. **Heath-Jarrow-Morton (HJM)**: Modelo de evolución de curvas de rendimiento para representar la estructura temporal de tasas de interés.
3. **Simulaciones Monte Carlo**: Generación de trayectorias de variables de mercado para evaluar escenarios de riesgo.

## 3. Horizonte temporal y escenarios
El horizonte temporal de análisis comprende 10 años, segmentado en:
- **Entrenamiento**: 70% de los datos históricos.
- **Validación**: 20% de los datos recientes.
- **Prueba**: 10% de los datos más recientes para evaluar la robustez del modelo.

Se consideran tres escenarios de proyección:
- **Escenario Central**: Basado en la evolución histórica.
- **Escenario Optimista**: Basado en los percentiles superiores de las simulaciones.
- **Escenario Pesimista**: Basado en los percentiles inferiores, considerando shocks económicos negativos.

## 4. Evaluación y selección de modelos
Se han aplicado diversas métricas para evaluar el rendimiento de los modelos:
- **RMSE (Root Mean Squared Error)**: Error cuadrático medio para evaluar precisión.
- **MAE (Mean Absolute Error)**: Error absoluto medio.
- **Backtesting**: Validación de los modelos con datos fuera de muestra.

La selección final del modelo se basa en un balance entre precisión predictiva y estabilidad de las proyecciones.

## 5. Optimización de hiperparámetros
Para mejorar la precisión de los modelos, se han optimizado hiperparámetros utilizando:
- **Grid Search con validación cruzada**: Evaluación exhaustiva de combinaciones de hiperparámetros.
- **Optimización Bayesiana**: Ajuste iterativo de parámetros mediante distribuciones probabilísticas.

## 6. Conclusión
La combinación de modelos econométricos, estadísticos y de machine learning ha permitido desarrollar una metodología robusta para la proyección de variables macroeconómicas y de mercado. Este enfoque facilita la toma de decisiones estratégicas basada en datos y mejora la capacidad de anticipar cambios en el entorno económico y financiero.

---
**Autor:** Héctor Daniel González Vargas 
**Fecha:** Febrero 2025
