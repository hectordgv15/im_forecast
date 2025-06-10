# Librerías
import pandas as pd
import numpy as np

# Módulos propios
from IMForecast.ts_functions import ts_functions

class VasicekIM:
    
    def __init__(self):
        pass
    
    # Calibración del modelo de Vasicek
    def calibrate_vasicek_parameters(self, rates, dt):
        '''
        Esta función calibra los parámetros del modelo de Vasicek (kappa, theta, sigma y r0)
        a partir de una serie histórica de tasas de interés.
        
        Parámetros:
            rates: Serie temporal de tasas de interés (pandas Series).
            dt: Intervalo de tiempo entre observaciones (por defecto, trimestral).

        Retorna:
            [kappa, theta, sigma, r0]: Lista de parámetros calibrados.
        ''' 
        n = len(rates)  # Número de observaciones en la serie histórica.

        # Suma de las tasas de interés desde el inicio hasta el penúltimo periodo.
        Sx = sum(rates.iloc[0 : (n - 1)])
        
        # Suma de las tasas de interés desde el segundo periodo hasta el último.
        Sy = sum(rates.iloc[1 : n])
        
        # Producto escalar de las tasas de interés en distintos periodos.
        Sxx = np.dot(rates.iloc[0 : (n - 1)], rates.iloc[0 : (n - 1)])
        Sxy = np.dot(rates.iloc[0 : (n - 1)], rates.iloc[1 : n])
        Syy = np.dot(rates.iloc[1 : n], rates.iloc[1 : n])
        
        # Estimación de theta (media de reversión) usando mínimos cuadrados.
        theta = (Sy * Sxx - Sx * Sxy) / (n * (Sxx - Sxy) - (Sx**2 - Sx * Sy))
        
        # Estimación de kappa (velocidad de reversión a la media).
        kappa = -np.log((Sxy - theta * Sx - theta * Sy + n * theta**2) /
                        (Sxx - 2 * theta * Sx + n * theta**2)) / dt
        
        # Factor de amortiguación a través del tiempo.
        a = np.exp(-kappa * dt)
        
        # Estimación de sigma^2 (varianza del ruido) ajustada por la reversión a la media.
        sigmah2 = (Syy - 2 * a * Sxy + a**2 * Sxx - 2 * theta * (1 - a) * (Sy - a * Sx) +
                n * theta**2 * (1 - a)**2) / n
        
        # Conversión a desviación estándar de la volatilidad del ruido.
        sigma = np.sqrt(sigmah2 * 2 * kappa / (1 - a**2))
        
        # Tasa inicial para simulaciones (último dato de la serie histórica).
        r0 = rates.iloc[n-1]
        
        return [kappa, theta, sigma, r0]

    # Simulación de una tasa futura utilizando el modelo de Vasicek.
    def simulate_next_vasicek_rate(self, r, kappa, theta, sigma, dt):  
        '''
        Simula la siguiente tasa de interés utilizando el modelo de Vasicek.
        
        Parámetros:
            r: Tasa actual.
            kappa: Velocidad de reversión.
            theta: Nivel promedio al que las tasas convergen.
            sigma: Volatilidad del ruido estocástico.
            dt: Intervalo de tiempo.

        Retorna:
            Tasa de interés simulada para el siguiente periodo.
        ''' 
        # Cálculo del término determinístico de reversión a la media.    
        val1 = np.exp(-1 * kappa * dt)
        
        # Varianza del ruido estocástico.
        val2 = (sigma**2) * (1 - val1**2) / (2 * kappa)
        
        # Cálculo de la siguiente tasa de interés.
        out = r * val1 + theta * (1 - val1) + (np.sqrt(val2)) * np.random.normal()
        
        return out

    # Simulación de una trayectoria de tasas de interés.
    def simulate_vasicek_path(self, N, r0, kappa, theta, sigma, dt):
        '''
        Genera una trayectoria simulada de tasas de interés.
        
        Parámetros:
            N: Número de pasos de la simulación.
            r0: Tasa inicial.
            kappa: Velocidad de reversión.
            theta: Nivel promedio al que las tasas convergen.
            sigma: Volatilidad del ruido estocástico.
            dt: Intervalo de tiempo.

        Retorna:
            Lista con la trayectoria simulada de tasas de interés.
        ''' 
        short_r = [0] * N  # Inicializa la lista de tasas.
        short_r[0] = r0    # Configura la tasa inicial.
        
        # Genera las tasas para cada periodo basado en la anterior.
        for i in range(1, N):
            short_r[i] = self.simulate_next_vasicek_rate(short_r[i - 1], kappa, theta, sigma, dt)
        
        return short_r

    # Simulación de múltiples trayectorias de tasas de interés.
    def multiple_vacisek_sim(self, M, N, r0, kappa, theta, sigma, dt):
        '''
        Genera múltiples trayectorias simuladas de tasas de interés.
        
        Parámetros:
            M: Número de trayectorias a simular.
            N: Número de pasos en cada trayectoria.
            r0: Tasa inicial.
            kappa: Velocidad de reversión.
            theta: Nivel promedio al que las tasas convergen.
            sigma: Volatilidad del ruido estocástico.
            dt: Intervalo de tiempo.

        Retorna:
            Matriz (N x M) con las trayectorias simuladas de tasas de interés.
        '''
        sim_arr = np.ndarray((N, M))  # Inicializa una matriz vacía para almacenar las simulaciones.
        
        # Genera cada trayectoria de manera independiente.        
        for i in range(0, M):
            sim_arr[:, i] = self.simulate_vasicek_path(N, r0, kappa, theta, sigma, dt)
        
        return sim_arr