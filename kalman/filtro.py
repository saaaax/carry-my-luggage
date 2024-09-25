import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter

# Función para crear el filtro de Kalman
def create_kalman_filter():
    kf = KalmanFilter(dim_x=2, dim_z=1)
    
    # Estado inicial [pos, velocidad]
    kf.x = np.array([[0.],    # Posición inicial
                     [0.]])   # Velocidad inicial

    # Matriz de transición del estado (supone movimiento constante)
    kf.F = np.array([[1., 1.], 
                     [0., 1.]])
    
    # Matriz de medición (solo estamos midiendo la posición)
    kf.H = np.array([[1., 0.]])
    
    # Matriz de covarianza del proceso
    kf.P *= 1000.  # Alta incertidumbre inicial
    
    # Ruido del proceso
    kf.Q = np.array([[1., 0.],
                     [0., 1.]])
    
    # Ruido de medición (supón que la medición de la distancia tiene ruido)
    kf.R = np.array([[5.]])
    
    return kf

# Generación de datos simulados (distancias medidas con ruido)
def generate_data(true_distance, num_steps, noise_std):
    true_positions = true_distance * np.ones(num_steps)  # La distancia verdadera
    measurements = true_positions + np.random.normal(0, noise_std, size=num_steps)  # Añadimos ruido
    return true_positions, measurements

# Aplicar filtro de Kalman
def apply_kalman_filter(kf, measurements):
    estimates = []
    
    for z in measurements:
        kf.predict()
        kf.update(np.array([[z]]))
        estimates.append(kf.x[0, 0])
    
    return estimates

# Configuraciones iniciales
true_distance = 10  # Distancia real constante al objeto (en metros)
num_steps = 50  # Número de pasos de simulación
noise_std = 2.0  # Desviación estándar del ruido en la medición

# Crear filtro de Kalman
kf = create_kalman_filter()

# Generar datos simulados
true_positions, measurements = generate_data(true_distance, num_steps, noise_std)

# Aplicar el filtro de Kalman
filtered_estimates = apply_kalman_filter(kf, measurements)

# Graficar resultados
plt.figure(figsize=(10, 6))
plt.plot(true_positions, label='Distancia real', linestyle='--')
plt.plot(measurements, label='Mediciones ruidosas', linestyle=':', color='orange')
plt.plot(filtered_estimates, label='Estimación (Filtro de Kalman)', color='green')
plt.legend()
plt.xlabel('Tiempo (pasos)')
plt.ylabel('Distancia (m)')
plt.title('Estimación de la distancia con Filtro de Kalman')
plt.grid()
plt.show()
