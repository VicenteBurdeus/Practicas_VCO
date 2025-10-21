import numpy as np
import matplotlib.pyplot as plt

# Crear un conjunto de puntos que forman un cuadrado
puntos = np.array([
    [0, 1],    # Punto inferior izquierdo
    [1, 0],    # Punto inferior derecho
    [0,-1],    # Punto superior derecho
    [-1, 0],    # Punto superior izquierdo
    [0, 1]     # Cerrar el cuadrado
])

# Crear figura
plt.figure(figsize=(15, 5))

# Dibujar puntos originales
plt.subplot(131)
plt.plot(puntos[:, 0], puntos[:, 1], 'b-o')
plt.title('Puntos Originales')
plt.grid(True)
plt.axis('equal')
plt.xlim(-8, 8)
plt.ylim(-8, 8)

# Matriz de traslación t=(-3,3)
T = np.array([
    [1, 0, -3],
    [0, 1, 3],
    [0, 0, 1]
])

# Agregar columna de unos para hacer transformación homogénea
puntos_homogeneos = np.hstack((puntos, np.ones((puntos.shape[0], 1))))

# Aplicar traslación
puntos_trasladados = np.dot(puntos_homogeneos, T.T)

# Dibujar puntos trasladados
plt.subplot(132)
plt.plot(puntos_trasladados[:, 0], puntos_trasladados[:, 1], 'r-o')
plt.title('Puntos Trasladados (-3,3)')
plt.grid(True)
plt.axis('equal')
plt.xlim(-5, 5)
plt.ylim(-5, 5)

# Matriz de rotación 30 grados
theta = np.radians(30)
R = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta), np.cos(theta), 0],
    [0, 0, 1]
])

# Aplicar rotación a los puntos trasladados
puntos_rotados = np.dot(puntos_homogeneos, np.dot(T, R).T)

# Dibujar puntos rotados
plt.subplot(133)
plt.plot(puntos_rotados[:, 0], puntos_rotados[:, 1], 'g-o')
plt.title('Puntos Rotados 30°')
plt.grid(True)
plt.axis('equal')
plt.xlim(-5, 5)
plt.ylim(-5, 5)

plt.tight_layout()
plt.show()