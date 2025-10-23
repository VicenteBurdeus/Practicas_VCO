import matplotlib.pyplot as plt 
from matplotlib import colors 
import numpy as np 
from PIL import Image 
import cv2 


# Intentar varias rutas posibles para la imagen
possible_paths = [
    "images/nemo0.jpg",
    "../Practica 5/images/nemo0.jpg",
    "./images/nemo0.jpg",
    "Practica 5/images/nemo0.jpg"
]
nemo = None

# Imprimir el directorio actual para debug

for path in possible_paths:
        nemo = cv2.imread(path)

# Convertir a HSV
nemo_hsv = cv2.cvtColor(nemo, cv2.COLOR_BGR2HSV)

# Crear la figura
fig = plt.figure(figsize=(12, 6)) 

# Mostrar imagen original
ax1 = fig.add_subplot(1, 2, 1, title="Original") 
plt.imshow(cv2.cvtColor(nemo, cv2.COLOR_BGR2RGB))  # Convertir BGR a RGB para mostrar

# Plotting the image on 3D plot (HSV)
h, s, v = cv2.split(nemo_hsv)
axis = fig.add_subplot(1, 2, 2, projection="3d") 

# Preparar los colores para el scatter plot
# Convertimos HSV a RGB para mostrar los colores correctamente
pixel_colors_hsv = nemo_hsv.reshape((-1, 3))
pixel_colors_rgb = cv2.cvtColor(pixel_colors_hsv.reshape(1, -1, 3), cv2.COLOR_HSV2RGB).reshape(-1, 3)

pixel_colors = pixel_colors_rgb.astype(float) / 255.0

# Crear el scatter plot 3D
axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".", alpha=0.1)
axis.set_xlabel("Hue") 
axis.set_ylabel("Saturation") 
axis.set_zlabel("Value")

# Ajustar los límites de los ejes
axis.set_xlim([0, 180])  # Hue en OpenCV va de 0 a 180
axis.set_ylim([0, 255])  # Saturation va de 0 a 255
axis.set_zlim([0, 255])  # Value va de 0 a 255

# Ajustar el layout y mostrar
plt.tight_layout()
plt.show()