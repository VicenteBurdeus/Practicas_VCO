import cv2
import numpy as np
import matplotlib.pyplot as plt

def mostrar_canales(img, titulos, nombre_espacio):
    """
    Muestra los tres canales de una imagen en escala de grises
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Canales del espacio de color {nombre_espacio}')
    
    for i, (canal, titulo) in enumerate(zip(cv2.split(img), titulos)):
        axs[i].imshow(canal, cmap='gray')
        axs[i].set_title(titulo)
        axs[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Leer la imagen
img = cv2.imread('Practica 2/AloeVera.jpg')
if img is None:
    print("Error: No se pudo cargar la imagen AloeVera.jpg")
    exit()

# 1. Mostrar canales RGB
# Convertir de BGR (OpenCV) a RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mostrar_canales(img_rgb, ['Canal R', 'Canal G', 'Canal B'], 'RGB')

# 2. Mostrar canales HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# Normalizar el canal H para visualización (originalmente 0-180)
h, s, v = cv2.split(img_hsv)
h = h.astype(float) * 255 / 180  # Escalar H de 0-180 a 0-255
img_hsv_display = cv2.merge([h.astype(np.uint8), s, v])
mostrar_canales(img_hsv_display, ['Canal H', 'Canal S', 'Canal V'], 'HSV')

# 3. Mostrar canales Lab
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
# Normalizar los canales a y b para visualización (originalmente 0-255 centrado en 128)
L, a, b = cv2.split(img_lab)
# Escalar a y b de [-127, 127] a [0, 255]
a = (a + 128).astype(np.uint8)
b = (b + 128).astype(np.uint8)
img_lab_display = cv2.merge([L, a, b])
mostrar_canales(img_lab_display, ['Canal L', 'Canal a', 'Canal b'], 'Lab')

# Mostrar la imagen original
plt.figure(figsize=(8, 6))
plt.imshow(img_rgb)
plt.title('Imagen Original (RGB)')
plt.axis('off')
plt.show()