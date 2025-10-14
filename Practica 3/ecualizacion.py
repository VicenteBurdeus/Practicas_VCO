import cv2
import numpy as np
import matplotlib.pyplot as plt

def imadjust(img, low_in, high_in):
    """
    Ajusta los niveles de intensidad de la imagen.
    :param img: imagen de entrada
    :param low_in: valor mínimo de entrada
    :param high_in: valor máximo de entrada
    :return: imagen ajustada
    """
    min_val = low_in
    max_val = high_in
    # Ajustar los valores de intensidad
    img_out = np.round(255.0 * (img - min_val)/(max_val - min_val + 1)).astype(np.uint8)
    # Recortar valores fuera del rango
    img_out[img < min_val] = 0
    img_out[img > max_val] = 255
    return img_out

# Leer la imagen y convertirla a escala de grises
img = cv2.imread('Practica 3/AloeVera.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Ecualizar el histograma
img_eq = cv2.equalizeHist(img_gray)

# Crear una figura con 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Mostrar imagen original en escala de grises
axs[0,0].imshow(img_gray, cmap='gray')
axs[0,0].set_title('Imagen Original (Escala de Grises)')
axs[0,0].axis('off')

# Mostrar histograma original
hist_gray = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
axs[0,1].plot(hist_gray, color='black')
axs[0,1].set_title('Histograma Original')
axs[0,1].set_xlim([0, 256])
axs[0,1].grid(True, alpha=0.3)
axs[0,1].set_xlabel('Valor de pixel')
axs[0,1].set_ylabel('Frecuencia')

# Mostrar imagen ecualizada
axs[1,0].imshow(img_eq, cmap='gray')
axs[1,0].set_title('Imagen Ecualizada')
axs[1,0].axis('off')

# Mostrar histograma ecualizado
hist_eq = cv2.calcHist([img_eq], [0], None, [256], [0, 256])
axs[1,1].plot(hist_eq, color='black')
axs[1,1].set_title('Histograma Ecualizado')
axs[1,1].set_xlim([0, 256])
axs[1,1].grid(True, alpha=0.3)
axs[1,1].set_xlabel('Valor de pixel')
axs[1,1].set_ylabel('Frecuencia')

# Ajustar el layout y mostrar
plt.tight_layout()
plt.show()