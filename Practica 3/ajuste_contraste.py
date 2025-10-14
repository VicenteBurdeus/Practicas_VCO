import cv2
import numpy as np
import matplotlib.pyplot as plt

def ajuste_contraste(img, min_val, max_val, modo=1):
    """
    Ajusta el contraste de la imagen según el modo especificado:
    modo 1: Dentro del rango -> 255, Fuera del rango -> 0
    modo 2: Dentro del rango -> 255, Fuera del rango -> mantener valores originales
    
    :param img: imagen de entrada
    :param min_val: valor mínimo del rango
    :param max_val: valor máximo del rango
    :param modo: 1 o 2, según el tipo de ajuste deseado
    :return: imagen ajustada
    """
    img_out = img.copy()
    
    # Máscara para valores dentro del rango
    mask_between = (img >= min_val) & (img <= max_val)
    
    if modo == 1:
        # Modo 1: Dentro -> 255, Fuera -> 0
        img_out[:] = 0  # Primero poner todo a 0
        img_out[mask_between] = 255  # Luego los valores dentro del rango a 255
    else:
        # Modo 2: Dentro -> 255, Fuera -> mantener valores originales
        img_out[mask_between] = 255  # Solo modificar los valores dentro del rango
    
    return img_out

# Leer la imagen y convertirla a escala de grises
img = cv2.imread('Practica 3/AloeVera.jpg', cv2.IMREAD_GRAYSCALE)

# Aplicar el ajuste de contraste con ambos modos
min_val = 100  # Valor mínimo
max_val = 200  # Valor máximo

# Aplicar ambos modos
img_modo1 = ajuste_contraste(img, min_val, max_val, modo=1)
img_modo2 = ajuste_contraste(img, min_val, max_val, modo=2)

# Crear una figura para mostrar las imágenes
plt.figure(figsize=(15, 10))

# Mostrar imagen original
plt.subplot(231)
plt.imshow(img, cmap='gray')
plt.title('Imagen Original')
plt.axis('off')

# Mostrar imagen ajustada modo 1
plt.subplot(232)
plt.imshow(img_modo1, cmap='gray')
plt.title(f'Modo 1: Dentro [255], Fuera [0]\n(Rango: {min_val}-{max_val})')
plt.axis('off')

# Mostrar imagen ajustada modo 2
plt.subplot(233)
plt.imshow(img_modo2, cmap='gray')
plt.title(f'Modo 2: Dentro [255], Fuera [original]\n(Rango: {min_val}-{max_val})')
plt.axis('off')


# Histograma original
plt.subplot(234)
plt.hist(img.ravel(), 256, [0, 256], color='black', alpha=0.7)
plt.title('Histograma Original')
plt.xlabel('Valor de pixel')
plt.ylabel('Frecuencia')

# Histograma modo 1
plt.subplot(235)
plt.hist(img_modo1.ravel(), 256, [0, 256], color='black', alpha=0.7)
plt.title('Histograma Modo 1')
plt.xlabel('Valor de pixel')
plt.ylabel('Frecuencia')

# Histograma modo 2
plt.subplot(236)
plt.hist(img_modo2.ravel(), 256, [0, 256], color='black', alpha=0.7)
plt.title('Histograma Modo 2')
plt.xlabel('Valor de pixel')
plt.ylabel('Frecuencia')

plt.tight_layout()
plt.show()