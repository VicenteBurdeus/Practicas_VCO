import matplotlib.pyplot as plt
import numpy as np
import cv2
from noise import gaussian_noise, saltAndPepper_noise

def aplicar_filtros(img_original, kernel_size=5):
    """
    Aplica ruido gaussiano y diferentes filtros a una imagen
    """
    # Hacer una copia de la imagen original para el ruido
    img_noise = img_original.copy()
    
    # Aplicar ruido gaussiano
    img_noise = gaussian_noise(img_noise)
    
    # Aplicar filtro gaussiano
    img_gaussian = cv2.GaussianBlur(img_noise, (kernel_size, kernel_size), 0)
    
    # Aplicar filtro de mediana
    img_median = cv2.medianBlur(img_noise, kernel_size)
    
    return img_noise, img_gaussian, img_median

# Leer la imagen
img = cv2.imread('AloeVera.jpg')
if img is None:
    print("Error: No se pudo cargar la imagen AloeVera.jpg")
    exit()

# Convertir de BGR a RGB para visualización
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Aplicar ruido y filtros
img_noise, img_gaussian, img_median = aplicar_filtros(img_rgb)

# Mostrar resultados
plt.figure(figsize=(15, 10))

# Imagen original
plt.subplot(221)
plt.imshow(img_rgb)
plt.title('Imagen Original')
plt.axis('off')

# Imagen con ruido gaussiano
plt.subplot(222)
plt.imshow(img_noise)
plt.title('Ruido Gaussiano')
plt.axis('off')

# Imagen con filtro gaussiano
plt.subplot(223)
plt.imshow(img_gaussian)
plt.title('Filtro Gaussiano')
plt.axis('off')

# Imagen con filtro de mediana
plt.subplot(224)
plt.imshow(img_median)
plt.title('Filtro de Mediana')
plt.axis('off')

plt.tight_layout()
plt.show()

# Mostrar comparación de diferencias
plt.figure(figsize=(15, 5))

# Diferencia entre imagen con ruido y filtro gaussiano
diff_gaussian = cv2.absdiff(img_noise, img_gaussian)
plt.subplot(121)
plt.imshow(diff_gaussian)
plt.title('Diferencia: Ruido vs Filtro Gaussiano')
plt.axis('off')

# Diferencia entre imagen con ruido y filtro de mediana
diff_median = cv2.absdiff(img_noise, img_median)
plt.subplot(122)
plt.imshow(diff_median)
plt.title('Diferencia: Ruido vs Filtro Mediana')
plt.axis('off')

plt.tight_layout()
plt.show()

Suavizado = np.array([
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9] ])

Agudizado = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0] ])

img_emboss = cv2.filter2D(src, -1, Emboss)
img_sepia = cv2.transform(src, Sepia)
img_paso_alto = cv2.filter2D(src, -1, Paso_alto)
img_suavizado = cv2.filter2D(src, -1, Suavizado)
img_agudizado = cv2.filter2D(src, -1, Agudizado)


plt.subplot(221)
plt.imshow(src)
plt.title('Imagen Original')
plt.axis('off')

plt.subplot(222)
plt.imshow(img_sepia)
plt.title('Sepia')
plt.axis('off')

plt.subplot(223)
plt.imshow(img_suavizado)
plt.title('Suavizado')
plt.axis('off')

plt.subplot(224)
plt.imshow(img_agudizado)
plt.title('Agudizado')
plt.axis('off')

plt.show()

