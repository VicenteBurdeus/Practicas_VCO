import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def redimensionar_imagen(img, size=(256, 256)):
    """
    Redimensiona una imagen al tamaño especificado
    """
    return cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)

def mostrar_imagenes(imagenes, titulos):
    """
    Muestra múltiples imágenes en una fila
    """
    n = len(imagenes)
    plt.figure(figsize=(15, 5))
    
    for i in range(n):
        plt.subplot(1, n, i+1)
        if len(imagenes[i].shape) == 2:  # Imagen en escala de grises
            plt.imshow(imagenes[i], cmap='gray')
        else:  # Imagen en color
            plt.imshow(cv2.cvtColor(imagenes[i], cv2.COLOR_BGR2RGB))
        plt.title(titulos[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Definir la ruta de las imágenes relativa al directorio del script
script_dir = os.path.dirname(os.path.abspath(__file__))
dirimg = os.path.join(script_dir, "Imagenes")

# 1. Leer las imágenes (usar rutas absolutas para evitar problemas de cwd)
img1_path = os.path.join(dirimg, "zorro.tiff")
img2_path = os.path.join(dirimg, "fish.gif")
img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("Error: No se pudieron cargar las imágenes")
    exit()

# 2. Redimensionar ambas imágenes a 256x256
img1_resized = redimensionar_imagen(img1)
img2_resized = redimensionar_imagen(img2)

# Crear una figura con todos los resultados
plt.figure(figsize=(15, 10))

# Imágenes originales
plt.subplot(231)
plt.imshow(img1_resized, cmap='gray')
plt.title('Imagen 1 (zorro)')
plt.axis('off')

plt.subplot(232)
plt.imshow(img2_resized, cmap='gray')
plt.title('Imagen 2 (pez)')
plt.axis('off')

# Suma
suma = cv2.add(img1_resized, img2_resized)
plt.subplot(233)
plt.imshow(suma, cmap='gray')
plt.title('Suma')
plt.axis('off')

# Resta
resta = cv2.subtract(img1_resized, img2_resized)
plt.subplot(234)
plt.imshow(resta, cmap='gray')
plt.title('Resta')
plt.axis('off')

# Multiplicación (normalizada)
mult = cv2.multiply(img1_resized, img2_resized, scale=1/255)
plt.subplot(235)
plt.imshow(mult, cmap='gray')
plt.title('Multiplicación')
plt.axis('off')

# Combinación lineal
img1_scaled = cv2.multiply(img1_resized.astype(float), 1.8)
img2_scaled = cv2.multiply(img2_resized.astype(float), 1.2)
combinacion = img1_scaled - img2_scaled + 128
combinacion = np.clip(combinacion, 0, 255).astype(np.uint8)

plt.subplot(236)
plt.imshow(combinacion, cmap='gray')
plt.title('Combinación Lineal\n(1.8*IMG1 - 1.2*IMG2 + 128)')
plt.axis('off')

plt.tight_layout()
plt.show()

# Guardar resultados
cv2.imwrite('resultado_suma.jpg', suma)
cv2.imwrite('resultado_resta.jpg', resta)
cv2.imwrite('resultado_multiplicacion.jpg', mult)
cv2.imwrite('resultado_combinacion.jpg', combinacion)

# Visualización 3D de la imagen
def visualizar_3d(imagen, titulo):
    # Crear una malla de coordenadas
    y, x = np.mgrid[0:imagen.shape[0], 0:imagen.shape[1]]
    
    # Crear la figura 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Crear el mapa de superficie
    surf = ax.plot_surface(x, y, imagen, cmap='viridis')
    
    # Añadir barra de color
    fig.colorbar(surf)
    
    # Configurar la visualización
    ax.set_title(titulo)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Intensidad')
    
    plt.show()
    
    # Crear también una visualización con contourf
    plt.figure(figsize=(10, 8))
    plt.contourf(x, y, imagen, levels=20, cmap='viridis')
    plt.colorbar(label='Intensidad')
    plt.title(f'{titulo} (Mapa de contorno)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# Visualizar la primera imagen en 3D (redimensionada para mejor rendimiento)
img_small = cv2.resize(img1_resized, (128, 128))
visualizar_3d(img_small, 'Visualización 3D de la imagen')
cv2.imwrite('resultado_resta.jpg', resta)
cv2.imwrite('resultado_multiplicacion.jpg', mult)
cv2.imwrite('resultado_combinacion.jpg', combinacion)

# Visualización 3D de la imagen
def visualizar_3d(imagen, titulo):
    # Crear una malla de coordenadas
    y, x = np.mgrid[0:imagen.shape[0], 0:imagen.shape[1]]
    
    # Crear la figura 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Crear el mapa de contorno
    surf = ax.plot_surface(x, y, imagen, cmap='viridis')
    
    # Añadir barra de color
    fig.colorbar(surf)
    
    # Configurar la visualización
    ax.set_title(titulo)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Intensidad')
    
    plt.show()
    
    # Crear también una visualización con contourf
    plt.figure(figsize=(10, 8))
    plt.contourf(x, y, imagen, levels=20, cmap='viridis')
    plt.colorbar(label='Intensidad')
    plt.title(f'{titulo} (Mapa de contorno)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# Visualizar la primera imagen en 3D (redimensionada para mejor rendimiento)
img_small = cv2.resize(img1_resized, (128, 128))
visualizar_3d(img_small, 'Visualización 3D de la imagen')
    

