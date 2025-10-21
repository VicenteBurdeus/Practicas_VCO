import cv2
import numpy as np
import matplotlib.pyplot as plt

def aplicar_sobel(img):
    """
    Aplica los operadores Sobel para detectar bordes horizontales y verticales
    """
    # Convertir a escala de grises si la imagen es en color
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Aplicar Sobel en x e y
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calcular la magnitud del gradiente
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalizar para visualización
    sobelx = cv2.normalize(sobelx, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    sobely = cv2.normalize(sobely, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return sobelx, sobely, magnitude

def aplicar_log(img, sigma=1):
    """
    Aplica el operador LoG (Laplaciano del Gaussiano)
    """
    # Convertir a escala de grises si la imagen es en color
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # 1. Aplicar filtro Gaussiano
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    
    # 2. Aplicar Laplaciano
    laplacian = cv2.Laplacian(blur, cv2.CV_64F)
    
    # Normalizar para visualización
    laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return laplacian

# Leer la imagen
img = cv2.imread('AloeVera.jpg')
if img is None:
    print("Error: No se pudo cargar la imagen AloeVera.jpg")
    exit()

# Convertir de BGR a RGB para visualización
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Aplicar Sobel
sobelx, sobely, sobel_magnitude = aplicar_sobel(img)

# Aplicar LoG con diferentes valores de sigma
log_sigma1 = aplicar_log(img, sigma=1)
log_sigma2 = aplicar_log(img, sigma=2)

# Mostrar resultados
plt.figure(figsize=(15, 10))

# Imagen original
plt.subplot(231)
plt.imshow(img_rgb)
plt.title('Imagen Original')
plt.axis('off')

# Sobel X
plt.subplot(232)
plt.imshow(sobelx, cmap='gray')
plt.title('Sobel X (Bordes Verticales)')
plt.axis('off')

# Sobel Y
plt.subplot(233)
plt.imshow(sobely, cmap='gray')
plt.title('Sobel Y (Bordes Horizontales)')
plt.axis('off')

# Magnitud del gradiente Sobel
plt.subplot(234)
plt.imshow(sobel_magnitude, cmap='gray')
plt.title('Magnitud del Gradiente Sobel')
plt.axis('off')

# LoG con sigma=1
plt.subplot(235)
plt.imshow(log_sigma1, cmap='gray')
plt.title('LoG (σ=1)')
plt.axis('off')

# LoG con sigma=2
plt.subplot(236)
plt.imshow(log_sigma2, cmap='gray')
plt.title('LoG (σ=2)')
plt.axis('off')

plt.tight_layout()
plt.show()