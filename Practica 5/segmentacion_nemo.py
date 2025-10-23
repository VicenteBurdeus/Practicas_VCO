import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os
import glob

def create_color_mask(image_hsv, lower_bound, upper_bound):
    """Crear máscara para un rango de color específico"""
    mask = cv2.inRange(image_hsv, lower_bound, upper_bound)
    return mask

def plot_3d_color_space(image, image_hsv, color_masks=None, title="Distribución de colores"):
    """Visualizar la distribución de colores en 3D con máscaras marcadas"""
    fig = plt.figure(figsize=(15, 5))
    
    # Imagen original
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title("Imagen Original")
    ax1.axis('off')
    
    # Máscara(s)
    ax2 = fig.add_subplot(1, 3, 2)
    if color_masks is not None:
        combined_mask = np.zeros_like(color_masks[0])
        for mask in color_masks:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        ax2.imshow(combined_mask, cmap='gray')
        ax2.set_title("Máscara Combinada")
    ax2.axis('off')
    
    # Distribución de colores 3D
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    
    # Preparar datos
    h, s, v = cv2.split(image_hsv)
    
    # Convertir a RGB para visualización
    pixel_colors = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).reshape(-1, 3)
    pixel_colors = pixel_colors.astype(float) / 255.0
    
    # Crear scatter plot
    ax3.scatter(h.flatten(), s.flatten(), v.flatten(),
                facecolors=pixel_colors, marker=".", alpha=0.1)
    
    # Si hay máscaras, resaltar los puntos seleccionados
    if color_masks is not None:
        for mask, color in zip(color_masks, ['red', 'white']):
            masked_h = h[mask > 0]
            masked_s = s[mask > 0]
            masked_v = v[mask > 0]
            ax3.scatter(masked_h, masked_s, masked_v,
                       c=color, marker=".", alpha=0.3)
    
    ax3.set_xlabel("Hue")
    ax3.set_ylabel("Saturation")
    ax3.set_zlabel("Value")
    ax3.set_title("Espacio de Color HSV")
    
    plt.tight_layout()
    return fig

def process_image(image_path):
    """Procesar una imagen con segmentación de múltiples colores"""
    # Leer imagen
    image = cv2.imread(image_path)
    if image is None:
        print(f"No se pudo cargar la imagen: {image_path}")
        return
    
    # Convertir a HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Definir rangos de color para Nemo (naranja y blanco)
    # Naranja (dos rangos para cubrir mejor el espectro del naranja)
    lower_orange1 = np.array([0, 120, 100])    # Naranja rojizo
    upper_orange1 = np.array([10, 255, 255])
    lower_orange2 = np.array([170, 120, 100])  # Naranja rojizo (continuación del espectro)
    upper_orange2 = np.array([180, 255, 255])
    
    # Blanco (ajustado para ser más selectivo)
    lower_white = np.array([0, 0, 180])       # Aumentado el valor mínimo
    upper_white = np.array([180, 40, 255])
    
    # Crear máscaras
    # Combinar los dos rangos de naranja
    mask_orange1 = create_color_mask(image_hsv, lower_orange1, upper_orange1)
    mask_orange2 = create_color_mask(image_hsv, lower_orange2, upper_orange2)
    mask_orange = cv2.bitwise_or(mask_orange1, mask_orange2)
    
    # Crear máscara para blanco
    mask_white = create_color_mask(image_hsv, lower_white, upper_white)
    
    # Mejorar las máscaras con operaciones morfológicas más sofisticadas
    # Usar kernel elíptico para preservar mejor las formas curvas
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

    # Procesar máscara naranja
    # Primero eliminar ruido pequeño
    mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_OPEN, kernel_small)
    # Suavizar bordes
    mask_orange = cv2.GaussianBlur(mask_orange, (5,5), 0)
    # Binarizar nuevamente con umbral adaptativo
    mask_orange = cv2.threshold(mask_orange, 127, 255, cv2.THRESH_BINARY)[1]
    # Rellenar huecos
    mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_CLOSE, kernel_medium)
    # Suavizar los bordes finales
    mask_orange = cv2.medianBlur(mask_orange, 5)

    # Procesar máscara blanca de manera similar
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel_small)
    mask_white = cv2.GaussianBlur(mask_white, (5,5), 0)
    mask_white = cv2.threshold(mask_white, 127, 255, cv2.THRESH_BINARY)[1]
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel_medium)
    mask_white = cv2.medianBlur(mask_white, 5)

    # Opcional: dilatar ligeramente para conectar regiones cercanas
    mask_orange = cv2.dilate(mask_orange, kernel_small, iterations=1)
    mask_white = cv2.dilate(mask_white, kernel_small, iterations=1)
    
    # Visualizar resultados
    fig = plot_3d_color_space(image, image_hsv, [mask_orange, mask_white], 
                             title=f"Análisis de {os.path.basename(image_path)}")
    
    # Mostrar resultado de la segmentación
    result = image.copy()
    result[mask_orange > 0] = [0, 165, 255]  # Naranja brillante
    result[mask_white > 0] = [255, 255, 255] # Blanco
    
    # Mostrar resultado final
    cv2.imshow(f"Segmentación - {os.path.basename(image_path)}", 
               np.hstack([image, result]))
    
    return fig

# Buscar imágenes en diferentes rutas posibles
possible_paths = [
    "images/nemo*.jpg",
    "../Practica 5/images/nemo*.jpg",
    "Practica 5/images/nemo*.jpg",
    "./images/nemo*.jpg"
]

image_paths = []
for path_pattern in possible_paths:
    found_images = glob.glob(path_pattern)
    if found_images:
        image_paths.extend(found_images)
        break

if not image_paths:
    print("No se encontraron imágenes. Se buscó en:")
    for path in possible_paths:
        print(f"- {os.path.abspath(path)}")
    print("\nDirectorio actual:", os.getcwd())
    exit(1)

print(f"Procesando {len(image_paths)} imágenes...")
for image_path in image_paths:
    fig = process_image(image_path)
    plt.show()
    cv2.waitKey(0)

cv2.destroyAllWindows()