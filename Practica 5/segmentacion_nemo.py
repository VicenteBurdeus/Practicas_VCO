import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def create_color_mask(image_hsv, lower_bound, upper_bound):
    """Crear máscara para un rango de color específico"""
    return cv2.inRange(image_hsv, lower_bound, upper_bound)

def apply_morphology(mask):
    """Aplicar operaciones morfológicas para mejorar la máscara"""
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    
    # Eliminar ruido pequeño
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    # Rellenar huecos
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium)
    return mask

def segment_nemo(image):
    """Segmentar a Nemo usando múltiples rangos de color"""
    # Convertir a HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Definir rangos de color
    # Naranja (cuerpo principal)
    lower_orange1 = np.array([0, 120, 100])
    upper_orange1 = np.array([10, 255, 255])
    lower_orange2 = np.array([170, 120, 100])
    upper_orange2 = np.array([180, 255, 255])
    
    # Blanco (franjas)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 40, 255])
    
    # Crear máscaras individuales
    mask_orange1 = create_color_mask(image_hsv, lower_orange1, upper_orange1)
    mask_orange2 = create_color_mask(image_hsv, lower_orange2, upper_orange2)
    mask_white = create_color_mask(image_hsv, lower_white, upper_white)
    
    # Combinar máscaras naranjas
    mask_orange = cv2.bitwise_or(mask_orange1, mask_orange2)
    
    # Aplicar mejoras morfológicas
    mask_orange = apply_morphology(mask_orange)
    mask_white = apply_morphology(mask_white)
    
    # Combinar todas las máscaras
    mask_combined = cv2.bitwise_or(mask_orange, mask_white)
    
    # Aplicar la máscara a la imagen original
    result = cv2.bitwise_and(image, image, mask=mask_combined)
    
    return mask_orange, mask_white, mask_combined, result

def process_all_images():
    """Procesar todas las imágenes de Nemo"""
    # Obtener la ruta del script actual
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, "images")
    
    # Buscar todas las imágenes de Nemo
    image_paths = glob.glob(os.path.join(images_dir, "nemo*.jpg"))
    
    for image_path in sorted(image_paths):
        # Leer imagen
        image = cv2.imread(image_path)
        if image is None:
            print(f"No se pudo cargar la imagen: {image_path}")
            continue
            
        # Obtener el nombre base de la imagen
        image_name = os.path.basename(image_path)
        
        # Segmentar la imagen
        mask_orange, mask_white, mask_combined, result = segment_nemo(image)
        
        # Crear visualización
        plt.figure(figsize=(15, 10))
        
        # Imagen original
        plt.subplot(231)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Imagen Original')
        plt.axis('off')
        
        # Máscara naranja
        plt.subplot(232)
        plt.imshow(mask_orange, cmap='gray')
        plt.title('Máscara Naranja')
        plt.axis('off')
        
        # Máscara blanca
        plt.subplot(233)
        plt.imshow(mask_white, cmap='gray')
        plt.title('Máscara Blanca')
        plt.axis('off')
        
        # Máscara combinada
        plt.subplot(234)
        plt.imshow(mask_combined, cmap='gray')
        plt.title('Máscara Combinada')
        plt.axis('off')
        
        # Resultado final
        plt.subplot(235)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title('Resultado Final')
        plt.axis('off')
        
        plt.suptitle(f'Segmentación de {image_name}')
        plt.tight_layout()
        plt.show()
        
        # Guardar resultados
        output_dir = os.path.join(script_dir, "resultados")
        os.makedirs(output_dir, exist_ok=True)
        
        cv2.imwrite(os.path.join(output_dir, f"{image_name}_mask_orange.jpg"), mask_orange)
        cv2.imwrite(os.path.join(output_dir, f"{image_name}_mask_white.jpg"), mask_white)
        cv2.imwrite(os.path.join(output_dir, f"{image_name}_mask_combined.jpg"), mask_combined)
        cv2.imwrite(os.path.join(output_dir, f"{image_name}_result.jpg"), result)

if __name__ == "__main__":
    process_all_images()

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