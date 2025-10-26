import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob

def distance_matrix(X, centers):
    """Calcular matriz de distancias entre todos los puntos y centros"""
    # Reshape para broadcasting: (n_samples, 1, n_features) - (1, k, n_features)
    diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))

def initialize_centers_plusplus(X, k):
    """Inicializar centroides usando k-means++
    
    Algoritmo:
    1. Elegir primer centro aleatoriamente
    2. Para cada centro adicional:
        - Calcular D(x)² para cada punto (distancia al cuadrado al centro más cercano)
        - Elegir nuevo centro con probabilidad proporcional a D(x)²
    3. Retornar los k centros
    """
    n_samples, n_features = X.shape
    centers = np.zeros((k, n_features))
    
    # Elegir primer centro aleatoriamente
    first_center_idx = np.random.randint(n_samples)
    centers[0] = X[first_center_idx]
    
    # Elegir los centros restantes
    for c in range(1, k):
        # Calcular distancias al cuadrado al centro más cercano
        closest_dist_sq = np.min(distance_matrix(X, centers[:c])**2, axis=1)
        
        # Calcular probabilidades
        probs = closest_dist_sq / closest_dist_sq.sum()
        
        # Elegir siguiente centro
        cumsum = np.cumsum(probs)
        r = np.random.rand()
        next_center_idx = np.searchsorted(cumsum, r)
        centers[c] = X[next_center_idx]
    
    return centers

def assign_clusters(X, centers):
    """Asignar puntos a clusters usando la distancia más cercana"""
    distances = distance_matrix(X, centers)
    return np.argmin(distances, axis=1)

def update_centers(X, labels, k):
    """Actualizar posición de los centros como media de los puntos asignados"""
    centers = np.zeros((k, X.shape[1]))
    for i in range(k):
        mask = labels == i
        if np.any(mask):
            centers[i] = X[mask].mean(axis=0)
    return centers

def kmeans_plusplus(X, k, max_iters=100, tol=1e-4):
    """Implementación de k-means++
    
    Parámetros:
    -----------
    X : array, shape (n_samples, n_features)
        Datos de entrada
    k : int
        Número de clusters
    max_iters : int, opcional (default=100)
        Número máximo de iteraciones
    tol : float, opcional (default=1e-4)
        Tolerancia para determinar convergencia
        
    Retorna:
    --------
    centers : array, shape (k, n_features)
        Centroides finales
    labels : array, shape (n_samples,)
        Etiquetas de cluster para cada punto
    """
    # Inicializar centros usando k-means++
    centers = initialize_centers_plusplus(X, k)
    
    for iteration in range(max_iters):
        old_centers = centers.copy()
        
        # Asignar puntos a clusters
        labels = assign_clusters(X, centers)
        
        # Actualizar centros
        centers = update_centers(X, labels, k)
        
        # Verificar convergencia
        if np.all(np.abs(centers - old_centers) < tol):
            break
    
    return centers, labels

def apply_kmeans_to_image(image_path, k_values, color_space='RGB'):
    """Aplicar k-means++ a una imagen"""
    # Leer imagen
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")
    
    # Convertir de BGR a RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convertir al espacio de color especificado
    if color_space == 'HSV':
        img_color = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    else:
        img_color = img_rgb
    
    # Reshape la imagen para k-means
    height, width = img_color.shape[:2]
    X = img_color.reshape(-1, 3)
    X = X.astype(np.float32) / 255.0  # Normalizar valores
    
    # Crear figura
    n_cols = len(k_values) + 1
    plt.figure(figsize=(5*n_cols, 5))
    
    # Mostrar imagen original
    plt.subplot(1, n_cols, 1)
    plt.imshow(img_rgb)
    plt.title('Imagen Original')
    plt.axis('off')
    
    # Aplicar k-means++ para diferentes valores de k
    for idx, k in enumerate(k_values, 1):
        print(f"Aplicando k-means++ con k={k}")
        
        # Ejecutar k-means++
        centers, labels = kmeans_plusplus(X, k)
        
        # Crear imagen segmentada
        segmented = centers[labels].reshape(height, width, 3)
        
        if color_space == 'HSV':
            segmented = cv2.cvtColor((segmented * 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
        else:
            segmented = (segmented * 255).astype(np.uint8)
        
        # Mostrar resultado
        plt.subplot(1, n_cols, idx + 1)
        plt.imshow(segmented)
        plt.title(f'K-means++ (k={k})')
        plt.axis('off')
    
    plt.tight_layout()
    return plt.gcf()

def main():
    # Rutas de las imágenes
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, 'images')
    results_dir = os.path.join(script_dir, 'resultados')
    
    # Crear directorio de resultados si no existe
    os.makedirs(results_dir, exist_ok=True)
    
    # Obtener imágenes
    image_paths = glob.glob(os.path.join(images_dir, '*nemo*.jpg'))
    
    if not image_paths:
        print("No se encontraron imágenes de nemo en la carpeta 'images'")
        return
    
    # Valores de k a probar
    k_values = [3, 4, 5]
    
    # Procesar cada imagen
    for img_path in image_paths:
        print(f"\nProcesando imagen: {os.path.basename(img_path)}")
        
        # Aplicar k-means++ en espacio RGB
        fig_rgb = apply_kmeans_to_image(img_path, k_values, color_space='RGB')
        fig_rgb.suptitle(f'Segmentación K-means++ en RGB - {os.path.basename(img_path)}')
        
        # Aplicar k-means++ en espacio HSV
        fig_hsv = apply_kmeans_to_image(img_path, k_values, color_space='HSV')
        fig_hsv.suptitle(f'Segmentación K-means++ en HSV - {os.path.basename(img_path)}')
        
        # Guardar resultados
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        rgb_output = os.path.join(results_dir, f'kmeans_plusplus_{base_name}_RGB.png')
        hsv_output = os.path.join(results_dir, f'kmeans_plusplus_{base_name}_HSV.png')
        
        fig_rgb.savefig(rgb_output)
        fig_hsv.savefig(hsv_output)
        print(f"Resultados guardados en:\n{rgb_output}\n{hsv_output}")

if __name__ == "__main__":
    main()