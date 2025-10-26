import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob

def distance_matrix(X, centers):
    """Calcular matriz de distancias entre todos los puntos y centros"""
    diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))

def assign_clusters(X, clusters, k):
    """Asignar puntos a clusters de manera vectorizada"""
    # Obtener centros como array
    centers = np.array([clusters[i]['center'] for i in range(k)])
    
    # Calcular todas las distancias de una vez
    distances = distance_matrix(X, centers)
    
    # Encontrar el cluster más cercano para cada punto
    cluster_indices = np.argmin(distances, axis=1)
    
    # Resetear los puntos en cada cluster
    for i in range(k):
        clusters[i]['points'] = []
    
    # Asignar puntos a sus clusters
    for idx, cluster_idx in enumerate(cluster_indices):
        clusters[cluster_idx]['points'].append(X[idx])
    
    return clusters

def update_clusters(clusters, k):
    """Actualizar centros de clusters"""
    for i in range(k):
        points = np.array(clusters[i]['points'])
        if points.shape[0] > 0:
            new_center = points.mean(axis=0)
            clusters[i]['center'] = new_center
            clusters[i]['points'] = []
    return clusters

def pred_cluster(X, clusters, k):
    """Predecir cluster para cada punto de manera vectorizada"""
    centers = np.array([clusters[i]['center'] for i in range(k)])
    distances = distance_matrix(X, centers)
    return np.argmin(distances, axis=1)

def initialize_clusters(k, X):
    """Inicializar clusters aleatoriamente"""
    clusters = {}
    np.random.seed(23)
    
    # Inicializar centros usando puntos aleatorios de los datos
    indices = np.random.choice(X.shape[0], k, replace=False)
    
    for idx in range(k):
        cluster = {
            'center': X[indices[idx]],
            'points': []
        }
        clusters[idx] = cluster
    return clusters

def run_kmeans(X, k, max_iters=100):
    """Ejecutar algoritmo k-means"""
    # Inicializar clusters
    clusters = initialize_clusters(k, X)
    
    # Iterar hasta convergencia o número máximo de iteraciones
    for _ in range(max_iters):
        old_centers = np.array([clusters[i]['center'] for i in range(k)])
        
        # Asignar puntos a clusters
        clusters = assign_clusters(X, clusters, k)
        
        # Actualizar centros
        clusters = update_clusters(clusters, k)
        
        # Verificar convergencia
        new_centers = np.array([clusters[i]['center'] for i in range(k)])
        if np.allclose(old_centers, new_centers):
            break
    
    return clusters, pred_cluster(X, clusters, k)

def plot_color_space(X, labels, centers, ax, color_space='RGB'):
    """Visualizar datos en el espacio de color"""
    scatter = ax.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', 
                        alpha=0.5, s=1)
    
    # Plotear centroides
    ax.scatter(centers[:,0], centers[:,1], c='red', marker='^', 
              s=200, label='Centroides')
    
    if color_space == 'RGB':
        ax.set_xlabel('R')
        ax.set_ylabel('G')
        ax.set_title('Espacio RGB (R vs G)')
    else:  # HSV
        ax.set_xlabel('H')
        ax.set_ylabel('S')
        ax.set_title('Espacio HSV (H vs S)')
    
    ax.legend()
    return scatter

def apply_kmeans_to_image(image_path, k_values, color_space='RGB'):
    """Aplicar k-means a una imagen con visualización mejorada"""
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
    
    # Preparar datos para k-means
    height, width = img_color.shape[:2]
    X = img_color.reshape(-1, 3)
    X = X.astype(np.float32) / 255.0
    
    # Configurar subplots
    n_cols = len(k_values) + 1
    fig = plt.figure(figsize=(5*n_cols, 12))
    
    # Mostrar imagen original
    ax_orig = plt.subplot2grid((3, n_cols), (0, 0))
    ax_orig.imshow(img_rgb)
    ax_orig.set_title('Imagen Original')
    ax_orig.axis('off')
    
    # Visualizar espacio de color original
    ax_color = plt.subplot2grid((3, n_cols), (1, 0))
    ax_color.scatter(X[:,0], X[:,1], c=X, alpha=0.5, s=1)
    if color_space == 'RGB':
        ax_color.set_title('Espacio RGB Original')
        ax_color.set_xlabel('R')
        ax_color.set_ylabel('G')
    else:
        ax_color.set_title('Espacio HSV Original')
        ax_color.set_xlabel('H')
        ax_color.set_ylabel('S')
    
    # Aplicar k-means para diferentes valores de k
    for idx, k in enumerate(k_values, 1):
        print(f"Aplicando k-means con k={k}")
        
        # Ejecutar k-means
        clusters, labels = run_kmeans(X, k)
        centers = np.array([clusters[i]['center'] for i in range(k)])
        
        # Reconstruir imagen segmentada
        segmented = centers[labels].reshape(height, width, 3)
        if color_space == 'HSV':
            segmented = cv2.cvtColor((segmented * 255).astype(np.uint8), 
                                   cv2.COLOR_HSV2RGB)
        else:
            segmented = (segmented * 255).astype(np.uint8)
        
        # Mostrar imagen segmentada
        ax_img = plt.subplot2grid((3, n_cols), (0, idx))
        ax_img.imshow(segmented)
        ax_img.set_title(f'K-means (k={k})')
        ax_img.axis('off')
        
        # Visualizar clusters en espacio de color
        ax_cluster = plt.subplot2grid((3, n_cols), (1, idx))
        plot_color_space(X, labels, centers, ax_cluster, color_space)
        
        # Mostrar histograma de asignaciones
        ax_hist = plt.subplot2grid((3, n_cols), (2, idx))
        ax_hist.hist(labels, bins=k, rwidth=0.8)
        ax_hist.set_title(f'Distribución de Clusters (k={k})')
        ax_hist.set_xlabel('Cluster')
        ax_hist.set_ylabel('Número de píxeles')
    
    plt.tight_layout()
    return plt.gcf()

def main():
    # Configuración
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, 'images')
    results_dir = os.path.join(script_dir, 'resultados')
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
        
        # Procesar en RGB
        fig_rgb = apply_kmeans_to_image(img_path, k_values, color_space='RGB')
        fig_rgb.suptitle(f'Análisis K-means en RGB - {os.path.basename(img_path)}')
        
        # Procesar en HSV
        fig_hsv = apply_kmeans_to_image(img_path, k_values, color_space='HSV')
        fig_hsv.suptitle(f'Análisis K-means en HSV - {os.path.basename(img_path)}')
        
        # Guardar resultados
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        rgb_output = os.path.join(results_dir, f'kmeans_analysis_{base_name}_RGB.png')
        hsv_output = os.path.join(results_dir, f'kmeans_analysis_{base_name}_HSV.png')
        
        fig_rgb.savefig(rgb_output, dpi=150, bbox_inches='tight')
        fig_hsv.savefig(hsv_output, dpi=150, bbox_inches='tight')
        print(f"Resultados guardados en:\n{rgb_output}\n{hsv_output}")
        
        plt.close('all')
    
    plt.show()

if __name__ == "__main__":
    main()