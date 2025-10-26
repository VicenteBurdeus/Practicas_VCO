import numpy as np
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
import os
import glob

def apply_meanshift(image, bandwidth=None, quantile=0.06, n_samples=3000):
    """Aplicar MeanShift a una imagen con un bandwidth específico"""
    # Convertir a RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Aplicar filtro para reducir ruido
    image_filtered = cv2.medianBlur(image_rgb, 3)
    
    # Aplanar la imagen
    flat_image = image_filtered.reshape((-1,3))
    flat_image = np.float32(flat_image)
    
    # Estimar bandwidth si no se proporciona
    if bandwidth is None:
        bandwidth = estimate_bandwidth(flat_image, quantile=quantile, n_samples=n_samples)
        print(f"Bandwidth estimado: {bandwidth:.2f}")
    
    # Aplicar MeanShift
    ms = MeanShift(bandwidth=bandwidth, max_iter=800, bin_seeding=True)
    ms.fit(flat_image)
    labeled = ms.labels_
    
    # Obtener número de segmentos
    segments = np.unique(labeled)
    n_segments = segments.shape[0]
    print(f"Número de segmentos: {n_segments}")
    
    # Obtener el color promedio de cada segmento
    total = np.zeros((segments.shape[0], 3), dtype=float)
    count = np.zeros(total.shape, dtype=float)
    for i, label in enumerate(labeled):
        total[label] = total[label] + flat_image[i]
        count[label] += 1
    avg = total/count
    avg = np.uint8(avg)
    
    # Reconstruir la imagen segmentada
    res = avg[labeled]
    result = res.reshape(image_rgb.shape)
    
    return result, n_segments, labeled

def analyze_image(image_path, bandwidths=[20, 30, 40]):
    """Analizar una imagen con diferentes valores de bandwidth"""
    # Leer imagen
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")
    
    # Crear figura
    n_bandwidths = len(bandwidths)
    fig = plt.figure(figsize=(5*(n_bandwidths + 1), 10))
    
    # Mostrar imagen original
    ax_orig = plt.subplot2grid((2, n_bandwidths + 1), (0, 0))
    ax_orig.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax_orig.set_title('Imagen Original')
    ax_orig.axis('off')
    
    # Histograma de color original (2D)
    ax_hist_orig = plt.subplot2grid((2, n_bandwidths + 1), (1, 0))
    flat_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1, 3)
    ax_hist_orig.scatter(flat_img[:,0], flat_img[:,1], c=flat_img/255.0, 
                        alpha=0.5, s=1)
    ax_hist_orig.set_title('Distribución de Color Original')
    ax_hist_orig.set_xlabel('R')
    ax_hist_orig.set_ylabel('G')
    
    # Procesar cada bandwidth
    for idx, bw in enumerate(bandwidths, 1):
        print(f"\nProcesando bandwidth = {bw}")
        
        # Aplicar MeanShift
        result, n_segments, labels = apply_meanshift(img, bandwidth=bw)
        
        # Mostrar resultado
        ax_result = plt.subplot2grid((2, n_bandwidths + 1), (0, idx))
        ax_result.imshow(result)
        ax_result.set_title(f'MeanShift (bw={bw})\n{n_segments} segmentos')
        ax_result.axis('off')
        
        # Mostrar distribución de colores segmentada
        ax_hist = plt.subplot2grid((2, n_bandwidths + 1), (1, idx))
        flat_result = result.reshape(-1, 3)
        scatter = ax_hist.scatter(flat_result[:,0], flat_result[:,1], 
                                c=flat_result/255.0, alpha=0.5, s=1)
        ax_hist.set_title(f'Distribución de Color\nSegmentada')
        ax_hist.set_xlabel('R')
        ax_hist.set_ylabel('G')
    
    plt.tight_layout()
    return fig

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
    
    # Valores de bandwidth a probar
    bandwidths = [20, 30, 40]
    
    # Procesar cada imagen
    for img_path in image_paths:
        print(f"\nProcesando imagen: {os.path.basename(img_path)}")
        
        # Analizar imagen con diferentes bandwidths
        fig = analyze_image(img_path, bandwidths)
        
        # Guardar resultados
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(results_dir, f'meanshift_analysis_{base_name}.png')
        
        # Ajustar y guardar figura
        fig.suptitle(f'Análisis MeanShift - {os.path.basename(img_path)}', y=1.02)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Resultados guardados en: {output_path}")
        
        plt.close('all')

if __name__ == "__main__":
    main()