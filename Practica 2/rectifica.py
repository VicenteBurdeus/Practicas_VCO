import cv2
import numpy as np
import matplotlib.pyplot as plt

def ginput(window_name, image, n_points):
    """
    Función personalizada para capturar puntos con el ratón.
    Retorna los puntos seleccionados en formato numpy array.
    """
    points = []
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            # Dibujar el punto en la imagen
            cv2.circle(img_display, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(window_name, img_display)
            
    img_display = image.copy()
    cv2.imshow(window_name, img_display)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    while len(points) < n_points:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyWindow(window_name)
    return np.array(points, dtype=np.float32)

def process_image(image_path, output_size=200):
    # Leer la imagen
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: No se pudo cargar la imagen {image_path}")
        return
    
    # Redimensionar si la imagen es muy grande
    max_dim = 800
    height, width = img.shape[:2]
    if max(height, width) > max_dim:
        scale = max_dim / max(height, width)
        img = cv2.resize(img, None, fx=scale, fy=scale)
    
    # Obtener 4 puntos de control
    print("Por favor, seleccione 4 puntos en la imagen (esquinas de una región rectangular)")
    pts = ginput("Seleccionar puntos", img, 4)
    
    # Definir los puntos de salida para formar un cuadrado
    outs = np.array([
        [0, 0],
        [output_size, 0],
        [output_size, output_size],
        [0, output_size]
    ], dtype=np.float32)
    
    # Obtener la matriz de transformación perspectiva
    M = cv2.getPerspectiveTransform(pts, outs)
    
    # Aplicar la transformación perspectiva
    warped = cv2.warpPerspective(img, M, (output_size, output_size))
    
    # Dibujar el polígono en la imagen original
    img_with_polygon = img.copy()
    cv2.polylines(img_with_polygon, [pts.astype(np.int32)], True, (0, 255, 0), 2)
    
    # Mostrar resultados
    plt.figure(figsize=(12, 5))
    
    # Convertir de BGR a RGB para matplotlib
    img_with_polygon_rgb = cv2.cvtColor(img_with_polygon, cv2.COLOR_BGR2RGB)
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    
    plt.subplot(121)
    plt.title('Imagen Original con Región Seleccionada')
    plt.imshow(img_with_polygon_rgb)
    plt.axis('off')
    
    plt.subplot(122)
    plt.title('Imagen Rectificada')
    plt.imshow(warped_rgb)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Lista de imágenes para procesar
    images = [
        'Building4.jpg',
        'Torredelmar.jpg',
        'Santa-Maria-Micaela-2.jpg',
        'edificio-caledonia-zona-16-3.jpg'
    ]
    
    print("Imágenes disponibles:")
    for i, img in enumerate(images):
        print(f"{i+1}. {img}")
    
    while True:
        try:
            idx = int(input("\nSeleccione el número de la imagen a procesar (0 para salir): ")) - 1
            if idx == -1:
                break
            if 0 <= idx < len(images):
                process_image(images[idx])
            else:
                print("Número de imagen inválido")
        except ValueError:
            print("Por favor, ingrese un número válido")
        except Exception as e:
            print(f"Error: {str(e)}")
        
        resp = input("\n¿Desea procesar otra imagen? (s/n): ")
        if resp.lower() != 's':
            break

if __name__ == "__main__":
    main()