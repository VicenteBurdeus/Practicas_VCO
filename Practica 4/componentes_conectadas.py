import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog

# Valores iniciales para detectar el fondo verde de las cartas (HSV)
max_value = 255
max_value_H = 360//2
low_H = 35   # Verde amarillento (bajado para ser más permisivo)
low_S = 30   # Saturación mínima (bajada para capturar verdes menos saturados)
low_V = 30   # Valor mínimo (bajado para capturar verdes más oscuros)
high_H = 85  # Verde azulado (subido para capturar más tonos de verde)
high_S = max_value
high_V = max_value

# Valor inicial para área mínima
min_area = 500

# Variables globales
min_area = 500
kernel_size = 3
iterations = 1
use_inv_threshold = True  # True para THRESH_BINARY_INV, False para THRESH_BINARY

# Generar colores aleatorios para las etiquetas
np.random.seed(42)  # Para reproducibilidad
colors = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)  # Preparar suficientes colores
colors[0] = [0, 0, 0]  # Fondo en negro

def on_threshold_type_change(val):
    global use_inv_threshold
    use_inv_threshold = bool(val)

def on_min_area_change(val):
    global min_area
    min_area = val

def on_kernel_change(val):
    global kernel_size
    kernel_size = 2 * val + 1  # Siempre impar: 3, 5, 7, ...

def on_iterations_change(val):
    global iterations
    iterations = val

def on_low_H_change(val):
    global low_H
    low_H = min(val, high_H-1)
    cv2.setTrackbarPos('Low H', 'Controls', low_H)

def on_high_H_change(val):
    global high_H
    high_H = max(val, low_H+1)
    cv2.setTrackbarPos('High H', 'Controls', high_H)

def on_low_S_change(val):
    global low_S
    low_S = min(val, high_S-1)
    cv2.setTrackbarPos('Low S', 'Controls', low_S)

def on_high_S_change(val):
    global high_S
    high_S = max(val, low_S+1)
    cv2.setTrackbarPos('High S', 'Controls', high_S)

def on_low_V_change(val):
    global low_V
    low_V = min(val, high_V-1)
    cv2.setTrackbarPos('Low V', 'Controls', low_V)

def on_high_V_change(val):
    global high_V
    high_V = max(val, low_V+1)
    cv2.setTrackbarPos('High V', 'Controls', high_V)

def procesar_mascara(mask):
    """
    Aplica operaciones morfológicas para mejorar la máscara
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Cerrar para rellenar huecos
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    # Abrir para eliminar ruido
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    
    return mask

def crear_imagen_etiquetas(labels, num_labels):
    """
    Crea una imagen con las etiquetas coloreadas aleatoriamente
    """
    # Generar colores aleatorios para cada etiqueta
    colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Fondo en negro
    
    # Crear imagen de etiquetas coloreadas
    labels_img = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    for i in range(num_labels):
        labels_img[labels == i] = colors[i]
    
    return labels_img

def analizar_componentes(mask, min_area=500, connectivity=8, frame_original=None):
    """
    Analiza las componentes conectadas en la máscara binaria.
    
    Args:
        mask: Máscara binaria
        min_area: Área mínima para considerar una componente
        connectivity: 4 u 8 para conectividad
        frame_original: Imagen original para dibujar los rectángulos y centroides
    Returns:
        output: Imagen con componentes marcadas
        frame_with_boxes: Imagen original con rectángulos y centroides
        labels_img: Imagen de etiquetas coloreadas
    """
    # Encontrar componentes conectadas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=connectivity)
    
    # Crear imagen de etiquetas coloreadas
    labels_img = crear_imagen_etiquetas(labels, num_labels)
    
    # Crear imagen para visualización de componentes
    output = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    # Si tenemos la imagen original, hacer una copia para dibujar
    if frame_original is not None:
        frame_with_boxes = frame_original.copy()
    
    # Filtrar por área y colorear componentes
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            # Colorear componente en la máscara
            output[labels == i] = colors[i]
            
            # Obtener coordenadas del rectángulo y centroide
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            cx = int(centroids[i][0])
            cy = int(centroids[i][1])
            
            # Dibujar en la imagen de salida
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(output, (cx, cy), 4, (0, 255, 0), -1)
            cv2.putText(output, f'Area: {area}', (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Si tenemos la imagen original, dibujar en ella
            if frame_original is not None:
                # Rectángulo azul claro, grosor 10px
                cv2.rectangle(frame_with_boxes, (x, y), (x+w, y+h), (255, 255, 0), 10)
                # Centroide amarillo, radio 4, grosor 5
                cv2.circle(frame_with_boxes, (cx, cy), 4, (255, 255, 0), 5)
    
    if frame_original is not None:
        return output, frame_with_boxes, labels_img
    return output, None, labels_img

# Crear ventanas
cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
cv2.namedWindow('Components', cv2.WINDOW_NORMAL)
cv2.namedWindow('Labels', cv2.WINDOW_NORMAL)  # Nueva ventana para etiquetas coloreadas
cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)

# Organizar las ventanas
cv2.moveWindow('Original', 0, 0)
cv2.moveWindow('Mask', 400, 0)
cv2.moveWindow('Components', 800, 0)
cv2.moveWindow('Labels', 1200, 0)
cv2.moveWindow('Controls', 400, 500)

# Crear trackbars para control
cv2.createTrackbar('Low H', 'Controls', low_H, max_value_H, on_low_H_change)
cv2.createTrackbar('High H', 'Controls', high_H, max_value_H, on_high_H_change)
cv2.createTrackbar('Low S', 'Controls', low_S, max_value, on_low_S_change)
cv2.createTrackbar('High S', 'Controls', high_S, max_value, on_high_S_change)
cv2.createTrackbar('Low V', 'Controls', low_V, max_value, on_low_V_change)
cv2.createTrackbar('High V', 'Controls', high_V, max_value, on_high_V_change)
cv2.createTrackbar('Kernel Size', 'Controls', 1, 5, on_kernel_change)
cv2.createTrackbar('Iterations', 'Controls', iterations, 5, on_iterations_change)
cv2.createTrackbar('Min Area', 'Controls', min_area, 5000, on_min_area_change)
cv2.createTrackbar('Inv Threshold', 'Controls', 1, 1, on_threshold_type_change)

# Seleccionar carpeta
folder_name = filedialog.askdirectory(initialdir='./Baraja_p_ker_1/Training/')

if not folder_name:
    print("No se seleccionó ninguna carpeta")
    exit()

# Inicializar tkinter
root = tk.Tk()
root.withdraw()  # Ocultar la ventana principal de tkinter

# Procesar imágenes
key = -1  # Inicializar key fuera del bucle
for filename in os.listdir(folder_name):
    if filename.endswith('.jpg'):
        # Leer imagen
        filepath = os.path.join(folder_name, filename)
        frame = cv2.imread(filepath)
        if frame is None:
            print(f"No se pudo cargar la imagen: {filepath}")
            continue
            
        try:
            # Convertir a HSV y aplicar umbralización
            frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(frame_HSV, 
                             (low_H, low_S, low_V),
                             (high_H, high_S, high_V))
            
            # Procesar y mejorar la máscara
            mask = procesar_mascara(mask)
            
            # Analizar componentes conectadas
            components, frame_with_boxes, labels_img = analizar_componentes(
                mask, min_area, connectivity=8, frame_original=frame)
            
            # Mostrar resultados iniciales
            cv2.imshow('Original', frame_with_boxes)
            cv2.imshow('Mask', mask)
            cv2.imshow('Components', components)
            cv2.imshow('Labels', labels_img)
            
            # Esperar tecla y actualizar la visualización continuamente
            while True:
                key = cv2.waitKey(100)  # Esperar 100ms
                
                # Actualizar máscara y componentes con los nuevos valores
                mask = cv2.inRange(frame_HSV, 
                                 (low_H, low_S, low_V),
                                 (high_H, high_S, high_V))
                
                # Invertir la máscara si es necesario
                if not use_inv_threshold:
                    mask = cv2.bitwise_not(mask)
                    
                mask = procesar_mascara(mask)
                components, frame_with_boxes, labels_img = analizar_componentes(
                    mask, min_area, connectivity=8, frame_original=frame.copy())
                
                # Mostrar resultados actualizados
                cv2.imshow('Original', frame_with_boxes)
                cv2.imshow('Mask', mask)
                cv2.imshow('Components', components)
                cv2.imshow('Labels', labels_img)
                
                # Salir si se presiona 'q' o ESC
                if key == ord('q') or key == 27:
                    break
                elif key != -1:  # Cualquier otra tecla pasa a la siguiente imagen
                    break
                    
        except Exception as e:
            print(f"Error procesando la imagen {filename}: {str(e)}")
            continue
            
        if key == ord('q') or key == 27:  # q o ESC para salir
            break

# Liberar recursos
cv2.destroyAllWindows()
root.quit()  # Cerrar la aplicación tkinter
            break

cv2.destroyAllWindows()