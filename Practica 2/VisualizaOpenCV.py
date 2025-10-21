import cv2
import numpy as np
import matplotlib.pyplot as plt

# Crear una imagen de ejemplo (tablero de ajedrez con colores)
def create_checkerboard(rows=8, cols=8, square_size=50):
    board = np.zeros((rows * square_size, cols * square_size, 3), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 0:
                # En BGR: [Blue, Green, Red]
                board[i*square_size:(i+1)*square_size, 
                     j*square_size:(j+1)*square_size] = np.array([0, 0, 255], dtype=np.uint8)  # Rojo en BGR
            else:
                board[i*square_size:(i+1)*square_size, 
                     j*square_size:(j+1)*square_size] = np.array([255, 0, 0], dtype=np.uint8)  # Azul en BGR
    return board

# Crear imagen original
#coje la imangen shark.jpg
img = cv2.imread('shark.jpg')

# 1. Mostrar la imagen original sin conversión (problema de color)
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.title('Sin conversión BGR->RGB\n(Colores incorrectos)')
plt.imshow(img)
plt.axis('off')

# 2. Mostrar la imagen con conversión BGR a RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(132)
plt.title('Con conversión BGR->RGB\n(Colores correctos)')
plt.imshow(img_rgb)
plt.axis('off')

# 3. Aplicar transformaciones y mostrar
# Matriz de traslación
rows, cols = img.shape[:2]
T = np.float32([[1, 0, 50],
                [0, 1, 30]])
img_translated = cv2.warpAffine(img_rgb, T, (cols + 50, rows + 30))

# Matriz de rotación
center = (cols/2, rows/2)
R = cv2.getRotationMatrix2D(center, 30, 1.0)
img_rotated = cv2.warpAffine(img_rgb, R, (cols, rows))

# Redimensionar la imagen (ejemplo de resize)
img_resized = cv2.resize(img_rgb, None, fx=0.5, fy=0.5, 
                        interpolation=cv2.INTER_CUBIC)

# Mostrar imagen transformada
plt.subplot(133)
plt.title('Imagen Rotada 30°\n(Con conversión BGR->RGB)')
plt.imshow(img_rotated)
plt.axis('off')

plt.tight_layout()
plt.show()

# Demostración de diferentes métodos de interpolación en resize
plt.figure(figsize=(15, 5))

methods = [
    (cv2.INTER_NEAREST, 'INTER_NEAREST'),
    (cv2.INTER_LINEAR, 'INTER_LINEAR'),
    (cv2.INTER_CUBIC, 'INTER_CUBIC')
]

for idx, (method, title) in enumerate(methods):
    resized = cv2.resize(img_rgb, None, fx=0.5, fy=0.5, interpolation=method)
    plt.subplot(1, 3, idx + 1)
    plt.title(f'Resize con {title}')
    plt.imshow(resized)
    plt.axis('off')

plt.tight_layout()
plt.show()