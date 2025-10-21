import matplotlib.pyplot as plt
import numpy as np  
from PIL import Image
import cv2

def checkboard(grid=10, square_size=20):
    # create a n * n matrix
    x = np.zeros((grid, grid), dtype=np.uint8)
 
    # fill with 255 the alternate rows and columns
    x[1::2, ::2] = 255
    x[::2, 1::2] = 255
    
    sz = grid * square_size 
    size = (sz,sz)
    img = Image.fromarray(x, mode='L')
    img_res = img.resize(size, resample=Image.Resampling.NEAREST)
    return np.array(img_res)

# Crear el tablero de ajedrez 10x10 con cuadrados de 20px
img = checkboard(10, 20)

# Mostrar imagen original
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.title('Tablero Original')
plt.imshow(img, cmap='gray')
plt.axis('off')

# Crear y aplicar matriz de traslación t=(100,50)
rows, cols = img.shape
T = np.float32([[1, 0, 100],
                [0, 1, 50]])
img_translated = cv2.warpAffine(img, T, (cols + 100, rows + 50))

# Mostrar imagen trasladada
plt.subplot(132)
plt.title('Traslación (100,50)')
plt.imshow(img_translated, cmap='gray')
plt.axis('off')

# Crear y aplicar matriz de rotación 30 grados
center = (cols/2, rows/2)
R = cv2.getRotationMatrix2D(center, 30, 1.0)
img_rotated = cv2.warpAffine(img, R, (cols, rows))

# Mostrar imagen rotada
plt.subplot(133)
plt.title('Rotación 30°')
plt.imshow(img_rotated, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()