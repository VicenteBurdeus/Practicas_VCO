import matplotlib.pyplot as plt 
import numpy as np 
from PIL import Image 
import cv2 
 
plt.figure(1) 
src = np.array (Image.open('Practica 3/AloeVera.jpg')) 

Emboss = np.array([ 
    [0,-1,-1], 
    [1, 0, -1], 
    [1, 1, 0] ]) 

Sepia = np.array([
    [0.393, 0.769, 0.189], 
    [0.349, 0.686, 0.168], 
    [0.272, 0.534, 0.131] ])

Paso_alto = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1] ])

Suavizado = np.array([
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9] ])

Agudizado = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0] ])

img_emboss = cv2.filter2D(src, -1, Emboss)
img_sepia = cv2.transform(src, Sepia)
img_paso_alto = cv2.filter2D(src, -1, Paso_alto)
img_suavizado = cv2.filter2D(src, -1, Suavizado)
img_agudizado = cv2.filter2D(src, -1, Agudizado)


plt.subplot(221)
plt.imshow(src)
plt.title('Imagen Original')
plt.axis('off')

plt.subplot(222)
plt.imshow(img_sepia)
plt.title('Sepia')
plt.axis('off')

plt.subplot(223)
plt.imshow(img_suavizado)
plt.title('Suavizado')
plt.axis('off')

plt.subplot(224)
plt.imshow(img_agudizado)
plt.title('Agudizado')
plt.axis('off')

plt.show()

