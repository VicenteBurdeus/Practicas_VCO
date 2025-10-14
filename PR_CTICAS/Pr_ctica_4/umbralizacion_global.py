# umbralizacon_global.py
#
# Programa pasa realizar operaciones de umbrallización global con imágenes de niveles de gris.
#
# Autor: José M Valiente    Fecha: marzo 2023
#
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from tkinter import filedialog
import os

window_original = 'Original_image'
window_threshold = 'Thresholded_image'
cv2.namedWindow(window_original,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
cv2.namedWindow(window_threshold,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

low_H = 155

def on_thresh_trackbar(val):
    global low_H, window_threshold, img
    low_H = val
    cv2.setTrackbarPos('trackbar', window_threshold , low_H)
    ret, thresh1 = cv2.threshold(img, val, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cv2.imshow(window_threshold, thresh1) #, cmap='gray')


# Selección de una carpeta mediante un diálogo de la biblioteca 'tkinter'
path = filedialog.askdirectory(initialdir="./../", title="Seleccione una carpeta")

# Creación de una barra de deslizamiento (trackbar)
cv2.createTrackbar('trackbar', window_threshold , low_H, 255, on_thresh_trackbar)

for root,  dirs, files in os.walk(path, topdown=False):
    for name in files:
        if not(name.endswith('.jpg')):
            continue
        filename = os.path.join(root, name)
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        cv2.imshow(window_original,img)
        
        key = -1
        while (key == -1):
             key=cv2.pollKey()
             # Aquí va la función  cv2.inRange(....)
             fret,thresh1 = cv2.threshold(img,low_H,255,cv2.THRESH_BINARY)
             cv2.imshow(window_threshold, thresh1) #, cmap='gray')
        if key == ord('q') or key == 27:    # 'q' o ESC para acabar
             break

 
cv2.destroyWindow(window_original)             
cv2.destroyWindow(window_threshold)       

 