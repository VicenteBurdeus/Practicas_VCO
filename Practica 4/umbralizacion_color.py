# umbralizacon_global.py
#
# Programa pasa realizar operaciones de umbralización por rangos de color con imágenes RGB
# convertidas a HSV.
#
# Autor: José M Valiente    Fecha: marzo 2023
#

from __future__ import print_function
import cv2 
import argparse
import os 
import tkinter as tk
from tkinter import filedialog
import numpy as np


max_value = 255
max_value_H = 360//2
# Valores iniciales para detectar el fondo verde de las cartas
# En HSV, el verde está alrededor de H=60 (en escala 0-180)
low_H = 40   # Verde amarillento
low_S = 50   # Saturación mínima para evitar grises
low_V = 50   # Valor mínimo para evitar negros
high_H = 80  # Verde azulado
high_S = max_value  # Máxima saturación
high_V = max_value  # Máximo brillo
window_name = 'Original image'
window_thresholded = 'Thresholded result'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'
   
def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv2.setTrackbarPos(low_H_name, window_thresholded, low_H)
def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv2.setTrackbarPos(high_H_name, window_thresholded, high_H)
def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv2.setTrackbarPos(low_S_name, window_thresholded, low_S)
def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv2.setTrackbarPos(high_S_name, window_thresholded, high_S)
def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv2.setTrackbarPos(low_V_name, window_thresholded, low_V)
def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv2.setTrackbarPos(high_V_name, window_thresholded, high_V)
 

frame_HSV = np.zeros((256, 256, 1), dtype = "uint8")
 
cv2.namedWindow(window_name, flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
#cv2.resizeWindow(window_name,520,1040)
cv2.namedWindow(window_thresholded,  flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.createTrackbar(low_H_name, window_thresholded , low_H, max_value_H, on_low_H_thresh_trackbar)
cv2.createTrackbar(high_H_name, window_thresholded , high_H, max_value_H, on_high_H_thresh_trackbar)
cv2.createTrackbar(low_S_name, window_thresholded , low_S, max_value, on_low_S_thresh_trackbar)
cv2.createTrackbar(high_S_name, window_thresholded , high_S, max_value, on_high_S_thresh_trackbar)
cv2.createTrackbar(low_V_name, window_thresholded , low_V, max_value, on_low_V_thresh_trackbar)
cv2.createTrackbar(high_V_name, window_thresholded , high_V, max_value, on_high_V_thresh_trackbar)


folders = './Baraja_p_ker_1/Training/'  # Carpeta con las cartas de poker

folder_name = filedialog.askdirectory(initialdir=folders)

f2Name = folder_name + '/'
list_files = os.scandir(f2Name)
for ent in list_files:
     if ent.is_file() and ent.name.endswith('.jpg'):
         filename = f2Name + ent.name
         frame = cv2.imread(filename)
         cv2.imshow(window_name, frame)
         
         frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
         key = -1
         while (key == -1):
             key=cv2.pollKey()
             # Umbralización por rango de color usando HSV
             # cv2.inRange(src, lowerb, upperb) -> máscara
             # src: imagen fuente en HSV
             # lowerb: límite inferior [H,S,V]
             # upperb: límite superior [H,S,V]
             frame_threshold = cv2.inRange(frame_HSV, 
                                        (low_H, low_S, low_V), 
                                        (high_H, high_S, high_V))
             
             cv2.imshow(window_thresholded, frame_threshold)
         if key == ord('q') or key == 27:    # 'q' o ESC para acabar
             break

cv2.destroyWindow(window_name)
cv2.destroyWindow(window_thresholded) 