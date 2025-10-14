import numpy as np 
import cv2  
from matplotlib import pyplot as plt 
img = cv2.imread('Practica 3/AloeVera.jpg') 
color = ('b','g','r') 
for i,col in enumerate(color): 
    histr = cv2.calcHist([img],[i],None,[16],[0,256])  # Reducido a 16 bins
    histr = histr/np.sum(histr)  # Normalización (density=True)
    plt.plot(histr, color=col, label=f'{col} channel (16 bins, normalized)')
plt.xlim([0,256]) 
plt.legend()
plt.title('Histogramas Normalizados de los canales BGR (16 bins)')
plt.xlabel('Valor de pixel')
plt.ylabel('Densidad (Frecuencia normalizada)')
plt.show()