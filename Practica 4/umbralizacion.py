import matplotlib.pyplot as plt 
import numpy as np 
from PIL import Image 
import cv2

path_img='Practica 4\Baraja_p_ker_1\Training\IMG_20210321_121516.jpg'

img = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE) 
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY) 
titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV'] 
images = [img, thresh1] 
for i in range(2): 
 plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255) 
 plt.title(titles[i]) 
 plt.xticks([]),plt.yticks([]) 
plt.show()