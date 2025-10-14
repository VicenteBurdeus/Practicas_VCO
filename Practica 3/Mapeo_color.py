import matplotlib.pyplot as plt  
import numpy as np  
import string
from PIL import Image
import math
import cv2

img = np.asarray(Image.open('Practica 3/AloeVera.jpg')  )
ro,co,ch = img.shape #  (640, 480, 3)

plt.figure(figsize=(9,5))
imgplot = plt.imshow(img)
plt.colorbar()

hsv_all = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv = hsv_all[:,:,0]
dst = np.array((ro,co,ch),type(img))

col = 30
r = 20
h1 = np.int32(((col-r/2) + 360) % 360)
h2 = np.int32(((col+r/2) + 360) % 360)
rng = range(h1,h2+1)
he, wi, ch = img.shape
for row in range(he):
    for col in range(wi):
        h = hsv[row,col]
        if h in rng:
            dst[row,col,:] = img[row,col,:]
        else:
            gray = np.sum(img[row,col,:])
            dst[row,col,:] = gray

                              
    
    

