import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np  
import string
from PIL import Image
import math

#nos movemos a la carpeta de practica 2

os.chdir('Practica 2')

img = Image.open('shark.jpg')  
img.size #  (640, 480)



# Obtener la imagen transformada

img_t = img 
plt.figure(figsize=(10, 6))
plt.subplot(121)
plt.imshow(img)
#plt.colorbar()

img_t = img_t.resize( (int(img_t.size[0]*1.5), int(img_t.size[1]*1.5)) )  # Redimensiona la imagen al 150% de su tamaño original
img_t = img_t.rotate(10)

plt.subplot(122)
plt.imshow(np.asarray(img_t))
#plt.colorbar()
plt.show()