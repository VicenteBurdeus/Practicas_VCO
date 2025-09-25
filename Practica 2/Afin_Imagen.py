import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np  
import string
from PIL import Image
import math

img = Image.open('building4.jpg')  
img.size #  (640, 480)



# Obtener la imagen transformada


img_t = img
plt.figure(figsize=(10, 6))
plt.subplot(121)
plt.imshow(img)
#plt.colorbar()
plt.subplot(122)
plt.imshow(np.asarray(img_t))
#plt.colorbar()
plt.show()  