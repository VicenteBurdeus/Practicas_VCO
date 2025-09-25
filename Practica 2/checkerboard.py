# Python program to print nXn
# checkerboard pattern using numpy
 
import matplotlib.pyplot as plt
import numpy as np  
import string
from PIL import Image
import math


# function to return a Checkerboard graylevel image of n x n squares each of size 'sz'
def checkboard(grid= 10, square_size = 10):
     
    print("Checkerboard pattern:")
 
    # create a n * n matrix
    x = np.zeros((grid, grid), dtype = np.uint8)
 
    # fill with 255 the alternate rows and columns
    x[1::2, ::2] = 255
    x[::2, 1::2] = 255
    
    sz = grid * square_size 
    size = (sz,sz)
    img = Image.fromarray(x, mode='L')
    img_res = img.resize(size, resample=Image.Resampling.NEAREST)
    return img_res

# Main code

img =checkboard(10,20)   # 8 x 8 squares of 40x40 pixels
plt.figure(figsize=(9,5))


img_t = img       # Obtener la imagen transformada
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(img_t)

plt.show()

img.save('checkerboard.jpg')