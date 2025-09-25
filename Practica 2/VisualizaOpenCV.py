import matplotlib.pyplot as plt
import cv2 
import numpy as np

src = cv2.imread('building4.jpg')
dst = cv2.resize(src, (256, 256), interpolation=cv2.INTER_CUBIC)

cv2.imshow('Imagen Original', src)
cv2.imshow('Imagen Escalada', dst)
cv2.waitKey()
cv2.destroyAllWindows()

src_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
dst_rgb = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
plt.subplot(121),plt.imshow(src),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()