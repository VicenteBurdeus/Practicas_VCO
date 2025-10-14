import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2


def saltAndPepper_noise(img, percent):
# img: Image to introduce the noise
# percent [0,1) percentaje of noise
    per = int(percent * img.size)
    for k in range(per):
        i = int(np.random.random() * img.shape[1])
        j = int(np.random.random() * img.shape[0])
        if img.ndim == 2:
            img[j,i] = 255
        elif (img.ndim == 3):
            img[j,i,0] = 255
            img[j,i,1] = 255
            img[j,i,2] = 255
    return img

def clamp(num, min_value=0, max_value=255): return int(max(min(num, max_value), min_value))

def gaussian_noise(img):
    global s
    h,w,c = img.shape
    for row in range(h):
        for col in range(w):
            s = np.random.normal(0,20,3)
            b = img[row,col,0]
            g = img[row,col,1]
            r = img[row,col,2]
            img[row,col,0] = clamp(b + s[0])
            img[row,col,1] = clamp(g + s[1])
            img[row,col,2] = clamp(r + s[2])
    return img       



if __name__ == '__main__':
    
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    img = np.array(Image.open('Practica 3/AloeVera.jpg'))
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')

    
    plt.subplot(132)
    img_noise = gaussian_noise(img)
    plt.imshow(img_noise)
    plt.title('Gaussian Noise')
    plt.axis('off')
    
    img_filtered = cv2.medianBlur(img_noise, 3)
    plt.subplot(133)
    plt.imshow(img_filtered)
    plt.title('Filtered Image (Median Blur)')
    plt.axis('off')
    
    plt.show()


                