import os

print("Carpeta actual:", os.getcwd())

from PIL import Image
import numpy as np
import cv2
from scipy import ndimage
from skimage import data, img_as_float

#Utilizando cualquiera de las imágenes introducidas en el ejercicio anterior, aplique distintas transformaciones y compruebe el resultado. 

dirimg = "Practica 1/Imagenes/"
imagenes = ["zorro.tiff", "fish.gif", "paarthurmax.jpg", "diamante.png"]
def imgname(formato):
    if formato == "tif":
        #return "Imagenes/zorro.tiff"
        return f"{dirimg}{imagenes[0]}"
    elif formato == "gif":
        return f"{dirimg}{imagenes[1]}"
    elif formato == "jpg":
        return f"{dirimg}{imagenes[2]}"
    elif formato == "png":
        return f"{dirimg}{imagenes[3]}"



if __name__ == "__main__":

    img = Image.open(imgname("tif"))
    img.show()
    

