import os

print("Carpeta actual:", os.getcwd())

from PIL import Image
import numpy as np
import cv2
from scipy import ndimage
from skimage import data, img_as_float

#Lea alguno de los archivos de color de tipo tif descargado muéstrelo en una ventana mediante Pillow. Si 
#no se visualiza bien intente poner el mapa de colores o paleta que se corresponda. Muestre también la 
#barra de colores al lado de la imagen. ¿Cuántos colores tiene la imagen?

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
    #solo usaremos el zorro 
    img = Image.open(imgname("tif"))
    print("Colores en la imagen:", img.getcolors())
