from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt


# Cargar imagen con PIL
img_pil = Image.open('AloeVera.jpg')
img_rgb = np.array(img_pil.convert('RGB'))

# 1) Conversión a escala de grises (PIL)
img_gray = np.array(img_pil.convert('L'))

# 2) Sepia (operación por canal)
sepia_matrix = np.array([
    [0.393, 0.769, 0.189],
    [0.349, 0.686, 0.168],
    [0.272, 0.534, 0.131]
])

def apply_sepia(img_arr):
    h, w, _ = img_arr.shape
    flat = img_arr.reshape((-1, 3)).astype(np.float32)
    sepia_flat = np.dot(flat, sepia_matrix.T)
    sepia_flat = np.clip(sepia_flat, 0, 255)
    return sepia_flat.reshape((h, w, 3)).astype(np.uint8)

img_sepia = apply_sepia(img_rgb)

# 3) Emboss usando PIL
img_emboss = np.array(img_pil.filter(ImageFilter.EMBOSS))

# 4) Blur (Gaussian) con PIL
img_blur = np.array(img_pil.filter(ImageFilter.GaussianBlur(radius=3)))

# 5) Sharpen con PIL
img_sharp = np.array(img_pil.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)))

# Mostrar resultados
plt.figure(figsize=(12, 8))
plt.subplot(231); plt.imshow(img_rgb); plt.title('Original (RGB)'); plt.axis('off')
plt.subplot(232); plt.imshow(img_gray, cmap='gray'); plt.title('Escala de grises'); plt.axis('off')
plt.subplot(233); plt.imshow(img_sepia); plt.title('Sepia'); plt.axis('off')
plt.subplot(234); plt.imshow(img_emboss); plt.title('Emboss'); plt.axis('off')
plt.subplot(235); plt.imshow(img_blur); plt.title('Blur (Gaussian)'); plt.axis('off')
plt.subplot(236); plt.imshow(img_sharp); plt.title('Sharpen'); plt.axis('off')

plt.tight_layout()
plt.show()

# Mostrar los canales R, G, B como imágenes en gris separadas
r = img_rgb[:, :, 0]
g = img_rgb[:, :, 1]
b = img_rgb[:, :, 2]

plt.figure(figsize=(9, 3))
plt.subplot(131); plt.imshow(r, cmap='gray'); plt.title('Canal R (gris)'); plt.axis('off')
plt.subplot(132); plt.imshow(g, cmap='gray'); plt.title('Canal G (gris)'); plt.axis('off')
plt.subplot(133); plt.imshow(b, cmap='gray'); plt.title('Canal B (gris)'); plt.axis('off')
plt.tight_layout(); plt.show()