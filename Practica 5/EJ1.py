import matplotlib.pyplot as plt 
from matplotlib import colors 
import numpy as np 
from PIL import Image 
import cv2 
 
nemo = cv2.imread("images/nemo0.jpg") 
fig = plt.figure() 
fig.add_subplot(1, 2, 1, title="Original") 
plt.imshow(nemo) 
 
    # Plotting the image on 3D plot 
r, g, b = cv2.split(nemo) 
axis = fig.add_subplot(1, 2, 2, projection="3d") 
pixel_colors = nemo.reshape((np.shape(nemo)[0] * np.shape(nemo)[1], 3)) 
norm = colors.Normalize(vmin=-1.0, vmax=1.0) 
norm.autoscale(pixel_colors) 
pixel_colors = norm(pixel_colors).tolist() 
 
axis.scatter( r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".") 
axis.set_xlabel("Red") 
axis.set_ylabel("Green") 
axis.set_zlabel("Blue") 
plt.show() 