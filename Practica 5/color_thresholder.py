import cv2 
import os 
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider, RangeSlider, RadioButtons, CheckButtons
import math

# Create figure. Exit on window close.
fig= plt.figure('Color thresholder', (8.5, 5.5))
def on_close(event):
    exit()
fig.canvas.mpl_connect('close_event', on_close)

# Create Axes for src and thresh images
ax_img_source = fig.add_axes([0.0042, 0.5, 0.495, 0.495])
ax_img_thresh = fig.add_axes([0.5020, 0.5, 0.495, 0.495])

# Create Axes for histograms
ax_hist_ch0 = fig.add_axes([0.01, 0.21, 0.23, 0.14], facecolor='#d0d0d0')
ax_hist_ch1 = fig.add_axes([0.345, 0.21, 0.23, 0.14], facecolor='#d0d0d0')
ax_hist_ch2 = fig.add_axes([0.68, 0.21, 0.23, 0.14], facecolor='#d0d0d0')

# Create Axes for color scales
ax_img_scale_ch0 = fig.add_axes([0.01, 0.13, 0.23, 0.03])
ax_img_scale_ch1 = fig.add_axes([0.345, 0.13, 0.23, 0.03])
ax_img_scale_ch2 = fig.add_axes([0.68, 0.13, 0.23, 0.03])

# Create Axes for sliders
ax_slider_ch0 = fig.add_axes([0.01, 0.08, 0.23, 0.03])
ax_slider_ch1 = fig.add_axes([0.345, 0.08, 0.23, 0.03])
ax_slider_ch2 = fig.add_axes([0.68, 0.08, 0.23, 0.03])
ax_slider_bright = fig.add_axes([0.34, 0.45, 0.1, 0.03])

#ax_text_clip = fig.add_axes([0.36, 0.42, 0.1, 0.03], xticks=[], yticks=[])
#for spine in ['top', 'bottom', 'left', 'right']:
#    ax_text_clip.spines[spine].set(edgecolor=None)
#text_clip= ax_text_clip.text(x=0, y=0, s='', c='r')

# Create Axes for buttons
ax_but_reset = fig.add_axes([0.34, 0.42, 0.1, 0.03])
ax_but_prev = fig.add_axes([0.08, 0.015, 0.2, 0.04])
ax_but_clear_samples = fig.add_axes([0.4, 0.015, 0.2, 0.04])
ax_but_next = fig.add_axes([0.72, 0.015, 0.2, 0.04])
ax_but_prev.set_visible(False)
ax_radbut_color_space = fig.add_axes([0.5020, 0.395, 0.1, 0.095], facecolor='#d8e0ff')
ax_checkbut_visual_options = fig.add_axes([0.0042, 0.41, 0.2, 0.08])
ax_checkbut_invert = fig.add_axes([0.85, 0.45, 0.145, 0.04])
for ax in [ax_radbut_color_space, ax_checkbut_visual_options, ax_checkbut_invert]:
    for pos in ['right', 'top', 'bottom', 'left']:
        ax.spines[pos].set_visible(False)

# Create axes for color scales images (1x128x3 RGB)
imx_scale_ch0= ax_img_scale_ch0.imshow(np.zeros((1,128,3), np.uint8), aspect='auto')
imx_scale_ch1= ax_img_scale_ch1.imshow(np.zeros((1,128,3), np.uint8), aspect='auto')
imx_scale_ch2= ax_img_scale_ch2.imshow(np.zeros((1,128,3), np.uint8), aspect='auto')
for ax in (ax_img_scale_ch0, ax_img_scale_ch1, ax_img_scale_ch2):
    ax.set_xticks([])
    ax.set_yticks([])

fit= False
low_res= True
inv= False

# Make buttons
but_next = Button(ax_but_next, 'Siguiente', hovercolor='0.975')
but_prev = Button(ax_but_prev, 'Anterior', hovercolor='0.975')
but_clear_samples = Button(ax_but_clear_samples, 'Borrar muestras', hovercolor='0.975')
but_reset = Button(ax_but_reset, 'Reset', hovercolor='0.975')
radbut_color_space = RadioButtons(ax_radbut_color_space, ('RGB', 'HSV', 'Lab'))
checkbut_visual_options = CheckButtons(ax=ax_checkbut_visual_options, labels= ['Ajustar a ventana', 'Baja resolución'], actives=[fit, low_res])
checkbut_invert = CheckButtons(ax=ax_checkbut_invert, labels= ['Invertir'], actives=[inv])

def onClickVisualOptions(label):
    global fit, low_res
    fit, low_res = checkbut_visual_options.get_status()
    setImageAxes()
    makeImages()
checkbut_visual_options.on_clicked(onClickVisualOptions)

def onClickInvert(label):
    global inv
    inv, = checkbut_invert.get_status()
    makeImages()
checkbut_invert.on_clicked(onClickInvert)

slider_bright = Slider(ax= ax_slider_bright, label='Luminosidad', valmin=0.5, valmax=1.5, valinit=1.0, valfmt="x%.2f")    

def onChangedBright(val):
    global bright
    bright= val
    makeImages()
    #text_clip.set_text('Saturado' if max(frame_rgb.flatten()) == 255 or min(frame_rgb.flatten()) == 0 else '')
slider_bright.on_changed(onChangedBright)

def onClickColorSpace(label):
    global color_space
    #Save current slider positions
    slider_init[color_space]['valinit']= [slider_ch0.val, slider_ch1.val, slider_ch2.val]
    color_space= label
    initSliders()
    makeImages()
    #Update figure
    plotHistograms()
    sliderUpdate(None)
    fig.canvas.draw()
radbut_color_space.on_clicked(onClickColorSpace)

# Manage mouse click events for sample selection on src and thresh images
def OnClickImage(event):
    global colorSamples
    if event.inaxes == ax_img_source or event.inaxes == ax_img_thresh:
        (col, row)= (int(event.xdata + 0.5), int(event.ydata + 0.5))
        colorRGB= frame_rgb[row, col]
        pixRGB= np.array([[colorRGB]], dtype=np.uint8)
        pixHSV= cv2.cvtColor(pixRGB, cv2.COLOR_RGB2HSV)
        pixLab= cv2.cvtColor(pixRGB, cv2.COLOR_RGB2Lab)
        colorSamples.append({'RGB': colorRGB, 'HSV':pixHSV[0,0], 'Lab':pixLab[0,0]})
        markColorSample(colorSamples[-1])
    fig.canvas.draw_idle()
cid_click = fig.canvas.mpl_connect('button_press_event', OnClickImage)

def OnMotionImage(event):
    if event.inaxes == ax_img_source or event.inaxes == ax_img_thresh:
        (col, row)= (int(event.xdata + 0.5), int(event.ydata + 0.5))
        viewColorSample(frame_space[row, col], frame_rgb[row,col])
        fig.canvas.draw_idle()
cid_motion = fig.canvas.mpl_connect('motion_notify_event', OnMotionImage)

plot0=plot1=plot2=[]

def OnLeaveImage(event):
    if event.inaxes == ax_img_source or event.inaxes == ax_img_thresh:
        if len(plot0):
           plot0.pop(0).remove()
           plot1.pop(0).remove()
           plot2.pop(0).remove()
        fig.canvas.draw_idle()
cid_leave= fig.canvas.mpl_connect('axes_leave_event', OnLeaveImage)

def markColorSample(sample):
    ax_hist_ch0.plot((sample[color_space][0],sample[color_space][0]), (0, max_hist_ch0), '-', color=sample['RGB']/255, zorder=2)
    ax_hist_ch1.plot((sample[color_space][1],sample[color_space][1]), (0, max_hist_ch1), '-', color=sample['RGB']/255, zorder=2)
    ax_hist_ch2.plot((sample[color_space][2],sample[color_space][2]), (0, max_hist_ch2), '-', color=sample['RGB']/255, zorder=2)
    
def viewColorSample(color, colorRGB):
    global plot0, plot1, plot2
    if len(plot0):
       plot0.pop(0).remove()
       plot1.pop(0).remove()
       plot2.pop(0).remove()
    plot0= ax_hist_ch0.plot((color[0],color[0]), (0, max_hist_ch0), '--', color=colorRGB/255, zorder=2)
    plot1= ax_hist_ch1.plot((color[1],color[1]), (0, max_hist_ch1), '--', color=colorRGB/255, zorder=2)
    plot2= ax_hist_ch2.plot((color[2],color[2]), (0, max_hist_ch2), '--', color=colorRGB/255, zorder=2)

#Update figure when sliders adjusted
def sliderUpdate(val):
    global mid_slider_ch0, mid_slider_ch1, mid_slider_ch2
    mid_slider_ch0 = (slider_ch0.val[0] + slider_ch0.val[1]) // 2
    mid_slider_ch1 = (slider_ch1.val[0] + slider_ch1.val[1]) // 2
    mid_slider_ch2 = (slider_ch2.val[0] + slider_ch2.val[1]) // 2
    updateSelectedIntervals()
    updateColorScales()
    showThresholdedImage(int(slider_ch0.val[0]), int(slider_ch0.val[1]), int(slider_ch1.val[0]), int(slider_ch1.val[1]), int(slider_ch2.val[0]), int(slider_ch2.val[1]))

def initSliders():
    global slider_ch0, slider_ch1, slider_ch2, color_space
    ax_slider_ch0.clear()
    ax_slider_ch1.clear()
    ax_slider_ch2.clear()
    slider_ch0 = RangeSlider(ax= ax_slider_ch0, label='', valmin=0, valmax=slider_init[color_space]['valmax'][0], valinit=slider_init[color_space]['valinit'][0], valfmt="%d")
    slider_ch1 = RangeSlider(ax= ax_slider_ch1, label='', valmin=0, valmax=slider_init[color_space]['valmax'][1], valinit=slider_init[color_space]['valinit'][1], valfmt="%d")
    slider_ch2 = RangeSlider(ax= ax_slider_ch2, label='', valmin=0, valmax=slider_init[color_space]['valmax'][2], valinit=slider_init[color_space]['valinit'][2], valfmt="%d")    
    slider_ch0.on_changed(sliderUpdate)
    slider_ch1.on_changed(sliderUpdate)
    slider_ch2.on_changed(sliderUpdate)

def updateSelectedIntervals():
    rect_ch0.update({'xy':(slider_ch0.val[0],0), 'width': slider_ch0.val[1] - slider_ch0.val[0]})
    rect_ch1.update({'xy':(slider_ch1.val[0],0), 'width': slider_ch1.val[1] - slider_ch1.val[0]})
    rect_ch2.update({'xy':(slider_ch2.val[0],0), 'width': slider_ch2.val[1] - slider_ch2.val[0]})

# Make images for color scales (128 x 1 pixels)
img_scale_ch0 = np.zeros((1,128,3), np.uint8)
img_scale_ch1 = np.zeros((1,128,3), np.uint8)
img_scale_ch2 = np.zeros((1,128,3), np.uint8)

def updateColorScales():
    global img_scale_ch0, img_scale_ch1, img_scale_ch2
    img_scale_ch0[:]= [np.uint8(mid_slider_ch0), np.uint8(mid_slider_ch1), np.uint8(mid_slider_ch2)]
    img_scale_ch1= img_scale_ch0.copy()
    img_scale_ch2= img_scale_ch0.copy()
    img_scale_ch0[0,:,0] = np.linspace(0, slider_ch0.valmax, 128, dtype=np.uint8)
    img_scale_ch1[0,:,1] = np.linspace(0, slider_ch1.valmax, 128, dtype=np.uint8)
    img_scale_ch2[0,:,2] = np.linspace(0, slider_ch2.valmax, 128, dtype=np.uint8)
    if color_space=='HSV':
        img_scale_ch0 = cv2.cvtColor(img_scale_ch0, cv2.COLOR_HSV2RGB)
        img_scale_ch1 = cv2.cvtColor(img_scale_ch1, cv2.COLOR_HSV2RGB)
        img_scale_ch2 = cv2.cvtColor(img_scale_ch2, cv2.COLOR_HSV2RGB)
    elif color_space=='Lab':
        img_scale_ch0 = cv2.cvtColor(img_scale_ch0, cv2.COLOR_LAB2RGB)
        img_scale_ch1 = cv2.cvtColor(img_scale_ch1, cv2.COLOR_LAB2RGB)
        img_scale_ch2 = cv2.cvtColor(img_scale_ch2, cv2.COLOR_LAB2RGB)
    imx_scale_ch0.set_data(img_scale_ch0)
    imx_scale_ch1.set_data(img_scale_ch1)
    imx_scale_ch2.set_data(img_scale_ch2)

def showThresholdedImage(low_ch0, high_ch0, low_ch1, high_ch1, low_ch2, high_ch2):
    #Make thresholded image
    lower_bound = np.array([low_ch0, low_ch1, low_ch2])
    upper_bound = np.array([high_ch0, high_ch1, high_ch2])
    frame_thresh = cv2.inRange(frame_space, lower_bound, upper_bound)
    if inv:
        frame_thresh= 255-frame_thresh
    imx_thresh.set_data(frame_thresh)

def plotHistograms():
    global rect_ch0, rect_ch1, rect_ch2, max_hist_ch0, max_hist_ch1, max_hist_ch2
    ax_hist_ch0.clear()
    ax_hist_ch1.clear()
    ax_hist_ch2.clear() 
    #Plot histograms
    (hr_ch0, edges_ch0, hr_ch1, edges_ch1, hr_ch2, edges_ch2) = histo_data
    ax_hist_ch0.stairs(hr_ch0, edges_ch0, ec='k', linewidth=1.0, zorder=3)
    ax_hist_ch0.set_xlim(0.0, len(hr_ch0), auto=False)
    ax_hist_ch1.stairs(hr_ch1, edges_ch1, ec='k', linewidth=1.0, zorder=3)
    ax_hist_ch1.set_xlim(0.0, len(hr_ch1), auto=False)
    ax_hist_ch2.stairs(hr_ch2, edges_ch2, ec='k', linewidth=1.0, zorder=3)
    ax_hist_ch2.set_xlim(0.0, len(hr_ch2), auto=False)
    for axis in (ax_hist_ch0, ax_hist_ch1, ax_hist_ch2):
        plt.sca(axis)
        plt.yticks([])
    ax_hist_ch0.set_title(color_space[0])
    ax_hist_ch1.set_title(color_space[1])
    ax_hist_ch2.set_title(color_space[2])
    ax_hist_ch0.set_xlim(left=-2, right=slider_ch0.valmax+2)
    ax_hist_ch1.set_xlim(left=-2, right=slider_ch1.valmax+2)
    ax_hist_ch2.set_xlim(left=-2, right=slider_ch2.valmax+2)
    #Draw selected intervals
    max_hist_ch0 = max(hr_ch0)
    max_hist_ch1 = max(hr_ch1)
    max_hist_ch2 = max(hr_ch2)
    rect_ch0= ax_hist_ch0.add_patch(Rectangle((slider_ch0.val[0],0),slider_ch0.val[1]-slider_ch0.val[0],max_hist_ch0,facecolor='#e0e0e0',zorder=1))
    rect_ch1= ax_hist_ch1.add_patch(Rectangle((slider_ch1.val[0],0),slider_ch1.val[1]-slider_ch1.val[0],max_hist_ch1,facecolor='#e0e0e0',zorder=1))
    rect_ch2= ax_hist_ch2.add_patch(Rectangle((slider_ch2.val[0],0),slider_ch2.val[1]-slider_ch2.val[0],max_hist_ch2,facecolor='#e0e0e0',zorder=1))
    #Draw color samples
    for hsvSample in colorSamples:
        markColorSample(hsvSample)  

def setImageAxes():
    global frame_src_sized, imx_source, imx_thresh
    if low_res:
        area= frame_src_rgb.shape[0]*frame_src_rgb.shape[1]
        if area>65536:
            scale= math.sqrt(65536.0/area)
            frame_src_sized = cv2.resize(frame_src_rgb, (int(frame_src_rgb.shape[1]*scale),int(frame_src_rgb.shape[0]*scale)),interpolation=cv2.INTER_NEAREST)
        else:
            frame_src_sized= frame_src_rgb
    aspect= 'auto' if fit else 'equal'
    ax_img_source.cla()
    ax_img_thresh.cla()
    imx_source= ax_img_source.imshow(frame_src_sized, aspect=aspect)
    imx_thresh= ax_img_thresh.imshow(np.zeros(frame_src_sized.shape[0:2]), cmap='gray', vmin=0, vmax=255, aspect=aspect)
    ax_img_source.set_xticks([])
    ax_img_source.set_yticks([])
    ax_img_thresh.set_xticks([])
    ax_img_thresh.set_yticks([])

def makeImages():
    global frame_rgb, frame_space, histo_data
    if bright!=1.0:
        frame_rgb= frame_src_sized*bright;
        frame_rgb= np.clip(frame_rgb, a_min=0.0, a_max=255.0).astype(np.uint8)
    else:
        frame_rgb= frame_src_sized
    imx_source.set_data(frame_rgb)
    
    if color_space=='HSV':
        frame_space = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    elif color_space=='Lab':
        frame_space = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
    else:
        frame_space = frame_rgb
    #Compute histograms for new image
    hr_ch0, edges_ch0 = np.histogram(frame_space[:,:,0],slider_ch0.valmax+1,(0,slider_ch0.valmax))
    hr_ch1, edges_ch1 = np.histogram(frame_space[:,:,1],slider_ch1.valmax+1,(0,slider_ch1.valmax))
    hr_ch2, edges_ch2 = np.histogram(frame_space[:,:,2],slider_ch2.valmax+1,(0,slider_ch2.valmax))  
    histo_data= (hr_ch0, edges_ch0, hr_ch1, edges_ch1, hr_ch2, edges_ch2)
    #Update figure
    plotHistograms()
    sliderUpdate(None)

#Load new image
def loadImage():
    global frame_src_rgb
    ent= list_images[iFrame]
    fig.canvas.manager.set_window_title('Color thresholder ['+ent.name+']')
#     filename = f2Name + ent.name
    filename = './images/' + ent.name
    frame_src_rgb = cv2.imread(filename, cv2.IMREAD_COLOR)
    setImageAxes()
    makeImages()

#Define button handlers
def onClickPrev(clic):
    global iFrame
    if ax_but_prev.get_visible():
        if iFrame > 0:
            iFrame = iFrame-1
            ax_but_next.set_visible(True)
        if iFrame == 0:
            ax_but_prev.set_visible(False)
        loadImage()
        fig.canvas.draw_idle()
but_prev.on_clicked(onClickPrev)

def onClickNext(clic):
    global iFrame
    if ax_but_next.get_visible():
        if iFrame < nFrames-1:
            iFrame = iFrame+1
            ax_but_prev.set_visible(True)
        if iFrame == nFrames-1:
            ax_but_next.set_visible(False)
        loadImage()
        fig.canvas.draw_idle()
but_next.on_clicked(onClickNext)


def onClickClearSamples(clic):
    colorSamples.clear()
    plotHistograms()
    fig.canvas.draw_idle()
but_clear_samples.on_clicked(onClickClearSamples)

def onClickResetBright(clic):
    slider_bright.set_val(1.0)
but_reset.on_clicked(onClickResetBright)
# Ask for image directory
init_folder = './images/'
folder_name = filedialog.askdirectory(initialdir=init_folder)
#folder_name = '../cartas'
f2Name = folder_name + '/'
list_files = list(os.scandir(f2Name))
list_images= [file for file in list_files if file.is_file() and file.name.lower().endswith(('.jpg','.png','tif','bmp'))]
nFrames= len(list_images)
print(nFrames)
colorSamples=[]
color_space='RGB'
bright= 1.0
slider_init={'RGB':{'valmax':[255,255,255], 'valinit':[[85,170],[85,170],[85,170]]}, 'HSV':{'valmax':[180,255,255], 'valinit':[[60,120],[85,170],[85,170]]}, 'Lab':{'valmax':[255,255,255], 'valinit':[[85,170],[85,170],[85,170]]}}
initSliders()
iFrame= 0
loadImage()
radbut_color_space.set_active(0)

# Now wait for events (mouse clics, buttons, sliders, close window)
plt.show()