import cv2
import numpy as np

def ginput(window_name, img, n_clks):
 # Wait 'number' mouse clicks and draw lines between them
 # the last point is the same as the first one
 # paramas:
 #         window_name: nema of window to receive mouse events
 #         img:  imagen showed in the window
 #         n_clks:  Number od mouse clicks to make a 'n_clks-1' polygon
 #              
 #return:
 #        []: List of (x,y)-tuples with coordinates of succesive points.
 #            the last point is the same of the first one.
    
    indx = 0    # local variables   
    (x0,y0) = (0,0)
    (x_fin,y_fin) = (0,0)
    n_clicks = n_clks
    points = []
    
    def onMouseClick(event, x, y, flags, img):	
    # to check if left mouse button was clicked
        nonlocal indx, x0, y0, x_fin, y_fin, n_clicks, points
        if ((event == cv2.EVENT_RBUTTONDOWN) or (event == cv2.EVENT_LBUTTONDOWN)):
            cv2.drawMarker(img, (x, y), (255, 255, 0), cv2.MARKER_SQUARE, 5,2)
            indx = indx+1
            if (indx==1):
                (x_fin, y_fin) = (x, y)
            if (indx>1) and (indx<n_clicks):
                  cv2.line(img,(x0,y0),(x,y),(0,0,255),2,8)
            if (indx == n_clicks):
                  cv2.line(img,(x0,y0),(x_fin,y_fin),(0,0,255),2,8)
                  (x,y) = (x_fin,y_fin)
            (x0,y0) = (x,y)
            points += [[x,y]]
            
    
    cv2.setMouseCallback(window_name, onMouseClick, img)
    while indx < n_clks:
        cv2.pollKey()
        cv2.imshow(window_name, img)
        # print(indx)
    cv2.setMouseCallback(window_name, lambda *args : None)  # Null callback
    cv2.pollKey()
    points = np.array(points[0:n_clks-1],np.float32)
    return points


