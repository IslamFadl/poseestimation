import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import io, measure
from scipy import ndimage as nd
#from matplotlib.colors import rgb_to_hsv


img = io.imread("temp/cam_1_Z_76.png")
#gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.imshow(img)


# TODO: define region of interest:
"""
# define ROI of RGB image 'img'
roi = img[r1:r2, c1:c2]

# convert it into HSV
hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
"""
# todo: get histogram

"""
https://dsp.stackexchange.com/questions/5922/how-to-determine-range-of-hsv-values-of-the-image
"""


## convert to hsv
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#mask = cv2.inRange(hsv, (100, 90, 90), (120, 255, 255))  #Blue
#mask = cv2.inRange(hsv, (160, 90, 20), (180, 255, 255))  #Red
mask = cv2.inRange(hsv, (0,0,75), (150, 150, 255))   #gray
plt.imshow(mask)



closed_mask = nd.binary_closing(mask, np.ones((7,7)))
plt.imshow(closed_mask)

label_image = measure.label(closed_mask)
plt.imshow(label_image)

print('done')

"""
rgb range for drill
44,44,44
123,123,123
x,x,x
y,y,y
z,z,z
etc
"""

"""
hsv values range for drill
10, 20, 90
40, 10,90
85,19,160
80, 20, 100
80, 20, 150
90, 30, 150

10,10,90
90, 30, 160
"""