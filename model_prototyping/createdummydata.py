import numpy as np
import cv2
from math import atan2, pi, hypot, sqrt, acos
import os
import shutil
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


RESNET_INPUT_SIZE = 224
IMG_SIZE = RESNET_INPUT_SIZE
THICKNESS = 5
B, G, R = 255, 0, 0

def twopoints3d(IMG_SIZE):
    """
    Creates two points in 3d space, using an if condition to make sure no coordinates are equal. This is necessary
    to prevent division by zero in xyplaneintersection(point1, point2), yzplaneintersection(point1, point2) and
    zxplaneintersection(point1, point2).

    :param IMG_SIZE: = Network input size, used to create points within the image canvas.
    :return: two lists: point1, point2, each of three elements.
    """
    # Create two points in 3D coordinates
    x1 = int(np.random.randint(1, high=IMG_SIZE - 1, size=None, dtype=int))
    y1 = int(np.random.randint(1, high=IMG_SIZE - 1, size=None, dtype=int))
    z1 = int(np.random.randint(1, high=IMG_SIZE - 1, size=None, dtype=int))
    #np.random.seed(41)
    x2 = int(np.random.randint(1, high=IMG_SIZE - 1, size=None, dtype=int))
    y2 = int(np.random.randint(1, high=IMG_SIZE - 1, size=None, dtype=int))
    z2 = int(np.random.randint(1, high=IMG_SIZE - 1, size=None, dtype=int))
    #
    if x1 != x2 and y1 != y2 and z1 != z2:
        point1, point2 = [x1, y1, z1], [x2, y2, z2]
    else:
        x1 = int(np.random.randint(1, high=IMG_SIZE - 1, size=None, dtype=int))
        y1 = int(np.random.randint(1, high=IMG_SIZE - 1, size=None, dtype=int))
        z1 = int(np.random.randint(1, high=IMG_SIZE - 1, size=None, dtype=int))
        x2 = int(np.random.randint(1, high=IMG_SIZE - 1, size=None, dtype=int))
        y2 = int(np.random.randint(1, high=IMG_SIZE - 1, size=None, dtype=int))
        z2 = int(np.random.randint(1, high=IMG_SIZE - 1, size=None, dtype=int))
        point1, point2 = [x1, y1, z1], [x2, y2, z2]
    return point1, point2

def xyplaneintersection(point1, point2):
    """
    From two points this function calculates the intersection point of the line with xy-plane
    :param point1: list = [x1, y1, z1]
    :param point2: list = [x2, y2, z2]
    :return: intersection point with xy plane = [xp, yp, zp]
    """
    # first get the equation of the line passing through the two points.
    [x1, y1, z1], [x2, y2, z2] = point1, point2
    # Secondly, calculate direction rations (DRs).
    l = x2 - x1
    m = y2 - y1
    n = z2 - z1
    # use point 1 or 2 to substitute in the equation below.
    # equation of the line is: Q = (x-x1)/l = (y-y1)/m = (z-z1)/n
    # This line intersects with xy-plane in (xp,yp,zp=0)
    # substitute: Q = (xp-x1)/l = (yp-y1)/m = (zp-z1)/n
    xp = x1 + (0 - z1) * l / n
    yp = y1 + (0 - z1) * m / n
    zp = z1 + (xp - x1) * n / l  # zp should = 0
    return [xp, yp, zp]

def yzplaneintersection(point1, point2):
    """
    From two points this function calculates the intersection point of the line with yz-plane
    :param point1: list = [x1, y1, z1]
    :param point2: list = [x2, y2, z2]
    :return: intersection point with xy plane = [xp, yp, zp]
    """
    (x1, y1, z1), (x2, y2, z2) = point1, point2
    l = x2 - x1
    m = y2 - y1
    n = z2 - z1
    # use point 1 or 2 to substitute in the equation below.
    # equation of the line is: Q = (x-x1)/l = (y-y1)/m = (z-z1)/n
    # This line intersects with xy-plane in (xp=0,yp,zp)
    # substitute: Q = (xp-x1)/l = (yp-y1)/m = (zp-z1)/n
    yp = y1 + (0 - x1) * m / l
    zp = z1 + (0 - x1) * n / l
    xp = x1 + (zp - z1) * l / n  # xp should = 0
    return [xp, yp, zp]

def zxplaneintersection(point1, point2):
    """
    From two points this function calculates the intersection point of the line with zx-plane
    :param point1: list = [x1, y1, z1]
    :param point2: list = [x2, y2, z2]
    :return: intersection point with xy plane = [xp, yp, zp]
    """
    (x1, y1, z1), (x2, y2, z2) = point1, point2
    l = x2 - x1
    m = y2 - y1
    n = z2 - z1
    # use point 1 or 2 to substitute in the equation below.
    # equation of the line is: Q = (x-x1)/l = (y-y1)/m = (z-z1)/n
    # This line intersects with xy-plane in (xp,yp=0,zp)
    # substitute: Q = (xp-x1)/l = (yp-y1)/m = (zp-z1)/n
    xp = x1 + (0 - y1) * l / m
    zp = z1 + (0 - y1) * n / m
    yp = y1 + (xp - x1) * m / l  # yp should = 0
    return [xp, yp, zp]

def twovectors(xp, yp, zp, point):
    """
    Get two vectors to perform a dot product afterwards between them.

    :param xp: intersection of line connecting point1 and point 2 with the yz-plane
    :param yp: intersection of line connecting point1 and point 2 with the zx-plane
    :param zp: intersection of line connecting point1 and point 2 with the xy-plane
    :param point1: list
    :return:
    """
    vec1 = np.array(point) - np.array([xp, yp, zp])
    vec2 = np.array([0, 0, 0]) - np.array([xp, yp, zp])
    return vec1, vec2

def angle3d(vec1, vec2):
    """
    From two vectors get angle by dot product.

    :param vec1: list
    :param vec2: list
    :return: alpha is the angle between the two vectors in 3d space, which is the Ground Truth and resembels what the
    sensor measures.
    """
    [x1, y1, z1] = vec1
    [x2, y2, z2] = vec2
    nom = x1 * x2 + y1 * y2 + z1 * z2
    denom = sqrt((x1 ** 2 + y1 ** 2 + z1 ** 2) * (x2 ** 2 + y2 ** 2 + z2 ** 2))
    alpha = acos(nom / denom)
    return alpha

def linelength3d(point1, point2):
    """
    Get the length of the line connecting point 1 and point 2 in 3d space.

    :param point1: list
    :param point2: list
    :return: float
    """
    (x1, y1, z1), (x2, y2, z2) = point1, point2
    l = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
    return l

def linelength2d(point1, point2):
    """
    Get the length of the line connecting point 1 and point 2 in 2d space.

    :param point1: list
    :param point2: list
    :return: float
    """
    (x1, y1), (x2, y2) = point1, point2
    l = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return l

def xy2dpoints(point1, point2):
    """
    Projection of a 3d point on a 3d plane means that one of the coordinates will be zero.
    Returns 2d point from a 3d point: (x, y, z) >> (x, y)
    :param point1: list
    :param point2: list
    :return: two 2d points
    """
    x1, y1, z1 = point1
    point1 = x1, y1
    x2, y2, z2 = point2
    point2 = x2, y2
    return point1, point2

def yz2dpoints(point1, point2):
    """
    Projection of a 3d point on a 3d plane means that one of the coordinates will be zero.
    Returns 2d point from a 3d point: (x, y, z) >> (y, z)
    :param point1: list
    :param point2: list
    :return: two 2d points
    """
    x1, y1, z1 = point1
    point1 = y1, z1
    x2, y2, z2 = point2
    point2 = y2, z2
    return point1, point2

def zx2dpoints(point1, point2):
    """
    Projection of a 3d point on a 3d plane means that one of the coordinates will be zero.
    Returns 2d point from a 3d point: (x, y, z) >> (z, x)
    :param point1: list
    :param point2: list
    :return: two 2d points
    """
    x1, y1, z1 = point1
    point1 = x1, z1
    x2, y2, z2 = point2
    point2 = x2, z2
    return point1, point2

def drawcircle(x1, y1, x2, y2, img):
    """
    Draws a circle with a diameter equal the line connecting points 1 and 2 and the center in the middle.
    Draws a line between the two points.
    :return: nothing
    """
    diameter = hypot(x2 - x1, y2 - y1)
    r = int(diameter / 2)
    x_center, y_center = int(0.5 * (x1 + x2)), int(0.5 * (y1 + y2))
    cv2.circle(img, (x_center, y_center), r, (190, 190, 190), -1)  # -1 solid circle
    cv2.line(img, (x1, y1), (x2, y2), (B, G, R), THICKNESS)
    pass

def angle2d(point1, point2):
    # Calculate angle
    (x1, y1) = point1
    (x2, y2) = point2
    b = abs(y2 - y1)
    a = abs(x2 - x1)
    angle = atan2(b, a) * 180 / pi
    """
    if angle<0:
        angle = 180 - angle
    else:
        pass
    """
    return angle

# todo: add .png files and weight file to git ignore.

np.random.seed(42)
images_path = os.path.expanduser("~/PycharmProjects/syndataset/")
if os.path.exists(images_path) and os.path.isdir(images_path):
    # deletes the directory
    shutil.rmtree(images_path)
    pass
os.makedirs(images_path, exist_ok=True)

for i in range(300):
    point1, point2 = twopoints3d(IMG_SIZE)
    
    # l_3d = linelength3d(point1, point2)
    [xp_xy, yp_xy, zp_xy] = xyplaneintersection(point1, point2)
    vec1_xy, vec2_xy = twovectors(xp_xy, yp_xy, zp_xy, point1)
    # get angle for file name
    alpha_xy_3d = angle3d(vec1_xy, vec2_xy)
    print(alpha_xy_3d)
    point1_xy, point2_xy = xy2dpoints(point1, point2)
    angle_xy_2d = angle2d(point1_xy, point2_xy)
    # Draw a blue line over a gray circle
    x1, y1 = point1_xy
    x2, y2 = point2_xy
    # Create a black image
    # img_xy = np.zeros((IMG_SIZE,IMG_SIZE,3), np.uint8)
    # drawcircle(x1,y1,x2,y2,img_xy)

    # yz-plane image
    [xp_yz, yp_yz, zp_yz] = yzplaneintersection(point1, point2)
    vec1_yz, vec2_yz = twovectors(xp_yz, yp_yz, zp_yz, point1)
    alpha_yz_3d = angle3d(vec1_yz, vec2_yz)
    point1_yz, point2_yz = yz2dpoints(point1, point2)
    angle_yz_2d = angle2d(point1_yz, point2_yz)
    # Draw a blue line over a gray circle
    x1, y1 = point1_yz
    x2, y2 = point2_yz
    # Create a black image
    img_yz = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
    drawcircle(x1, y1, x2, y2, img_yz)

    # flipped yz-plane image
    flipHorizontal = cv2.flip(img_yz, 1)    # horizontal flip around y-axis

    # zx-plane image
    [xp_zx, yp_zx, zp_zx] = zxplaneintersection(point1, point2)
    vec1_zx, vec2_zx = twovectors(xp_yz, yp_zx, zp_zx, point1)
    alpha_zx_3d = angle3d(vec1_zx, vec2_yz)
    point1_zx, point2_zx = zx2dpoints(point1, point2)
    angle_zx_2d = angle2d(point1_zx, point2_zx)
    # Draw a blue line over a gray circle
    x1, y1 = point1_zx
    x2, y2 = point2_zx
    # Create a black image
    img_zx = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
    drawcircle(x1, y1, x2, y2, img_zx)

    # save images
    # cv2.imwrite(f'{images_path}/frame_{i}_angle_{alpha_xy_3d}_xyplane.png', img_xy)
    cv2.imwrite(f'{images_path}/frame_{i}_angle_{alpha_xy_3d}_yzplane.png', img_yz)
    cv2.imwrite(f'{images_path}/frame_{i}_angle_{alpha_xy_3d}_flippedyzplane.png', flipHorizontal)
    cv2.imwrite(f'{images_path}/frame_{i}_angle_{alpha_xy_3d}_zxplane.png', img_zx)
    print(f'Iteration {i + 1}\n')
   
    # Plot in 3D and visualize:
    """
    [x1, y1, z1], [x2, y2, z2] = point1, point2
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z');
    # Data for a three-dimensional line
    xline = np.linspace(x1, x2, 100)
    yline = np.linspace(y1, y2, 100)
    zline = np.linspace(z1, z2, 100)
    ax.plot3D(xline, yline, zline, 'gray')
    # Data for three-dimensional scattered points
    datapoints=30
    xdata = np.linspace(x1, x2, datapoints) + 0.1 * np.random.randn(datapoints)
    ydata = np.linspace(y1, y2, datapoints) + 0.1 * np.random.randn(datapoints)
    zdata = np.linspace(z1, z2, datapoints) + 0.1 * np.random.randn(datapoints)
    ax.scatter3D(xdata, ydata, zdata, cmap='Greens');
    plt.show()
    """
