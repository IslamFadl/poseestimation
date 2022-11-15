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
    P = [xp, yp, zp]
    return P

def twovectors(P, A):
    """
    Get two vectors
    :param P: list [x, y, z]
    :param A: list [x, y, z]
    :return: np.array
    """
    vec1 = np.array(A) - np.array(P)
    vec2 = np.array([A[0], A[1], 0]) - np.array(P)
    return vec1, vec2

def angle3d(vec1, vec2):
    """
    Get angle by dot product from two vectors.
    :param vec1: list
    :param vec2: list
    :return: alpha is the angle between the two vectors in 3d space, which is the Ground Truth and resembels what the
    sensor measures.
    """
    [x1, y1, z1] = vec1
    [x2, y2, z2] = vec2
    nom = x1 * x2 + y1 * y2 + z1 * z2
    denom = sqrt((x1 ** 2 + y1 ** 2 + z1 ** 2) * (x2 ** 2 + y2 ** 2 + z2 ** 2))
    alpha = acos(nom / denom) * 180 /pi
    return alpha

def xyimage(A, B, img):
    """
    Get points projection on 2d planes and draw the lines b/w them.
    Projection of a 3d point on a 2d plane means that one of the coordinates will be zero.
    Drwas circle and line on the images.
    :param point1: list
    :param point2: list
    :return:
    """
    x1, y1, z1 = A
    x2, y2, z2 = B
    # diameter = hypot(z2 - z1, y2 - y1)
    # r = int(diameter / 2)
    # x_center, y_center = int(0.5 * (x1 + x2)), int(0.5 * (y1 + y2))
    # cv2.circle(img, (x_center, y_center), r, (190, 190, 190), -1)  # -1 solid circle
    cv2.line(img, (x1, y1), (x2, y2), (B, G, R), THICKNESS)
    pass

def yzimage(A, B, img):
    """
    Get points projection on 2d planes and draw the lines b/w them.
    Projection of a 3d point on a 2d plane means that one of the coordinates will be zero.
    Drwas circle and line on the images.
    :param point1: list
    :param point2: list
    :return:
    """
    x1, y1, z1 = A
    x2, y2, z2 = B
    # diameter = hypot(z2 - z1, y2 - y1)
    # r = int(diameter / 2)
    # x_center, y_center = int(0.5 * (x1 + x2)), int(0.5 * (y1 + y2))
    # cv2.circle(img, (x_center, y_center), r, (190, 190, 190), -1)  # -1 solid circle
    cv2.line(img, (y1, z1), (y2, z2), (B, G, R), THICKNESS)
    pass

# todo: add .png files and weight file to git ignore.

np.random.seed(42)
images_path = os.path.expanduser("~/PycharmProjects/syndataset/")
if os.path.exists(images_path) and os.path.isdir(images_path):
    # deletes the directory
    shutil.rmtree(images_path)
    pass
os.makedirs(images_path, exist_ok=True)

for i in range(1):
    A, B = twopoints3d(IMG_SIZE)
    P = xyplaneintersection(A, B)
    vec1, vec2 = twovectors(P, A)
    alpha = angle3d(vec1, vec2)
    img_xy = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
    xyimage(A, B, img_xy)
    img_yz = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
    yzimage(A, B, img_yz)

    # flipped xy-plane image
    flipped_xy = cv2.flip(img_xy, 1)  # horizontal flip.

    # save images
    # cv2.imwrite(f'{images_path}/frame_{i}_angle_{alpha_xy_3d}_xyplane.png', img_xy)
    cv2.imwrite(f'{images_path}/frame_{i}_angle_{alpha}_xyplane.png', img_xy)
    cv2.imwrite(f'{images_path}/frame_{i}_angle_{alpha}_flippedxyplane.png', flipped_xy)
    cv2.imwrite(f'{images_path}/frame_{i}_angle_{alpha}_yzplane.png', img_yz)
    print(f'Iteration {i + 1}\n')
   
