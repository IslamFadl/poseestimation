import os
import cv2
from pathlib import Path
import shutil

# Input of images to be converted from .png format to .jpg format.
rgb_PATH = os.path.expanduser("~/repos/labelme/LabelMeAnnotationTool/Images/segmentexample")
# Output path for the generated .jpg images.
rgb_OUTPATH = os.path.expanduser("~/repos/labelme/LabelMeAnnotationTool/Images/segmentexamplejpg")
if os.path.exists(rgb_OUTPATH) and os.path.isdir(rgb_OUTPATH):
    shutil.rmtree(rgb_OUTPATH)
os.makedirs(rgb_OUTPATH, exist_ok=True)

rgb_DIR_PATH = sorted(os.listdir(rgb_PATH))     # Order images in 'PATH' alphabetically.

print("Before saving images:")
print(os.listdir(rgb_OUTPATH))

for i, image_name in enumerate(rgb_DIR_PATH):
    # Index every image in 'DIR_PATH'.
    input_path = os.path.join(rgb_PATH, image_name)
    image = cv2.imread(input_path)
    # change directory
    os.chdir(rgb_OUTPATH)
    # Saving the image
    image_name = Path(input_path).stem
    cv2.imwrite(image_name + '.jpg', image)

# List files
print("After saving rgb images:")
print(os.listdir(rgb_OUTPATH))
print('Successfully saved')