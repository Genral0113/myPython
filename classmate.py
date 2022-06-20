import cv2
import numpy as np
from myutils import *

img_file = r'C:\Users\asus\Desktop\classmates\01.jpg'

img = cv2.imread(img_file)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv_show('gray', gray)

