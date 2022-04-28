from imutils import contours
import argparse
import imutils
import myutils
import cv2
import matplotlib.pyplot as plt
import numpy as np


def display_image_and_press_any_key_to_continue(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-t", "--template", required=True, help="path to template OCR-A image")
ap.add_argument("-min", "--min_value", required=True, help="input minimum value")
args = vars(ap.parse_args())
print(args['min_value'])

#
FIRST_NUMBER = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card"
}


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread(args['template'])
cv_show('original image', img)

ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv_show('ref', ref)

ref = cv2.threshold(ref, int(args['min_value']), 255, cv2.THRESH_BINARY)[1]
cv_show('binary image', ref)

kernel = np.ones((3, 3), np.uint8)
ref = cv2.erode(ref, kernel, iterations=1)
cv_show('erode', ref)

ref_, refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, refCnts, -1, (0, 0, 255), 3)
cv_show('img', img)