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
ap.add_argument("-d", "--debug", required=True, help="input if debug")
args = vars(ap.parse_args())
if args['debug'] == 'True':
    args['debug'] = True
else:
    args['debug'] = False

if args['debug']:
    print(args)

#
FIRST_NUMBER = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card"
}


def cv_show(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread(args['template'])
if args['debug']:
    cv_show('original template', img)

ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
if args['debug']:
    cv_show('ref', ref)

ref_binary = cv2.threshold(ref, int(args['min_value']), 255, cv2.THRESH_BINARY_INV)[1]
if args['debug']:
    cv_show('binary image', ref_binary)

# kernel = np.ones((3, 3), np.uint8)
# ref = cv2.erode(ref, kernel, iterations=1)
# if args['debug']:
    # cv_show('erode', ref)

ref_, refCnts, hierarchy = cv2.findContours(ref_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_img = cv2.drawContours(img.copy(), refCnts, -1, (0, 0, 255), 1)
if args['debug']:
    cv_show('contour', contour_img)

refCnts = myutils.sort_contours(refCnts, method="left-to-right")[0]

digits = {}
for (i, c) in enumerate(refCnts):
    (x, y, w, h) = cv2.boundingRect(c)
    roi = ref[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88))
    digits[i] = roi

if args['debug']:
    for i in range(len(digits)):
        cv_show(str(i), digits[i])

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

image = cv2.imread(args["image"])
if args['debug']:
    cv_show('image', image)
image = myutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
if args['debug']:
    cv_show('gray', gray)

tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
if args['debug']:
    cv_show('tophat', tophat)

gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)

gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")

print(np.array(gradX).shape)
if args['debug']:
    cv_show('gradX', gradX)

gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
if args['debug']:
    cv_show('gradX', gradX)
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
if args['debug']:
    cv_show('thresh', thresh)

thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
if args['debug']:
    cv_show('thresh', thresh)

thresh_, threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = threshCnts
cur_img = image.copy()
cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 3)
if args['debug']:
    cv_show('img', cur_img)

locs = []
for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

    if 2.5 < ar < 4.0:
        if (40 < w < 55) and (10 < h < 20):
            locs.append((x, y, w, h))

locs = sorted(locs, key=lambda x: x[0])

output = []

for (i, (gX, gY, gW, gH)) in enumerate(locs):
    groupOutput = []
    group = gray[gY-5:gY + gH + 5, gX - 5: gX + gW + 5]
    if args['debug']:
        cv_show('group', group)

    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    if args['debug']:
        cv_show('group', group)

    group_, digitCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]

    for c in digitCnts:
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))
        if args['debug']:
            cv_show('roi', roi)

        scores = []

        for(digit, digitROI) in digits.items():
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)

    groupOutput.append(str(np.argmax(scores)))

    cv2.rectangle(image, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
    cv2.putText(image, "".join(groupOutput), (gX, gY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    output.extend(groupOutput)

# print('Credit Card Type: {}'.format(FIRST_NUMBER[output[0]]))
# print('Credit Card # {}'.format(''.join(output)))
cv2.imshow('Image', image)
cv2.waitKey(0)
