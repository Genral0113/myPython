import cv2
import matplotlib.pyplot as plt
import numpy as np


def display_image_and_press_any_key_to_continue(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img_file = r'D:\Workspace\pic\img1.jpg'
img = cv2.imread(img_file)
print(img.shape)

# display origin image
# display_image_and_press_any_key_to_continue('origin', img)

# for i in range(3):
#     display_image_and_press_any_key_to_continue('', img[:, :, i])

# to gray image
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
# print(gray_img.shape)
display_image_and_press_any_key_to_continue('gray', gray_img)

#
# top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)

# replicate_img = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
# replicate_reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REFLECT)
# replicate_reflect_101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REFLECT_101)
# replicate_wrap = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_WRAP)
# replicate_constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_CONSTANT, value=0)

# plt.subplot(231), plt.imshow(img, 'gray'), plt.title('original')
# plt.subplot(232), plt.imshow(replicate_img, 'gray'), plt.title('replicate')
# plt.subplot(233), plt.imshow(replicate_reflect, 'gray'), plt.title('reflect')
# plt.subplot(234), plt.imshow(replicate_reflect_101, 'gray'), plt.title('reflect101')
# plt.subplot(235), plt.imshow(replicate_wrap, 'gray'), plt.title('wrap')
# plt.subplot(236), plt.imshow(replicate_constant, 'gray'), plt.title('constant')

# plt.show()
# plt.close()

#
thresh_hold = 50
max_val = 255
ret, thresh_binary = cv2.threshold(gray_img, thresh_hold, max_val, cv2.THRESH_BINARY)
ret, thresh_binary_inv = cv2.threshold(gray_img, thresh_hold, max_val, cv2.THRESH_BINARY_INV)
ret, thresh_mask = cv2.threshold(gray_img, thresh_hold, max_val, cv2.THRESH_MASK)
ret, thresh_otsu = cv2.threshold(gray_img, thresh_hold, max_val, cv2.THRESH_OTSU)
ret, thresh_tozero = cv2.threshold(gray_img, thresh_hold, max_val, cv2.THRESH_TOZERO)
ret, thresh_tozero_inv = cv2.threshold(gray_img, thresh_hold, max_val, cv2.THRESH_TOZERO_INV)
ret, thresh_triangle = cv2.threshold(gray_img, thresh_hold, max_val, cv2.THRESH_TRIANGLE)
ret, thresh_trunc = cv2.threshold(gray_img, thresh_hold, max_val, cv2.THRESH_TRUNC)
#
# titles = ['Gray', 'BINARY', 'BINARY_INV', 'MASK', 'OTSU', 'TOZERO', 'TOZERO_INV', 'TRIANGLE', 'TRUNC']
# images = [gray_img, thresh_binary, thresh_binary_inv, thresh_mask, thresh_otsu, thresh_tozero, thresh_tozero_inv, thresh_triangle, thresh_trunc]
#
# for i in range(len(titles)):
#     plt.subplot(3, 3, i + 1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()

#
blur = cv2.blur(img, (3, 3))
title = 'blur'
# display_image_and_press_any_key_to_continue(title, blur)

blur_morm_true = cv2.boxFilter(img, -1, (3, 3), normalize=True)
title = 'blur_normalize_true'
# display_image_and_press_any_key_to_continue(title, blur_morm_true)

blur_norm_flase = cv2.boxFilter(img, -1, (3, 3), normalize=False)
title = 'blur_normalize_false'
# display_image_and_press_any_key_to_continue(title, blur_norm_flase)

blur_gaussian = cv2.GaussianBlur(img, (3, 3), 1)
title = 'blur_Gaussian'
# display_image_and_press_any_key_to_continue(title, blur_gaussian)

blur_median = cv2.medianBlur(img, 3)
title = 'blur_median'
# display_image_and_press_any_key_to_continue(title, blur_median)

res = np.hstack((blur, blur_gaussian, blur_median))
title = 'median vs average'
# display_image_and_press_any_key_to_continue(title, res)


kernel = np.ones((3, 3), np.uint8)
erosion = cv2.erode(thresh_binary, kernel, iterations=1)
title = 'erosion'
# display_image_and_press_any_key_to_continue(title, erosion)

dilate = cv2.dilate(thresh_binary, kernel, iterations=1)
title = 'dilate'
# display_image_and_press_any_key_to_continue(title, erosion)

morph_close = cv2.morphologyEx(thresh_binary, cv2.MORPH_CLOSE, kernel)
title = 'morphology close'
# display_image_and_press_any_key_to_continue(title, morph_close)

morph_open = cv2.morphologyEx(thresh_binary, cv2.MORPH_OPEN, kernel)
title = 'morphology open'
# display_image_and_press_any_key_to_continue(title, morph_open)

morph_gradient = cv2.morphologyEx(thresh_binary, cv2.MORPH_GRADIENT, kernel)
title = 'morphology gradient'
# display_image_and_press_any_key_to_continue(title, morph_gradient)

morph_tophat = cv2.morphologyEx(thresh_binary, cv2.MORPH_TOPHAT, kernel)
title = 'morphology tophat'
# display_image_and_press_any_key_to_continue(title, morph_tophat)

morph_blackhat = cv2.morphologyEx(thresh_binary, cv2.MORPH_BLACKHAT, kernel)
title = 'morphology blankhat'
# display_image_and_press_any_key_to_continue(title, morph_blackhat)

sobelx = cv2.Sobel(thresh_binary, cv2.CV_64F, 1, 0, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
title = 'sobel x'
# display_image_and_press_any_key_to_continue(title, sobelx)

sobely = cv2.Sobel(thresh_binary, cv2.CV_64F, 0, 1, ksize=3)
sobely = cv2.convertScaleAbs(sobely)
title = 'sobel y'
# display_image_and_press_any_key_to_continue(title, sobely)

sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
title = 'sobel xy'
# display_image_and_press_any_key_to_continue(title, sobelxy)


scharrx = cv2.Scharr(gray_img, cv2.CV_64F, 1, 0)
scharrx = cv2.convertScaleAbs(scharrx)
scharry = cv2.Scharr(gray_img, cv2.CV_64F, 0, 1)
scharry = cv2.convertScaleAbs(scharry)
scharrxy = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)
display_image_and_press_any_key_to_continue('scharrxy', scharrxy)

laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

res = np.hstack((sobelxy, scharrxy, laplacian))
# display_image_and_press_any_key_to_continue('res', res)

canny = cv2.Canny(gray_img, 50, 100)
title = 'canny'
# display_image_and_press_any_key_to_continue(title, canny)
