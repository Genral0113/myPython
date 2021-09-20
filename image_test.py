from scipy import misc
import matplotlib.pyplot as plt
import imageio
from numpy import linalg

img_file1 = r'D:\Workspace\pic\img1.jpg'
img_file2 = r'D:\Workspace\pic\DSC_0020.jpg'

img = imageio.imread(img_file1)

img_gray = img @ [0.2126, 0.7152, 0.0722]
# print(img_gray.shape)

plt.imshow(img_gray, cmap="gray")
plt.show()

img_red = img[:, :, 0]
img_green = img[:, :, 1]
img_blue = img[:, :, 2]

U, s, Vt = linalg.svd(img_gray)
print(U.shape)
print(s.shape)
print(Vt)
