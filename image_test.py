from scipy import misc
import matplotlib.pyplot as plt
import imageio

img_file1 = r'C:\Users\asus\Desktop\img1.jpg'
img_file2 = r'C:\Users\asus\Desktop\DSC_0020.jpg'

img = imageio.imread(img_file1)
# plt.imshow(img)
# plt.show()
# print(img.shape)
# print(img)

img_red = img[:, :, 0]
img_green = img[:, :, 1]
img_blue = img[:, :, 2]

print(img_red)