import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import classification as cl



test_image =  cv2.imread("modified_images/m_40.png")
im_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

pca = cl.Principal_components_analysis(im_gray)

labelled_image = cl.classification(pca)
plt.imshow(labelled_image)
plt.show()





# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(pca["x"], pca["w"],pca["z"], c = , s=50, cmap='viridis')
# plt.show()
