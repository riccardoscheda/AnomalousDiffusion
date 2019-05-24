import cv2
import pylab as plt

import classification as cl


test_image =  cv2.imread("modified_images/m_15.png")
im_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

pca = cl.Principal_components_analysis(im_gray)
plt.scatter(pca["x"],pca["y"])
