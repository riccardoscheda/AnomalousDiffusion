import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import os

import classification as cl
import fronts as fr

# outfolder = "fronts/"
# if not os.path.exists(outfolder):
#     os.makedirs(outfolder)
#
# if not os.path.exists("labelled_images/"):
#     os.makedirs("labelled_images/")
#
#
# for i in range(1):
#     test_image =  cv2.imread("modified_images/m_"+str(i)+".png")
#     im_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
#     pca = cl.Principal_components_analysis(im_gray)
#     labelled_image = cl.classification(pca)
#     plt.imsave("labelled_images/labelled_image_"+str(i)+".png",labelled_image)

coord = fr.fronts("labelled_images/labelled_image_0.png","prova.txt")
plt.plot(coord[0],coord[1])

#
# struct = [0,1,1,1,1,1,1,1,0]
# kernel = fr.make_kernel(struct,40)
# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(15,15))
# kernel = np.array(kernel,np.uint8)
# opening = cv2.morphologyEx(labelled_image,cv2.MORPH_OPEN,kernel)
# plt.imshow(opening)
