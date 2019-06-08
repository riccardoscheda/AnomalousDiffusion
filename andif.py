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


dx, _= fr.fronts("Data/labelled_images/labelled_m_30.png")
#plt.plot(coord[0],coord[1])
dx
sx , dx = fr.divide(dx)
len(dx)!=len(sx)
len(dx)
len(sx)

dx = dx[:106]
len(dx)

df = pd.DataFrame(pd.read_csv( "Data/labelled_images/fronts/fronts_labelled_m_0.png.txt", delimiter=' '))
df.columns=["0","1"]
df
#reduced = coord[coord[1] > 60]
#reduced = reduced[reduced[1]<1180]

#flipped = pd.DataFrame(np.flip(np.array(coord)))
#
# plt.ylim((0,1200))
# plt.xlim((0,1600))
# plt.plot(sx[0],sx[1])
# plt.plot(sx[0],sx[1])
# #plt.plot(coord[0],coord[1])
# plt.show()


#
# struct = [0,1,1,1,1,1,1,1,0]
# kernel = fr.make_kernel(struct,40)
# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(15,15))
# kernel = np.array(kernel,np.uint8)
# opening = cv2.morphologyEx(labelled_image,cv2.MORPH_OPEN,kernel)
# plt.imshow(opening)
3
