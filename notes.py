#%%
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import os

import classification as cl
import fronts as fr
import analysis as an

from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline


from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
# #%%
# sx = pd.DataFrame(pd.read_csv(("Results/labelled_images/fronts/fronts_labelled_m_0.png.txt"),delimiter =' '))
# sx.columns = ["x", "y"]
# #%%
# plt.xlim(0,1200)
# plt.ylim(0,1600)
# plt.plot(sx["y"],sx["x"])
#
# #%%
#
#
# #%%
# sx = pd.DataFrame(pd.read_csv(("Results/labelled_images/fronts/divided_fronts/sxfronts_labelled_m_0.png.txt"),delimiter =' '))
#
#
# plt.xlim(0,1200)
# plt.ylim(0,1600)
# plt.plot(sx["y"],sx["x"])
# len(sx)
# #%%
# y = savgol_filter(sx["x"], 5, 3)
# plt.plot(y)
# #%%
#
#

#%%

# for im in os.listdir(image_folder):
#     im2 = im
#     im = im.replace("_",".")
#     os.rename(image_folder+im2,image_folder+im)

# import cv2
# import os
# import re
# image_folder = "Results/labelled_images1216/"
# images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
# images
#
#
#
# images = sorted(images, key = lambda x: int(x.split('.')[2]))
# for im in images:
#     frame = cv2.imread(os.path.join(image_folder, im))
#     height, width, layers = frame.shape
# video = cv2.VideoWriter("video1216.avi", 0, 1, (width,height))
#
#
# for image in images:
#     video.write(cv2.imread(os.path.join(image_folder, image)))
# video.release()
# #%%

#
# _ , im = fr.fronts("Results/labelled_images/labelled_m_8.png")
#
# plt.imsave("fr8.png",im)
# #%%
#
#
#
# def atoi(text):
#     if text.isdigit():
#         return int(text)

#%%
#
# train_image =  cv2.imread("Data/images/31.png")
# im_gray = cv2.cvtColor(train_image, cv2.COLOR_BGR2GRAY)
# pca_train = cl.Principal_components_analysis(im_gray,10,10)
#
# labels = GaussianMixture(2).fit_predict(pca_train)
# fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# # ax.scatter(pca_train["x"],pca_train["y"],pca_train["z"],c=labels)
#
# labels = labels.reshape(120,160)
# plt.imshow(labels)
# plt.show()
# #plt.imsave("3cluster.png",labels)
#

#%%
path = "Results/labelled_images1010/fronts1/"
areas = []
for value in list(os.listdir(path)):
    if str(value).endswith("png.txt"):
        pol, area = an.area(path + value)
        areas.append(area)

areas_hand = []
path = "Data/data_fronts/"
i = 1
df_sx = pd.DataFrame()
df_dx = pd.DataFrame()

for i in range(int(len(os.listdir(path))/2-1)):
    path = "Data/data_fronts/"
    value = "Sham_8-2-18_Field 5_"+str(i+1)+"_sx.txt"
    value1 = "Sham_8-2-18_Field 5_"+str(i+1)+"_dx.txt"
    areas_hand.append(an.area_btw_fronts(path + value,path + value1))

areas = np.array(areas)
areas = sorted(areas,reverse=True)
areas = areas/areas[0]
areas = areas[:40]
areas_hand = np.array(areas_hand)
areas_hand = sorted(areas_hand,reverse=True)
areas_hand = areas_hand/areas_hand[0]
areas_hand = areas_hand[:40]

plt.plot(areas)
plt.plot(areas_hand)
error = an.error(areas,areas_hand)

plt.plot(error)
plt.show()


#%%
path = "Data/data_fronts/"
for file in os.listdir(path):
        polsx = pd.DataFrame(pd.read_csv(path + file,sep ='\t'))
        polsx.columns = ["y","x"]
        # poldx = pd.DataFrame(pd.read_csv(df_dx,sep ='\t'))
        # poldx.columns = ["y","x"]
        polsx["y"] = polsx["y"]*1600/844
        polsx["x"] = polsx["x"]*1200/630
polsx = pd.DataFrame(pd.read_csv(path + "Sham_8-2-18_Field 5_27_sx.txt",sep ='\t'))
polsx.columns = ["y","x"]
# poldx = pd.DataFrame(pd.read_csv(df_dx,sep ='\t'))
# poldx.columns = ["y","x"]
polsx["y"] = polsx["y"]*1600/844
polsx["x"] = polsx["x"]*1200/630
        #plt.plot(polsx["x"],polsx["y"])
plt.ylim(0,1600)
plt.xlim(0,1200)
        # plt.plot(poldx["y"]["x"])

pol = pd.DataFrame(pd.read_csv("Results/labelled_images1010/fronts/fronts_labelled.m.26.png.txt",sep =' '))
pol.columns = ["y","x"]
plt.plot(pol["x"],pol["y"])
plt.plot(polsx["x"],polsx["y"])
plt.show()
#%%

plt.figure(dpi=160)
plt.xlabel("frames")
plt.plot(areas,label="areas")
plt.plot(areas_hand, label= "hand drawn areas")
plt.legend()
plt.savefig("areas.png")

error = an.error(np.array(areas),np.array(areas_hand))
plt.figure(dpi=160)
plt.xlabel("frames")
plt.ylabel("error")
plt.plot(error)
plt.savefig("error_btw_areas.png")
