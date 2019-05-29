import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


from sklearn.cluster import KMeans
import classification as cl


for i in range(30,31):
    test_image =  cv2.imread("modified_images/m_"+str(i)+".png")
    im_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    pca = cl.Principal_components_analysis(im_gray)
    labelled_image = cl.classification(pca)
    #plt.imsave("prova"+str(i)+".png",labelled_image)

plt.imshow(labelled_image)
