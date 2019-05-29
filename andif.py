import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import os

import classification as cl
import fronts

outfolder = "fronts/"
if not os.path.exists(outfolder):
    os.makedirs(outfolder)


for i in range(0,60):
    test_image =  cv2.imread("modified_images/m_"+str(i)+".png")
    im_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    pca = cl.Principal_components_analysis(im_gray)
    labelled_image = cl.classification(pca)
    fronts.fronts(labelled_image,"fronts/fronts"+str(i)+".txt")
