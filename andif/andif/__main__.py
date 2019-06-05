from plumbum import cli, colors
import os
from plumbum import local
from PIL import Image as imreader
from plumbum.cli.image import Image as imdisplay
from plumbum.cli.terminal import Progress

############################################
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import os

#####################################################
import sys
sys.path.insert(0, '/home/riccardo/git/AnomalousDiffusion')
import classification as cl
import fronts as fr
#import test_classification
######################################################

path = "."

class AnomalousDiffusion(cli.Application):
    """Application for Cell migration
    """
    PROGNAME = "andif"
    VERSION = "0.1.0"

    def main(self):
        if not self.nested_command:           # will be ``None`` if no sub-command follows
            print("No command given")
            return 1   # error exit code

@AnomalousDiffusion.subcommand("label")
class Label(cli.Application):
    "Saves the labelled image using pca and K-means algorithms"
    all = cli.Flag(["all", "every image"], help = "If given, I will label all the images in the current directory")
    def main(self, value : str = ""):
        if not os.path.exists("labelled_images"):
            os.makedirs("labelled_images")

        if(value == ""):
            if (self.all):
                cont = 0
                for value in list(os.listdir(".")):
                    if str(value).endswith(".png"):
                        test_image =  cv2.imread(value)
                        im_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
                        print("analyzing image " +str(cont+1)+"/"+str(len(os.listdir("."))))
                        #print("i'm doing PCA on the LBP image")
                        pca = cl.Principal_components_analysis(im_gray)
                        #print("PCA finished")
                        #print("Now i'm using K-means to classify the subimages")
                        labelled_image = cl.classification(im_gray, pca)
                        #print("K-means finished")
                        plt.imsave("labelled_images/labelled_"+str(cont)+".png",labelled_image)
                        cont = cont + 1
                print(colors.green|"Saved the labelled images in dir 'labelled_images/'")
            else:
                print(colors.red|"image not given")
        else:
            if(value not in list(os.listdir(path))):
                print(colors.red|"this image does not exists")
            else:
                test_image =  cv2.imread(value)
                im_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
                print("image taken")
                print("i'm doing PCA on the LBP image")
                pca = cl.Principal_components_analysis(im_gray)
                print("PCA finished")
                print("Now i'm using K-means to classify the subimages")
                labelled_image = cl.classification(im_gray, pca)
                print("K-means finished")
                plt.imsave("labelled_images/labelled_"+value,labelled_image)
                print(colors.green|"Saved the labelled image in dir 'labelled_images/'")

@AnomalousDiffusion.subcommand("fronts")
class Fronts(cli.Application):
    "Tracks the borders of the cells and saves the coordinates"
    all = cli.Flag(["all", "every image"], help = "If given, I will save the fronts of all the images in the current directory")
    def main(self, value : str = ""):
        if not os.path.exists("fronts"):
            os.makedirs("fronts")

        if(value == ""):
            if (self.all):
                cont = 0
                for value in list(os.listdir(".")):
                    if str(value).endswith(".png"):
                        fr.fronts(value,"fronts" + value + ".txt")
                        cont = cont + 1
                print(colors.green|"Saved the fronts of the images in dir 'fronts/'")
            else:
                print(colors.red|"image not given")
        else:
            if(value not in list(os.listdir(path))):
                print(colors.red|"this image does not exists")
            else:
                print("image taken")
                fr.fronts(value,"fronts/fronts" + value + ".txt")
                print(colors.green|"Saved the fronts of the image in dir 'fronts/'")




if __name__ == "__main__":
    AnomalousDiffusion.run()
