from plumbum import cli, colors
import os
from plumbum import local
#from PIL import Image as imreader
#from plumbum.cli.image import Image as imdisplay
#from plumbum.cli.terminal import Progress

############################################
import cv2
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import os

#####################################################
import sys
sys.path.insert(0, '/home/riccardo/git/AnomalousDiffusion')
import classification as cl
import fronts as fr
import analysis as an
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

@AnomalousDiffusion.subcommand("read")
class Read(cli.Application):
    "Reads the nd2 file and create a new folder with the images in png format"
    #all = cli.Flag(["all", "every image"], help = "If given, I will save the fronts of all the images in the current directory")
    #s = cli.Flag(["s", "save"], help = "If given, I will save the image with the borders in the current directory")
    def main( self,n_images : int  , value : str = ""):
        cl.create_set(n_images, path = value)
        print(colors.green|"Saved the images in dir 'images/")

@AnomalousDiffusion.subcommand("modify")
class Modify(cli.Application):
    "Modify an image with histogram equalization and saves it in a new folder with the images in png format"
    #all = cli.Flag(["all", "every image"], help = "If given, I will save the fronts of all the images in the current directory")
    #s = cli.Flag(["s", "save"], help = "If given, I will save the image with the borders in the current directory")
    def main( self,n_images : int = 1, value : str = ""):
        cl.create_modified_images(path = value,n_images = n_images)
        print(colors.green|"Saved the images in dir 'modified_images/")


@AnomalousDiffusion.subcommand("label")
class Label(cli.Application):
    "Saves the binary image using pca and K-means algorithms"
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
                        pca = cl.Principal_components_analysis(im_gray,window_sizeX=10,window_sizeY=10)
                        labelled_image = cl.classification(im_gray, pca,window_sizeX=10,window_sizeY=10)
                        plt.imsave("labelled_images/labelled_"+value,labelled_image)
                        cont = cont + 1
                print(colors.green|"Saved the binary images in dir 'labelled_images/'")
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
                pca = cl.Principal_components_analysis(im_gray,window_sizeX=10,window_sizeY=10)
                print("PCA finished")
                print("Now i'm using K-means to classify the subimages")
                labelled_image = cl.classification(im_gray, pca,window_sizeX=10,window_sizeY=10)
                print("K-means finished")
                plt.imsave("labelled_images/labelled_"+value,labelled_image)
                print(colors.green|"Saved the labelled image in dir 'labelled_images/'")

@AnomalousDiffusion.subcommand("fronts")
class Fronts(cli.Application):
    "Tracks the longest borders in the images and saves the coordinates in a txt file"
    all = cli.Flag(["all", "every image"], help = "If given, I will save the fronts of all the images in the current directory")
    s = cli.Flag(["s", "save"], help = "If given, I will save the image with the borders in the current directory")
    def main(self, value : str = ""):
        if not os.path.exists("fronts"):
            os.makedirs("fronts")

        if(value == ""):
            if (self.all):
                cont = 0
                for value in list(os.listdir(".")):
                    if str(value).endswith(".png"):
                        coord, im = fr.fronts(value)
                        np.savetxt("fronts/fronts_"+ value + ".txt", coord,fmt = '%d', delimiter=' ')
                        cont = cont + 1
                print(colors.green|"Saved the fronts of the images in dir 'fronts/'")
            else:
                print(colors.red|"image not given")
        else:
            if(value not in list(os.listdir(path))):
                print(colors.red|"this image does not exists")
            else:
                print("image taken")
                coord, im = fr.fronts(value)
                if (self.s):
                    plt.imsave("front_"+ value, im)
                np.savetxt("fronts/fronts_"+ value +".txt", coord,fmt = '%d', delimiter=' ')
                print(colors.green|"Saved the fronts of the image in dir 'fronts/'")

@AnomalousDiffusion.subcommand("fast")
class Fast(cli.Application):
    "Tracks the longest borders in the images and saves the coordinates in a txt file"
    all = cli.Flag(["all", "every image"], help = "If given, I will save the fronts of all the images in the current directory")
    s = cli.Flag(["s", "save"], help = "If given, I will save the image with the borders in the current directory")
    def main(self, value : str = ""):
        if not os.path.exists("fronts"):
            os.makedirs("fronts")

        if(value == ""):
            if (self.all):
                for value in list(os.listdir(".")):
                    if str(value).endswith(".png"):
                        fr.fast_fronts(value)
                print(colors.green|"Saved the fronts of the images in dir 'fronts/'")
            else:
                print(colors.red|"image not given")
        else:
            if(value not in list(os.listdir(path))):
                print(colors.red|"this image does not exists")
            else:
                print("image taken")
                fr.fast_fronts(value)
                print(colors.green|"Saved the fronts of the images in dir 'fronts/'")


@AnomalousDiffusion.subcommand("divide")
class Divide(cli.Application):
    "Divides the front in two piecies, one left and one right and save them in two txt files"
    all = cli.Flag(["all", "every text file"], help = "If given, I will save the dx and sx fronts of all the images in the current directory")
    def main(self, value : str = ""):
        if not os.path.exists("divided_fronts"):
            os.makedirs("divided_fronts")

        if(value == ""):
            if (self.all):
                cont = 0
                for value in list(os.listdir(".")):
                    if str(value).endswith(".txt"):
                        df = pd.DataFrame(pd.read_csv(value , delimiter=' '))
                        df.columns = ["x","y"]
                        sx, dx = fr.divide(df)
                        np.savetxt("divided_fronts/"+ value+"dx.txt" , dx.values, fmt='%d')
                        np.savetxt("divided_fronts/"+ value+"sx.txt" , sx.values, fmt='%d')
                        cont = cont + 1
                print(colors.green|"Divided the fronts of the images in dir 'divided_fronts/'")
            else:
                print(colors.red|"file not given")
        else:
            if(value not in list(os.listdir(path))):
                print(colors.red|"this file does not exists")
            else:
                print("image taken")
                df = pd.DataFrame(pd.read_csv(value , delimiter=' '))
                df.columns = ["x","y"]
                sx, dx = fr.divide(df)
                np.savetxt("divided_fronts/"+ value +"dx.txt" , dx.values, fmt='%d')
                np.savetxt("divided_fronts/"+ value +"sx.txt", sx.values, fmt='%d')
                print(colors.green|"Divided the fronts and saved in dir 'divided_fronts/'")

@AnomalousDiffusion.subcommand("area")
class Area(cli.Application):
    "Computes the area of the polygon formed by the two borders of the cells"
    all = cli.Flag(["all", "every text file"], help = "If given, I will save area of all the images in a txt file")
    def main(self, value : str = ""):

        if(value == ""):
            if (self.all):
                areas = []
                for value in list(os.listdir(".")):
                    if str(value).endswith("png.txt"):
                        pol, area = an.area(value)
                        areas.append(area)

                with open('Areas.txt', 'w+') as f:
                    for item in areas:
                        f.write("%s\n" % item)
                print(colors.green|"areas saved in file 'Areas.txt'")
            else:
                print(colors.red|"file not given")
        else:
            if(value not in list(os.listdir(path))):
                print(colors.red|"this file does not exists")
            else:
                pol, area = fr.area(value)
                print("The area of the polygon is "+ str(area))

@AnomalousDiffusion.subcommand("error")
class Error(cli.Application):
    "Computes the error between two areas between the fronts"
    def main(self, value : str = ""):

        if(value == ""):
                errors = []
                i = 0
                areas = pd.DataFrame(pd.read_csv("Areas.txt"))
                areas_hand = pd.DataFrame(pd.read_csv("Areas_hand.txt"))
                print(colors.green|"errors saved in file 'error.txt'")




if __name__ == "__main__":
    AnomalousDiffusion.run()
