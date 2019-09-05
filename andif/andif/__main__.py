from plumbum import cli, colors
import os
from plumbum import local
from nd2reader import ND2Reader

############################################
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from shapely.geometry import Polygon

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
    all = cli.Flag(["all", "every image"], help = "If given, I will save the fronts of all the images in the current directory")

    def main( self,n_images : int  , value : str = ""):
        if self.all:
            for direct in os.listdir("."):
                if str(direct).endswith("9") or str(direct).endswith("8"):
                    print("reading images in directory: "+ str(direct))
                    ## The files in the directories have the same images so i take the last one first
                    for value in ["003.nd2","002.nd2","001.nd2"]:
                        try:
                            cl.create_set(n_images, path = direct + "/" + value)
                            break
                        except: pass
        print(colors.green|"Saved the images in dir 'images/")

@AnomalousDiffusion.subcommand("modify")
class Modify(cli.Application):
    "Modify an image with histogram equalization and saves it in a new folder with the images in png format"
    all = cli.Flag(["all", "every image"], help = "If given, I will save the modified images in the directory modified_images/")

    def main( self, value : str = ""):

                if(value == ""):
                    if (self.all):
                        cont = 0
                        for value in list(os.listdir(".")):
                            if str(value).endswith(".png"):
                                cl.create_modified_images(path = value)
                        print(colors.green|"Saved the binary images in dir 'labelled_images/'")
                else:
                        cl.create_modified_images(path = value)
                        print(colors.green|"Saved the image in dir 'modified_images/")

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
        if(value == ""):
            if (self.all):
                for direct in os.listdir("."):
                    if str(direct).endswith("9") or str(direct).endswith("8"):
                        cont = 0
                        outdir = direct + "/images/fronts/"
                        if not os.path.exists(outdir):
                            os.makedirs(outdir)
                        for value in list(os.listdir(direct+"/images/")):
                            if str(direct + "/images/" + value).endswith(".png"):
                                fr.fast_fronts(direct + "/images/" + value,outdir = outdir,save = True,iterations = 3)
                                print("image "+str(cont))
                                cont = cont  + 1
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

@AnomalousDiffusion.subcommand("areas")
class Area(cli.Application):
    "Computes the areas for all the directories with the images!"

    def main(self):
        df = pd.DataFrame()
        j = 0
        d = []
        for direct in os.listdir("."):
            if str(direct).endswith("9") or str(direct).endswith("8"):
                if not os.path.exists(direct + "/images"):
                    print(colors.yellow|"images/ doesn't exist in directory " +str(direct))
                    pass
                else:
                    print("reading images in directory: "+ str(direct))
                    d.append(direct)
                    areas = []

                    for i in range(0,300):
                        try:


                            filesx = direct+ "/images/fronts/"+ str(i)+".png_sx.txt"
                            filedx = direct+ "/images/fronts/"+ str(i)+".png_dx.txt"
                            pol, area = an.area(filesx,filedx)
                            areas.append(area)
                        except:pass
                    areas = np.array(areas)/areas[0]
                    areas = areas[areas<1.2]
                    #areas = areas[areas>0.1]
                    areas = pd.Series(areas)
                    df[j] = areas
                    j = j+1

            df.columns = d
            plt.figure(dpi = 200)
            for i in list(df.columns):
                plt.plot(df[i])
                plt.legend()
            plt.savefig("aree.png")

            df.to_csv("aree.txt", sep=' ')
        if j > 0:
            print(colors.green|"areas saved in file 'areas.txt'")
        else:
            print(colors.red|"areas don't saved. There was a problem, maybe wrong directory")



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

@AnomalousDiffusion.subcommand("msd")
class MSD(cli.Application):
    "Computes the Mean Squared Displacement (MSD) for all the directories with the images!"

    def main(self):
        MSDX = []
        MSDY = []
        mean = []
        for direct in os.listdir("."):
            d = []

            if str(direct).endswith("9") or str(direct).endswith("8"):
                if not os.path.exists(direct + "/images/fronts"):
                    print(colors.yellow|"fronts/ doesn't exist in directory " +str(direct))
                    pass
                else:
                    print("reading files in directory: "+ str(direct))
                    d.append(direct)

                    msdx , msdy = an.MSD(direct+"/images/fronts/", nframes = 97,delimiter = " ")
                    msdx = pd.DataFrame(msdx)
                    msdy = pd.DataFrame(msdy)
                    msdx.to_csv(direct + "/msdx.txt", sep=' ')
                    msdy.to_csv(direct + "/msdy.txt", sep=' ')
                    print(colors.green|"msd saved in files 'msdx.txt' and 'msdy.txt'")
                    MSDX.append(np.mean(msdx))
                    plt.plot(np.mean(msdx),label = direct)
                    #plt.plot(np.mean(msdy),label = direct)
                    plt.legend()
                    mean.append(np.mean(msdx))

        plt.savefig("MSD.png")
        plt.figure()
        plt.plot(np.mean(pd.DataFrame(mean)))
        plt.savefig("mean.png")
        MSDX = pd.DataFrame(MSDX)
        MSDX.to_csv("MSDX.txt", sep=' ')




@AnomalousDiffusion.subcommand("fit")
class FIT(cli.Application):
    """Computes the fit parameters (D,a) for the Mean Squared Displacement for all the directories and saves
    in a txt file
    """

    def main(self):
        D = []
        alpha = []
        cont = 0
        for direct in os.listdir("."):

            if str(direct).endswith("9") or str(direct).endswith("8"):
                if not os.path.exists(direct + "/msd.txt"):
                    print(colors.yellow|"file 'msd.txt' doesn't exist in directory " +str(direct))
                    pass
                else:
                    path = direct+ "/msd.txt"
                    df = pd.DataFrame(pd.read_csv(path,sep = " "))
                    #del df[0]
                    msd = np.mean(df)
                    popt, popcv = an.fit(msd)
                    D.append(popt[0])
                    alpha.append(popt[1])
                    cont += 1

        D = pd.Series(D)
        alpha = pd.Series(alpha)
        fit = pd.DataFrame(columns = ["D","alpha"])
        fit["D"] = D
        fit["alpha"] = alpha

        fit.to_csv("fit.txt",sep = " ")
        if cont > 0 :
            print(colors.green | "fit parameters saved in file fit.txt")
        else : print(colors.red| "probably the file fit.txt is empty")


@AnomalousDiffusion.subcommand("faster")
class Faster(cli.Application):
    "Tracks the longest borders in the images and saves the coordinates in a txt file"
    all = cli.Flag(["all", "every image"], help = "If given, I will save the fronts of all the images in the current directory")
    s = cli.Flag(["s", "save"], help = "If given, I will save the image with the borders in the current directory")
    def main(self, value : str = "", fields : int = 1):
        if(value == ""):
            if (self.all):
                for direct in os.listdir("."):
                    if str(direct).endswith("9") or str(direct).endswith("8"):
                        cont = 0
                        outdir = direct + "/fronts/"
                        if not os.path.exists(outdir):
                            os.makedirs(outdir)
                        for value in ["003.nd2","002.nd2","001.nd2"]:
                            try:
                                with ND2Reader(direct + "/" + value) as images:
                                    images.iter_axes = "vt"
                                    fields = images.sizes["v"]
                                    frames = images.sizes["t"]
                                    for i in range(1,fields+1):
                                        for j in range(frames- 90):
                                            im0 = np.matrix(images[0]).astype("uint8")
                                            im = np.matrix(images[i*j]).astype("uint8")
                                            im0 = np.asarray(im0)
                                            im = np.asarray(im)
                                            ct = fr.cdf(im0)
                                            c = fr.cdf(im)
                                            gray = fr.hist_matching(c,ct,im)
                                            fr.fast_fronts(gray,size = 5, threshold = 200,outdir = outdir,save = True, fname = str(i)+"-"+str(j), length_struct = 1,iterations = 1)
                                            print("field "+str(cont)+" ["+"#"*int(j/10)+"-"*int(20-int(j/10))+"] "+str(j/2)+"% ", end="\r")
                                        print("field "+str(cont)+" ["+"#"*20+"] 100%")
                                        cont = cont + 1
                                break
                            except:
                                pass
                        print(colors.green|"Saved the fronts of the images in dir 'fronts/'")
            else:
                cont = 0
                outdir = "fronts/"
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                for value in ["003.nd2","002.nd2","001.nd2"]:
                    try:
                        with ND2Reader(value) as images:
                            images.iter_axes = "vt"
                            #fields = images.sizes["v"]
                            frames = images.sizes["t"]
                            for i in range(1,fields+1):
                                if not os.path.exists(outdir + str(i)):
                                    os.makedirs(outdir + str(i))
                                #don't need all the frames
                                xs = pd.DataFrame()
                                ys = pd.DataFrame()
                                xr = pd.DataFrame()
                                yr = pd.DataFrame()
                                for j in range(frames - 90):
                                    im0 = np.matrix(images[i*j]).astype("uint8")
                                    im = np.matrix(images[i*j]).astype("uint8")
                                    im0 = np.asarray(im0)
                                    im = np.asarray(im)
                                    ct = fr.cdf(im0)
                                    c = fr.cdf(im)
                                    gray = fr.hist_matching(ct,c,im)
                                    dfs, _ , _ = fr.fast_fronts(gray,size = 5, threshold = 127,outdir = outdir,save = False, length_struct = 1,iterations = 1)
                                    xr.insert(j,str(j),dfs[0]["x"],True)
                                    xs.insert(j,str(j),dfs[1]["x"],True)
                                    yr.insert(j,str(j),dfs[0]["y"],True)
                                    ys.insert(j,str(j),dfs[1]["y"],True)
                                    print("field "+str(cont)+" ["+"#"*int(j/10)+"-"*int(20-int(j/10))+"] "+str(j/2)+"% ", end="\r")
                                print("field "+str(cont)+" ["+"#"*20+"] 100%")
                                cont = cont + 1
                                xs.to_csv(outdir + str(i) + "/xs.txt", sep = " ")
                                xr.to_csv(outdir + str(i) + "/xr.txt", sep = " ")
                                ys.to_csv(outdir + str(i) + "/ys.txt", sep = " ")
                                yr.to_csv(outdir + str(i) + "/yr.txt", sep = " ")
                        break
                    except:
                        pass
                print(colors.green|"Saved the fronts of the images in dir 'fronts/'")
        else:
            if(value not in list(os.listdir(path))):
                print(colors.red|"this image does not exists")
            else:
                print("image taken")
                fr.fast_fronts(value)
                print(colors.green|"Saved the fronts of the images in dir 'fronts/'")


if __name__ == "__main__":
    AnomalousDiffusion.run()
