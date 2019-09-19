from plumbum import cli, colors
import os
from plumbum import local
from nd2reader import ND2Reader

############################################
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from shapely.geometry import Polygon

#####################################################
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

    def main( self,n_images : int  , value : str = "",field : int = 1):
        if self.all:
            for direct in os.listdir("."):
                #Rough way to detect only the directories of the experiments
                if str(direct).endswith("9") or str(direct).endswith("8"):
                    print("reading images in directory: "+ str(direct))
                    ## The files in the same directories have the same images so i take the last one first
                    for value in ["003.nd2","002.nd2","001.nd2"]:
                        try:
                            #saves the images taken from the nd2 file
                            cl.create_set(n_images, field ,path = direct + "/" + value)
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
                        #idex for the images name
                        cont = 0
                        for value in list(os.listdir(".")):
                            #modifies every png image in the directory and saves them in png format
                            if str(value).endswith(".png"):
                                cl.create_modified_images(path = value)
                        print(colors.green|"Saved the modified images in dir 'modified_images/'")
                else:
                    try:
                        cl.create_modified_images(path = value)
                        print(colors.green|"Saved the image in dir 'modified_images/")
                    except:
                        print(colors.red|"this image does not exists")

@AnomalousDiffusion.subcommand("label")
class Label(cli.Application):
    "Saves the binary image using pca and Gaussian-mixture algorithms"
    all = cli.Flag(["all", "every image"], help = "If given, I will label all the images in the current directory")
    def main(self, value : str = ""):
        if not os.path.exists("labelled_images"):
            os.makedirs("labelled_images")

        if(value == ""):
            if (self.all):
                #idex for the images name
                cont = 0
                for value in list(os.listdir(".")):
                    if str(value).endswith(".png"):
                        #reading the image
                        test_image =  cv2.imread(value)
                        #make it gray
                        im_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

                        print("analyzing image " +str(cont+1)+"/"+str(len(os.listdir("."))))
                        #labels the images using PCA and GaussianMixture algorithms
                        pca = cl.Principal_components_analysis(im_gray,window_sizeX=10,window_sizeY=10)
                        labelled_image = cl.classification(im_gray, pca,window_sizeX=10,window_sizeY=10)
                        #saves them
                        plt.imsave("labelled_images/labelled_"+value,labelled_image)
                        cont = cont + 1
                print(colors.green|"Saved the binary images in dir 'labelled_images/'")
            else:
                print(colors.red|"image not given")
        else:
            if(value not in list(os.listdir(path))):
                print(colors.red|"this image does not exists")
            else:
                #reading the image
                test_image =  cv2.imread(value)
                #make it gray
                im_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
                print("image taken")
                print("i'm doing PCA on the LBP image")
                #labels the images using PCA and GaussianMixture algorithms
                pca = cl.Principal_components_analysis(im_gray,window_sizeX=10,window_sizeY=10)
                print("PCA finished")
                print("Now i'm using Gaussian-mixture to classify the subimages")
                labelled_image = cl.classification(im_gray, pca,window_sizeX=10,window_sizeY=10)
                print("finished")
                #saves them
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
                #index for the images name
                cont = 0
                for value in list(os.listdir(".")):
                    if str(value).endswith(".png"):
                        #reading the image
                        test_image =  cv2.imread(value)
                        #make it gray
                        im_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
                        #finds the coordinates of the longest border in the image
                        coord, im = fr.fronts(im_gray)
                        #saves it
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
                #reading the image
                test_image =  cv2.imread(value)
                #make it gray
                im_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
                #finds the coordinates of the longest border in the image
                coord, im = fr.fronts(im_gray)
                if (self.s):
                    plt.imsave("front_"+ value, im)
                np.savetxt("fronts/fronts_"+ value +".txt", coord,fmt = '%d', delimiter=' ')
                print(colors.green|"Saved the fronts of the image in dir 'fronts/'")

@AnomalousDiffusion.subcommand("divide")
class Divide(cli.Application):
    "Divides the front in two piecies, one left and one right and save them in two txt files"
    all = cli.Flag(["all", "every text file"], help = "If given, I will save the dx and sx fronts of all the images in the current directory")
    def main(self, value : str = ""):
        if not os.path.exists("divided_fronts"):
            os.makedirs("divided_fronts")

        if(value == ""):
            if (self.all):

                for value in list(os.listdir(".")):
                    if str(value).endswith(".txt"):
                        #reading the file with the coordinates of the longest border
                        df = pd.DataFrame(pd.read_csv(value , delimiter=' '))
                        df.columns = ["x","y"]
                        #divide the longest border in two, the left one and the right one
                        sx, dx = fr.divide(df)
                        np.savetxt("divided_fronts/"+ value+"dx.txt" , dx.values, fmt='%d')
                        np.savetxt("divided_fronts/"+ value+"sx.txt" , sx.values, fmt='%d')

                print(colors.green|"Divided the fronts of the images in dir 'divided_fronts/'")
            else:
                print(colors.red|"file not given")
        else:
            if(value not in list(os.listdir(path))):
                print(colors.red|"this file does not exists")
            else:
                print("image taken")
                #reading the file with the coordinates of the longest border
                df = pd.DataFrame(pd.read_csv(value , delimiter=' '))
                df.columns = ["x","y"]
                #divide the longest border in two, the left one and the right one
                sx, dx = fr.divide(df)
                np.savetxt("divided_fronts/"+ value +"dx.txt" , dx.values, fmt='%d')
                np.savetxt("divided_fronts/"+ value +"sx.txt", sx.values, fmt='%d')
                print(colors.green|"Divided the fronts and saved in dir 'divided_fronts/'")


##This command takes the images from the directories 'images/' while command 'faster' Takes
## the images directly from the nd2 file
@AnomalousDiffusion.subcommand("fast")
class Fast(cli.Application):
    "Tracks the longest borders in the images and saves the coordinates in a txt file"
    all = cli.Flag(["all", "every image"], help = "If given, I will save the fronts of all the images in the current directory")
    s = cli.Flag(["s", "save"], help = "If given, I will save the image with the borders in the current directory")
    def main(self, value : str = ""):
        if(value == ""):
            if (self.all):
                #make the dataframe which will be saved with all the coordinates for all the experiment
                df = pd.DataFrame(columns = ["i","x","y","side","frame","experiment","field"])
                df.to_csv("coordinates.txt",index = False,header = df.columns, sep = " ")
                #two directories for the two experiments
                for outdir in ["EF/","Sham/"]:
                    #index for the field of view
                    cont = 0
                    #number of frames to analyze
                    frames = 100
                    for path in ["22-11-18","14-02-19","21-11-18","22-02-19","21-02-19","16-11-18","20-11-18","07-02-19","08-02-19","11-02-19"]:
                        try:
                            for frame in range(frames):
                                #reading the image from the directory
                                image =  cv2.imread(outdir + path + "/images/" +str(frame)+".png")
                                #make it grays
                                im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                                #modify the image with adaptive histogram equalization
                                new = cl.adaptive_contrast_enhancement(im,(100,100))
                                #blurring the image with gaussian filter
                                blur = cv2.GaussianBlur(new,(5,5),0)
                                #Otsu Threshold to binarize it
                                ret3,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                                #finds the two longest borders inside the image
                                dfs, b, c = fr.fast_fronts(thresh,length_struct=5,iterations=1)

                                for i in [0,1]:
                                    try:
                                        #interpolation of the two borders
                                        df = an.necklace_points(dfs[i], N = 1000)

                                        #making the dataframe which will be saved in txt file
                                        if i == 0:
                                            df["side"] = "dx"
                                        else: df["side"] = "sx"
                                        df["frame"] = frame
                                        df["experiment"] = outdir
                                        df["field"] = cont

                                        #saving the dataframe with the coordinates
                                        df.to_csv("coordinates.txt", header = None , sep = " ", mode= "a")
                                    except: pass
                                #status bar
                                print("field "+ path+" ["+"#"*int(frame/frames*20)+"-"*int(20-int(frame/frames*20))+"] "+str(int(frame/frames*100))+"% ", end="\r")
                            cont = cont + 1
                            print("field "+ path+" ["+"#"*20+"] 100%")

                        except:
                            pass
            else:
                print(colors.red|"image not given")
        else:
            if(value not in list(os.listdir(path))):
                print(colors.red|"this image does not exists")
            else:
                print("image taken")
                fr.fast_fronts(value)
                print(colors.green|"Saved the fronts of the images in dir 'fronts/'")



##IMPORTANT this command is different form 'fast' because takes the images directly from the nd2 file
## while 'fast' takes the saved images from the directories 'imges/'
@AnomalousDiffusion.subcommand("faster")
class Faster(cli.Application):
    "Tracks the longest borders in the images and may save the coordinates in a txt file"
    all = cli.Flag(["all", "every image"], help = "If given, I will save the fronts of all the images in the current directory")
    s = cli.Flag(["s", "save"], help = "If given, I will save the image with the borders in the current directory")
    def main(self, value : str = "", fields : int = 8):
        if(value == ""):
            if (self.all):
                for outdir in os.listdir("."):
                    #Rough way to detect only the directories of the experiments
                    if str(outdir).endswith("9") or str(outdir).endswith("8"):
                        #index for the field of view
                        cont = 0
                        ## The files in the same directories have the same images so i take the last one first
                        for value in ["003.nd2","002.nd2","001.nd2"]:
                            try:
                                with ND2Reader(outdir + "/" + value) as images:
                                    print("directory " + outdir)
                                    #iterations of the images in the nd2 file
                                    images.iter_axes = "vt"
                                    fields = images.sizes["v"]
                                    frames = images.sizes["t"]
                                    #making the dataframe which contain all the coordinates for all the experiments will be saved in txt file
                                    df = pd.DataFrame(columns = ["i","x","y","side","frame","field"])
                                    df.to_csv(outdir + "/" + outdir + ".txt",index = False,header = df.columns, sep = " ")
                                    for field in range(fields):
                                        for frame in range(frames):
                                            #histogram matching maybe is not necessary

                                            #im0 = cv2.convertScaleAbs(images[0],alpha=(255.0/65535.0))
                                            #im = cv2.convertScaleAbs(images[frame + frames*(field)],alpha=(255.0/65535.0))
                                            # im0 = np.asarray(im0)
                                            # im = np.asarray(im)
                                            # ct = fr.cdf(im0)
                                            # c = fr.cdf(im)
                                            # gray = fr.hist_matching(c,ct,im)
                                            #dfs, _ , _ = fr.fast_fronts(im,size = 50, length_struct = 1,iterations = 1)

                                            #converts the image in uint8
                                            thresh = cv2.convertScaleAbs(images[frame + frames*(field)],alpha=(255.0/65535.0))
                                            max = np.max(thresh)
                                            thresh[0:2,] = max
                                            thresh[len(thresh)-2:len(thresh)-1,:] = max
                                            #add two white bands in the image to make sure opencv recognisez the right fronts
                                            thresh[:,0:400] = max
                                            thresh[:,-400:] = max
                                            #modifying the image with adaptive histogram equalization
                                            new = cl.adaptive_contrast_enhancement(thresh,(100,100))
                                            #blurring the image
                                            blur = cv2.GaussianBlur(new,(5,5),0)
                                            #binarize the image with Otsu thresholding
                                            ret3,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

                                            #finding the two longest borders in the image

                                            dfs, b, c = fr.fast_fronts(thresh,length_struct=10,iterations=1)
                                            for i in [0,1]:
                                                try:
                                                    #interpolation of the coordinates to have always the same number of points
                                                    df = an.necklace_points(dfs[i])
                                                    #making the dataframe with all the coordinates
                                                    if i == 0:
                                                        df["side"] = "dx"
                                                    else: df["side"] = "sx"
                                                    df["frame"] = frame
                                                    df["field"] = field
                                                    df.to_csv(outdir + "/" + outdir +".txt", header = None , sep = " ", mode= "a")
                                                except: pass
                                            #status bar
                                            print("field "+str(cont)+" ["+"#"*int(frame/frames*20)+"-"*int(20-int(frame/frames*20))+"] "+str(int(frame/frames*100))+"% ", end="\r")
                                        print("field "+str(cont)+" ["+"#"*20+"] 100%")
                                        cont = cont + 1
                                break
                            except:
                                pass
                        print(colors.green|"Saved the fronts in the file " + outdir + ".txt")
            else:
                #index for the field of view
                cont = 0
                outdir = os.getcwd()
                ## The files in the same directories have the same images so i take the last one first
                for value in ["003.nd2","002.nd2","001.nd2"]:
                    try:
                        with ND2Reader(value) as images:
                            #iteration of the images in the nd2 file
                            images.iter_axes = "vt"
                            frames = images.sizes["t"]
                            #making the dataframe which contains the coordinates of the borders for all the experiment
                            df = pd.DataFrame(columns = ["i","x","y","side","frame","field"])
                            df.to_csv("coordinates.txt",index = False,header = df.columns, sep = " ")
                            for field in range(fields):
                                for frame in range(frames):
                                    #convert the image in type uint8
                                    thresh = cv2.convertScaleAbs(images[frame + frames*(field)],alpha=(255.0/65535.0))
                                    max = np.max(thresh)
                                    thresh[0:2,] = max
                                    thresh[len(thresh)-2:len(thresh)-1,:] = max
                                    #adding two white bands on the left and on the right to make sure opencv recognizes the right borders
                                    thresh[:,0:400] = max
                                    thresh[:,-400:] = max
                                    #modifying the image with adaptive histogram equalization
                                    new = cl.adaptive_contrast_enhancement(thresh,(100,100))
                                    #blurring the image with Gaussian filter
                                    blur = cv2.GaussianBlur(new,(9,9),1)
                                    #binarize the image with Otsu thresholding
                                    ret3,thresh = cv2.threshold(new,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
                                    #finds the two longest borders inside the image
                                    dfs,b,c = fr.fast_fronts(thresh,length_struct=7,iterations=2)
                                    for i in [0,1]:
                                        try:
                                            #interpolation of the borders to have always the same number of points
                                            df = an.necklace_points(dfs[i])
                                            #making the dataframe which will contain all the coordinates for all the experiments
                                            if i == 0:
                                                df["side"] = "dx"
                                            else: df["side"] = "sx"
                                            df["frame"] = frame
                                            df["field"] = field
                                            df.to_csv("coordinates.txt", header = None , sep = " ", mode= "a")
                                        except: pass
                                    #status bar
                                    print("field "+str(cont)+" ["+"#"*int(frame/frames*20)+"-"*int(20-int(frame/frames*20))+"] "+str(int(frame/frames*100))+"% ", end="\r")
                                print("field "+str(cont)+" ["+"#"*20+"] 100%")
                                cont = cont + 1
                        break
                    except:
                        pass
                print(colors.green|"Saved the fronts in the file " + outdir + ".txt")
        else:
            if(value not in list(os.listdir(path))):
                print(colors.red|"this image does not exists")
            else:
                print("image taken")
                fr.fast_fronts(value)
                print(colors.green|"Saved the fronts of the images in dir 'fronts/'")



@AnomalousDiffusion.subcommand("areas")
class Area(cli.Application):
    "Computes the areas for all the directories with the images!"

    def main(self):
        df = pd.DataFrame()
        j = 0
        d = []
        for direct in os.listdir("."):
            #Rough way to detect only the directories of the experiments
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
                            #reading the coordinates from the txt files
                            filesx = direct+ "/images/fronts/"+ str(i)+".png_sx.txt"
                            filedx = direct+ "/images/fronts/"+ str(i)+".png_dx.txt"
                            #make it dataframes
                            sx = pd.read_csv(filesx, sep = " ")
                            dx = pd.read_csv(filedx, sep = " ")
                            #computes the area between the two fronts
                            pol, area = an.area(sx,dx)
                            areas.append(area)
                        except:pass
                    #normalize the area with the area of the first frame
                    areas = np.array(areas)/areas[0]
                    areas = areas[areas<1.2]
                    #areas = areas[areas>0.1]
                    areas = pd.Series(areas)
                    df[j] = areas
                    j = j+1

            df.columns = d
            #making the plot of the areas
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
        MSD = []
        mean = []
        for direct in os.listdir("."):
            d = []
            #Rough way to detect only the directories of the experiments
            if str(direct).endswith("9") or str(direct).endswith("8"):
                if not os.path.exists(direct + "/images/fronts"):
                    print(colors.yellow|"fronts/ doesn't exist in directory " +str(direct))
                    pass
                else:
                    print("reading files in directory: "+ str(direct))
                    x = pd.DataFrame()
                    d.append(direct)
                    for i in range(len(os.listdir(direct))//2):
                        #reading the x coordinates from the txt files
                        s = pd.read_csv(direct + "/images/fronts/"+str(i)+".png_sx.txt", sep = " ")
                        s.columns = [0,1]
                        x[i] = s[0]
                    #computes the MSD of the dataframe with the x coordinates
                    msd = an.MSD(x)
                    msd = pd.DataFrame(msd)
                    #saving it
                    msd.to_csv(direct + "/msd.txt", sep=' ')
                    print(colors.green|"msd saved in files 'msd.txt'")
                    MSD.append(np.mean(msd))
                    plt.plot(np.mean(msd),label = direct)
                    #plt.plot(np.mean(msdy),label = direct)
                    plt.legend()
                    mean.append(np.mean(msd))

        #making the plot with the MSD for all the experiments
        plt.savefig("MSD.png")
        plt.figure()
        plt.plot(np.mean(pd.DataFrame(mean)))
        plt.savefig("mean.png")
        MSD = pd.DataFrame(MSD)
        MSD.to_csv("MSD.txt", sep=' ')




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
            #Rough way to detect only the directories of the experiments
            if str(direct).endswith("9") or str(direct).endswith("8"):
                if not os.path.exists(direct + "/msd.txt"):
                    print(colors.yellow|"file 'msd.txt' doesn't exist in directory " +str(direct))
                    pass
                else:
                    path = direct+ "/msd.txt"
                    #making the dataframe with the msd from the txt file
                    df = pd.DataFrame(pd.read_csv(path,sep = " "))
                    #del df[0]
                    msd = np.mean(df)
                    #fit the msd
                    popt, popcv = an.fit(msd)
                    D.append(popt[0])
                    alpha.append(popt[1])
                    cont += 1

        D = pd.Series(D)
        alpha = pd.Series(alpha)
        #making the dataframe with the parameters found with the fit
        fit = pd.DataFrame(columns = ["D","alpha"])
        fit["D"] = D
        fit["alpha"] = alpha

        fit.to_csv("fit.txt",sep = " ")
        if cont > 0 :
            print(colors.green | "fit parameters saved in file fit.txt")
        else : print(colors.red| "probably the file fit.txt is empty")


if __name__ == "__main__":
    AnomalousDiffusion.run()
