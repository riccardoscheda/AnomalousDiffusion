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
    all = cli.Flag(["all", "every image"], help = "If given, I will read the first nd2 file in the direcories and save all the images in the directory 'images/'")

    def read(n_images, value, direct, field ):
        """
        Reads the images from nd2 file

        Parameters:
        ----------------
        n_images : integer for the number of images to read from nd2
        value : string with the nd2 file name
        direct : string for the name of the directory
        field : integer for the number of fields of view
        """
        #saves the images taken from the nd2 file
        cl.create_set(n_images, field ,path = direct + "/" + value)
        print(colors.green|"Saved the images in dir '"+ direct+"/images/")

    def main( self, value : str, n_images : int = 100   , field : int = 1):

        #if not given a nd2 file, prints error
        if value.endswith(".nd2"):
            if self.all:
                #searching only the directories in the current directory
                dirs = [x[0] for x in os.walk(".")]
                #try to read the nd2 file given; if doesn't find it, passes
                for direct in dirs:
                    try:
                        Read.read(n_images, value, direct, field)
                    except : pass
            else:
                #if not Flag, search only in the current directory
                try:
                    Read.read(n_images, value, ".", field)
                except:
                    print(colors.red|"File not found")

        else: print(colors.red|"Wrong name, i need nd2 format file")




@AnomalousDiffusion.subcommand("modify")
class Modify(cli.Application):
    "Modify an image with histogram equalization and saves it in a new folder with the images in png format"

    def main( self, value : str ):
                #if value is . will modify all the images in the current directory
                if(value == "."):
                    for image in list(os.listdir(value)):
                        #modifies every png image in the directory and saves them in png format
                        if str(image).endswith(".png") or str(image).endswith(".jpg"):
                            cl.create_modified_images(path = image)
                    print(colors.green|"Saved the modified images in dir 'modified_images/'")
                else:
                    try:
                        #modifies every png image in the directory and saves them in png format
                        cl.create_modified_images(path = value)
                        print(colors.green|"Saved the image in dir 'modified_images/")
                    except:
                        print(colors.red|"this image does not exists")

@AnomalousDiffusion.subcommand("label")
class Label(cli.Application):
    "Saves the binary image using pca and Gaussian-mixture algorithms"

    def label(value):
        """
        Reads the image from the given value and label it

        Parameters:
        -----------------------
        value : string for the file name of the image
        """
        #reading the image
        test_image =  cv2.imread(value)
        #make it gray
        im_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        #labels the images using PCA and GaussianMixture algorithms
        pca = cl.Principal_components_analysis(im_gray,window_sizeX=10,window_sizeY=10)
        labelled_image = cl.classification(im_gray, pca,window_sizeX=10,window_sizeY=10)
        return value, labelled_image

    def save(value , labelled_image):
        #saves them
        plt.imsave("labelled_images/labelled_" + value, labelled_image)

    def main(self, value : str ):
        #if there is no the directory, it creates it
        if not os.path.exists("labelled_images"):
            os.makedirs("labelled_images")
        #if given ".", it does it for all the directories in the current directory
        if (value == "."):
            #idex for the images name
            cont = 0
            for image in list(os.listdir(value)):
                if str(image).endswith(".png") or str(image).endswith(".jpg") :
                    print("analyzing image " +str(cont+1)+"/"+str(len(os.listdir(value))-1))
                    value, labelled_image = Label.label(image)
                    Label.save(value, labelled_image)
                    cont = cont + 1
            print(colors.green|"Saved the binary images in dir 'labelled_images/'")
        else:
            try:
                value, labelled_image = Label.label(value)
                Label.save(value, labelled_image)
                print(colors.green|"Saved the labelled image in dir 'labelled_images/'")
            except:
                print(colors.red|"this image does not exists")

@AnomalousDiffusion.subcommand("fronts")
class Fronts(cli.Application):
    "Tracks the longest borders in the images and saves the coordinates in a txt file"
    s = cli.Flag(["s", "save"], help = "If given, I will save the image with the borders in the current directory")

    def fronts(self, value, cont ):
        """
        Reads the image from the given string and finds the fronts and save it
        in a txt file

        Parameters:
        -----------------------------
        self : for the flag --s
        value : string for the image file name
        cont : integers for indexing the images
        """
        #reading the image
        test_image =  cv2.imread(value)
        #make it gray
        im_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        #finds the coordinates of the longest border in the image
        coord, im = fr.fronts(im_gray)
        np.savetxt("fronts/"+ str(cont) + ".txt", coord,fmt = '%d', delimiter=' ')
        if (self.s):
            plt.imsave(value, im)

    def main(self, value : str):
        #index for the frames
        frame = 0
        #if there is no the directory, it creates it
        if not os.path.exists("fronts"):
            os.makedirs("fronts")

        if(value == "."):
            frames = len(os.listdir(".")) - 1
            for im in list(os.listdir(".")):
                if str(im).endswith(".png") or str(im).endswith(".jpg"):
                    #finds the fronts of the image and saves it in text file
                    Fronts.fronts(self, im, frame)
                    frame = frame + 1
                #status bar
                print("image " + str(frame) +": ["+"#"*int(frame/frames*20)+"-"*int(20-int(frame/frames*20))+"] "+str(int(frame/frames*100))+"% ", end="\r")
            print("image " + str(frame) +": ["+"#"*20+"] 100%")
            print(colors.green|"Saved the fronts of the images in dir 'fronts/'")
        else:
            try:
                #finds the fronts of the image and saves it in text file
                Fronts.fronts(self, value, frame)
                print(colors.green|"Saved the fronts of the image in dir 'fronts/'")
            except:
                print(colors.red|"this image does not exists")


@AnomalousDiffusion.subcommand("divide")
class Divide(cli.Application):
    "Divides the front in two piecies, one left and one right and save them in two txt files"

    def divide(value):
        """
        Reads the coordinates from the given file txt
        and divide it into two files sx and dx for the left and right borders

        Parameters:
        -------------------------
        value : string for the txt file name
        """
        #reading the file with the coordinates of the longest border
        df = pd.DataFrame(pd.read_csv(value , delimiter=' '))
        df.columns = ["x","y"]
        #divide the longest border in two, the left one and the right one
        sx, dx = fr.divide(df)
        #save it
        np.savetxt("divided_fronts/"+ value+"dx.txt" , dx.values, fmt='%d')
        np.savetxt("divided_fronts/"+ value+"sx.txt" , sx.values, fmt='%d')

    def main(self, value : str ):
        #if there is no the directory, it creates it
        if not os.path.exists("divided_fronts"):
            os.makedirs("divided_fronts")
        #if given ".", it does it for all the directories in the current directory
        if(value == "."):
            for value in list(os.listdir(".")):
                if str(value).endswith(".txt"):
                    Divide.divide(value)
            print(colors.green|"Divided the fronts and saved in dir 'divided_fronts/'")
        else:
            try:
                Divide.divide(value)
                print(colors.green|"Divided the fronts and saved in dir 'divided_fronts/'")
            except:
                print(colors.red|"File does not exists")

##This command takes the images from the directories 'images/' while command 'faster' takes
## the images directly from the nd2 file
@AnomalousDiffusion.subcommand("fast")
class Fast(cli.Application):
    "Tracks the longest borders in the images and saves the coordinates in a txt file"

    def fast(path, frame):
        """
        Reads the image from the given path and finds the borders sx and dx

        Parameters:
        --------------------------------
        path : string for the directory
        frame : integer for indexing the frame

        Returns two pandas dataframe with the coordinates of the borders
        """
        #reading the image from the directory
        image =  cv2.imread( path +str(frame)+".png")
        #make it grays
        imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #finds the two longest borders inside the image
        dfs, b, c = fr.fast_fronts(imgray,length_struct=5,iterations=1)
        #interpolation of the two borders
        dx = an.necklace_points(dfs[0], N = 1000)
        sx = an.necklace_points(dfs[1], N = 1000)

        return sx, dx

    def to_dataframe(directory,frames,field):
        """
        Saves the coordinates for all the frames in a single
        dataframe in tidy format

        Paramaters:
        -----------------------
        directory : string for the directory name
        frames : integers for the number of the frames
        field : integer for the number of fields of view
        """
        path = directory + "images/"
        for frame in range(frames):
            #making the dataframe in tidy format
            sx, dx = Fast.fast(path, frame)
            dx["side"] = "dx"
            sx["side"] = "sx"
            df = pd.concat([dx,sx])
            df["frame"] = frame
            df["field"] = field
            df["experiment"] = directory
            df.to_csv(directory + "/" + "coordinates.txt",index = True,header = None, sep = " ", mode = "a")
            #status bar
            print("directory " + directory +": ["+"#"*int(frame/frames*20)+"-"*int(20-int(frame/frames*20))+"] "+str(int(frame/frames*100))+"% ", end="\r")


    def main(self, directory : str ,frames : int = 100):
        #saving an empty dataframe to append later the coordinates
        df = pd.DataFrame(columns = ["i","x","y","side","frame","field","experiment"])
        df.to_csv(directory + "coordinates.txt",index = False,header = df.columns, sep = " ")
        if (directory == "."):
            #if given ".", it does it for all the directories in the current directory
            for value in os.listdir(directory):
                try:
                    #find the fronts ans save it into a dataframe
                    Fast.to_dataframe(value +"/",frames,0)
                    print(value + " ["+"#"*20+"] 100%")
                except:
                    print(colors.yellow|"images/ doesn't exist in directory " +str(value))
        else:
            try:
                #find the fronts ans save it into a dataframe
                Fast.to_dataframe(directory, frames,0)
                print(directory + " ["+"#"*20+"] 100%")
            except:
                print(colors.red|"File does not exists")



##IMPORTANT this command is different form 'fast' because takes the images directly from the nd2 file
## while 'fast' takes the saved images from the directories 'imges/'
@AnomalousDiffusion.subcommand("faster")
class Faster(cli.Application):
    "Tracks the longest borders in the images and may save the coordinates in a txt file"
    all = cli.Flag(["all", "every directoies"], help = "If given, I will read the nd2 file in all the direcories and save all the coordinates in the file 'coordinates.txt'")

    def faster(im):
        """
        Finds the borders from the image and returns two pandas dataframes with the left
        and right borders

        Parameters:
        im : the image in matrix format

        Returns the two dataframes sx and dx with the left and right coordinates
        """
        try:

            _ , im = Label.label(im)
            dfs, b, c = fr.fast_fronts(im,length_struct=5,iterations=1)
            #interpolation of the two borders
            dx = an.necklace_points(dfs[0], N = 100)
            sx = an.necklace_points(dfs[1], N = 100)

            return sx, dx

        except:
            #sometimes doesn't recognizes the borders (in the last frames there are not borders)
            #so it saves empty dataframes
            dx = pd.DataFrame(columns = ["x","y"])
            sx = pd.DataFrame(columns = ["x","y"])
            return sx, dx

    def to_dataframe(directory,im,frame,field):
        """
        Saves the coordinates for all the frames in a single
        dataframe in tidy format

        Paramaters:
        -----------------------
        directory : string for the directory name
        im : image in matrix format
        frames : integers for the number of the frames
        field : integer for the number of fields of view
        """
        #making the dataframe in tidy format

        sx, dx = Faster.faster(im)
        dx["side"] = "dx"
        sx["side"] = "sx"
        df = pd.concat([dx,sx])
        df["frame"] = frame
        df["field"] = field
        df["experiment"] = directory
        df.to_csv("coordinates.txt",index = True,header = None, sep = " ", mode = "a")

    def read(direct, value):
        """
        Reads the images from nd2 file and saves the dataframe

        Parameters:
        --------------------------
        direct : string for the directory name
        value : string for the nd2 file name
        """
        with ND2Reader(direct + "/" + value) as images:
            print(colors.yellow|"directory " + direct)
            #iterations of the images in the nd2 file


            # ERRORE QUI#############################################################
            images.iter_axes = "vt"
            fields = images.sizes["v"]
            frames = images.sizes["t"]

            print(fields)
            for field in range(fields):
                for frame in range(frames):
                    #making the image of type uint8
                    im = np.asmatrix(images[frame + frame*field]).astype(np.uint8)
                    Faster.to_dataframe(direct, im,frame, field)
                    #status bar
                    print("field " + str(field) +": ["+"#"*int(frame/frames*20)+"-"*int(20-int(frame/frames*20))+"] "+str(int(frame/frames*100))+"% ", end="\r")
                print("field " + str(field) +": ["+"#"*20+"] 100%")

    def main(self, value : str):
        #saving an empty dataframe to append later the coordinates
        df = pd.DataFrame(columns = ["i","x","y","side","frame","field","experiment"])
        df.to_csv("coordinates.txt",index = False,header = df.columns, sep = " ")
        if(self.all):
            #searching only the directories in the current directory
            dirs = os.listdir(".")
            for direct in dirs:
                try:
                    Faster.read(direct, value)
                except: pass
            print(colors.green|"Saved coordinates in file 'coordinates.txt'")
        else:
            try:
                direct = "."
                Faster.read(direct, value)
                print(colors.green|"Saved coordinates in file 'coordinates.txt'")
            except:
                print(colors.red|"File does not exists")


@AnomalousDiffusion.subcommand("areas")
class Area(cli.Application):
    "Computes the areas for all the directories with the images!"

    def findarea(direct, i):
        """
        Finds the area between the two borders from the two txt files

        Parameters:
        ------------------------
        direct : string for the directory name
        i : integer for indexing the txt files

        Returns the area between the two borders
        """
        #reading the coordinates from the txt files
        filesx = direct+ "/images/fronts/"+ str(i)+".png_sx.txt"
        filedx = direct+ "/images/fronts/"+ str(i)+".png_dx.txt"
        #make it dataframes
        sx = pd.read_csv(filesx, sep = " ")
        dx = pd.read_csv(filedx, sep = " ")
        sx = an.necklace_points(sx, N = 100)
        dx = an.necklace_points(dx, N = 100)
        #computes the area between the two fronts
        pol, area = an.area(sx,dx)

        return area

    def make_dataframe(direct, frames,df,j):
        """
        Computes the area between the two borders and make a single dataframe
        with all the areas for all the frames

        Parameters:
        --------------------------
        direct : string for the directory name
        frames : integer for the number of frames
        df : pandas dataframe with the coordinates
        j : integer for indexing the different directories

        Return the dataframe with all the areas for all the directories
        """
        print("reading images in directory: "+ str(direct))

        areas = []
        for i in range(0,frames):
            area = Area.findarea(direct, i)
            areas.append(area)
        #normalize the area with the area of the first frame
        areas = np.array(areas)/areas[0]
        areas = pd.Series(areas)
        df[j] = areas
        return df

    def plot(df):
        """
        Plot the areas and saves the plot

        Paramaters:
        --------------
        df : pandas dataframe with the areas
        """
        #making the plot of the areas
        plt.figure(dpi = 200)
        for i in list(df.columns):
            plt.plot(df[i])
            plt.legend()
        plt.savefig("aree.png")


    def main(self, frames : int = 100):
        df = pd.DataFrame()
        j = 0
        d = []
        for direct in os.listdir("."):
            #if there is no the directory, it passes to the next directory
            if not os.path.exists(direct + "/images"):
                print(colors.yellow|"images/ doesn't exist in directory " +str(direct))
                pass
            else:
                try:
                    df = Area.make_dataframe(direct, frames,df,j)
                    d.append(direct)
                    j = j+1
                    df.columns = d
                    Area.plot(df)
                    df.to_csv("aree.txt", header = None, index = False ,sep=' ')
                except: pass

        if j > 0:
            print(colors.green|"areas saved in file 'areas.txt'")
        else:
            print(colors.red|"areas don't saved. There was a problem, maybe wrong directory")


#now this command is useless
@AnomalousDiffusion.subcommand("error")
class Error(cli.Application):
    "Computes the error between two areas between the fronts"
    def main(self,file1 = str, file2 = str):

            areas = pd.DataFrame(pd.read_csv(file1))
            areas_hand = pd.DataFrame(pd.read_csv(file2))
            error = an.error(areas, areas_hand)
            error = pd.DataFrame(error)
            error.to_csv("error.txt", sep = " ")
            print(colors.green|"errors saved in file 'error.txt'")


@AnomalousDiffusion.subcommand("msd")
class MSD(cli.Application):
    "Computes the Mean Squared Displacement (MSD) for all the directories with the images!"

    def find_msd(direct, frames):
        """
        Finds the msd for all the frames

        Paramaters:
        -----------------------
        direct : string for the directory name
        frames: integers for the number of frames

        Returns the dataframe with the msd
        """
        x = pd.DataFrame()
        for i in range(frames):
            #reading the x coordinates from the txt files
            s = pd.read_csv(direct + "/images/fronts/"+str(i)+".png_sx.txt", sep = " ")
            s.columns = [0,1]
            x[i] = s[0]
        #computes the MSD of the dataframe with the x coordinates
        msd = an.MSD(x)
        #saving it
        msd.to_csv(direct + "/msd.txt", header = None, index = False,sep=' ')
        print(colors.green|"msd saved in files 'msd.txt'")

        return msd

    def plot(msd):
        """
        Plots the msd and save it

        Paramaters:
        -------------------------
        msd : pandas dataframe with the msd
        """
        #makes the plot for the msd and the mean of the msd
        try:
            for i in range(len(msd.T)):
                plt.plot(msd.T[i])
                plt.savefig("msd.png")
        except: pass

        plt.figure(dpi = 200)
        plt.plot(np.mean(msd))
        plt.savefig("MSD.png")


    def main(self, frames : int = 100):
        mean = []
        msd = []
        for direct in os.listdir("."):
            #if there is no the directory, it passes to the next directory
            if not os.path.exists(direct + "/images/fronts"):
                print(colors.yellow|"fronts/ doesn't exist in directory " +str(direct))
                pass
            else:
                print("reading files in directory: "+ str(direct))
                try:
                    msdsx = MSD.find_msd(direct, frames)
                    msd.append(msdsx)
                    mean.append(np.mean(msdsx))
                except: pass

        mean = pd.DataFrame(mean)
        mean.to_csv("meanMSD.txt", header = None, index = False,sep=' ')
        MSD.plot(mean)



@AnomalousDiffusion.subcommand("fit")
class Fit(cli.Application):
    """Computes the fit parameters (D,a) for the Mean Squared Displacement for all the directories and saves
    in a txt file
    """
    def fit(direct):
        """
        Find the fit for the msd from the txt file

        Paramaters:
        --------------------
        direct : string for the directory name

        Return two float for the parameters of the fit
        """
        path = direct+ "/msd.txt"
        #making the dataframe with the msd from the txt file
        df = pd.DataFrame(pd.read_csv(path,sep = " "))
        msd = np.mean(df)
        #fit the msd
        popt = an.fit(msd)
        D = popt[0]
        alpha = popt[1]

        return D, alpha

    def to_file(D,alpha):
        """
        Saves to file the parameters of the fit

        Paramaters:
        D : float, parameter of the fit
        alpha : float, parameter of the fit
        """
        D = pd.Series(D)
        alpha = pd.Series(alpha)
        #making the dataframe with the parameters found with the fit
        fit = pd.DataFrame(columns = ["D","alpha"])
        fit["D"] = D
        fit["alpha"] = alpha
        fit.to_csv("fit.txt",sep = " ")

    def main(self):
        D = []
        alpha = []
        cont = 0
        for direct in os.listdir("."):
            #if there is no the file, it passes to the next directory
            if not os.path.exists(direct + "/msd.txt"):
                print(colors.yellow|"file 'msd.txt' doesn't exist in directory " +str(direct))
                pass
            else:
                print("reading file 'msd.txt' in directory " +str(direct))
                D.append(Fit.fit(direct)[0])
                alpha.append(Fit.fit(direct)[1])
                cont += 1

        Fit.to_file(D,alpha)
        print(colors.green | "fit parameters saved in file fit.txt")


if __name__ == "__main__":
    AnomalousDiffusion.run()
