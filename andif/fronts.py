import cv2
import pandas as pd
import numpy as np
import pylab as plt
from scipy import stats
import itertools as it
from skimage.exposure import cumulative_distribution

import classification as cl

def fronts(im, threshold = 127, iterations = 1):
    """

    Takes the longest border inside an image.
    The input image is modified by morphological transformation in order
    to have a smoother border.


    Parameters
    --------------------

    im : the image in matrix format
    threshold : integer for the threshold to make the image binary
    iterations : integer for the number of iterations for the erosion

    Returns:

    a dataframe with the x and y coordinates of the longest border;
    the image with the drawn found front.

    References
    ------------
    [1] https://docs.opencv.org/3.1.0/d4/d73/tutorial_py_contours_begin.html
    [2] https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html

    """
    #reading the image
    imgray = np.asmatrix(im).astype("uint8")
    #apply adaptive histogram histogram_equalization
    imgray = cl.adaptive_contrast_enhancement(imgray)
    #make it binary
    ret, thresh = cv2.threshold(imgray, threshold, 255, 0)

    #create the kernel for the morphological transformations
    struct = [0,0,0,1,1,1,1,1,1,1,0,0,0]
    kernel = make_kernel(struct,100)
    kernel = np.array(kernel,np.uint8)

    #using erosion to enhance the fronts
    erode = cv2.erode(thresh,kernel,iterations = iterations)

    #finding all the contours and select the longest one
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lencontours = np.array([len(x) for x in contours])
    sel = [x in np.sort(lencontours)[-1:] for x in lencontours]
    maxcontours = np.array(contours)
    maxcontours = maxcontours[sel]

    image_with_fronts = cv2.drawContours(imgray, maxcontours, -1, (0,255,0), 3)

    #making maxcontours as array so it can ben putted in a dataframe
    maxcontours = list(it.chain.from_iterable(maxcontours))
    maxcontours = list(it.chain.from_iterable(maxcontours))
    maxcontours = np.array(maxcontours)
    #making it dataframe
    coordinates = pd.DataFrame(maxcontours)
    coordinates.columns = ["x", "y"]

    return coordinates, image_with_fronts

def divide(coordinates):
    """
    Divides the found border in two different borders: the left one and the
    right one.

    Parameters
    ------------
    coord : pandas Dataframe which contains the coordinates of the border

    Returns (first sx and second dx) two pandas dataframes one for the left border and one for the right
    border.
    """
    coordinates.columns = ["x", "y"]
    #takes the left upper corner and keep what there is before
    leftup = np.min(np.where(coordinates["y"] == np.max(coordinates["y"])))
    leftdown = np.max(np.where(coordinates["y"]== np.min(coordinates["y"])))
    dx = coordinates.iloc[leftdown:leftup , :]
    #takes the right upper corner and takes what there is after
    rightup = np.max(np.where(coordinates["y"] == np.max(coordinates["y"])))
    sx = coordinates.iloc[rightup: ,:]

    return sx, dx


def make_kernel(struct,length):
    """
    Makes the Structuring Element for the morphological operations

    Parameters:
    -------------------------

    struct : list of 0s and 1s ex. ([0,1,1,1,0])

    Returns a binary numpy matrix
    """
    kernel = []
    #making a matrix with rows all equal to the struct
    for i in range(length):
        kernel.append(struct)
    return np.matrix(kernel)



def cdf(im):
 '''
 computes the Cumulative distribution function of an image as 2D numpy ndarray

 Parameters:
 ---------------------

 im : image in matrix format

 Returns a numpy array with the cumulative distribution function

 References:
 -----------------------------

 [1] https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
 '''
 c, b = cumulative_distribution(im)
 # pad the beginning and ending pixels and their CDF values
 c = np.insert(c, 0, [0]*b[0])
 c = np.append(c, [1]*(255-b[-1]))
 return c

def hist_matching(c, c_t, im):
 '''
 Returns the image with the histogram similar to the sample image, using operations on the cumulative distributions
 of the two images.

 Parameters:
 -------------------------

 c: CDF of input image computed with the function cdf()
 c_t: CDF of template image computed with the function cdf()
 im: input image as 2D numpy ndarray

 Returns the modified pixel values

 References:
 -----------------------------

 [1] https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
 '''
 pixels = np.arange(256)
 # find closest pixel-matches corresponding to the CDF of the input image, given the value of the CDF H of
 # the template image at the corresponding pixels, s.t. c_t = H(pixels) <=> pixels = H-1(c_t)
 new_pixels = np.interp(c, c_t, pixels)
 im = (np.reshape(new_pixels[im.ravel()], im.shape)).astype(np.uint8)
 return im

def fast_fronts(im, outdir = "fronts/", size = 50, length_struct = 10,iterations = 2, bands = True, save = False, fname = ""):
    """
    Takes the two longest borders inside an image and it may save in a text file
    the (x,y) coordinates.
    The input image is binarized with Otsu thresholding and modified by morphological transformation in order to have a smoother border

    Parameters
    ---------

    im : the image matrix in uint8 format
    outdir : string for the ouput directory where it saves the txt files
    size : integer for the size of the window for the adaptive histogram equalization
    length_struct : integer for the length for the kernel for the morphological transformations
    iterations : integer for the number of times the dilation is applied
    bands : boolean var for adding two white bands on the left and on the right in the image
    save : boolean var for saving the coordinates in a txt file
    fname : string for the name of the output files

    Returns:
    --------
    a list with the two dataframes with the coordinates of the longest borders
    the maxcontours computed by openCV
    the final binary image after the morphological transformations

    References
    --------------------------------------

    [1] Contours: https://docs.opencv.org/3.1.0/d4/d73/tutorial_py_contours_begin.html
    [2] Morphological transformations: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    [3] Otsu thresholding : https://docs.opencv.org/3.4.0/d7/d4d/tutorial_py_thresholding.html
    """


    #the struct is for the kernel for the morphological trasnformations
    struct = [0,0,0,1,1,1,1,0,0,0]
    kernel = make_kernel(struct,length_struct)
    kernel = np.array(kernel,np.uint8)

    #apply adaptive histogram histogram_equalization
    #grid_size = (int(size),int(size))
    #gray = cl.adaptive_contrast_enhancement(gray, grid_size= grid_size)

    #blurring the image will give better results for the Otsu thresholding
    blur = cv2.GaussianBlur(im,(5,5),0)
    ret3, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    ###################
    #In order to track the right central border, i give to the image a border for each
    #side, so opencv doesn't consider the border of the image as a contour
    if bands:
        thresh[0:2,] = 255
        thresh[len(thresh)-2:len(thresh)-1,:] = 255
        thresh[:,0:400] = 255
        thresh[:,-400:] = 255

    #now i try using dilate, opening and closing to make the cells more uniform
    cstruct = np.ones(length_struct)
    ckernel = make_kernel(cstruct,length_struct*2)
    ckernel = np.array(ckernel,np.uint8)
    dilate = cv2.dilate(thresh,ckernel,iterations = iterations)
    opening = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, ckernel)
    cstruct = np.ones(length_struct*3)
    ckernel = make_kernel(cstruct,length_struct*3)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, ckernel)

    #now opencv finds the longest border
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lencontours = np.array([len(x) for x in contours])
    sel = [x in np.sort(lencontours)[-1:] for x in lencontours]
    maxcontours = np.array(contours)
    maxcontours = maxcontours[sel]

    coord = maxcontours
    image_with_fronts = cv2.drawContours(thresh, maxcontours, -1, (255,255,255), 3)
    #dfs is a list which will contain the left and right borders
    dfs = []

    #now make the maxcontours as dataframes (seems to work only in this way)
    maxcontours = list(it.chain.from_iterable(maxcontours))
    maxcontours = list(it.chain.from_iterable(maxcontours))
    maxcontours = np.array(maxcontours)
    maxcontours = pd.DataFrame(maxcontours)
    coordinates = maxcontours
    coordinates.columns = ["x", "y"]
    #now divide the longest border in the left and right borders
    #takes the left upper corner and keep what there is before,
    #taking only the borders which i need
    leftup = np.min(np.where(coordinates["y"] == np.max(coordinates["y"])))
    leftdown = np.max(np.where(coordinates["y"]== np.min(coordinates["y"])))
    dx = coordinates.iloc[leftdown:leftup , :]
    dfs.append(dx)
    #takes the right upper corner and takes what there is after
    rightup = np.max(np.where(coordinates["y"] == np.max(coordinates["y"])))
    sx = coordinates.iloc[rightup: ,:]
    dfs.append(sx)

    #saves the right and left borders i need
    if save:
        np.savetxt(outdir + fname + "_dx.txt", dx,fmt = '%d', delimiter=' ')
        np.savetxt(outdir + fname + "_sx.txt", sx,fmt = '%d', delimiter=' ')

    return  dfs , coordinates  , closing
