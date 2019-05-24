import cv2
import os # To perform path manipulations
from skimage.feature import local_binary_pattern # Local Binary Pattern function
import matplotlib.pyplot as plt # For plotting
import numpy as np
from nd2reader import ND2Reader
import pandas as pd
from sklearn.decomposition import PCA

def create_set(n_images = 60):
    #creating output directories
    outfolder = "images/"
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    #Reading images from nd2 file
    image = []
    with ND2Reader("../data_prova/Sham_8_2_18_Field_5.nd2") as images:
        for i in range(n_images):
            image.append(np.asmatrix(images[i]).astype('uint16'))

    #creating the set of images i will modify

        for i in range(n_images):
            plt.imsave(outfolder+str(i)+".png",image[i],cmap="gray")


def adaptive_contrast_enhancement(image,grid_size):
    """
    This function uses adaptive histogram equalization in order
    to enhance fronts of the cells. Returns the modified image.
    Parameters
    -----------------------
    image : image in matrix format
    grid_size : tuple with the size of the grid
    References
    ---------------------
    [1] https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html
    """
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=grid_size)
    cl1 = clahe.apply(image)
    return cl1



def create_modified_images(image , n_images = 60, grid_size = (100,100)):
    modimg = []
    modified_images = "modified_images/"
    if not os.path.exists(modified_images):
        os.makedirs(modified_images)

    for i in range(n_images):
        #contrast enhancement on the images
        im = adaptive_contrast_enhancement(image[i],grid_size)
        modimg.append(im)
        plt.imsave("m_"+str(i)+".png",im,cmap="gray")


def LBP(image):
    """
    Computes the local binary pattern of an image
    Returns the normalized histogram of the local binary pattern image.
    Parameters
    --------------------------------
    image : image in matrix format
    References
    -------------------------------
    [1] http://hanzratech.in/2015/05/30/local-binary-patterns.html
    [2] https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
    """

    radius = 1
    # Number of points to be considered as neighbours
    no_points = 8 * radius
    # Uniform LBP is used
    lbp = local_binary_pattern(image,no_points,radius,method="uniform")
    #make the histogram of pixel intensities
    (hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, no_points + 3),range=(0, no_points + 2))
    hist = hist.astype("float")
    # Normalize the histogram
    hist /= (hist.sum() + 1e-7)
    return hist


def Principal_components_analysis(image , window_sizeX = 12, window_sizeY = 16):
    """
    Computes the principal components analysis (PCA) of an image:
    - The image is divided in subimages
    - the locally binary pattern histogram is computed for each subimage
    - all of the histograms are moved in a Dataframe
    - PCA is computed on that Dataframe where the samples are the subimages and the histogram bins are
    the features
    Parameters:
    ----------------------------------
    image : image in matrix format
    window_sizeX : the size of the width of the subimages
    window_sizeY : the size of the height of the subimages
    """
    rows = int(len(image)/window_sizeX)
    cols = int(len(image.T)/window_sizeY)
    labels = np.empty((rows,cols))
    testdf = pd.DataFrame()
    cont = 0

    for i in range(rows):
        for j in range(cols):
            subimage = image[i*window_sizeX:(i+1)*window_sizeX,j*window_sizeY:(j+1)*window_sizeY]
            series = pd.Series(LBP(subimage))
            testdf[cont] = series
            cont = cont + 1

    pca = PCA(2)
    principal_components = pca.fit_transform(testdf.T)
    principalDF = pd.DataFrame(principal_components, columns = ["x","y"])

    return principalDF
