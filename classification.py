import cv2
import os # To perform path manipulations
from skimage.feature import local_binary_pattern # Local Binary Pattern function
import matplotlib.pyplot as plt # For plotting
import numpy as np
from nd2reader import ND2Reader
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import itertools

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
    #divides the image in many subimages, and for these the LBP histogram is computed
    rows = int(len(image)/window_sizeX)
    cols = int(len(image.T)/window_sizeY)
    labels = np.empty((rows,cols))
    testdf = pd.DataFrame()
    cont = 0
    #The LBP histogram for each subimage is appended in a dataframe
    for element in itertools.product(range(rows),range(cols)):
        subimage = image[element[0]*window_sizeX:(element[0]+1)*window_sizeX,element[1]*window_sizeY:(element[1]+1)*window_sizeY]
        series = pd.Series(LBP(subimage))
        testdf[cont] = series
        cont = cont + 1

    #The PCA is computed for the dataframe given by the LBP histograms, where the subimages are
    #the samples and the bin of the histograms ( which are 10 ) are the features
    #I choose to take 5 components because all of the first 5 variances are between 10% and 20%

    pca = PCA(5)
    principal_components = pca.fit_transform(testdf.T)
    principalDF = pd.DataFrame(principal_components, columns = ["x","y","z","u","w"])

    #return the dataframe of the first 5 principal components
    return principalDF

def classification(data, window_sizeX = 12, window_sizeY = 16):
    """
    Computes the classification of the subimages of the total image through the K-means algorithm.
    Returns the binary image, where a label corresponds to the cells and one
    label corresponds to the background.

    Parameters
    -----------------------------
    data : pandas dataframe
    window_sizeX : the size of the width of the subimages
    window_sizeY : the size of the height of the subimages

    References
    ----------------------
    [1] https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
    """
    rows = int(1200/window_sizeX)
    cols = int(1600/window_sizeY)

    #using the K-means algorithm to classify the 2 clusters given by the principal components
    kmeans = KMeans(2)
    kmeans.fit(data)
    labels = kmeans.predict(data)
    labels = labels.reshape(rows,cols)
    #sometimes Kmeans predict inverted labels, so i make all the images with the same ordered labels
    if (labels[0][0] == 1):
        labels = abs(1-labels)

    #since the result of K-means is a smaller binary image (100*100 pixels) i resize them as the original images
    labell2 = np.empty((1200,1600))
    for i in range (1200):
        for j in range (1600):
            labell2[i][j] = labels[int(i/window_sizeX)][int(j/window_sizeY)]

    #returns the labelled image in uint8
    return np.array(labell2,np.uint8)
