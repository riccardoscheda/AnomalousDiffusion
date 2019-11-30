import cv2 #for applications to images
import os # To perform path manipulations
from skimage.feature import local_binary_pattern # Local Binary Pattern function
import matplotlib.pyplot as plt # For plotting
import numpy as np
from nd2reader import ND2Reader #to read from nd2 files
import pandas as pd
from sklearn.decomposition import PCA #for principal components analysis
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import itertools

def create_set( frames , field ,path = "../data_prova/Sham_8_2_18_Field_5.nd2"):
    """
    Reads the images from the nd2 file and saves them in the directory 'images/'

    Paramters:
    -----------------------

    frames : number of images to read from the nd2 file
    field : the field of view in the nd2 file
    path : the path of the nd2 file

    """
    #creating output directories
    outfolder = "images/"
    name = path.split("/")[0]
    if not os.path.exists(name + "/" + outfolder):
        os.makedirs(name + "/" + outfolder)

    #Reading images from nd2 file
    #creating the set of images i will modify

    with ND2Reader(path) as images:
        #choosing the indexing of the images in the nd2 file
        images.iter_axes = "vt"
        fields = images.sizes["v"]
        #frames = images.sizes["t"]
        for frame in range(frames):
            #status bar
            print("image "+str(frame+1)+"/"+str(frames)+" ["+"#"*int(frame/frames*20)+"-"*int(20-int(frame/frames*20))+"] "+str(int(frame/frames*100))+"% ", end="\r")
            image = np.asmatrix(images[frame + frame*(field)]).astype('uint16')
            plt.imsave(name + "/" + outfolder+str(frame)+".png",image,cmap="gray")
        print("images "+str(frame+1)+"/"+str(frames)+" ["+"#"*20+"] 100%")

def adaptive_contrast_enhancement(image,grid_size = (50,50)):
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



def create_modified_images(path , grid_size = (100,100)):
    """
    Reads the images from the directory 'images/'and modify them with adaptive histogram
    equalization and then saves them in the directory 'modified_images/''
    Parameters
    -----------------------

    path : the path of the image in png
    grid_size : tuple with the size of the grid

    """
    modified_images = "modified_images/"
    if not os.path.exists(modified_images):
        os.makedirs(modified_images)
    #reading the image
    image =  cv2.imread(path)
    #make it gray
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #contrast enhancement on the images
    im = adaptive_contrast_enhancement(imgray,grid_size)
    plt.imsave("modified_images/"+ path ,im,cmap="gray")


def LBP(image):
    """
    Computes the local binary pattern of an image
    Returns:
    the lbp image;
    the normalized histogram of the local binary pattern image.

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
    #ret, thresh = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
    newim = adaptive_contrast_enhancement(image)
    # Uniform LBP is used
    lbp = local_binary_pattern(newim,no_points,radius,method="uniform")
    #make the histogram of pixel intensities
    (hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, no_points + 3),range=(0, no_points + 2))
    hist = hist.astype("float")
    # Normalize the histogram
    hist /= (hist.sum() + 1e-7)
    return lbp , hist


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
    window_sizeX : the size of the width of the subimages, IMPORTANT: depends on the image size
    window_sizeY : the size of the height of the subimages, IMPORTANT: depends on the image size

    Returns a dataframe with the first 5 principal components
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
        _, hist = LBP(subimage)
        series = pd.Series(hist)
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

def classification(image, data, window_sizeX = 12, window_sizeY = 16, model = "gaussianmixture"):
    """
    Computes the classification of the subimages of the total image through the GaussianMixture algorithm.
    Returns the binary image, where a label corresponds to the cells and one
    label corresponds to the background.

    Parameters
    -----------------------------
    image : image in matrix format
    data : pandas dataframe
    window_sizeX : the size of the width of the subimages, IMPORTANT: depends on the image size
    window_sizeY : the size of the height of the subimages, IMPORTANT: depends on the image size
    model : the model used to cluster the data; it can be "gaussianmixture" (default) or "kmeans"

    Returns the labelled image in uint8

    References
    ----------------------
    [1] https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
    """

    rows = int(len(image)/window_sizeX)
    cols = int(len(image.T)/window_sizeY)

    if model == "kmeans":
        #using the K-means algorithm to classify the 2 clusters given by the principal components
        kmeans = KMeans(2)
        kmeans.fit(data)
        labels = kmeans.predict(data)
        labels = labels.reshape(rows,cols)
    else:
        if model == "gaussianmixture":
            #Using Gaussian Mixture instead of K-means gives a better result
            labels = GaussianMixture(2).fit_predict(data)
            labels = labels.reshape(rows,cols)
        else:
            raise ValueError("model can be only kmeans or gaussianmixture")

    #sometimes Kmeans predict inverted labels, so i make all the images with the same ordered labels
    if (labels[0][0] == 1):
        labels = abs(1-labels)

    #since the result of K-means is a smaller binary image (100*100 pixels) i resize them as the original images
    labell2 = np.empty((len(image),len(image.T)))
    for i in range (len(image)):
        for j in range (len(image.T)):
            labell2[i][j] = labels[int(i/window_sizeX)][int(j/window_sizeY)]

    #returns the labelled image in uint8
    return np.array(labell2,np.uint8)
