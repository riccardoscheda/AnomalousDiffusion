## Classification.py

Classification library is a library that makes binary grayscale images using PCA algorithm and GaussianMixture algorithm.

## Adaptive Histogram Equalization
## `adaptive_contrast_enhancement(image,grid_size = (50,50))`

The function uses [Adaptive
Histogram Equalization](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html
) in order to enhance the borders inside an image.
The equalization is adaptive because is not a global histogram equalization,
but the equalization is divided for different windows, where you can decide the size with the tuple `giride_size`. The reuslt changes with the size of the window.

#### Parameters


-image: the image in matrix format

-grid_size : a tuple with two integer which are the width and height of the subimages

#### Example:

```
import classification as cl
import cv2

im = cv2.imread("docs/images/lena.png")
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
new = cl.adaptive_contrast_enhancement(imgray,grid_size=(100,100))
```
And we obtain:

<p align="center">
  <img src="https://github.com/riccardoscheda/AnomalousDiffusion/blob/master/docs/images/lena.png" width = "200" >
  <img src="https://github.com/riccardoscheda/AnomalousDiffusion/blob/master/docs/images/equalizedlena.png" width = "200">
</p>
Fig.1: original image (left) and the modified image with adaptive histogram equalization (right)

### Locally Binary pattern
## `LBP(image)`

Computes the [Locally Binary Pattern](https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_local_binary_pattern.html) of an image and
returns the normalized histogram of the local binary pattern image.
In this function we used the method 'uniform' to have always 10 bins for the histograms, which are the bins that refers to the uniform patterns in the image. We need 10 bins because they are used as features for the PCA algrorithm in the function `Principal_components_analysis` (below).

#### Parameters

-image : the image in matrix format


#### Example:
```
import classification as cl
import cv2

im = cv2.imread("docs/images/lena.png")
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
lbp , hist = cl.LBP(imgray)
```

<p align="center">
  <img src="https://github.com/riccardoscheda/AnomalousDiffusion/blob/master/docs/images/lena.png" width = "200" >
  <img src="https://github.com/riccardoscheda/AnomalousDiffusion/blob/master/docs/images/lbplena.png" width = "200">
</p>
Fig.2: original image (left) and the modified image with locally binary pattern with the method 'uniform' (right)

## PCA
## `Principal_components_analysis(image,window_sizeX = 12, window_sizeY = 16)`

This function computes the PCA algorithm for the locally binary pattern subimages of the original image, and then takes in a dataframe the first 5 principal components components.

#### Parameters:

-image : image in matrix format

-window_sizeX : the size of the width of the subimages

-window_sizeY : the size of the height of the subimages


#### Example:
```
import classification as cl
import cv2

im = cv2.imread("docs/images/lena.png")
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
dataframe = cl.Principal_components_analysis(imgray)
```
what we obtain is a dataframe like this:

x|y|z|u|w
---|---|---|---|---
0.12996983128369194|-0.06285114901795194|0.009630236673870328|-0.0013368769283709939|-0.0023434951040339404
0.12996983128369125|-0.06285114901795226|0.009630236673869755|-0.0013368769283715611|-0.0023434951040339526
0.12996983128369125|-0.06285114901795247|0.009630236673870415|-0.0013368769283712643|-0.0023434951040346634
0.08312408370945534|-0.03549650246107342|0.003607495986024477|-0.0025576345014921794|-0.00247981733049018
...|...|...|...|...

## Binarization
## `classification(image, data, window_sizeX = 12, window_sizeY = 16)`

Computes the classification of the subimages of the total image through the GaussianMixture algorithm.
Returns the binary image, where a label corresponds to the cells and one
label corresponds to the background.

#### Parameters:


image : the image in matrix format

data : pandas dataframe

window_sizeX : the size of the width of the subimages

window_sizeY : the size of the height of the subimages

#### Example:

```
import classification as cl
import cv2

im = cv2.imread("docs/images/lena.png")
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
dataframe = cl.Principal_components_analysis(imgray, window_sizeX = 8, window_sizeY = 8)

labelled_image = cl.classification(imgray, dataframe, window_sizeX = 8, window_sizeY = 8)
```
And we obtain is something like this:
<p align="center">
  <img src="https://github.com/riccardoscheda/AnomalousDiffusion/blob/master/docs/images/lena.png" width = "200" >
  <img src="https://github.com/riccardoscheda/AnomalousDiffusion/blob/master/docs/images/labelledlena.png" width = "200">
</p>
Fig.3: original image (left) and the binary image using PCA and GaussianMixture algorithms. (right)


References:
------------------

[1] [Adaptive
Histogram Equalization](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html
)

[2] [Locally Binary Pattern](https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_local_binary_pattern.html)
