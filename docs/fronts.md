## Fronts.py

This library have tools to track the longest border inside an image and save them in a csv file.
There are three main functions : `fronts`,`divide` and `fast_fronts`.
The last one is more or less the union of the first two, because `fronts` only track one longest border, which can be divided in two by the function `divide`.
`fast_fronts` instead return immediately two borders in two different dataframes.


#### `fronts(im)`

Takes the longest border inside an image.
The input image is modified by morphological transformation in order
to have a smoother border.

Parameters
---------------

im : the image in matrix format
threshold : integer for the threshold to make the image binary
iterations : integer for the number of iterations for the erosion

Returns:
a dataframe with the x and y coordinates of the longest border;
the image with the drawn found front.

Example:
----------------
```
import fronts as fr
import cv2

im = cv2.imread("docs/images/coin.png")
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
data , im = fr.fronts(imgray,threshold = 240, iterations = 0)
```
And we obtain the longest border:

<p align="center">
  <img src="https://github.com/riccardoscheda/AnomalousDiffusion/blob/master/docs/images/coin.png" width = "200" >
  <img src="https://github.com/riccardoscheda/AnomalousDiffusion/blob/master/docs/images/maxcoin.png" width = "200">
</p>
Fig.1: original image (left) and the image with the longest border found (right)

The longest border is returned in the dataframe `data`:

x|y
---|---
415|16
416|15
417|15
418|16
424|16
...|...

#### `divide(coordinates)`

Divides the found border in two different borders: the left one and the
right one.

Parameters
------------
coord : pandas Dataframe which contains the coordinates of the border

Returns (first sx and second dx) two pandas dataframes one for the left border and one for the right
border.

Example:
---------
```
import fronts as fr
import cv2

im = cv2.imread("docs/images/coin.png")
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
data , im = fr.fronts(imgray,threshold = 240, iterations = 0)

sx, dx = fr.divide(data)

```

<p align="center">
  <img src="https://github.com/riccardoscheda/AnomalousDiffusion/blob/master/docs/images/coordinates.png" width = "200" >
  <img src="https://github.com/riccardoscheda/AnomalousDiffusion/blob/master/docs/images/dividedcoordinates.png" width = "200">
</p>
Fig.2: unique longest border (left) and the border divided in left one and right one (right)

#### `make_kernel(struct,length)`

Makes the Structuring Element for the morphological operations

Parameters:
-------------------------

struct : list of 0s and 1s ex. ([0,1,1,1,0])
length : integer for the dimension of the matrix

Returns a binary numpy matrix

Example:
-------------------------

```
import fronts as fr

struct = [0,1,1,1,0,0]
kernel = fr.make_kernel(struct, 4)

kernel
```
```
>>> matrix([[0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0, 0]])
```
#### `cdf(im)`

Computes the Cumulative Distribution Function  (CDF) of an image as 2D numpy ndarray

Parameters:
---------------------

im : image in matrix format

Returns a numpy array with the cumulative distribution function

#### `hist_matching(c, c_t, im)`:

 Returns the image with the histogram similar to the sample image, using operations on the cumulative distributions
 of the two images.

 Parameters:
 -------------------------

 c: CDF of input image computed with the function cdf()
 c_t: CDF of template image computed with the function cdf()
 im: input image as 2D numpy ndarray

 Returns the modified pixel values

Example:
------------------------------

```
import fronts as fr

lena = cv2.imread("docs/images/lena.png")
lena = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)

coin = cv2.imread("docs/images/coin.png")
coin = cv2.cvtColor(coin, cv2.COLOR_BGR2GRAY)

c = fr.cdf(coin)
c_t = fr.cdf(lena)

newcoin = fr.hist_matching(c,c_t,coin)
```

And we obtain:
<p align="center">
  <img src="https://github.com/riccardoscheda/AnomalousDiffusion/blob/master/docs/images/coin.png" width = "200" >
  <img src="https://github.com/riccardoscheda/AnomalousDiffusion/blob/master/docs/images/lena.png" width = "200">
<img src="https://github.com/riccardoscheda/AnomalousDiffusion/blob/master/docs/images/histcoin.png" width = "200">
</p>
Fig.3:  coin (left), lena (center) and the new coin with histogram similar to the lena's histogram (right).


#### `fast_fronts(im, outdir = "fronts/", size = 50, length_struct = 10,iterations = 2, save = False, fname = "")`

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
  a list with the two dataframes with the coordinates of the two longest borders
  the maxcontours computed by openCV
  the final binary image after the morphological transformations

  References
  --------------------------------------

  [1] [Contours](https://docs.opencv.org/3.1.0/d4/d73/tutorial_py_contours_begin.html)

  [2] [Morphological transformations](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html)

  [3] [Otsu thresholding](https://docs.opencv.org/3.4.0/d7/d4d/tutorial_py_thresholding.html)

Example:
--------------------
```
import fronts as fr
import cv2

coin = cv2.imread("docs/images/histcoin.png")
coin = cv2.cvtColor(coin, cv2.COLOR_BGR2GRAY)

dfs , coordinates , binary = fr.fast_fronts(coin,size = 20,bands = False,length_struct=0,iterations = 1)
```
And we obtain:
<p align="center">
  <img src="https://github.com/riccardoscheda/AnomalousDiffusion/blob/master/docs/images/fastcoordinates.png" width = "200" >
  <img src="https://github.com/riccardoscheda/AnomalousDiffusion/blob/master/docs/images/fastcoin.png" width = "200">
</p>
Fig.4: divided borders (left) and the binary image obtained with Otsu threshold and morphological operations (right)
