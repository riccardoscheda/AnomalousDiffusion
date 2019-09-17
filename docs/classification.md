## Classification.py

Classification library is a library that makes binary grayscale images using PCA algorithm and GaussianMixture algorithm.

### Adaptive Histogram Equalization
#### `adaptive_contrast_enhancement(image,grid_size = (50,50))`
The function uses adaptive
histogram equalization in order to enhance the borders inside an image.
The equalization is adaptive because is not a global histogram equalization,
but the equalization is divided for different windows, where you can decide the size with the tuple `giride_size`. The reuslt changes with the size of the window.

<p align="center">
  <img src="https://github.com/riccardoscheda/AnomalousDiffusion/blob/master/docs/images/8.png" >
</p>
<p align="center">
  <img src="https://github.com/riccardoscheda/AnomalousDiffusion/blob/master/docs/images/m_8.png">
</p>
*Fig.1: original image (left) and the modified image with adaptive histogram equalization (right)*

### Locally Binary pattern
#### `LBP(image)`
Computes the [Locally Binary Pattern](https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_local_binary_pattern.html) of an image and
returns the normalized histogram of the local binary pattern image.
In this function we used the method 'uniform' to have always 10 bins for the histograms, which are the bins that refers to the uniform patterns in the image.

Example:
```
import classification as cl
import cv2

im = cv2.imread("docs/images/lena.png")
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
lbp , hist = cl.LBP(imgray)
```

<p align="center">
  <img src="https://github.com/riccardoscheda/AnomalousDiffusion/blob/master/docs/images/lena.png" >
</p>
<p align="center">
  <img src="https://github.com/riccardoscheda/AnomalousDiffusion/blob/master/docs/images/lbplena.png">
</p>
*Fig.1: original image (left) and the modified image with locally binary pattern with the method 'uniform' (right)*
