import os
import numpy as np
import matplotlib as plt
import cv2
import pandas as pd
from shapely.affinity import rotate,scale
from shapely.geometry import Polygon


from hypothesis import given,settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as enp
from hypothesis.extra import pandas as epd


import classification as cl
import fronts as fr
import analysis as an

#######################################################
##TESTS FOR classification.py

@given(dim = st.integers(min_value=1000,max_value=1600))
@settings(max_examples = 50)
def test_adaptive_contrast_enhancement(dim):
    """
    Tests:
    if the function returns a np.ndarray
    if the function returns an image different from the original one
    """
    im_gray = np.random.randint(0,255,dtype="uint8",size =(dim,dim))
    new_im = cl.adaptive_contrast_enhancement(im_gray,(10,10))


    assert isinstance(new_im, np.ndarray) == True

    #making a boolean array True or False if each pixel of the original image
    #is different from the pixels of the new image
    a = np.array([a!=b for (a,b) in zip(im_gray,new_im)])
    assert a.any() == True

@given(dim = st.integers(min_value=100,max_value=200))
@settings(max_examples = 50)
def test_LBP(dim):
    """
    Tests:
    if the output histogram is normalized
    if the length of the histogram is 10
    if the output is a np.ndarray
    if the function returns an image different from the original one
    """
    im_gray = np.random.randint(0,255,dtype="uint8",size =(dim,dim))
    new_im, hist = cl.LBP(im_gray)
    assert isinstance(hist, np.ndarray) == True
    assert isinstance(new_im, np.ndarray) == True
    assert sum(hist) <= 1.0
    assert sum(hist) >= 0.999
    assert len(hist) == 10
    #making a boolean array True or False if each pixel of the original image
    #is different from the pixels of the new image
    a = np.array([a!=b for (a,b) in zip(im_gray,new_im)])
    assert a.any() == True


@given(dim = st.integers(min_value=5,max_value=10))
@settings(max_examples = 50)
def test_Principal_components_analysis(dim):
    """
    Tests:
    if the function returns a dataframe
    if does not return Nan values
    If the number of the columns is always 5
    If the length of a column is the number of the pixels inside a window
    If the pca for different images gives different results
    """

    im_gray = np.random.randint(0,255,dtype="uint8",size =(dim*10,dim*10))
    df = cl.Principal_components_analysis(im_gray, window_sizeX = 10, window_sizeY = 10)

    im_gray2 = np.random.randint(0,255,dtype="uint8",size =(dim*10,dim*10))
    df2 = cl.Principal_components_analysis(im_gray2, window_sizeX = 10, window_sizeY = 10)

    assert isinstance(df, pd.DataFrame) == True
    assert any(np.where(np.isnan(df))) == False
    #has the firste 5 principal components
    assert len(df.T) == 5
    #the length of a column is the number of the pixels inside a window
    assert len(df) == dim*dim
    #the two dataframes for two different images are different
    assert df.equals(df2) == False



def test_classification():
    """
    Tests:
    if the output is numpy array
    if the output image is binary (there are only 0s and 1s)
    if the output for the same image is equal
    """
    im_gray = np.random.randint(0,255,dtype="uint8",size =(1200,1600))
    binary = cl.classification(im_gray, cl.Principal_components_analysis(im_gray))
    #doing the same function a second time
    binary2 = cl.classification(im_gray, cl.Principal_components_analysis(im_gray))

    assert isinstance(binary,np.ndarray) == True
    assert len(np.where(binary == 0 )[1]) + len(np.where(binary == 1)[1]) == len(binary)*len(binary.T)
    #if the two images are equal
    a = np.array([a!=b for (a,b) in zip(binary,binary2)])
    assert a.all() == False


###############################################################################
#TESTS FOR fronts.py

@given(im = st.lists(min_size=10, max_size = 100,elements = st.integers(min_value=0,max_value=255)))
def test_fronts(im):
    """
    Tests:
    if the output is a pandas DataFrame and a numpy array
    If the length of the dataframe is bigger than 2
    If the length of the new image is equal to the length of the original image
    """

    im_gray = np.asmatrix(im)
    fronts, image_with_fronts = fr.fronts(im_gray)

    assert isinstance(fronts, pd.DataFrame) == True
    assert isinstance(image_with_fronts, np.ndarray) == True
    assert len(fronts) >=  2
    assert len(image_with_fronts) == len(im_gray)


@given(struct = st.lists(min_size=1, elements = st.integers(min_value=0,max_value=1)),length = st.integers(min_value = 1, max_value=5))
def test_make_kernel(struct,length):
    """
    Tests:
    if the output of make_kernel is a numpy matrix of 0s and 1s
    """

    binary = fr.make_kernel(struct, length)

    assert isinstance(binary, np.matrix) == True
    #the sum of 0s and 1s gives the total number of the elements of the matrix
    assert len(np.where(binary == 0 )[1]) + len(np.where(binary == 1)[1]) == len(binary)*len(binary.T)

@given(dim = st.integers(min_value=1000,max_value=1600))
@settings(max_examples = 50)
def test_fast_fronts(dim):
    """
    Tests :
    if the output is a list of two pandas dataframes
    if the len of the binary image is equal to the length of the original image
    if the two dataframes with the coordinates are different
    """

    im_gray = np.random.randint(0,255,dtype="uint8",size =(dim,dim))
    df, im, im2 = fr.fast_fronts(im_gray)
    dx = df[0]
    sx = df[1]

    assert isinstance(df, list) == True
    assert len(df) == 2
    assert len(im2) == len(im_gray)
    assert sx.equals(dx) == False


@given(dim = st.integers(min_value = 2,max_value=100),max = st.integers(min_value = 500,max_value=1000))
@settings(max_examples = 50)
def test_divide(dim,max):
    """
    Tests:
    if returns two pandas Dataframes
    if the two output dataframes are different
    if the x coordinates are always different from 0
    if the y coordinates are always less than the max
    """
    #we want only two columns that refer to x and y coordinates
    y =  np.linspace(0,max,num=dim)
    x = y + np.random.randint(600,800,dtype="uint16",size =(dim))
    coord = pd.DataFrame()
    coord[0] = x
    coord[1] = y
    sx , dx = fr.divide(coord)

    assert len(sx) != 0
    assert len(dx) != 0
    assert isinstance(sx, pd.DataFrame) == True
    assert isinstance(dx, pd.DataFrame) == True

    assert all(sx["x"] == 0) == False
    assert all(dx["x"] == 0) == False
    assert all(sx["y"] <= max) == True
    assert all(dx["y"] <= max) == True
    assert sx.equals(dx) == False


##############################################################################
## TESTS FOR analysis.py


@given(dx = enp.arrays(int,(1,100)),sx = enp.arrays(int,(1,100)))
def test_area(dx,sx):
    """
    Tests:
    if returns a type Polygon and a nonnegative number
    """
    #they have to be different points in space to make a polygon!!!!
    if (a == b for (a,b) in zip(dx,sx)):
        pass
    else:
        dx = pd.DataFrame(dx)
        sx = pd.DataFrame(sx)

        pol , area = an.area(dx,sx)
        assert isinstance(pol, Polygon) == True
        assert isinstance(area, float)
        assert area >=0

@given(dim = st.integers(min_value = 10,max_value=100))
@settings(max_examples = 50)
def test_error(dim):
    """
    Tests:
    if returns an array of  positive numbers
    if it's commutative
    """

    #it shall be commutative ?
    a =  np.random.randint(600,800,dtype="uint16",size =(dim))
    b =  np.random.randint(600,800,dtype="uint16",size =(dim))
    error1 = an.error(a, b)
    error2 = an.error(b, a)
    assert all(error1 - error2 < 1e-5) == True
    assert isinstance(error1, np.ndarray) == True

@given(dim = st.integers(min_value = 10,max_value=100),N = st.integers(min_value = 10,max_value=100),l = st.integers(min_value = 10,max_value=100))
@settings(max_examples = 50)
def test_grid(dim,N,l):

    x =  np.random.randint(600,800,dtype="uint16",size =(dim))
    y =  np.random.randint(600,800,dtype="uint16",size =(dim))

    df = pd.DataFrame()
    df[0] = x
    df[1] = y
    grid = an.grid(df, N , l)

    assert isinstance(grid, pd.DataFrame) == True
    assert len(grid) == N

@given(dim = st.integers(min_value = 10,max_value=100),N = st.integers(min_value = 10,max_value=100))
@settings(max_examples = 50)
def test_necklace_points(dim,N):
    """
    Tests if:
    The output is a dataframe and if the values are int
    """
    #we want only two columns that refer to x and y coordinates
    x =  np.random.randint(600,800,dtype="uint16",size =(dim))
    y =  np.linspace(0,1200,num=dim)
    coord = pd.DataFrame()
    coord[0] = x
    coord[1] = y
    df = an.necklace_points(coord,N)
    assert isinstance(df, pd.DataFrame) == True
    assert isinstance(df.values[0][0], np.int32)
    assert len(df) == N


@given(df0 = enp.arrays(int,(1,100)),df1 = enp.arrays(int,(1,100)))
def test_velocity(df0,df1):
    """
    Tests:
    if the velocity is a DataFrame
    if the length of the velocity dataframe is equal to the length of the
    input DataFrames
    """
    df = pd.DataFrame()
    if (a == b for (a,b) in zip(df0,df1)):
        pass
    else:
        df[0] = pd.DataFrame(df0)
        df[1] = pd.DataFrame(df1)

        vel = an.velocity(df0,df1)
        assert isinstance(vel, pd.Series) == True
        assert len(vel) == len(df0)
        assert len(vel) == len(df1)

@given(df0 = enp.arrays(int,(1,100)),df1 = enp.arrays(int,(1,100)),df2 = enp.arrays(int,(1,100)))
def test_VACF(df0,df1,df2):
    """
    Tests:
    if the output is a numpy array
    if the output array elements are positive
    """
    df = pd.DataFrame()
    if (a == b for (a,b) in zip(df0,df1)):
        pass
    else:
        df[0] = pd.DataFrame(df0)
        df[1] = pd.DataFrame(df1)
        df[2] = pd.DataFrame(df2)

        msd = an.VACF(df)
        assert isinstance(msd, pd.DataFrame) == True
        assert all(msd.all() >= 0)

@given(dim = st.integers(min_value = 2,max_value=100))
@settings(max_examples = 50)
def test_MSD(dim):
    """
    Tests:
    if the output is a pandas dataframe
    if all of the output elements are positive
    """
    #we want only two columns that refer to x and y coordinates
    x = pd.DataFrame()
    for i in range(4):
        x[i] =  np.random.randint(600,800,dtype="uint16",size =(dim))

    msd = an.MSD(x)

    assert isinstance(msd, pd.DataFrame) == True
    assert all(msd.all() >= 0)


#
# def test_MSD_Sham():
#     """
#     Tests:
#     if the output is a numpy array
#     if the output array elements are positive
#     """
#     path = "Data/data_fronts/"
#     msd, pos2, y2 = an.MSD_Sham(path,side = "sx",delimiter = "\t")
#
#
#     assert isinstance(msd, np.ndarray) == True
#     assert np.isclose(msd[0], 0, rtol = 10e-5) == True
#     assert all(msd > 0) == True

@given(dim = st.integers(min_value = 10,max_value=100))
@settings(max_examples = 50)
def test_fit(dim):
    """
    Tests
    if return an array of the parameters
    """
    x = np.linspace(0,2*np.pi,num = dim)
    fit = an.fit(x)
    assert isinstance(fit, np.ndarray) == True

@given(dim = st.integers(min_value = 2,max_value=100))
@settings(max_examples = 50)
def test_cdf(dim):
    """
    Tests
    if return a numpy array
    if the last element of the output array is 1.0

    """
    im_gray = np.random.randint(0,255,dtype="uint8",size =(dim,dim))
    c = fr.cdf(im_gray)
    assert isinstance(c, np.ndarray)
    assert np.isclose(c[-1],1) == True

@given(dim = st.integers(min_value = 2,max_value=100))
@settings(max_examples = 50)
def test_hist_matching(dim):
    """
    Tests
    if return an image of the same shape

    """

    im_gray = np.random.randint(0,255,dtype="uint8",size =(dim,dim))
    ct=fr.cdf(im_gray)
    c=fr.cdf(im_gray)
    im=fr.hist_matching(c,ct,im_gray)

    assert im_gray.shape == im_gray.shape
