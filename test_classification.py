import os
import numpy as np
import matplotlib as plt
import cv2
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

from hypothesis.extra import numpy as enp

import classification as cl
import fronts as fr
import analysis as an
filepath = "Data/"
test_image =  cv2.imread( filepath + "images/1.png")
im_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

#######################################################
##TESTS FOR classification.py



def test_create_set():
    """
    Tests if the number of images in the set is correct
    """
    assert len(os.listdir(filepath + "images/")) == 60

def test_create_modified_images():
    """
    Tests if the number of images in the set is correct
    """
    assert len(os.listdir("Results/modified_images/")) == 59

def test_adaptive_contrast_enhancement():
    """
    Tests if the function returns a np.ndarray
    """
    assert isinstance(cl.adaptive_contrast_enhancement(im_gray,(10,10)), np.ndarray) == True

def test_LBP():
    """
    Tests:
    if the output histogram is normalized
    if the length of the histogram is 10
    if the output is a np.ndarray
    """
    assert isinstance(cl.LBP(im_gray), np.ndarray) == True
    assert sum(cl.LBP(im_gray)) <= 1.0
    assert sum(cl.LBP(im_gray)) >= 0.999
    assert len(cl.LBP(im_gray)) == 10

def test_Principal_components_analysis():
    """
    Tests:
    if the function returns a dataframe
    if does not return Nan values
    """
    assert isinstance(cl.Principal_components_analysis(im_gray), pd.DataFrame) == True
    assert any(np.where(np.isnan(cl.Principal_components_analysis(im_gray)))) == False

def test_classification():
    """
    Tests:
    if the output is numpy array
    if the output image is binary (there are only 0s and 1s)
    """
    binary = cl.classification(im_gray, cl.Principal_components_analysis(im_gray))
    assert isinstance(binary,np.ndarray) == True
    assert len(np.where(binary == 0 )[1]) + len(np.where(binary == 1)[1]) == len(binary)*len(binary.T)

###############################################################################
#TESTS FOR fronts.py


def test_fronts():
    """
    Tests:
    if the output is a pandas DataFrame
    """
    fronts, _ = fr.fronts(filepath + "images/1.png")
    assert isinstance(fronts, pd.DataFrame) == True



def test_make_kernel():
    """
    Tests:
    if the output of make_kernel is a numpy matrix of 0s and 1s
    """
    struct = [0,0,0,1,0,0,0]
    binary = fr.make_kernel(struct, 1)
    assert isinstance(binary, np.matrix) == True
    assert len(np.where(binary == 0 )[1]) + len(np.where(binary == 1)[1]) == len(binary)*len(binary.T)

def test_fast_fronts():
    """
    Tests :
    if the output is a list of two pandas dataframes
    if it saves two txt file for each input image
    """
    df, im, im2 = fr.fast_fronts(im_gray, outdir = "")
    assert isinstance(df, list) == True
    assert len(df) == 2 or len(df) == 0

@given(df = enp.arrays(int,(1,100)))
def test_divide(df):
    """
    Tests:
    if return two pandas Dataframes
    if the two output dataframes are different
    """
    if (a == b for (a,b) in zip(df,df)):
        pass
    else:
        coord = pd.DataFrame(df)
        coord.columns = ["x","y"]
        sx , dx = fr.divide(coord)
        assert len(sx) != 0
        assert len(dx) != 0
        assert isinstance(sx, pd.DataFrame) == True
        assert isinstance(dx, pd.DataFrame) == True
        assert all(sx["x"] == 0) == False
        assert all(dx["x"] == 0) == False
        assert sx.equals(dx) == False

##############################################################################
## TESTS FOR analysis.py


from shapely.affinity import rotate,scale
from shapely.geometry import Polygon

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


def test_comparison():
    """
    Tests:
    if returns two np.ndarrays
    if returns arrays with positive numbers
    """
    areas, areas_hand = an.comparison()
    assert all(areas >=0 ) == True
    assert all(areas_hand >=0 ) == True

import math

def test_error():
    """
    Tests:
    if returns an array of  positive numbers
    if it's commutative
    """


    #it shall be commutative ?
    a , b = an.comparison()
    error1 = an.error(a, b)
    error2 = an.error(b, a)
    assert all(error1 - error2 < 1e-5) == True
    assert isinstance(error1, np.ndarray) == True
    assert all(error1 > 0) == True

def test_grid():
    path = "Results/labelled_images1010/fronts/divided_fronts/fronts_labelled.m.2.png.txtsx.txt"
    N = 100
    l = 1200
    grid = an.grid(path, N , l)
    assert isinstance(grid, pd.DataFrame) == True
    assert len(grid) == N

@given(df = enp.arrays(int,(1,100)))
def test_necklace_points(df):
    """
    Tests if:
    The output is a dataframe and if  the values are int
    """

    #they have to be different points in space
    df = pd.DataFrame(df)
    if (a == b for (a,b) in zip(df,df)):
        pass
    else:
        assert isinstance(df, pd.DataFrame) == True
        assert isinstance(df.values, "uint32") == True


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

@given(df = enp.arrays(int,(1,1000)))
def test_MSD(df):
    """
    Tests:
    if the output is a pandas dataframe
    if all of the output elements are positive
    """
    x = pd.DataFrame
    if (a == b for (a,b) in zip(df,df)):
        pass
    else:
        for i in range(10):
            x[i] = pd.DataFrame(df)
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

def test_fit():
    """
    Tests
    if return an array of the parameters
    """
    x = np.arange(0,100)
    fit = an.fit(x)
    assert isinstance(fit, np.ndarray) == True

def test_cdf():
    """
    Tests
    if return a numpy array
    if the last element of the output array is 1.0

    """
    c = fr.cdf(im_gray)
    assert isinstance(c, np.ndarray)
    assert np.isclose(c[-1],1) == True

def test_hist_matching():
    """
    Tests
    if return an image of the same shape

    """
    ct=fr.cdf(im_gray)
    c=fr.cdf(im_gray)
    im=fr.hist_matching(c,ct,im_gray)

    assert im_gray.shape == im.shape
