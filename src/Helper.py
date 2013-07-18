# -*- coding: UTF-8 -*-
'''
This moduel contains auxiliary functions for all other modules
'''
from matplotlib.pyplot import imshow
from matplotlib.cm import gray
from numpy import vstack

def transform1Dto2D(arr, rows, cols):
    '''
    Helper function that transform a 1D array into a rectangular 2D array.

    :param arr: 1D array
    :param rows: number of rows
    :param cols: number of cols

    :returns: 2D array
    '''
    return vstack([arr[cols * x:cols * (x + 1)] for x in xrange(rows)])

def display1DArrAsGrayImg(arr, pxlHeight, pxlWidth):
    '''
    This function will transform a 1-D array into rectangular array by stacking every pxlWidth numbers, and display the resulting 2-D array as grayscale image.
    '''
    arr2D = vstack([arr[pxlWidth * x:pxlWidth * (x + 1)] for x in xrange(pxlHeight)])
    imshow(arr2D, cmap=gray)

def displayLearnedEncoding(arrTuple):
    pass