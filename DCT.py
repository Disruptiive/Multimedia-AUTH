from scipy.fftpack import *

import numpy as np
from numpy import log10

def frameDCT(Y):
    return dct(Y.reshape(-1,1), norm='ortho',axis=-1)

def iframeDCT(c): 
    return idct(c, norm='ortho',axis=-1)

def DCTpower(c):
    return 10*log10(np.square(np.abs(c)))