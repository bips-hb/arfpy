# put helper functions in this file

import numpy as np 

def bincount(x, nbins): 
    res = np.bincount(x[x >= 0], minlength=nbins+1)
    res[res == 1] = 0 # Avoid terminal nodes with just one obs
    return res/np.sum(res)

def nodeid_choice(p, a, size, replace):
    return np.random.choice(p = p, a = a, size = size, replace = replace)