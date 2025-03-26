import numpy as np
from scipy import ndimage

def n_sites_to_avalanche_size_and_time(n_sites):
    """Returns the avalanche size from a vector containing number of sites
    failed in each iteration."""
    connected_compon, n_comp = ndimage.label(n_sites)
    avalanche_size = []
    avalanche_time = []
    for i in range(1,n_comp):
        avalanche_size.append(n_sites[connected_compon==i].sum())
        avalanche_time.append(np.argwhere(connected_compon==i).ravel()[0])
    avalanche_size = np.array(avalanche_size)
    avalanche_time = np.array(avalanche_time)
    return avalanche_size, avalanche_time