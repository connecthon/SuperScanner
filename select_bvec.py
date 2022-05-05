"""
Connecthon code 2022
Code to select the closet b vector from the bvec file to that defined by the user

Carolyn McNabb

What I want to do:
load the bvec file
define the *ideal* bvec values I want to predict
get the nearest bvec values from the bvec file and print them
"""
#select_bvec.py
import os
from scipy.io import loadmat, savemat
import numpy as np
import pandas as pd

#change directory to the data dir
os.chdir('/cubric/scratch/sapcm15/connecthon_wand/charmed_train/derivatives/dwi/sub-06400')

#load the bvec and bval data and transpose
bvec = np.loadtxt('bvec.eddy_rotated_bvec').T
print(bvec.shape)

bval = np.loadtxt('/cubric/scratch/sapcm15/connecthon_wand/charmed_train/sub-06400/dwi/314_06400_CHARMED_2.bval')
#print(bval)

#define ideal bvec coordinates from the bval = 4000 or 6000 data 

#get the rows where bval == e.g., 4000
mask_6000 = bval[:] == 6000
mask_4000 = bval[:] == 4000
mask_2400 = bval[:] == 2400
mask_1200 = bval[:] == 1200
mask_500 = bval[:] == 500
mask_200 = bval[:] == 200

#mask the bvec file to include only those vectors where bval = 4000 (uses boolean masking)
bvec_6000 = bvec[mask_6000]
bvec_4000 = bvec[mask_4000]
high_b = [bvec_4000, bvec_6000]

bvec_2400 = bvec[mask_2400]
bvec_1200 = bvec[mask_1200]
bvec_500 = bvec[mask_500]
bvec_200 = bvec[mask_200]
low_b = [bvec_200, bvec_500, bvec_1200, bvec_2400]

    
#create a loop to get this for every b=4000 and b=6000 vector
closest = np.zeros((bvec_4000.shape[0],2,4,3))
volume_index = np.zeros((bvec_4000.shape[0],2,4,1))


for hi in range(len(high_b)):
    for lo in range(len(low_b)):     
        for vec in range(high_b[hi].shape[0]):
            eucl_dist = np.sum((low_b[lo] - high_b[hi][vec])**2, axis=1)
            idx = np.argmin(eucl_dist)
            closest[vec][hi][lo] = low_b[lo][idx]
            #get the index based on the whole data
            volume_index[vec][hi][lo] = np.where(bvec[:] == closest[vec][hi][lo])[0][0]
            
print(volume_index) 
index_array = np.array(volume_index)
#np.savetxt('./volume_index', index_array, delimiter = ',')

savemat('./volume_index.mat', {'indexes': index_array})


# change directory back to hackathon dir
os.chdir('../../../../hackathon_code')

