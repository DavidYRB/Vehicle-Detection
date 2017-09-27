import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2
import pickle
from feature_extraction import *
import time
# import images path
# store the address of image of cars and notcars into two arrays and make them equal length
car_path = './vehicles'
car_sub = ['/GTI_Far', '/GTI_Left', '/GTI_MiddleClose', '/GTI_Right', '/KITTI_extracted']
not_car_path = './non-vehicles'
not_car_sub = ['/GTI', '/Extras']

cars = []
notcars = []
car_len = 0
notcar_len = 0
for i in range(len(car_sub)):
    imgs = glob.glob(car_path+car_sub[i] + '/*.png')
    cars.append(imgs)

for i in range(len(not_car_sub)):
    imgs = glob.glob(not_car_path + not_car_sub[i] + '/*.png')
    #notcar_len += len(imgs)
    notcars.append(imgs)
cars = np.hstack(cars)
notcars = np.hstack(notcars)

print("The number of cars is ", cars.shape[0])
print("The number of notcars is ", notcars.shape[0])

# feature extraction
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
hog_channel = 'ALL'
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

####### CHANGE HERE WHEN USING DIFFERENT PARAMETER ########
addr = './models_parameters/YCrCb_LinearSVC/'
model_name = 'model_YCrCb_Linear.p'
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
###########################################################


print('color_space:', color_space, ' pixels per cell: ', pix_per_cell, ' orient number for hog:', orient, '\n'
      'cells per block:', cell_per_block, ' hog channels:', hog_channel, ' spatial histogram:', spatial_size, 
      ' histogram bins:', hist_bins, '\n', 'spatial_feat: ', spatial_feat, ' hist_feat: ', hist_feat, ' hog_feat: ', hog_feat)

t_start = time.time()
car_features = feature_extraction(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
t_car = time.time() 

notcar_features = feature_extraction(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

t_notcar = time.time()  
#### Train the model ###
print("time for feature extraction of cars and noncars are: ", t_car-t_start, t_notcar-t_start)
# get X features ready
X = np.vstack((car_features, notcar_features)).astype(np.float64)

# get y labels ready
y = np.concatenate((np.ones(len(car_features)),np.zeros(len(notcar_features))))

print("feature dimension is: ", X.shape)
print("label dimension is: ", y.shape)

write_obj = {"X": X, "y": y}
with open(addr + 'dataset.p', 'wb') as file:
    pickle.dump(write_obj, file)

para_obj = {"color_space": color_space, "orient": orient, "pix_per_cell": pix_per_cell, "cell_per_block": cell_per_block, "hog_channel": hog_channel,
            "spatial_size":spatial_size, "hist_bins": hist_bins, "spatial_feat": spatial_feat, "hog_feat": hog_feat, "hist_feat":hist_feat, "addr": addr, "model_name": model_name }
with open(addr+'parameters.p', 'wb') as file:
    pickle.dump(para_obj, file)

print("program execution time: ", time.time()-t_start)
