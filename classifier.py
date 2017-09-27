# import package 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2
from feature_extraction import *
from slide_window import *
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# When change the color space or any other parameters, it has to be changed in Training_data.py, 
# Since the way that training data features are generated has to be applied to test data.


####### CHANGE HERE WHEN USING DIFFERENT PARAMETER ########
add = './models_parameters/YCrCb_LinearSVC/'

###########################################################



para = pickle.load( open(add+'parameters.p', 'rb'))
addr = para["addr"]
model_name = para["model_name"]

color_space = para["color_space"] # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = para["orient"]  # HOG orientations
pix_per_cell = para["pix_per_cell"] # HOG pixels per cell
cell_per_block = para["cell_per_block"] # HOG cells per block
hog_channel = para["hog_channel"] # Can be 0, 1, 2, or "ALL"
spatial_size = para["spatial_size"] # Spatial binning dimensions
hist_bins = para["hist_bins"]    # Number of histogram bins
spatial_feat = para["spatial_feat"] # Spatial features on or off
hist_feat = para["hist_feat"]# Histogram features on or off
hog_feat = para["hog_feat"] # HOG features on or off

print('color_space:', color_space, ' pixels per cell: ', pix_per_cell, ' orient number for hog:', orient, '\n'
      'cells per block:', cell_per_block, ' hog channels:', hog_channel, ' spatial histogram:', spatial_size, 
      ' histogram bins:', hist_bins, '\n', 'spatial_feat: ', spatial_feat, ' hist_feat: ', hist_feat, ' hog_feat: ', hog_feat, '\n',
      'address:', addr, 'model_name:', model_name)

# Import training and testing dataset
dataset = pickle.load( open(addr + 'dataset.p', 'rb'))
X = dataset["X"]
y = dataset["y"]
# create a classifier

X_scaler = StandardScaler().fit(X)
scaled_x = X_scaler.transform(X)
print(scaled_x.shape)
#split training and testing set
rand_st = np.random.randint(0,100)
X_train, X_test, y_train, y_test = train_test_split(scaled_x, y, test_size=0.2, random_state=rand_st)

print("Training set has {0} insatances and each insatance has {1} features".format(X_train.shape[0], X_train.shape[1]))
print("Training labels in total: {}".format(len(y_train)))
print("Test set has {0} insatances and each insatance has {1} features".format(X_test.shape[0], X_test.shape[1]))
print("Test labels in total: {}".format(len(y_test)))


print("Start training...")
svc = LinearSVC()
t = time.time()
svc.fit(X_train, y_train)
train_time = time.time()-t
print("Training fisished")

pred = svc.predict(X_test)
accu = accuracy_score(y_test, pred)
print("Training time is {0}, test set results is {1:5.3f}".format(train_time, accu))

# saves the model to disk
print("Saving model...")
write_obj = {"X_scaler": X_scaler, "svc": svc}
with open(addr + model_name, 'wb') as file:
    pickle.dump(write_obj, file)
print("Model saved")





