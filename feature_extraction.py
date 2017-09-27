# This file is used for the defination of feature extraction functions and window slicing function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
import cv2

def bin_spatial(img, size=(32,32)):
	ch_0 = cv2.resize(img[:, :, 0], size).ravel()
	ch_1 = cv2.resize(img[:, :, 1], size).ravel()
	ch_2 = cv2.resize(img[:, :, 2], size).ravel()
	
	return np.hstack((ch_0,ch_1,ch_2))


def color_hist(img, nbins=32, bins_range = (0,256)):
	ch0_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
	ch1_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
	ch2_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)

	clc_feature = np.concatenate((ch0_hist[0], ch1_hist[0], ch2_hist[0]))
	return clc_feature


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
	if vis == True:
		features, hog_img = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
								cells_per_block=(cell_per_block,cell_per_block), transform_sqrt=True, 
								visualise=vis, feature_vector=feature_vec)
		return features, hog_img
	else:
		features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
		   			  cells_per_block=(cell_per_block,cell_per_block), transform_sqrt=True, 
					  visualise=vis, feature_vector=feature_vec)

		return features


def feature_extraction(names, color_space='RGB', spatial_size=(32,32), hist_bins=32, orient=9, pix_per_cell=8,
					   cell_per_block=2, hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True):
	features = []
	for name in names:
		image = mpimg.imread(name)
		image_features = []

		# convert color space
		if color_space != 'RGB':
			if color_space == 'HSV':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
			elif color_space == 'LUV':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
			elif color_space == 'HLS':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
			elif color_space == 'YUV':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
			elif color_space == 'YCrCb':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
		else: feature_image = np.copy(image) 
        # determine wheather use spatial features or not
		if spatial_feat==True:
			spatial_features = bin_spatial(feature_image, spatial_size)
			image_features.append(spatial_features)
        # determine wheather use color histogram features or not
		if hist_feat == True:
			hist_features = color_hist(feature_image, hist_bins)
			image_features.append(hist_features)
        # determine wheather use HOG features or not
		if hog_feat == True:
			if hog_channel == 'ALL':
				hog_features = []
				for i in range(feature_image.shape[2]):
					hog_features.append(get_hog_features(feature_image[:,:,i], orient, pix_per_cell, 
        												 cell_per_block, vis=False, feature_vec=True))
				hog_features = np.ravel(hog_features)
			else:
				hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, pix_per_cell, 
        												 cell_per_block, vis=False, feature_vec=True)
			image_features.append(hog_features)
       	# append image vector to features, image features are converted to a vector with np.concatenate().
		features.append(np.concatenate(image_features))
	print(color_space)

	print("Spatial features dimension: {0}. histogram features dimension: {1}. \
			HOG features dimension: {2}. Instance's features dimension: {3}.".\
			format(np.asarray(spatial_features).shape[0], np.asarray(hist_features).shape[0], 
				   np.asarray(hog_features).shape[0], np.asarray(features).shape))
	return features


def test_img_feature(win_img, color_space='RGB', spatial_size=(32,32), hist_bins=32, orient=9, pix_per_cell=8,
					   cell_per_block=2, hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True):
	image_features = []
	if color_space != 'RGB':
			if color_space == 'HSV':
				feature_image = cv2.cvtColor(win_img, cv2.COLOR_RGB2HSV)
			elif color_space == 'LUV':
				feature_image = cv2.cvtColor(win_img, cv2.COLOR_RGB2LUV)
			elif color_space == 'HLS':
				feature_image = cv2.cvtColor(win_img, cv2.COLOR_RGB2HLS)
			elif color_space == 'YUV':
				feature_image = cv2.cvtColor(win_img, cv2.COLOR_RGB2YUV)
			elif color_space == 'YCrCb':
				feature_image = cv2.cvtColor(win_img, cv2.COLOR_RGB2YCrCb)
	else: feature_image = np.copy(win_img) 
       # determine wheather use spatial features or not
	if spatial_feat==True:
		spatial_features = bin_spatial(feature_image, spatial_size)
		image_features.append(spatial_features)
    # determine wheather use color histogram features or not
	if hist_feat == True:
		hist_features = color_hist(feature_image, hist_bins)
		image_features.append(hist_features)
    # determine wheather use HOG features or not
	if hog_feat == True:
		if hog_channel == 'ALL':
			hog_features = []
			for i in range(feature_image.shape[2]):

				hog_features.append(get_hog_features(feature_image[:,:,i], orient, pix_per_cell, 
       												 cell_per_block, vis=False, feature_vec=True))
			hog_features = np.ravel(hog_features)
		else:
			hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, pix_per_cell, 
       												 cell_per_block, vis=False, feature_vec=True)
		image_features.append(hog_features)
      	# append image vector to features, image features are converted to a vector with np.concatenate().
	feature = np.concatenate(image_features)

	return feature