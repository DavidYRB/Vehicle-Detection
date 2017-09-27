# import package 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2
from feature_extraction import *
from slide_window import *
import time
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

####### CHANGE HERE WHEN USING DIFFERENT PARAMETER ########
add = './models_parameters/YCrCb_LinearSVC/'

###########################################################

# feature extraction
para = pickle.load( open(add + 'parameters.p', 'rb'))
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

class rect_tracker():
	"""docstring for rect_tracker"""
	def __init__(self):
		self.recent_rect = []

	def add_rect(self, rects):
		self.recent_rect.append(rects)
		if len(self.recent_rect)>7:
			self.recent_rect = self.recent_rect[-6:]




print('color_space:', color_space, ' pixels per cell: ', pix_per_cell, ' orient number for hog:', orient, '\n'
	  'cells per block:', cell_per_block, ' hog channels:', hog_channel, ' spatial histogram:', spatial_size, 
      ' histogram bins:', hist_bins, '\n', 'spatial_feat: ', spatial_feat, ' hist_feat: ', hist_feat, ' hog_feat: ', hog_feat)

# import trained classifiers
model = pickle.load( open(addr+model_name, 'rb'))
svc = model["svc"]
X_scaler = model["X_scaler"]

ystart = 400
ystop = 660
scale = [1.5, 2]

def generate_video(image):
	rectangles=[]
	
	out_img_1, boxes_1 = find_cars(image, ystart, ystop, 
                               scale[0], svc, X_scaler, 
							   orient, pix_per_cell, cell_per_block, 
							   spatial_size, hist_bins)



	if len(boxes_1)>0:
		tracker.add_rect(boxes_1)

	heat = np.zeros_like(image[:,:,0]).astype(np.float)
	# Add heat to each box in box list
	for rect in tracker.recent_rect:
		heat = add_heat(heat, rect)
    
	# Apply threshold to help remove false positives
	heat = apply_thresh(heat,3)

	# Visualize the heatmap when displaying    
	heatmap = np.clip(heat, 0, 255)
	labels = label(heatmap)

	draw_img = draw_labeled_bboxes(np.copy(image), labels)

	return draw_img

tracker = rect_tracker()

input_video = 'project_video.mp4'
output_video = 'output_video_test.mp4'

clip = VideoFileClip(input_video)
video_clip = clip.fl_image(generate_video)
video_clip.write_videofile(output_video, audio=False)