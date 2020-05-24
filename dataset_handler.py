# Utitlity file with functions for handling trajectory plots
#
# Author: Miguel Saavedra

import numpy as np
import pandas as pd
import os
import cv2
import glob
from operator import itemgetter

class DatasetHandler:

    def __init__(self, img_path):
        # Define number of frames
        self.num_frames = 0

        # Set up paths
        root_dir_path = os.path.dirname(os.path.realpath(__file__))
        self.image_dir = os.path.join(root_dir_path, img_path)

        # Set up data holders
        self.images = []
        self.images_rgb = []
        self.time_list = []

        self.k = np.array([[410, 0, 640],
                           [0, 410, 360],
                           [0,   0,   1]], dtype=np.float32)
        
        # Read first frame
        self.read_frame()
        
    def read_frame(self):
        self._read_image()
              
    def _read_image(self):
        for filename in glob.glob(self.image_dir + '*.jpeg'): #assuming gif
            base_name = os.path.basename(filename)
            # Split the time stamp of each image and save it in a list
            # nsecs is divided 1000000
            current_time = float(base_name.rsplit("secs_",2)[1].split('n')[0]) + float(base_name.rsplit("nsecs_",1)[1].split('.')[0])/1000000000
            # Inverted bits value, image is being wrongly readed
            self.images.append([current_time, cv2.bitwise_not(cv2.imread(filename, 0))])
            self.images_rgb.append([current_time, cv2.imread(filename)[:, :, ::-1]])
            self.time_list.append(current_time)
            self.num_frames += 1
        print("Data loaded: {0} image frames".format(int(self.num_frames)), end="\r")
        # Sort time stamps
        self._sort_timestamp()

    def _sort_timestamp(self):
        # Sorting grayscale images
        self.images = sorted(self.images, key = itemgetter(0))
        # Sorting rgb images
        self.images_rgb = sorted(self.images_rgb, key = itemgetter(0))

        # Obtain minimum to set zero time
        min_time = np.floor(np.amin(self.time_list))

        # Modify the timestamp column to seconds
        self.images = [[np.round(t - min_time, 3), img] for t, img in self.images]
        self.images_rgb = [[np.round(t - min_time, 3), img] for t, img in self.images_rgb]


