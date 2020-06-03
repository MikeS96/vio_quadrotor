# Class to handle images
#
# Author: Miguel Saavedra

import numpy as np
import pandas as pd
import os
import cv2
import glob
from operator import itemgetter

class DatasetHandler:

    def __init__(self, img_path, depth_path, imu_df):
        """
        Class to handle Grayscale, RGB and depth maps in a comprehensive way with lists.

        :param img_path: Images path
        :param depth_path: Depth maps path
        :param imu_df: IMU dataframe
        """

        # Define number of frames
        self.num_frames = 0
        self.depth_frames = 0

        # Define reference dataframe to sync timestamps
        self.imu = imu_df

        # Set up paths
        root_dir_path = os.path.dirname(os.path.realpath(__file__))
        self.image_dir = os.path.join(root_dir_path, img_path)
        self.depth_dir = os.path.join(root_dir_path, depth_path)

        # Set up data holders
        self.images = []
        self.images_rgb = []
        self.depth_maps = []
        self.time_list = []

        # Camera intrinsic
        self.k = np.array([[410, 0, 640],
                           [0, 410, 360],
                           [0,   0,   1]], dtype=np.float32)

        # Read first frame
        self.read_frame()
        
    def read_frame(self):
        """ Read Image and Depth maps """
        self._read_image()
        self._read_depth()
              
    def _read_image(self):
        """ Read RGB and Grayscale images with their timestamp and store
        them in two different lists where images[0] return a tuple. images[0][0] 
        is the current timestamp and images[0][1] is the image. """
        for filename in glob.glob(self.image_dir + '*.jpeg'): 
            base_name = os.path.basename(filename)
            # Split the time stamp of each image and save it in a list
            # nsecs is divided 1000000000
            current_time = float(base_name.rsplit("secs_",2)[1].split('n')[0]) + float(base_name.rsplit("nsecs_",1)[1].split('.')[0])/1000000000
            # Save Grayscale and RGB images in a list with their timestamp
            self.images.append([current_time, cv2.imread(filename, 0)])
            self.images_rgb.append([current_time, cv2.imread(filename)[:, :, ::-1]])
            self.time_list.append(current_time)
            self.num_frames += 1
        print("Data loaded: {0} image frames".format(int(self.num_frames)), end="\r")


    def _sort_timestamp(self):
        """ Sort the images and depth maps timestamps """
        # Sorting grayscale images
        self.images = sorted(self.images, key = itemgetter(0))
        # Sorting rgb images
        self.images_rgb = sorted(self.images_rgb, key = itemgetter(0))
        # Sorting depth maps
        self.depth_maps = sorted(self.depth_maps, key = itemgetter(0))

        # Obtain minimum to set zero time
        min_time = np.floor(np.amin(self.time_list))

        # Modify the timestamp column to seconds
        self.images = [[np.round(t - min_time, 2), img] for t, img in self.images]
        self.images_rgb = [[np.round(t - min_time, 2), img] for t, img in self.images_rgb]
        self.depth_maps = [[np.round(t - min_time, 2), img] for t, img in self.depth_maps]
        # Sync timestamps with IMU
        self._sync_timestamp()

    def _sync_timestamp(self):
        """ Synchronize the images and depth maps timestamps with the IMU timestamps """
        # Transform IMU dataframe into a numpy array
        list_of_inertias = self.imu.values.tolist()
        imu_array = np.array(list_of_inertias).T
        # Find the closest timestamp in the IMU array and replace it in the image time-stamp
        for i in range(self.num_frames):
            minArg = np.argmin(np.abs(imu_array[0,:] - self.images[i][0]))
            self.images[i][0] = imu_array[0, minArg]
            self.images_rgb[i][0] = imu_array[0, minArg]
            self.depth_maps[i][0] = imu_array[0, minArg]


    def _read_depth(self):
        """ Read Depth maps with their timestamp and store
        them in a list where depth_maps[0] return a tuple. depth_maps[0][0] 
        is the current timestamp and depth_maps[0][1] is the depth map. """
        for filename in glob.glob(self.depth_dir + '*.dat'): 
            base_name = os.path.basename(filename)
            # Split the time stamp of each depth map and save it in a list
            # nsecs is divided 1000000000
            current_time = float(base_name.rsplit("secs_",2)[1].split('n')[0]) + float(base_name.rsplit("nsecs_",1)[1].split('.')[0])/1000000000
            # Read depth map 
            depth = np.loadtxt(
                filename,
                delimiter=' ',
                dtype=np.float64)
            # Store depth map in a class' attribute
            self.depth_maps.append([current_time, depth])
            self.depth_frames += 1
        print("Data loaded: {0} depth frames".format(int(self.depth_frames)), end="\r")
        # Sort time stamps
        self._sort_timestamp()
