# Utitlity file with functions for handling trajectory plots
#
# Author: Miguel Saavedra

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import pandas as pd
import cv2

from rotations import Quaternion, skew_symmetric

def visualize_trajectory(trajectory, title = "Vehicle's trajectory"):
    """
    Plot the vehicle's trajectory

    :param trajectory: Numpy array (3 x M) where M is the number of samples
        with the trayectory of the vehicle.
    :param title: Name of the plot
    """

    # lists for x, y and z values
    locX = list(trajectory[0,:])
    locY = list(trajectory[1,:])
    locZ = list(trajectory[2,:])
    
    # Axis limits
    maxX = np.amax(locX) + 1
    minX = np.amin(locX) - 1
    maxY = np.amax(locY) + 1
    minY = np.amin(locY) - 1 
    maxZ = np.amax(locZ) + 1
    minZ = np.amin(locZ) - 1

    # Set styles
    mpl.rc("figure", facecolor="white")
    plt.style.use("seaborn-whitegrid")

    # Plot the figure
    fig = plt.figure(figsize=(8, 6), dpi=100)
    gspec = gridspec.GridSpec(2, 2)
    ZY_plt = plt.subplot(gspec[1,1])
    YX_plt = plt.subplot(gspec[0,1])
    ZX_plt = plt.subplot(gspec[1,0])
    D3_plt = plt.subplot(gspec[0,0], projection='3d')
    
    toffset = 1.0
    
    # Actual trajectory plotting ZX
    ZX_plt.set_title("Trajectory (X, Z)", y=toffset)
    ZX_plt.plot(locX, locZ, "--", label="Trajectory", zorder=1, linewidth=1.5, markersize=2)
    ZX_plt.set_xlabel("X [m]")
    ZX_plt.set_ylabel("Z [m]")
    # Plot vehicle initial location
    ZX_plt.scatter([0], [0], s=8, c="green", label="Start location", zorder=2)
    ZX_plt.scatter(locX[-1], locZ[-1], s=8, c="red", label="End location", zorder=2)
    ZX_plt.set_xlim([minX, maxX])
    ZX_plt.set_ylim([minZ, maxZ])
    ZX_plt.legend(bbox_to_anchor=(1.05, 1.0), loc=3, title="Legend", borderaxespad=0., fontsize="medium", frameon=True)
        
    # Plot ZY
    ZY_plt.set_title("Trajectory (Y, Z)", y=toffset)
    ZY_plt.set_xlabel("Y [m]")
    ZY_plt.plot(locY, locZ, "--", linewidth=1.5, markersize=2, zorder=1)
    ZY_plt.scatter([0], [0], s=8, c="green", label="Start location", zorder=2)
    ZY_plt.scatter(locY[-1], locZ[-1], s=8, c="red", label="End location", zorder=2)
    ZY_plt.set_xlim([minY, maxY])
    ZY_plt.set_ylim([minZ, maxZ])
    
    # Plot YX
    YX_plt.set_title("Trajectory (Y X)", y=toffset)
    YX_plt.set_ylabel("X [m]")
    YX_plt.set_xlabel("Y [m]")
    YX_plt.plot(locY, locX, "--", linewidth=1.5, markersize=2, zorder=1)
    YX_plt.scatter([0], [0], s=8, c="green", label="Start location", zorder=2)
    YX_plt.scatter(locY[-1], locX[-1], s=8, c="red", label="End location", zorder=2)
    YX_plt.set_xlim([minY, maxY])
    YX_plt.set_ylim([minX, maxX])

    # Plot 3D
    D3_plt.set_title("3D trajectory", y = 1.1)
    D3_plt.plot3D(xs = locX, ys = locY, zs = locZ, zorder=0)
    D3_plt.scatter(0, 0, 0, s=8, c="green", zorder=1)
    D3_plt.scatter(locX[-1], locY[-1], locZ[-1], s=8, c="red", zorder=1)
    D3_plt.set_xlim3d(minX, maxX)
    D3_plt.set_ylim3d(minY, maxY)
    D3_plt.set_zlim3d(minZ, maxZ)
    D3_plt.tick_params(direction='out', pad=-2)
    D3_plt.set_xlabel("X [m]", labelpad=0)
    D3_plt.set_ylabel("Y [m]", labelpad=0)
    D3_plt.set_zlabel("Z [m]", labelpad=-5)
    
    # Plotting the result
    fig.suptitle(title, fontsize=16, y = 1.05)
    D3_plt.view_init(35, azim=45)
    plt.tight_layout()
    plt.show()

def visualize_angles(rotations, title = "Vehicle's euler angles"):
    """
    Plot the vehicle's orientation (Euler angles)

    :param rotations: Numpy array (4 x M) where M is the number of samples. 
        This variable has a Quaternion at each time-step.
    :param title: Name of the plot
    """

    # lists to unpack roll, pitch, yaw angles
    roll = []
    pitch = []
    yaw = []
    x_axis = np.arange(rotations.shape[1])

    # Transform Quaternion into Euler angle representation
    for i in range(0, rotations.shape[1]):
        current_rot = rotations[:, i]
        q = Quaternion(*current_rot).to_euler()
        
        roll.append(q.item(0))
        pitch.append(q.item(1))
        yaw.append(q.item(2))
        
    # Axis limits for Roll, Pitch and Yaw
    maxR = np.amax(roll) + 0.1
    minR = np.amin(roll) - 0.1
    maxP = np.amax(pitch) + 0.1
    minP = np.amin(pitch) - 0.1
    maxY = np.amax(yaw) + 0.1
    minY = np.amin(yaw) - 0.1
    maxX = rotations.shape[1] + 10
    minX = -5

    # Set styles
    mpl.rc("figure", facecolor="white")
    plt.style.use("seaborn-whitegrid")

    # Plot the figure
    fig = plt.figure(figsize=(8, 6), dpi=100)
    gspec = gridspec.GridSpec(3, 1)
    R_plt = plt.subplot(gspec[0,0])
    P_plt = plt.subplot(gspec[1,0])
    Y_plt = plt.subplot(gspec[2,0])
    
    toffset = 1.0
    
    # Roll angle
    R_plt.set_title(r'Roll angle $\theta$', y=toffset)
    R_plt.plot(x_axis, roll, "-", label="Angle", zorder=1, linewidth=1.5, markersize=2)
    R_plt.axes.xaxis.set_ticklabels([])
    R_plt.set_ylabel("Roll [rads]")
    # Plot vehicle initial orientation
    R_plt.scatter([0], [0], s=8, c="green", label="Start location", zorder=2)
    R_plt.scatter(x_axis[-1], roll[-1], s=8, c="red", label="End location", zorder=2)
    R_plt.set_xlim([minX, maxX])
    R_plt.set_ylim([minR, maxR])
        
    # Pitch angle
    P_plt.set_title(r'Pitch angle $\beta$', y=toffset)
    P_plt.set_ylabel("Roll [rads]")
    P_plt.axes.xaxis.set_ticklabels([])
    P_plt.plot(x_axis, pitch, "-", linewidth=1.5, label="Angle", markersize=2, zorder=1)
    P_plt.scatter([0], [0], s=8, c="green", label="Start location", zorder=2)
    P_plt.scatter(x_axis[-1], pitch[-1], s=8, c="red", label="End location", zorder=2)
    P_plt.set_xlim([minX, maxX])
    P_plt.set_ylim([minP, maxP])
    
    # Yaw angle
    Y_plt.set_title(r'Yaw angle $\gamma$', y=toffset)
    Y_plt.set_ylabel("Yaw [rads]")
    Y_plt.axes.xaxis.set_ticklabels([])
    Y_plt.plot(x_axis, yaw, "-", linewidth=1.5, label="Angle", markersize=2, zorder=1)
    Y_plt.scatter([0], [0], s=8, c="green", label="Start location", zorder=2)
    Y_plt.scatter(x_axis[-1], yaw[-1], s=8, c="red", label="End location", zorder=2)
    Y_plt.set_xlim([minX, maxX])
    Y_plt.set_ylim([minY, maxY])
    Y_plt.legend(loc=4, title="Legend", borderaxespad=0., fontsize="medium", frameon=True)
    
    # Plotting the result
    fig.suptitle(title, fontsize=16, y = 1.05)
    plt.tight_layout()
    plt.show()

def visualize_camera_movement(image1, image1_points, image2, image2_points, is_show_img_after_move=False):
    """
    Plot the camera movement between two consecutive image frames

    :param image1: First image at time stamp t
    :param image1_points: Feature vector for the first image
    :param image2: First image at time stamp t + 1
    :param image2_points: Feature vectir for the second image
    :param is_show_img_after_move: Bool variable to plot movement or not
    """
    image1 = image1.copy()
    image2 = image2.copy()
    
    for i in range(0, len(image1_points)):
        # Coordinates of a point on t frame
        p1 = (int(image1_points[i][0]), int(image1_points[i][1]))
        # Coordinates of the same point on t+1 frame
        p2 = (int(image2_points[i][0]), int(image2_points[i][1]))

        cv2.circle(image1, p1, 5, (0, 255, 0), 1)
        cv2.arrowedLine(image1, p1, p2, (0, 255, 0), 1)
        cv2.circle(image1, p2, 5, (255, 0, 0), 1)

        if is_show_img_after_move:
            cv2.circle(image2, p2, 5, (255, 0, 0), 1)
    
    if is_show_img_after_move: 
        return image2
    else:
        return image1

def compare_3d(ground_truth, trajectory, title):
	"""
	Plot the vehicle's trajectory in 3D space

	:param ground_truth: Numpy array (3 x M) where M is the number of samples
	    with the ground truth trayectory of the vehicle.
	:param trajectory: Numpy array (3 x M) where M is the number of samples
	    with the estimated trajectory of the vehicle.
	:param title: Name of the plot
	"""

	# Axis limits
	maxX = np.amax(trajectory[0,:]) + 5
	minX = np.amin(trajectory[0,:]) - 5
	maxY = np.amax(trajectory[1,:]) + 5
	minY = np.amin(trajectory[1,:]) - 5 
	maxZ = np.amax(trajectory[2,:]) + 5
	minZ = np.amin(trajectory[2,:]) - 5

	est_traj_fig = plt.figure()
	ax = est_traj_fig.add_subplot(111, projection='3d')
	ax.plot(ground_truth[0,:], ground_truth[1,:], ground_truth[2,:], "-", label='Ground Truth')
	ax.plot(trajectory[0,:], trajectory[1,:], trajectory[2,:], "-", label='Estimated')
	ax.set_xlabel('X [m]')
	ax.set_ylabel('Y [m]')
	ax.set_zlabel('Z [m]')
	ax.set_title(title, y = 1.0)
	ax.legend()
	ax.set_xlim(minX, maxX)
	ax.set_ylim(minY, maxY)
	ax.set_zlim(minZ, maxZ)
	plt.tight_layout()
	plt.show()

def compare_2d_trajectory(ground_truth, trajectory, title = "VO vs Ground Truth Trajectory"):
    """
    Plot the comparison between the vehicle's ground truth trajectory and the estimated trajectory

    :param ground_truth: Numpy array (3 x M) where M is the number of samples
        with the ground truth trayectory of the vehicle.
    :param trajectory: Numpy array (3 x M) where M is the number of samples
        with the estimated trajectory of the vehicle.
    :param title: Name of the plot
    """

    # lists for x, y and z values estimation
    locX = list(trajectory[0,:])
    locY = list(trajectory[1,:])
    locZ = list(trajectory[2,:])
    # lists for x, y and z values ground truth
    locX_gt = list(ground_truth[0,:])
    locY_gt = list(ground_truth[1,:])
    locZ_gt = list(ground_truth[2,:])
    
    # Axis limits
    maxX = np.amax(locX) + 5
    minX = np.amin(locX) - 5
    maxY = np.amax(locY) + 5
    minY = np.amin(locY) - 5 
    maxZ = np.amax(locZ) + 5
    minZ = np.amin(locZ) - 5

    # Set styles
    mpl.rc("figure", facecolor="white")
    plt.style.use("seaborn-whitegrid")

    # Plot the figure
    fig = plt.figure(figsize=(8, 6), dpi=100)
    gspec = gridspec.GridSpec(2, 2)
    ZY_plt = plt.subplot(gspec[1,1])
    YX_plt = plt.subplot(gspec[0,1])
    ZX_plt = plt.subplot(gspec[1,0])
    D3_plt = plt.subplot(gspec[0,0], projection='3d')
    
    toffset = 1.0
    
    # Actual trajectory plotting ZX
    ZX_plt.set_title("Trajectory (X, Z)", y=toffset)
    ZX_plt.plot(locX_gt, locZ_gt, "--", label="Trajectory GT", zorder=1, linewidth=1.5, markersize=2)
    ZX_plt.plot(locX, locZ, "--", label="Trajectory Estimated", zorder=1, linewidth=1.5, markersize=2)
    ZX_plt.set_xlabel("X [m]")
    ZX_plt.set_ylabel("Z [m]")
    # Plot initial position
    ZX_plt.scatter([0], [0], s=8, c="green", label="Start location", zorder=2)
    ZX_plt.scatter(locX[-1], locZ[-1], s=8, c="purple", label="End location Estimation", zorder=2)
    ZX_plt.scatter(locX_gt[-1], locZ_gt[-1], s=8, c="red", label="End location GT", zorder=2)
    ZX_plt.set_xlim([minX, maxX])
    ZX_plt.set_ylim([minZ, maxZ])
    ZX_plt.legend(bbox_to_anchor=(1.05, 1.0), loc=3, title="Legend", borderaxespad=0., fontsize="medium", frameon=True)
        
    # Plot ZY
    ZY_plt.set_title("Trajectory (Y, Z)", y=toffset)
    ZY_plt.set_xlabel("Y [m]")
    ZY_plt.plot(locY_gt, locZ_gt, "--", linewidth=1.5, markersize=2, zorder=1)
    ZY_plt.plot(locY, locZ, "--", linewidth=1.5, markersize=2, zorder=1)
    ZY_plt.scatter([0], [0], s=8, c="green", label="Start location", zorder=2)
    ZY_plt.scatter(locY[-1], locZ[-1], s=8, c="purple", label="End location Estimation", zorder=2)
    ZY_plt.scatter(locY_gt[-1], locZ_gt[-1], s=8, c="red", label="End location GT", zorder=2)
    ZY_plt.set_xlim([minY, maxY])
    ZY_plt.set_ylim([minZ, maxZ])
    
    # Plot YX
    YX_plt.set_title("Trajectory (Y X)", y=toffset)
    YX_plt.set_ylabel("X [m]")
    YX_plt.set_xlabel("Y [m]")
    YX_plt.plot(locY_gt, locX_gt, "--", linewidth=1.5, markersize=2, zorder=1)
    YX_plt.plot(locY, locX, "--", linewidth=1.5, markersize=2, zorder=1)
    YX_plt.scatter([0], [0], s=8, c="green", label="Start location", zorder=2)
    YX_plt.scatter(locY[-1], locX[-1], s=8, c="purple", label="End location Estimation", zorder=2)
    YX_plt.scatter(locY_gt[-1], locX_gt[-1], s=8, c="red", label="End location GT", zorder=2)
    YX_plt.set_xlim([minY, maxY])
    YX_plt.set_ylim([minX, maxX])

    # Plot 3D
    D3_plt.set_title("3D trajectory", y = 1.1)
    D3_plt.plot3D(xs = locX_gt, ys = locY_gt, zs = locZ_gt, zorder=0)
    D3_plt.plot3D(xs = locX, ys = locY, zs = locZ, zorder=0)
    D3_plt.scatter(0, 0, 0, s=8, c="green", zorder=1)
    D3_plt.scatter(locX[-1], locY[-1], locZ[-1], s=8, c="purple", zorder=1)
    D3_plt.scatter(locX_gt[-1], locY_gt[-1], locZ_gt[-1], s=8, c="red", zorder=1)
    D3_plt.set_xlim3d(minX, maxX)
    D3_plt.set_ylim3d(minY, maxY)
    D3_plt.set_zlim3d(minZ, maxZ)
    D3_plt.tick_params(direction='out', pad=-2)
    D3_plt.set_xlabel("X [m]", labelpad=0)
    D3_plt.set_ylabel("Y [m]", labelpad=0)
    D3_plt.set_zlabel("Z [m]", labelpad=-5)
    
    # Plotting the result
    fig.suptitle(title, fontsize=16, y = 1.05)
    D3_plt.view_init(35, azim=45)
    plt.tight_layout()
    plt.show()

def compare_2d_angles(ground_truth, rotations, title = "VO vs Ground Truth angles"):
    """
    Plot the comparison between the vehicle's ground truth orientation and the estimated orientation

    :param ground_truth: Numpy array (4 x M) where M is the number of samples. 
        This variable has a Quaternion at each time-step and this is the ground truth.
    :param trajectory: Numpy array (4 x M) where M is the number of samples. 
        This variable has a Quaternion at each time-step and this is the estimated orientation.
    :param title: Name of the plot
    """

    # list to unpack roll, pitch, yaw angles from the ground truth
    roll = []
    pitch = []
    yaw = []
    # Unpack roll pitch and yaw estimations
    roll_es = []
    pitch_es = []
    yaw_es = []
    # X axis
    x_axis = np.arange(ground_truth.shape[1])
    x_axis_es = np.arange(rotations.shape[1])
    
    for i in range(0, ground_truth.shape[1]):
        current_rot = ground_truth[:, i]
        q = Quaternion(*current_rot).to_euler()
        
        roll.append(q.item(0))
        pitch.append(q.item(1))
        yaw.append(q.item(2))

    for i in range(0, rotations.shape[1]):
        current_rot = rotations[:, i]
        q = Quaternion(*current_rot).to_euler()
        
        roll_es.append(q.item(0))
        pitch_es.append(q.item(1))
        yaw_es.append(q.item(2))
        
    # Axis limits for Roll, Pitch and Yaw Ground Truth
    maxR = np.amax(roll) + 0.1
    minR = np.amin(roll) - 0.1
    maxP = np.amax(pitch) + 0.1
    minP = np.amin(pitch) - 0.1
    maxY = np.amax(yaw) + 0.1
    minY = np.amin(yaw) - 0.1
    maxX = ground_truth.shape[1] + 10
    minX = -5

    # Axis limits for Roll, Pitch and Yaw estimated
    maxR_es = np.amax(roll_es) + 0.1
    minR_es = np.amin(roll_es) - 0.1
    maxP_es = np.amax(pitch_es) + 0.1
    minP_es = np.amin(pitch_es) - 0.1
    maxY_es = np.amax(yaw_es) + 0.1
    minY_es = np.amin(yaw_es) - 0.1
    maxX_es = rotations.shape[1] + 10
    minX_es = -5

    # Set styles
    mpl.rc("figure", facecolor="white")
    plt.style.use("seaborn-whitegrid")

    # Plot the figure
    fig = plt.figure(figsize=(8, 6), dpi=100)
    gspec = gridspec.GridSpec(3, 2)
    R_plt = plt.subplot(gspec[0,0])
    P_plt = plt.subplot(gspec[1,0])
    Y_plt = plt.subplot(gspec[2,0])
    R_plt_es = plt.subplot(gspec[0,1])
    P_plt_es = plt.subplot(gspec[1,1])
    Y_plt_es = plt.subplot(gspec[2,1])
    
    toffset = 1.0
    
    # Roll Orientation
    R_plt.set_title(r'Roll angle GT $\theta$', y=toffset)
    R_plt.plot(x_axis, roll, "-", label="Angle GT", zorder=1, linewidth=1.5, markersize=2)
    R_plt.set_ylabel("Roll [rads]")
    # Plot initial location
    R_plt.scatter([0], [0], s=8, c="green", label="Start location", zorder=2)
    R_plt.scatter(x_axis[-1], roll[-1], s=8, c="red", label="End location", zorder=2)
    R_plt.set_xlim([minX, maxX])
    R_plt.set_ylim([minR, maxR])

    # Roll Orientation estimated
    R_plt_es.set_title(r'Roll angle Estimated $\theta$', y=toffset)
    R_plt_es.plot(x_axis_es, roll_es, "-", c="orange", label="Angle Estimated", zorder=1, linewidth=1.5, markersize=2)
    R_plt_es.axes.xaxis.set_ticklabels([])
    # Plot initial location
    R_plt_es.scatter([0], [0], s=8, c="blue", label="Start location", zorder=2)
    R_plt_es.scatter(x_axis_es[-1], roll_es[-1], s=8, c="purple", label="End location", zorder=2)
    R_plt_es.set_xlim([minX_es, maxX_es])
    R_plt_es.set_ylim([minR_es, maxR_es])
        
    # Pitch Orientation
    P_plt.set_title(r'Pitch angle GT $\beta$', y=toffset)
    P_plt.set_ylabel("Roll [rads]")
    P_plt.axes.xaxis.set_ticklabels([])
    P_plt.plot(x_axis, pitch, "-", linewidth=1.5, label="Angle GT", markersize=2, zorder=1)
    P_plt.scatter([0], [0], s=8, c="green", label="Start location", zorder=2)
    P_plt.scatter(x_axis[-1], pitch[-1], s=8, c="red", label="End location", zorder=2)
    P_plt.set_xlim([minX, maxX])
    P_plt.set_ylim([minP, maxP])

    # Pitch Orientation estimated
    P_plt_es.set_title(r'Pitch angle Estimated $\beta$', y=toffset)
    P_plt_es.axes.xaxis.set_ticklabels([])
    P_plt_es.plot(x_axis_es, pitch_es, "-", linewidth=1.5, c="orange", label="Angle Estimated", markersize=2, zorder=1)
    P_plt_es.scatter([0], [0], s=8, c="blue", label="Start location", zorder=2)
    P_plt_es.scatter(x_axis_es[-1], pitch_es[-1], s=8, c="purple", label="End location", zorder=2)
    P_plt_es.set_xlim([minX_es, maxX_es])
    P_plt_es.set_ylim([minP_es, maxP_es])
    
    # Yaw Trajectory
    Y_plt.set_title(r'Yaw angle GT $\gamma$', y=toffset)
    Y_plt.set_ylabel("Yaw [rads]")
    Y_plt.axes.xaxis.set_ticklabels([])
    Y_plt.plot(x_axis, yaw, "-", linewidth=1.5, label="Angle GT", markersize=2, zorder=1)
    Y_plt.scatter([0], [0], s=8, c="green", label="Start location", zorder=2)
    Y_plt.scatter(x_axis[-1], yaw[-1], s=8, c="red", label="End location", zorder=2)
    Y_plt.set_xlim([minX, maxX])
    Y_plt.set_ylim([minY, maxY])
    Y_plt.legend(loc=4, title="Legend", borderaxespad=0., fontsize="medium", frameon=True)

    # Yaw Orientation estimated
    Y_plt_es.set_title(r'Yaw angle Estimated $\gamma$', y=toffset)
    Y_plt_es.axes.xaxis.set_ticklabels([])
    Y_plt_es.plot(x_axis_es, yaw_es, "-", linewidth=1.5, c="orange", label="Angle Estimated", markersize=2, zorder=1)
    Y_plt_es.scatter([0], [0], s=8, c="blue", label="Start location", zorder=2)
    Y_plt_es.scatter(x_axis_es[-1], yaw_es[-1], s=8, c="purple", label="End location", zorder=2)
    Y_plt_es.set_xlim([minX_es, maxX_es])
    Y_plt_es.set_ylim([minY_es, maxY_es])
    Y_plt_es.legend(loc=4, title="Legend", borderaxespad=0., fontsize="medium", frameon=True)
    
    # Plotting the result
    fig.suptitle(title, fontsize=16, y = 1.05)
    plt.tight_layout()
    plt.show()

def compare_3d_all(ground_truth, trajectory_vo, trajectory_vio, title):
	"""
	Plot the vehicle's trajectory in 3D space for the ground truth, VO and VIO estimates

	:param ground_truth: Numpy array (3 x M) where M is the number of samples
	    with the ground truth trayectory of the vehicle.
	:param trajectory_vo: Numpy array (3 x M) where M is the number of samples
	    with the estimated VO trajectory of the vehicle.
	:param trajectory_vio: Numpy array (3 x M) where M is the number of samples
	    with the estimated VIO trajectory of the vehicle.
	:param title: Name of the plot
	"""

	# Axis limits
	maxX = np.amax(ground_truth[0,:]) + 5
	minX = np.amin(ground_truth[0,:]) - 5
	maxY = np.amax(ground_truth[1,:]) + 5
	minY = np.amin(ground_truth[1,:]) - 5 
	maxZ = np.amax(ground_truth[2,:]) + 5
	minZ = np.amin(ground_truth[2,:]) - 5

	est_traj_fig = plt.figure()
	ax = est_traj_fig.add_subplot(111, projection='3d')
	ax.plot(ground_truth[0,:], ground_truth[1,:], ground_truth[2,:], "-", label='Ground Truth')
	ax.plot(trajectory_vo[0,:], trajectory_vo[1,:], trajectory_vo[2,:], "-", label='VO Estimate')
	ax.plot(trajectory_vio[0,:], trajectory_vio[1,:], trajectory_vio[2,:], "-", label='VIO Estimate')
	ax.set_xlabel('X [m]')
	ax.set_ylabel('Y [m]')
	ax.set_zlabel('Z [m]')
	ax.set_title(title, y = 1.0)
	ax.legend()
	ax.set_xlim(minX, maxX)
	ax.set_ylim(minY, maxY)
	ax.set_zlim(minZ, maxZ)
	plt.tight_layout()
	plt.show()