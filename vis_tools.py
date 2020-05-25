# Utitlity file with functions for handling trajectory plots
#
# Author: Miguel Saavedra

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import pandas as pd

from rotations import Quaternion, skew_symmetric

def visualize_trajectory(df_values):
    
    # Convert dataframe into numpy array
    list_of_pos = df_values.values.tolist()
    trajectory = np.array(list_of_pos).T

    #list for x, y and z values
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
    ZX_plt.set_xlabel("X")
    ZX_plt.set_ylabel("Z")
    # Plot camera initial location
    ZX_plt.scatter([0], [0], s=8, c="green", label="Start location", zorder=2)
    ZX_plt.scatter(locX[-1], locZ[-1], s=8, c="red", label="End location", zorder=2)
    ZX_plt.set_xlim([minX, maxX])
    ZX_plt.set_ylim([minZ, maxZ])
    ZX_plt.legend(bbox_to_anchor=(1.05, 1.0), loc=3, title="Legend", borderaxespad=0., fontsize="medium", frameon=True)
        
    # Plot ZY
    ZY_plt.set_title("Trajectory (Y, Z)", y=toffset)
    ZY_plt.set_xlabel("Y")
    ZY_plt.plot(locY, locZ, "--", linewidth=1.5, markersize=2, zorder=1)
    ZY_plt.scatter([0], [0], s=8, c="green", label="Start location", zorder=2)
    ZY_plt.scatter(locY[-1], locZ[-1], s=8, c="red", label="End location", zorder=2)
    ZY_plt.set_xlim([minY, maxY])
    ZY_plt.set_ylim([minZ, maxZ])
    
    # Plot YX
    YX_plt.set_title("Trajectory (Y X)", y=toffset)
    YX_plt.set_ylabel("X")
    YX_plt.set_xlabel("Y")
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
    D3_plt.set_xlabel("X", labelpad=0)
    D3_plt.set_ylabel("Y", labelpad=0)
    D3_plt.set_zlabel("Z", labelpad=-5)
    
    # Plotting the result
    fig.suptitle("Vehicle's trajectory", fontsize=16, y = 1.05)
    D3_plt.view_init(35, azim=45)
    plt.tight_layout()
    plt.show()

def visualize_angles(rotations_df):
    
    list_of_rotations = rotations_df.values.tolist()
    rotations = np.array(list_of_rotations).T

    #list to unpack roll, pitch, yaw angles
    roll = []
    pitch = []
    yaw = []
    x_axis = np.arange(rotations.shape[1])
    
    for i in range(0, rotations.shape[1]):
        current_rot = rotations[:, i]
        q = Quaternion(w = current_rot[3], x = current_rot[0], 
                       y = current_rot[1], z = current_rot[2]).to_euler()
        
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
    
    # Roll Trajectory
    
    R_plt.set_title(r'Roll angle $\theta$', y=toffset)
    R_plt.plot(x_axis, roll, "-", label="Angle", zorder=1, linewidth=1.5, markersize=2)
    R_plt.axes.xaxis.set_ticklabels([])
    R_plt.set_ylabel("Roll [rads]")
    # Plot camera initial location
    R_plt.scatter([0], [0], s=8, c="green", label="Start location", zorder=2)
    R_plt.scatter(x_axis[-1], roll[-1], s=8, c="red", label="End location", zorder=2)
    R_plt.set_xlim([minX, maxX])
    R_plt.set_ylim([minR, maxR])
        
    # Pitch Trajectory
    P_plt.set_title(r'Pitch angle $\beta$', y=toffset)
    P_plt.set_ylabel("Roll [rads]")
    P_plt.axes.xaxis.set_ticklabels([])
    P_plt.plot(x_axis, pitch, "-", linewidth=1.5, label="Trajectory", markersize=2, zorder=1)
    P_plt.scatter([0], [0], s=8, c="green", label="Start location", zorder=2)
    P_plt.scatter(x_axis[-1], pitch[-1], s=8, c="red", label="End location", zorder=2)
    P_plt.set_xlim([minX, maxX])
    P_plt.set_ylim([minP, maxP])
    
    # Yaw Trajectory
    Y_plt.set_title(r'Yaw angle $\gamma$', y=toffset)
    Y_plt.set_ylabel("Yaw")
    Y_plt.axes.xaxis.set_ticklabels([])
    Y_plt.plot(x_axis, yaw, "-", linewidth=1.5, label="Trajectory", markersize=2, zorder=1)
    Y_plt.scatter([0], [0], s=8, c="green", label="Start location", zorder=2)
    Y_plt.scatter(x_axis[-1], yaw[-1], s=8, c="red", label="End location", zorder=2)
    Y_plt.set_xlim([minX, maxX])
    Y_plt.set_ylim([minY, maxY])
    Y_plt.legend(loc=4, title="Legend", borderaxespad=0., fontsize="medium", frameon=True)
    
    # Plotting the result
    fig.suptitle("Vehicle's euler angles", fontsize=16, y = 1.05)
    plt.tight_layout()
    plt.show()

def visualize_camera_movement(image1, image1_points, image2, image2_points, is_show_img_after_move=False):
    image1 = image1.copy()
    image2 = image2.copy()
    
    for i in range(0, len(image1_points)):
        # Coordinates of a point on t frame
        p1 = (int(image1_points[i][0]), int(image1_points[i][1]))
        # Coordinates of the same point on t+1 frame
        p2 = (int(image2_points[i][0]), int(image2_points[i][1]))

        cv.circle(image1, p1, 5, (0, 255, 0), 1)
        cv.arrowedLine(image1, p1, p2, (0, 255, 0), 1)
        cv.circle(image1, p2, 5, (255, 0, 0), 1)

        if is_show_img_after_move:
            cv.circle(image2, p2, 5, (255, 0, 0), 1)
    
    if is_show_img_after_move: 
        return image2
    else:
        return image1