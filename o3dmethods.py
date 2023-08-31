# Scripts for various methods in open3d
import numpy as np
import open3d as o3d
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# Main method
#   Detects a single bridge given an array with information about section of river with bridge
# Input: list with pcd_sel, midpoint, slope, bounding box
# Output: bounding box of final bridge, point cloud cropped from PCD_SEL
def single_bridge_detection(downpcd, bridges):
    # extract from output array
    pcd_sel = bridges[0]
    midPoint_gnss = bridges[1]
    slope_gnss = bridges[2] 
    
    # Calculate angle
    gnss_angle = np.arctan(slope_gnss)

    # Rotate to become parallel with X axis
    # note: preprocessing already done in multibridge detection
    rotParallelX = calc_rotation_matrix_from_angle(-gnss_angle)
    pcd_sel.rotate(rotParallelX, center=pcd_sel.get_center())
    gnss_angle = 0


    #RANSAC
    ransac_bb, average_z= ransac_plane(pcd_sel, 1.0, 3, 1000)

    average_z = np.mean(np.asarray(ransac_bb.get_box_points())[:,2])

    bb_mid_strip, bridge_slope, bridge_center = get_mid_strip(pcd_sel, ransac_bb, average_z, midPoint_gnss)

    # o3d.visualization.draw_geometries([pcd_sel, bb_mid_strip])

    bb_mid_strip_perp = get_mid_strip_perp(pcd_sel, bb_mid_strip, average_z)

    # o3d.visualization.draw_geometries([pcd_sel, bb_mid_strip_perp, bb_mid_strip])

    # Put together points of 2 bounding strips
    a1 = np.asarray(bb_mid_strip.get_box_points())
    a2 = np.asarray(bb_mid_strip_perp.get_box_points())
    allPoints = np.concatenate((a1, a2))

    # Calculate angle of bridge if slanted
    bridge_angle = np.arctan(bridge_slope)

    # Create final bounding box
    finalBox = bind_minmax(allPoints, pcd_sel, bridge_angle, average_z, bridge_center)

    o3d.visualization.draw_geometries([pcd_sel,finalBox])

    # Crope final bridge from pcd_sel (NOTE: can also crop from original pcd, downpcd)
    finalBRIDGE = pcd_sel.crop(finalBox)

    # Rotating back to original pose from parallel with x axis
    rotate_back = np.arctan(slope_gnss)
    rotParallelX = calc_rotation_matrix_from_angle(rotate_back)
    finalBox.rotate(rotParallelX, center=pcd_sel.get_center())
    finalBRIDGE.rotate(rotParallelX, center=pcd_sel.get_center())
    finalBRIDGE.paint_uniform_color([0, 1, 0])
    # o3d.visualization.draw_geometries([finalBRIDGE, finalBox])
    
    return finalBox, finalBRIDGE


# Section 2: Open 3d Methods
#   - Taken from Open 3d library / example code

# make pcd colorable
def color_pcd(xyz, color_axis=-1, rgb=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if color_axis >= 0:
        if color_axis == 3:
            axis_vis = np.arange(0, xyz.shape[0], dtype=np.float32)
        else:
            axis_vis = xyz[:, color_axis]
        min_ = np.min(axis_vis)
        max_ = np.max(axis_vis)

        colors = cm.flag((axis_vis - min_) / (max_ - min_))[:, 0:3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    pcd.paint_uniform_color([0, 0, 0])
    return pcd

# DB SCAN Clustering 
def db_scan(input_cloud, eps, min_points):

    db_cloud = copy.deepcopy(input_cloud)
    db_cloud.paint_uniform_color((0, 0, 0))

    # with o3d.utility.VerbosityContextManager(
    #         o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(db_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    db_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # assess the size of all clusters DBSCAN yields
    candidates=[len(np.where(labels==j)[0]) for j in np.unique(labels)]
    
    # find the largest cluster
    best_candidate=int(np.unique(labels)[np.where(candidates== np.max(candidates))[0]])
    # Filtering points that contain label for best canditate
    db_cloud_sub=db_cloud.select_by_index(list(np.where(labels== best_candidate)[0]))
    # Find outlier points
    outlier=db_cloud.select_by_index(list(np.where(labels== best_candidate)[0]), invert=True)

    outlier.paint_uniform_color((0, 1, 0))
    db_cloud_sub.paint_uniform_color((0, 1, 0))
    
    # o3d.visualization.draw_geometries([db_cloud_sub, db_cloud])

    return db_cloud_sub

# RANSAC plane segmentation
def ransac_plane(pcd_sel, distance_threshold, ransac_n, num_iterations):
    distance_threshold=1.0     # max distance between inlier point and plane
    ransac_n=3                  # number of points sampled for plane estimation
    num_iterations=1000         # how often plane is samples and verified

    pcd_ransac = copy.deepcopy(pcd_sel)

    plane_model, inliers = pcd_ransac.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)
    [a, b, c, d] = plane_model
    # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    inlier_cloud = pcd_ransac.select_by_index(inliers)

    inlier_cloud.paint_uniform_color([1, 0, 0])

    outlier_cloud = pcd_ransac.select_by_index(inliers, invert=True)

    outlier_cloud.paint_uniform_color([1, 0, 0])

    # print('Found ' + str(np.asarray(inlier_cloud.points).shape[0]) + ' inliers')
    # print('Found ' + str(np.asarray(outlier_cloud.points).shape[0]) + ' outliers')
    ransac_bb = inlier_cloud.get_oriented_bounding_box()
    ransac_bb.color = (0, 1, 0)
    o3d.visualization.draw_geometries([pcd_sel, ransac_bb])
    average_z = np.mean(np.asarray(ransac_bb.get_box_points())[:,2])
    return ransac_bb, average_z


# Section 3: Geometric Methods 

# Get the middle strip parallel to the river bank
# Returns: tight bounding box parallel to river bank
def get_mid_strip(pcd, bbox, average_z, midPoint_gnss):

    # Create large bounding box around section of bridge
    obb = cropStrip(bbox, average_z, midPoint_gnss, perp=False)
    # o3d.visualization.draw_geometries([pcd, obb])

    # Cropped poiont cloud of the bridge
    pcd_length_tight = pcd.crop(obb)
    pcd_length_tight.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_geometries([pcd, pcd_length_tight])

    # SECOND DB Scan to remove noise
    pcd_length_tight_dbscanned = db_scan(pcd_length_tight, eps=0.6, min_points=5)
    
    # Form smaller bounding tox tight around bridge
    bbtight_len = pcd_length_tight_dbscanned.get_oriented_bounding_box()
    bbtight_len.color = (0, 1, 0)

    # Find slope of bridge (TO USE FOR ROTATION)
    points = np.asarray(pcd_length_tight_dbscanned.points)[:,: 2]
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    slope = avg_slope(hull_points)
    # plot_convex_hull_points(points)

    # Find center of bridge (TO USE FOR ROTATION)
    bridge_center = pcd_length_tight_dbscanned.get_center()

    return bbtight_len, slope, bridge_center

# Get Tight Bounding box for perpindicular component
# Returns: tight bounding box in perpindicular direction
def get_mid_strip_perp(pcd, bbox, average_z):
    obb = cropStrip(bbox, average_z, None, perp=True)
    # o3d.visualization.draw_geometries([pcd, obb])
    
    # point cloud crop from larger array
    pcdVerticalCrop = pcd.crop(obb)
    pcdVerticalCrop.paint_uniform_color([1, 0, 0])
    
    # create bounding box based on crop
    bbtight_len = pcdVerticalCrop.get_oriented_bounding_box()
    bbtight_len.color = (0, 1, 0)

    return bbtight_len

# Creates a long thin bounding box that is later used for cropping
# Returns: bounding box that is thin across the bridge(not tight
def cropStrip(bbox, average_z, midPoint_gnss, perp):
    
    # mid point of line
    if not perp:
        midPoint = midPoint_gnss
    else:
        # perpindicular bounding box uses the parallel bounding box as reference for finding midpoint
        bbox_array = np.asarray(bbox.get_box_points())
        sortedPoints = bbox_array[np.argsort(bbox_array[:,0])]
        top4 = sortedPoints[::2]
        l, r = top4[:2], top4[2:]
        l = (l[0] + l[1] )/2
        r = (r[0] + r[1] )/2 
        midPoint = (l + r)/2
        midPoint = midPoint[0:2]
    
    # Find line extending from mid point, perp: y axis, not perp: x axis
    if perp:
        p1, p2 = [midPoint[0],midPoint[1] + 70], [midPoint[0],midPoint[1] - 70]
    else:
        p1, p2 =[midPoint[0]+ 60 ,midPoint[1]], [midPoint[0]-60 ,midPoint[1]]
    
    # Creating thin box, width = width*2
    width = 3
    perp2DArray = [[p1[0] + width, p1[1]+width], [p1[0]-width, p1[1]-width], [p2[0]+width, p2[1]-width], [p2[0]-width, p2[1]+width]]
    perpBound2D = np.array(perp2DArray)
    h1, h2 = average_z-10, average_z +10
    perpBound3D = np.vstack((np.hstack((perpBound2D, np.array([[h1, h1, h1, h1]]).T)), np.hstack((perpBound2D, np.array([[h2, h2, h2, h2]]).T))))
    vec = o3d.utility.Vector3dVector(perpBound3D)
    obb = o3d.geometry.OrientedBoundingBox.create_from_points(vec)
    obb.color =(0, 1, 0)

    return obb

# Calculates amount to lengthen bridge if bridge is slanted, minimum of length 10
def calculate_lengthen(yLen, bridge_angle):
    lengthen_amount = yLen/np.sin(bridge_angle) - yLen
    if lengthen_amount < -10:
        lengthen_amount = 0
    return lengthen_amount

# Binding final rectangle by min/max x y points of crops 
# Returns: bounding box
def bind_minmax(allPoints, pcd_sel, bridge_angle, average_z, bridge_center):
    # Create bounding box given x, y, restraints in 2d array
    min_x, max_x, min_y, max_y = min(allPoints[:,0]), max(allPoints[:,0]), min(allPoints[:,1]), max(allPoints[:,1])
    boundingArray = [[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]
    h1, h2 = average_z-30, average_z +15
    perpBound3D = np.vstack((np.hstack((boundingArray, np.array([[h1, h1, h1, h1]]).T)), np.hstack((boundingArray, np.array([[h2, h2, h2, h2]]).T))))
    vec = o3d.utility.Vector3dVector(perpBound3D)
    obb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(vec)
    obb.color =(0, 1, 0)

    # Visualize first bounding box, based on dimensions set by x, y, constraints
    o3d.visualization.draw_geometries([pcd_sel,obb])

    # Lengthening box for slanted matrix
    lengthen_amount = calculate_lengthen(obb.get_extent()[1], bridge_angle)
    lengthened_box = lengthen_box(obb, lengthen_amount/2 + 20)

    o3d.visualization.draw_geometries([pcd_sel,lengthened_box])
    
    # Rotate Bridge by angle of the slant
    bridge_angle += (np.pi/2)
    rotMax = calc_rotation_matrix_from_angle(bridge_angle)

    lengthened_box.rotate(rotMax, center=bridge_center)
    lengthened_box.color = (0, 1, 0)

    # Final bounding box
    # o3d.visualization.draw_geometries([pcd_sel, lengthened_box])

    return lengthened_box

# Lengthen a box along it's x coordinates (for slanted bridges)
def lengthen_box(tight_final, length):

    tight_final_points = np.array(tight_final.get_box_points())
    ymin = np.min(tight_final_points[:,1])

    for point in tight_final_points:
        if point[1] == ymin:
            point[1] -= length
        else:
            point[1] += length

    vec = o3d.utility.Vector3dVector(tight_final_points)
    obb = o3d.geometry.OrientedBoundingBox.create_from_points(vec)
    obb.color = (0, 1, 0)

    return obb

# Geometry Helpers
def calc_rotation_matrix_from_angle(angle_radians):
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)
    
    rotation_matrix = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])
    return rotation_matrix

# Random Plots for visualization


# mat plot lib checking
def plot_2d_points_with_lines(data):
    
    x = data[:, 0]
    y = data[:, 1]

    plt.scatter(x, y, color='blue', marker='o')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter Plot of 2D Points with Horizontal Lines')
    plt.legend()

    plt.show()

def plot_convex_hull_points(points):
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    plt.figure(figsize=(8, 6))
    
    # Plot the points within the convex hull
    plt.scatter(hull_points[:, 0], hull_points[:, 1], color='blue', label='Convex Hull Points')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Points in Convex Hull')
    plt.legend()
    plt.grid()
    plt.show()


# Calculates slope of vertical length of bridge by finding minimum X for max and min Y values
def avg_slope(points):
    ymin, ymax= np.min(points[:,1]),  np.max(points[:,1])
    xBotLeft, xTopLeft = [float('inf'), 0], [float('inf'), 0]

    for point in points:
        if abs(point[1] - ymin) < 1:
            if point[0] < xBotLeft[0]:
                xBotLeft = point

        if abs(point[1] - ymax) < 1:
            if point[0] < xTopLeft[0]:
                xTopLeft =point

    slope = (xTopLeft[1] - xBotLeft[1]) / (xTopLeft[0] - xBotLeft[0])
    return slope

