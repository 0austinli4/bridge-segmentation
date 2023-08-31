import numpy as np
import open3d as o3d
import copy
import o3dmethods as o3dmethods

# Main function for detecting river banks, returns point cloud of river bank
def detect_riverbank(inputpcd_sel, inputgnss):
    pcd_sel = copy.deepcopy(inputpcd_sel)
    gnss = copy.deepcopy(inputgnss)
    gnss.paint_uniform_color([0, 1, 0])
    # o3d.visualization.draw_geometries([pcd_sel, gnss])
    
    riverbank = find_bank(pcd_sel, gnss)
    riverbank.paint_uniform_color([0, 1, 0])

    o3d.visualization.draw_geometries([inputpcd_sel, riverbank])
    o3d.visualization.draw_geometries([riverbank])

    return riverbank

# Find river bank given pcd_sel and gnss, finding intersections between perpindicular lines from gnss and pcd
def find_bank(pcd_sel, gnss):

    gnss_points = np.array(gnss.points)
    slope = avg_slope(gnss_points)
    gnss_angle = np.arctan(slope)
    rotParallelX = calc_rotation_matrix_from_angle(-gnss_angle)
    pcd_sel.rotate(rotParallelX, center=pcd_sel.get_center())
    gnss.rotate(rotParallelX, center=pcd_sel.get_center())

    # pcd_edge_down = pcd_sel.voxel_down_sample(voxel_size=2)
    # o3d.visualization.draw_geometries([pcd_sel, downsampled_gnss])

    octree = o3d.geometry.Octree(max_depth=11)
    octree.convert_from_point_cloud(pcd_sel, size_expand=0.05)

    # o3d.visualization.draw_geometries([octree, downsampled_gnss])


    intersection_points = []
    prev = [0, 0 , 0]


    for point in gnss.points:
        slope = (point[1] - prev[1]) / (point[0] - prev[0])
        slope = -1 * (1/ slope)

        p1, p2 = find_intersection_points(octree, point, slope)
        if np.any(p1):
            intersection_points.append(p1)
        
        if np.any(p2):
            intersection_points.append(p2)
        
        prev = point

    vec = o3d.utility.Vector3dVector(intersection_points)
    final = o3d.geometry.PointCloud(vec)
    final.paint_uniform_color([0, 1, 0])
    filtered_final, ind  = final.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.0)
    filtered_final.paint_uniform_color([0, 0, 1])
    filtered_final = lift(filtered_final)
    backRotation = calc_rotation_matrix_from_angle(gnss_angle)
    filtered_final.rotate(backRotation, center=pcd_sel.get_center())

    return filtered_final


# get slope
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

# Move points in perpindicular direction and return first intersection 
# Goes in positive and negative direction
def find_intersection_points(octree, point, slope):
    unit_vector = np.array([1, slope, 0])
    unit_vector /= np.linalg.norm(unit_vector)
    iterations = 0

    pointReverse = np.copy(point)
    # vecP = o3d.utility.Vector3dVector(point)
    leafNode, nodeInfo = octree.locate_leaf_node(point)
    
    while nodeInfo is None and iterations < 70:
        point += unit_vector
        iterations +=1 
        leafNode, nodeInfo = octree.locate_leaf_node(point)

    leafNode, nodeInfo1 = octree.locate_leaf_node(pointReverse)

    iterations = 0
    while nodeInfo1 is None and iterations < 70:
        pointReverse -= unit_vector
        iterations +=1 
        leafNode, nodeInfo1 = octree.locate_leaf_node(pointReverse)

    if nodeInfo is not None and nodeInfo1 is not None:
        return nodeInfo.origin, nodeInfo1.origin
    elif nodeInfo is not None:
        return nodeInfo.origin, None
    elif nodeInfo1 is not None:
            return None, nodeInfo1.origin
    else:
        return None, None

# Stretch the z axis of the gnss point cloud
def lift(pcd):
    array = np.asarray(pcd.points)
    total = array.copy()
    copy_up = array.copy()

    for i in range(10):
        copy_up[:, 2] += 1
        total = np.vstack((total, copy_up))
    
    vec = o3d.utility.Vector3dVector(total)
    final = o3d.geometry.PointCloud(vec)
    return final

# Calculate rotation matrix given an angle
def calc_rotation_matrix_from_angle(angle_radians):
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)
    
    rotation_matrix = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])
    return rotation_matrix