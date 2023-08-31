import numpy as np
import open3d as o3d
import copy
import matplotlib.pyplot as plt
import o3dmethods as o3dmethoseditas

# clustering method
from sklearn.cluster import KMeans

# Preprocess of pcd and gnss: coloring, downsample, stastical outlier removal, dbscan
# Returns largest dbscan cluster, downpcd, and gnss point cloud 
def preprocess(pcd, gnss):

    pcd = o3dmethoseditas.color_pcd(np.asarray(pcd.points))
    gnss = o3dmethoseditas.color_pcd(np.asarray(gnss.points))
    gnss.paint_uniform_color([0, 1, 0])

    downpcd = pcd.voxel_down_sample(voxel_size=0.5)

    # o3d.visualization.draw_geometries([downpcd])

    filtered_cloud, ind = downpcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
    downpcd.paint_uniform_color([0, 1, 0])
    filtered_cloud.paint_uniform_color([0, 0, 0])

    # o3d.visualization.draw_geometries([downpcd, filtered_cloud])
    
    downpcd.paint_uniform_color([0, 0, 0])

    pcd_sel = o3dmethoseditas.db_scan(filtered_cloud, eps=0.6, min_points=5)
    pcd_sel.paint_uniform_color((0, 0, 0))

    return pcd_sel, downpcd, gnss

# Find bridges: matches odometry line to point cloud
def find_bridges(pcd_sel, gnss, num_bridges):
    coordinates_2d_array = matchGNSSPointCloud(gnss, pcd_sel)
    plot2dcoordinates(coordinates_2d_array)

    # o3d.visualization.draw_geometries([gnss, pcd_sel])

    bridges = clusterBridge(pcd_sel, gnss, coordinates_2d_array, num_bridges)

    return bridges

# Match GNSS and Point cloud intersections using octree
def matchGNSSPointCloud(gnss, pcd_sel):
    downsampled_gnss = gnss.voxel_down_sample(voxel_size=2)

    stretch_gnss = lift(downsampled_gnss)
    # o3d.visualization.draw_geometries([stretch_gnss])

    # # # To do  : test the difference between treating it as a voxel grid -> octree vs. diretly treating it as a octree?? -> looks the same

    # gnss = gnss.voxel_down_sample(voxel_size = 1)
    octree = o3d.geometry.Octree(max_depth=7)
    # octree2 = o3d.geometry.Octree(max_depth=6)

    octree.convert_from_point_cloud(pcd_sel, size_expand=0.05)
    # octree2.create_from_voxel_grid(voxel_grid)
    o3d.visualization.draw_geometries([stretch_gnss, octree])

    intersection_array = []

    for point in stretch_gnss.points:
        leafNode, nodeInfo = octree.locate_leaf_node(point)
        if nodeInfo is not None:
            intersection_array.append(nodeInfo.origin)

    coordinates_2d_array = np.vstack(intersection_array)[:,:2]
    return coordinates_2d_array

# Identifies bridges given 2d array of points of intersection
# Returns: array of "bridges", each index is one bridge containing a crop of the bridge, midpoint of that section, slope of gnss, and bounding box
def clusterBridge(pcd_sel, gnss, coordinates_2d_array, num_bridges):
    #K MEANS ALGORITHM
    num_clusters = num_bridges

    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(coordinates_2d_array)

    cluster_labels = kmeans.labels_
    cluster_points = [[] for _ in range(num_clusters)]

    
    # Assign points to clusters
    for i, label in enumerate(cluster_labels):
        cluster_points[label].append(coordinates_2d_array[i])

    cluster_points = [np.array(cluster) for cluster in cluster_points]

    xranges = []
    yranges = []

    for cluster in cluster_points:
        xmin, xmax = np.min(cluster[:,0]), np.max(cluster[:,0])
        ymin, ymax = np.min(cluster[:,1]), np.max(cluster[:,1])
        xranges += [[xmin, xmax]]
        yranges += [[ymin, ymax]]

    
    # Return the cropped bridge section, midpoint, slope, and object bounding box
    output = []
    for i in range(num_clusters):

        xmin, xmax, ymin, ymax = xranges[i][0], xranges[i][1], yranges[i][0], yranges[i][1]

        midPoint = [(xmin+xmax)/2, (ymin+ymax)/2]

        # Arbitrary box size of 120
        xmin,xmax,ymin,ymax = xmin -60, xmax +60,  ymin -60, ymax +60
        
        slope = gnss_angle(gnss, xranges[i], yranges[i])

        rotation_angle_rad = np.arctan(slope)

        rot_matrix = o3dmethoseditas.calc_rotation_matrix_from_angle(rotation_angle_rad)

        # Creating bounding box around min/max x, y coords of K-means cluster
        bb = [[xmin, ymin], [xmax, ymin], [xmin, ymax], [xmax, ymax]]
        bb3d = np.vstack((np.hstack((bb, np.array([[-100, -100, -100, -100]]).T)), np.hstack((bb, np.array([[100, 100, 100, 100]]).T))))
        vec = o3d.utility.Vector3dVector(bb3d)
        obb = o3d.geometry.OrientedBoundingBox.create_from_points(vec)
        obb.color = (0, 1, 0)
        obb.rotate(rot_matrix, center=obb.get_center())

        o3d.visualization.draw_geometries([pcd_sel, obb])

        # Crop this section of the point cloud and return it
        pcdCrop = copy.deepcopy(pcd_sel)
        pcd_cropped = pcdCrop.crop(obb)

        output.append([pcd_cropped, midPoint, slope, obb])
        
        # o3d.io.write_point_cloud(outputName, pcd_cropped)
        # o3d.visualization.draw_geometries([pcd_sel, obb])
    
    return output

# Finds angle of gnss at certain x and y range
def gnss_angle(gnss, xrange, yrange):
    xmin = xrange[0]
    xmax = xrange[1]
    ymin = yrange[0]
    ymax = yrange[1]
    
    gnss_array = np.asarray(gnss.points)
    x_coordinates = gnss_array[:,0]
    y_coordinates = gnss_array[:,1]
    inliers = np.where( (x_coordinates >= xmin) & (x_coordinates <= xmax) & (y_coordinates >= ymin) & (y_coordinates <= ymax) )
    
    filtered_x = x_coordinates[inliers]
    filtered_y = y_coordinates[inliers]

    slope, intercept = np.polyfit(filtered_x, filtered_y, 1)
    return slope


# Stretch the z axis of the point cloud
def lift(pcd):
    pcd_array = np.asarray(pcd.points)
    total = pcd_array.copy()

    copy_up = pcd_array.copy()
    copy_down = pcd_array.copy()

    for i in range(5):
        copy_up[:, 2] += 3
        total = np.vstack((total, copy_up))
    
    for i in range(5):
        copy_down[:, 2] -= 3
        total = np.vstack((total, copy_down))
    
    vec = o3d.utility.Vector3dVector(total)
    final = o3d.geometry.PointCloud(vec)
    final.paint_uniform_color([0, 1, 0])
    return final

# Convert .txt to .xyz
def convertToXYZ():
    newFile = open('data/slam_bridge_gnss/bridge_monday_6_1_dense.xyz', 'w')
    with open('data/odometry_txtfiles/bridge_monday_6_1.txt', 'r') as f:
        for line in f:
            line = line.replace(",", " ")
            line = line
            newFile.write(line)


# Plots scatter plot of coordinates of intersection
def plot2dcoordinates(coordinates_2d_array):
    # Extract x and y coordinates
    x_coordinates = coordinates_2d_array[:, 0]
    y_coordinates = coordinates_2d_array[:, 1]

    # Create a scatter plot
    plt.scatter(x_coordinates, y_coordinates, color='blue', label='Coordinates')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Scatter Plot of X, Y Coordinates')
    plt.legend()
    plt.grid(True)
    plt.show()
    print(len(coordinates_2d_array))
