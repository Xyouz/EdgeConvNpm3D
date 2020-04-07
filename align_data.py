import numpy as np
import argparse
import glob
from sklearn.neighbors import KDTree

from utils.ply import read_ply, write_ply


def local_PCA(points):

    bar = points.mean(axis=0)

    centered = (points - bar)[:,:,np.newaxis]
    cov = (np.matmul(centered, centered.transpose(0,2,1))).mean(axis=0)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    return eigenvalues, eigenvectors


def neighborhood_PCA(query_points, cloud_points, radius):

    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points
    
    kdtree = KDTree(cloud_points)

    neighborhoods = kdtree.query_radius(query_points, radius)

    all_eigenvalues = np.zeros((query_points.shape[0], 3))
    all_eigenvectors = np.zeros((query_points.shape[0], 3, 3))

    for i, ind in enumerate(neighborhoods):
        val, vec = local_PCA(cloud_points[ind,:])
        all_eigenvalues[i] = val
        all_eigenvectors[i] = vec

    return all_eigenvalues, all_eigenvectors

def grid_subsampling(points, voxel_size):

    #   Tips :
    #       > First compute voxel indices in each direction for all the points (can be negative).
    #       > Sum and count the points in each voxel (use a dictionaries with the indices as key).
    #         Remember arrays cannot be dictionary keys, but tuple can.
    #       > Divide the sum by the number of point in each cell.
    #       > Do not forget you need to return a numpy array and not a dictionary.
    #

    xyz_min = points.min(axis=0)

    out = {}
    for point in points:
        index = tuple(((point - xyz_min)//voxel_size).astype(int))
        try:
            out[index].append(point)
        except KeyError:
            out[index] = [point]

    subsampled_points = np.zeros((len(out), 3))
    for c, index in enumerate(out):
        position = np.array(out[index]).mean(axis=0)
        subsampled_points[c] = position

    return subsampled_points

def alignCloud(cloud):
    # Align point clouds using PCA
    subsampled = grid_subsampling(cloud, 2.5)
    mean = subsampled.mean(axis= 0,keepdims=True)

    cloud = cloud - mean
    subsampled = (subsampled - mean)
    C = subsampled.T @ subsampled / (subsampled.shape[0] - 1)
    _, basis = np.linalg.eig(C)
    cloud = cloud @ basis
    return cloud

def computenormals(cloud):
    _, vec = neighborhood_PCA(cloud, cloud, 0.5)
    normal = vec[:,:,0]
    return normal.astype(np.float32)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--path', type=str, default='data/MiniChallenge/test', metavar='N',
                        help='Folder containing the point clouds')
    parser.add_argument('--labels', type=bool, default=False)
    args = parser.parse_args()

    for fname in glob.glob(args.path + "/*.ply"):
        print(fname)
        cloud = read_ply(fname)
        points = np.vstack((cloud['x'], cloud['y'], cloud['z'])).T

        normals = computenormals(points)

        if args.labels:
            labels = cloud['class']
            data = [points, normals, labels] 
            colname = ['x','y','z', 'nx', 'ny', 'nz', 'class']
        else:
            data = [points, normals]
            colname = ['x','y','z', 'nx', 'ny', 'nz']
        
        write_ply(fname[:-4]+"_al.ply", data, colname)
        