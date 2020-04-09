import numpy as np

from utils.ply import read_ply, write_ply
from sklearn.neighbors import KDTree

from scipy import stats

cloudfile = "checkpoints/cloud.ply"

k = 10

cloud = read_ply(cloudfile)
points = np.vstack((cloud['x'], cloud['y'], cloud['z'])).T
labels = cloud['class'].astype(np.int32)

labeled = np.nonzero(labels)[0]
not_labeled = np.nonzero(labels==0)[0]

tree = KDTree(points[labeled])

dist, neighbors= tree.query(points[not_labeled],k=k)
neighborslabels = labels[labeled][neighbors]

ind = stats.mode(neighborslabels, axis=1)[0]
labels[not_labeled] = ind.reshape(-1)

write_ply("checkpoints/full_cloud.ply", (points,labels), ('x','y','z','scalar_class'))
np.savetxt('checkpoints/full_cloud.txt', labels, fmt='%d')
