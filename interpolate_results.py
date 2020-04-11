import numpy as np

from utils.ply import read_ply, write_ply
from sklearn.neighbors import KDTree

from scipy import stats
import argparse

cloudfile = "checkpoints/cloud.ply"

parser = argparse.ArgumentParser(description='Point Cloud Recognition')
parser.add_argument('--input_file', type=str, metavar='I', default="checkpoints/cloud.ply",
                    help='File containing the predictions.')
parser.add_argument('--output_prefix', type=str, default="checkpoints/full_cloud", metavar='O',
                    help='Labelled ground truth cloud.')
parser.add_argument('-k', type=int, default=10, metavar='k',
                    help='Number of neighbors.')
args = parser.parse_args()


cloud = read_ply(args.input_file)
points = np.vstack((cloud['x'], cloud['y'], cloud['z'])).T
labels = cloud['class'].astype(np.int32)

labeled = np.nonzero(labels)[0]
not_labeled = np.nonzero(labels==0)[0]

tree = KDTree(points[labeled])

dist, neighbors= tree.query(points[not_labeled],k=args.k)
neighborslabels = labels[labeled][neighbors]

ind = stats.mode(neighborslabels, axis=1)[0]
labels[not_labeled] = ind.reshape(-1)

# Labelled cloud
write_ply('{}.ply'.format(args.output_prefix), (points,labels), ('x','y','z','scalar_class'))

# Label list for IoU computation and benchmark submission
np.savetxt('{}.txt'.format(args.output_prefix), labels, fmt='%d')
