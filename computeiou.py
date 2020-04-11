from utils.ply import read_ply, write_ply
import numpy as np 
import matplotlib.pyplot as plt 
import argparse

def to6classes(labels):
    """
    Convert the 10 classes of the Paris-Lille-3D dataset to the 6 classes of the NPM3D benchmark dataset.
    """
    res =np.zeros_like(labels, dtype=int)
    res[labels==1] = 1 #Ground
    res[labels==2] = 2 #buildings
    res[labels==3] = 3 #poles
    res[labels==4] = 3 #poles
    # res[labels==5] = 0 #trashcan
    # res[labels==6] = 0 #barriers
    res[labels==7] = 4 #Ground
    res[labels==8] = 5 #Ground
    res[labels==9] = 6 #Ground
    return res
    
def confusionMatrix(pred, gt):
    """
    Compute the confusion matrix for predicted labels pred and ground truth gt.
    """
    res = np.zeros((6,6),dtype=int)
    for i in range(1,7):
        for j in range(1,7):
            res[i-1,j-1] = np.sum((gt==i) * (pred==j))
    return res

def iou(cm):
    """
    Given the confusion matrix, computes the IoU for every class.
    """
    res = []
    for i in range(6):
        inter = cm[i,i]
        union = np.sum(cm[i,:]) + np.sum(cm[:,i]) - cm[i,i]
        res.append(inter/union)
    return res


parser = argparse.ArgumentParser(description='Point Cloud Recognition')
parser.add_argument('--pred_file', type=str, metavar='P', default="checkpoints/full_cloud.txt",
                    help='File containing the predictions.')
parser.add_argument('--cloud_file', type=str, default="data/MiniChallenge/test/Lille2_al.ply", metavar='C',
                    help='Labelled ground truth cloud.')
parser.add_argument('--latex', type=bool, default=False, help="IoU printing format")
args = parser.parse_args()

pred = np.loadtxt(args.pred_file, dtype=int)

cloud = read_ply(args.cloud_file)

gt10c = cloud['scalar_class']

gt = to6classes(gt10c)

cm = confusionMatrix(pred, gt)

print(cm)

# plt.matshow(cm)
# plt.show()

iouc = iou(cm)

if args.latex:
    for v in iouc :
        print("{:.1f}".format(100*v),end=" & ")
    print("{:.1f}".format(100* np.mean(iouc)))
else:
    print('Mean IoU {:.1f}'.format(100* np.mean(iouc)))
    for v in iouc:
        print("{:.1f}".format(100*v),end=" ")