
import sys 
sys.path.append("./")
from util_pca import DataManager
import argparse
from skimage import io
import numpy as np

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       setting option                           '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
parser = argparse.ArgumentParser(description='Predict test for model.')
parser.add_argument('-id','--image_dir', dest='image_dir',type=str,required=True)
parser.add_argument('-i','--image', dest='image',type=str,required=True)
args = parser.parse_args()


dm=DataManager()

dm.load_image(args.image_dir+'/',415,'image')
dm.load_x_mean('image')
S=dm.construct_pca('image')
#print(dm.data['image'].shape)
#print(S.shape)

image=io.imread(args.image).flatten().reshape((-1,1))
pca_out=dm.pca(S,image,4)

data=dm.reconstruct(S,pca_out).reshape((600,600,3)).astype(np.uint8)
io.imsave('reconstruction.jpg',data)
