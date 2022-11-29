import net
import torch
import numpy as np
from scipy.io import loadmat
import time
from numpy import genfromtxt

start = time.time()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X=139
Y=139

use_pca=False

if use_pca:
    train_dl, test_dl, _ = net.load_data_voxels('./data/sub-06400_ndwi.mat', 139, Y, num_y_cols=1, batch_size=128,  pca_var=0.90, device=device)
    model_name='voxelnet_' + str(Y) +'_pca'
else :
    closest_dirs = genfromtxt('closest_dirs_0_to_265.txt' , delimiter=',')
    X=closest_dirs[Y,0:5]
    X=np.ndarray.tolist(X.astype(int))
    train_dl, test_dl, _ = net.load_data_voxels('./data/sub-06400_ndwi.mat', X, Y, num_y_cols=1, batch_size=128,  device=device)
    model_name='voxelnet_' + str(Y)


in_dim = train_dl.dataset[:][0].shape[1]
hidden_dim=10 # aprox 2/3 number of input dimensions + number output dims
model = net.voxelnet(in_dim, hidden_dim, 1).to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-4)

trained_model = net.train(model, train_dl, test_dl, opt, num_epochs=5)
torch.save(trained_model, './models/' + model_name )

end = time.time()

print(f'The training has been completed in {(end-start)/60} mins')
