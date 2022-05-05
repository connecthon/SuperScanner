import net
import torch
import numpy as np
from scipy.io import loadmat
import time

start = time.time()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dl, test_dl, _ = net.load_data_voxels('./data/sub-06400_ndwi.mat', 139, 139, num_y_cols=1, batch_size=128,  pca_var=0.8, device=device)

in_dim = train_dl.dataset[:][0].shape[1]
model = net.voxelnet(in_dim, 100, 1).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

trained_model = net.train(model, train_dl, test_dl, opt, num_epochs=5)
torch.save(trained_model, './models/voxelnet_pca')

end = time.time()

print(f'The training has been completed in {(end-start)/60} mins')
