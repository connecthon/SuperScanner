import net
from scipy.io import savemat
import numpy as np

_, _, valid_data = net.load_data_voxels('./data/sub-06400_ndwi.mat', 139, 140, num_y_cols=1, batch_size=50,  pca_var=0.9, device='cpu')

in_dim = valid_data.dataset[:][0].shape[1]
print(in_dim)
trained_model = net.voxelnet(in_dim,10,1).to('cpu')
predictions = net.evaluate_plot(trained_model, './models/voxelnet_pca', valid_data)

savemat('./data/prediction.mat', {'predictions':predictions})
np.savetxt('./data/prediction.csv', predictions, delimiter=',')
