import net
from scipy.io import savemat
import numpy as np

X=[0,34,55,27,137]
Y=265

_, _, valid_data = net.load_data_voxels('./data/sub-06400_ndwi.mat', X, Y, num_y_cols=1, batch_size=50,  device='cpu')

num_hidden_dims=10
trained_model = net.voxelnet(len(X),num_hidden_dims,1).to('cpu')
predictions = net.evaluate_plot(trained_model, './models/voxelnet_' + str(Y), valid_data)

savemat('./data/prediction' + str(Y) +'.mat', {'predictions':predictions})
np.savetxt('./data/prediction' + str(Y) + '.csv', predictions, delimiter=',')
