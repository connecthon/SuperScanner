import net
from scipy.io import savemat
import numpy as np
from numpy import genfromtxt

use_pca=False
Y=139

if use_pca:
    _, _, valid_data = net.load_data_voxels('./data/sub-06400_ndwi.mat', 139, Y, num_y_cols=1, batch_size=50,  pca_var=0.9, device='cpu')
    in_dim = valid_data.dataset[:][0].shape[1]
    print(in_dim)
    trained_model = net.voxelnet(in_dim,10,1).to('cpu')
    model_name='voxelnet_' + str(Y) + '_pca'
    #predictions = net.evaluate_plot(trained_model, './models/voxelnet_pca', valid_data)
    #savemat('./data/prediction_pca.mat', {'predictions':predictions})
    #np.savetxt('./data/prediction_pca.csv', predictions, delimiter=',')
else :
    closest_dirs = genfromtxt('closest_dirs_0_to_265.txt' , delimiter=',')
    X=closest_dirs[Y,0:5]
    X=np.ndarray.tolist(X.astype(int))
    _, _, valid_data = net.load_data_voxels('./data/sub-06400_ndwi.mat', X, Y, num_y_cols=1, batch_size=50,  device='cpu')
    num_hidden_dims=10
    trained_model = net.voxelnet(len(X),num_hidden_dims,1).to('cpu')
    model_name='voxelnet_' + str(Y)
    #predictions = net.evaluate_plot(trained_model, './models/voxelnet_' + str(Y), valid_data)
    #savemat('./data/prediction' + str(Y) +'.mat', {'predictions':predictions})
    #np.savetxt('./data/prediction' + str(Y) + '.csv', predictions, delimiter=',')

predictions = net.evaluate_plot(trained_model, './models/' + model_name, valid_data)
savemat('./data/prediction_' + model_name +'.mat', {'predictions':predictions})
np.savetxt('./data/prediction_' + model_name + '.csv', predictions, delimiter=',')
