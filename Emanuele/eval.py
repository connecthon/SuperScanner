import net

_, _, valid_data = net.load_data_voxels('./data/sub-06400_ndwi.mat', 139, 140, num_y_cols=1, batch_size=50,  device='cpu')

trained_model = net.voxelnet(139,100,1).to('cpu')
net.evaluate_plot(trained_model, './models/voxelnet', valid_data)
