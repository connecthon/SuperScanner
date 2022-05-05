import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from scipy.io import loadmat
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import os

class voxelnet(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(voxelnet, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.layer1  = nn.Linear(self.in_dim, self.hid_dim)
        self.relu1   = nn.ReLU()
        self.layer2  = nn.Linear(self.hid_dim, self.hid_dim)
        self.relu2   = nn.ReLU()
        self.layer3  = nn.Linear(self.hid_dim, self.hid_dim)
        self.relu3   = nn.ReLU()
        self.out_layer = nn.Linear(self.hid_dim, self.out_dim)

    def forward(self, x):
        output = self.relu1(self.layer1(x))
        output = self.relu2(self.layer2(output))
        output = self.relu3(self.layer3(output))
        output = self.out_layer(output)
        return output

def load_data_voxels(path, X_col_idx, y_col_idx, num_y_cols = 1, train_test_split=0.7, batch_size=25, pca_var=None, device='cpu'):
    """
    Load the data for training. Supported format: ".mat".

    Parameters
    ----------
    path - str: path of the data to load
    X_col_idx - int: index, all the previous columns to be used as X
    y_col_id  - int: index of the first column for y
    num_y_cols - int: number of columns of y data
    train_test_split - float: percentage of training data
    batch_size - int: number of samples per batch
    devide - str: device where to load the data (e.g. cpu or cuda)

    Returns
    -------
    train_data, test_data - tuple of torch.tensor: train and test datasets
    """

    if path[-4:] == ".mat":
        data = loadmat(path)['S']
    else:
        raise TypeError('Please select a valid format. Supported files: ".mat"')

    mask = np.ones(data.shape[0], dtype=bool)
    for i in range(data.shape[0]):
        mask[i] = ~(data[i].sum() == 1)
        
    new_data = data[mask]

    if isinstance(pca_var, float) :
        red_data = PCA_reduction(new_data[:,:X_col_idx], pca_var)
        X = torch.tensor(red_data).to(device).float()
        y = torch.tensor(new_data[:,y_col_idx:y_col_idx+num_y_cols]).to(device).float()
    else :
        red_data=new_data
        X = torch.tensor(new_data[:,X_col_idx]).to(device).float()
        y = torch.tensor(new_data[:,y_col_idx:y_col_idx+num_y_cols]).to(device).float()

    train_length = int(train_test_split*red_data.shape[0])
    test_length  = int((red_data.shape[0] - train_length)/2)
    valid_length = red_data.shape[0] - train_length - test_length

    tensor_data = TensorDataset(X,y)

    train_data, test_data, valid_data = tuple(random_split(tensor_data, [train_length, test_length, valid_length]))
    train_dl = DataLoader(train_data, batch_size=batch_size)
    test_dl = DataLoader(test_data, batch_size=None)
    valid_dl = DataLoader(valid_data, batch_size=None)
    return train_dl, test_dl, valid_dl

def PCA_reduction(data, exp_var=0.8):
    pca = PCA()
    pca.fit(data)
    variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = min(np.where(variance >= exp_var)[0]) + 1
    pca_reduce = PCA(n_components=n_components)
    reduced_data = pca_reduce.fit_transform(data)
    return reduced_data
    
def train(model, train_dl, test_dl, optimizer, loss_function=torch.nn.MSELoss(), num_epochs=5):
    """
    Function for model training

    Parameters
    ----------
    model - PyTorch model to be trained
    train_dl - PyTorch DataLoader: training data
    test_dl  - PyTorch DataLoader: testing data
    optimizer - PyTorch optimizer for loss minimization
    loss_function - PyTorch loss function to minimize
    num_epochs - number of training iterations
    Returns
    -------
    model - state dictionary of weights for the trained model
    """

    for epoch in range(0, num_epochs):

        print(f'Starting epoch {epoch+1}')

        current_loss = 0.0
        model.train()
        for i, (inputs, targets) in enumerate(train_dl, 0):             
            optimizer.zero_grad()
            outputs = model(inputs)
            train_loss = loss_function(outputs, targets)
            
            train_loss.backward()
            optimizer.step()

            current_loss += train_loss.item()

        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            for i, (test_inputs, test_targets) in enumerate(test_dl, 0):
                test_outputs = model(test_inputs)
                test_loss += loss_function(test_outputs, test_targets)
            
        print(f'Epoch {epoch}:\n'
              f'Train loss: {current_loss/len(train_dl):.4f}\n'
              f'Test loss: {test_loss/len(test_dl):.4f}\n'
              f'----------')

    print('Training process has finished. Saving the model...')
    return model.state_dict()
    
def evaluate_plot(trained_model, trained_model_path, valid_dl, save_fig=True, device='cpu'):

    valid_data = valid_dl.dataset
    trained_model.load_state_dict((torch.load(trained_model_path)))
    model_name=os.path.basename(trained_model_path)

    trained_model.eval()
    inputs = valid_dl.dataset[:][0]
    real_outputs = valid_dl.dataset[:][1]
    outputs = trained_model(inputs)

    fig, ax = plt.subplots()

    ax.set_title('Validation data')
    ax.set_xlabel('Real data')
    ax.set_ylabel('Predicted data')
    ax.plot(np.arange(0,1,0.1), np.arange(0,1,0.1), "r--", alpha=0.5, label = "y=x")
    ax.scatter(real_outputs.detach().cpu().numpy(), outputs.detach().cpu().numpy(), s=1, alpha=0.3, color='k')
    
    if save_fig:
        plt.savefig('./data/performance_' + model_name + '.jpg')
    plt.show()
    
    return outputs.detach().cpu().numpy()
