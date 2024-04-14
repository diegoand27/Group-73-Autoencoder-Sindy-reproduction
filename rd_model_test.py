import numpy as np
from example_rd import get_rd_data
from SINDy_utils import library_size
from training import train_network
import torch
import pickle
from autoencoder_SINDy import AutoencoderSINDy
from training import create_feed_dictionary

# For Plots
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Use GPU if available (There are some problems with multiple devices, since it is only for a single forward pass we can use CPU)
device = 'cpu'

# Load net params (used to create the model) (Introduce the net_parmas file name)
with open('rd_2OG_net_params.pkl', 'rb') as f:
    net_params = pickle.load(f)

# Load network parameters (non-traineable parameters obtained at the end of the training) (Introduce the network_parameters file name)
with open('rd_2OG_network_parameters.pkl', 'rb') as f:
    network = pickle.load(f)

network = {key: tensor.to(device) for key, tensor in network.items()}

net_params['p_dropout'] = 0

autoencoder_network = AutoencoderSINDy(net_params,device)

# Load autoencoder model (non-traineable parameters obtained at the end of the training) (Introduce the model file name)
autoencoder_network.load_state_dict(torch.load('rd_2OG_model.pth'))

autoencoder_network.network = network

# Generate testing data
noise_strength = 1e-6
_, _, test_data = get_rd_data()

# Create testing dictionary
test_dict = create_feed_dictionary(test_data, net_params, idxs=None)
test_dict = {key: tensor.to(device) for key, tensor in test_dict.items()}

# Forward pass
network_parameters = autoencoder_network.forward(test_dict['x'], test_dict['dx'], test_dict['ddx'], network, net_params)


t_test = test_data['t']
z_predict = network_parameters['z'].detach().numpy()

# Print SINDy Coeffs
print('SINDy Coeffs: ', torch.sum(network['coeff_mask']))
print(network['sindy_coeffs'])

# Spatial data
fig1 = plt.figure()
plt.plot(z_predict[:,0], z_predict[:,1])
plt.xlabel('z1')
plt.ylabel('z2')
plt.title('Encoded system')
plt.axis('equal')

# Time evolution
fig3 = plt.figure()
plt.plot(t_test, z_predict[:,0],'--',color='gray',label='Model z1')
plt.xlabel('t')
plt.ylabel('z')
plt.title('Time evolution z1')
plt.legend()

# Time evolution
fig4 = plt.figure()
plt.plot(t_test, z_predict[:,1],'--',color='gray',label='Model z2')
plt.xlabel('t')
plt.ylabel('z')
plt.title('Time evolution z2')
plt.legend()


plt.show()