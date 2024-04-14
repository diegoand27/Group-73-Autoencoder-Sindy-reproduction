import sys
sys.path.append("../src")
import os

import datetime
import numpy as np
from example_lorenz import get_lorenz_data
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
device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'

# Load results dict (Introduce the results_dict file name)
#with open('results_dict_14_coeffs.pkl', 'rb') as f:
#    results_dict = pickle.load(f)

# Load net params (used to create the model) (Introduce the net_parmas file name)
with open('lorenz_10_net_params.pkl', 'rb') as f:
    net_params = pickle.load(f)

# Load network parameters (non-traineable parameters obtained at the end of the training) (Introduce the network_parameters file name)
with open('lorenz_10_network_parameters.pkl', 'rb') as f:
    network = pickle.load(f)

network = {key: tensor.to(device) for key, tensor in network.items()}

net_params['p_dropout'] = 0

autoencoder_network = AutoencoderSINDy(net_params,device)

autoencoder_network.load_state_dict(torch.load('lorenz_10_model.pth'))

autoencoder_network.network = network

# Generate testing data
noise_strength = 1e-6
test_data = get_lorenz_data(1, noise_strength=noise_strength)

# Create testing dictionary
test_dict = create_feed_dictionary(test_data, net_params, idxs=None)
test_dict = {key: tensor.to(device) for key, tensor in test_dict.items()}

# Forward pass
network_parameters = autoencoder_network.forward(test_dict['x'], test_dict['dx'], test_dict['ddx'], network, net_params)

t_test  = t = np.arange(0, 5, .02)
z_test = test_data['z']
z_predict = network_parameters['z'].detach().numpy()

# Print SINDy Coeffs
print('SINDy Coeffs: ', torch.sum(network['coeff_mask']))
print(network['sindy_coeffs'])

# Original data
fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.plot(z_test[0,:,0], z_test[0,:,1], z_test[0,:,2])
ax.set_xlabel('z1')
ax.set_ylabel('z2')
ax.set_zlabel('z3')
ax.set_title('Original system')


# Prediction
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot(z_predict[:,0], z_predict[:,1], z_predict[:,2])
ax2.set_xlabel('z1')
ax2.set_ylabel('z2')
ax2.set_zlabel('z3')
ax2.set_title('Encoded sytem')


# Time evolution z1
fig4 = plt.figure()
plt.plot(t_test, z_test[0,:,0], label='Test z1')
plt.plot(t_test, z_predict[:,0],'--',color='gray',label='Model z1')
plt.xlabel('t')
plt.ylabel('z')
plt.title('Time evolution z1')
plt.legend()

# Time evolution z2
fig5 = plt.figure()
plt.plot(t_test, z_test[0,:,1], label='Test z2')
plt.plot(t_test, z_predict[:,1],'--',color='gray',label='Model z2')
plt.xlabel('t')
plt.ylabel('z')
plt.title('Time evolution')
plt.legend()

# Time evolution z3
fig3 = plt.figure()
plt.plot(t_test, z_test[0,:,2], label='Test z3')
plt.plot(t_test, z_predict[:,2],'--',color='gray',label='Model z3')
plt.xlabel('t')
plt.ylabel('z')
plt.title('Time evolution')
plt.legend()

plt.show()