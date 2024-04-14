import sys
sys.path.append("././src")

#import os
import datetime
import numpy as np
from example_pendulum import get_pendulum_data
import torch
import pickle
from autoencoder_SINDy import AutoencoderSINDy, define_loss
from training import create_feed_dictionary

# For Plots
import matplotlib.pyplot as plt


# Use GPU if available (There are some problems with multiple devices, since it is only for a single forward pass we can use CPU)
device = 'cpu' 

# Load net params (used to create the model) (Introduce the net_parmas file name)
with open('pendulum_A_net_params.pkl', 'rb') as f:
    net_params = pickle.load(f)


# Load network parameters (non-traineable parameters obtained at the end of the training) (Introduce the network_parameters file name)
with open('pendulum_A_network_parameters.pkl', 'rb') as f:
    network = pickle.load(f)

network = {key: tensor.to(device) for key, tensor in network.items()}

net_params['p_dropout'] = 0

autoencoder_network = AutoencoderSINDy(net_params,device)

autoencoder_network.load_state_dict(torch.load('pendulum_A_model.pth'))

autoencoder_network.network = network

loss, losses, loss_refinement = define_loss(network, net_params, device)

# Initialize figures outside the loop
fig1, ax1 = plt.subplots()

fig3, ax2 = plt.subplots()

for i in range(6):
    test_data = get_pendulum_data(1)

    # Create testing dictionary
    test_dict = create_feed_dictionary(test_data, net_params, idxs=None)
    test_dict = {key: tensor.to(device) for key, tensor in test_dict.items()}

    # Forward pass
    network_parameters = autoencoder_network.forward(test_dict['x'], test_dict['dx'], test_dict['ddx'], network, net_params)

    t_test  = test_data['t']
    z_test  = test_data['z']
    z_predict = network_parameters['z'].detach().numpy()
    dz_predict = network_parameters['dz'].detach().numpy()

    # Add new curves to existing plots
    ax1.plot(z_predict, dz_predict,color='#1f77b4')

ax2.plot(t_test, z_predict, '--', color='gray', label=f'Model {i+1}')
ax2.plot(t_test, z_test, '-', color='blue', label=f'Test {i+1}')

# Set labels and legends
ax1.set_xlabel('z')
ax1.set_ylabel('dz/dt')
ax1.set_title('Pendulum prediction')
ax1.legend()

ax2.set_xlabel('t')
ax2.set_ylabel('z')
ax2.set_title('Time evolution')
ax2.legend()

# Show the plots with all added curves
plt.show()