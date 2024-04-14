import torch
import numpy as np
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

from autoencoder_SINDy import AutoencoderSINDy,define_loss

def train_network(training_data, val_data, net_params, device):

    """
    Training process of the autoencoder SINDy.
      - First the network is created.
      - Then, a training if performed for a number of max_epochs. In this training,
      sequenctial thresholding is performed (if required)
      - The second training of refinement does not consider regularization losses
      so that the SINDy coeffs. adapt to the system

    Arguments:
        training_data: training data for the model generated from several
        initial conditions
        val_data: small test data set for validation purposes
        net_params: dictionary of network parameters including autoencoder parameters
        and SINDy parameters

    Return:
        results_dict: dictionary of results obtained from the training (losses,
        SINDy model terms, etc.)

    """

    ## Training set up
    #net_params = net_params

    # Set up network
    autoencoder_network = AutoencoderSINDy(net_params,device)
    autoencoder_network = autoencoder_network.to(device) # Move to GPU

    network = autoencoder_network.network
    network['coeff_mask'] =  network['coeff_mask'].to(device)
    network = {key: tensor.to(device) for key, tensor in network.items()}

    # Define losses
    loss, losses, loss_refinement = define_loss(network, net_params, device)
    # Get learning rate and choose Adam optimizer for both full and refinment training
    learning_rate = net_params['learning_rate']
    optimizer = optim.Adam(autoencoder_network.parameters(), lr=learning_rate)
    optimizer_refinement = optim.Adam(autoencoder_network.parameters(), lr=learning_rate)

    # Create validation dictionary
    validation_dict = create_feed_dictionary(val_data, net_params, idxs=None)
    validation_dict = {key: tensor.to(device) for key, tensor in validation_dict.items()}

    # Compute the norm of the validation data (don't know why right now -> see Model slection in the SI Appendix)
    x_norm = torch.mean(val_data['x']**2).item()
    if net_params['model_order'] == 1:
        sindy_predict_norm_x = torch.mean(val_data['dx']**2).item()
    else:
        sindy_predict_norm_x = torch.mean(val_data['ddx']**2).item()

    # Define list to save validation losses
    validation_losses = []

    # Compute the number of terms in the SINDy mask coefficient matrix (note
    # that they can only be 1's or 0's)
    sindy_model_terms = [torch.sum(network['coeff_mask'])]

    # Number of thresholds
    n_threshold = 0

    ## Begin the training process
    print('TRAINING')
    # For each epoch
    for i in range(net_params['max_epochs']):
        # For each batch
        for j in range(net_params['epoch_size']//net_params['batch_size']):
            # Select the batch indices to introduce in the Autoencoder + SINDy
            batch_idxs = torch.arange(j*net_params['batch_size'], (j+1)*net_params['batch_size'])
            # Generate the x, dx, ddx and lr values to introduce in the training
            train_dict = create_feed_dictionary(training_data, net_params, idxs=batch_idxs)
            # Reset all gradients of trainable parameters
            optimizer.zero_grad()
            # Update network inputs
            network['x'] = train_dict['x']
            network['dx'] = train_dict['dx']
            network['ddx'] = train_dict['ddx']
            # Make forward pass
            network = autoencoder_network.forward(train_dict['x'], train_dict['dx'], train_dict['ddx'], network, net_params)
            # Compute losses from forward pass
            loss, losses, _ = define_loss(network, net_params, device)
            # Backward - Gradients
            loss.backward()
            # Optimizer step
            optimizer.step()

        if i % 10 == 0:
            print("Epoch %d" % i)

        with torch.no_grad():
            # Print progress if desired (print_progress) and by a frequency (print_frequency)
            if net_params['print_progress'] and (i % net_params['print_frequency'] == 0):
                # Update network inputs
                network['x'] = validation_dict['x']
                network['dx'] = validation_dict['dx']
                network['ddx'] = validation_dict['ddx'] 
                # Losses from training are already computed
                # Next step is to calculate the validation dictionary losses
                # For that we need a forward pass for the validation_dic
                # The next line uses the training_network tuple and updates the ...
                # necessary values (z, predict, etc.) and saves them in a new tuple
                validation_forward = autoencoder_network.forward(validation_dict['x'], validation_dict['dx'], validation_dict['ddx'], network, net_params)
                # Compute validation losses
                val_loss, val_losses, _ = define_loss(validation_forward, net_params, device)
                # Function that prints progress and appends validation losses
                validation_losses.append(print_progress(i, loss, losses, val_loss, val_losses, x_norm, sindy_predict_norm_x))


        # Update coefficient mask if sequential thresholding is active
        # at a frequency (threshold_frequency)
        if net_params['sequential_thresholding'] and (i % net_params['threshold_frequency'] == 0) and (i > 0):
            # Set to zero the values of the coefficient if the sindy coefficients are below threshold
            network['coeff_mask'] = (torch.abs(network['sindy_coeffs']) > net_params['coefficient_threshold']).to(torch.float32).to(device)
            # Add new coeffcicient mask to validation dicctionary
            validation_dict['coeff_mask'] = network['coeff_mask']
            # Print that thresholding has been done and the number of active coefficients
            print('THRESHOLDING: %d active coefficients' % torch.sum(network['coeff_mask']))
            # Save the new number of active terms (coeff_mask = 1)
            sindy_model_terms.append(torch.sum(network['coeff_mask']))
            print('SINDy COEFFICIENTS:')
            print(torch.mul(network['coeff_mask'],network['sindy_coeffs']))
            # Start dropout after determined threshold
            n_threshold = n_threshold + 1
            if net_params['p_dropout_threshold'][0] == n_threshold:
                net_params['p_dropout'] = net_params['p_dropout_threshold'][1]
                print('DROPOUT ACTIVATED: p = ', net_params['p_dropout'])


    ## Begin the refinement process
    # Note that the refinement does not perform any threshold on the coefficient mask
    # and no loss for sindy coefficient regularization is computed
    print('REFINEMENT')
    # For each refinement epoch
    for i_refinement in range(net_params['refinement_epochs']):
        # For each refinement batch (equal to training)
        for j in range(net_params['epoch_size']//net_params['batch_size']):
            # Select the batch indices to introduce in the Autoencoder + SINDy
            batch_idxs = torch.arange(j*net_params['batch_size'], (j+1)*net_params['batch_size'])
            # Generate the x, dx, ddx and lr values to introduce in the training
            train_dict = create_feed_dictionary(training_data, net_params, idxs=batch_idxs)
            # Reset all gradients of trainable parameters
            optimizer_refinement.zero_grad()
            # Update network inputs
            network['x'] = train_dict['x']
            network['dx'] = train_dict['dx']
            network['ddx'] = train_dict['ddx'] 
            # Make forward pass
            network = autoencoder_network.forward(train_dict['x'], train_dict['dx'], train_dict['ddx'], network, net_params)
            # Compute losses from forward pass
            _, losses, loss_refinement = define_loss(network, net_params, device)
            # Backward - Gradients on loss_refinement
            loss_refinement.backward()
            # Optimizer step
            optimizer_refinement.step()

        if i_refinement % 10 == 0:
            print("Epoch %d" % i_refinement)

        with torch.no_grad():
            # Print progress if desired (print_progress) and by a frequency (print_frequency)
            if net_params['print_progress'] and (i_refinement % net_params['print_frequency'] == 0):
                # Update network inputs
                network['x'] = validation_dict['x']
                network['dx'] = validation_dict['dx']
                network['ddx'] = validation_dict['ddx'] 
                # Losses from training are already computed
                # Next step is to calculate the validation dictionary losses
                # For that we need a forward pass for the validation_dic
                # The next line uses the training_network tuple and updates the ...
                # necessary values (z, predict, etc.) and saves them in a new tuple
                validation_forward = autoencoder_network.forward(validation_dict['x'], validation_dict['dx'], validation_dict['ddx'], network, net_params)
                # Compute validation losses
                _, val_losses, val_loss_refinement = define_loss(validation_forward, net_params, device)
                # Function that prints progress and appends validation losses
                validation_losses.append(print_progress(i_refinement, loss_refinement, losses, val_loss_refinement, val_losses, x_norm, sindy_predict_norm_x))


    ## Save torch state and picke file (dunno what's this for, just copied and
    # translated it to Pytorch hehe)
    torch.save(autoencoder_network.state_dict(), net_params['data_path'] + net_params['save_name'] + '_model.pth')
    pickle.dump(net_params, open(net_params['data_path'] + net_params['save_name'] + '_net_params.pkl', 'wb'))
    pickle.dump(network, open(net_params['data_path'] + net_params['save_name'] + '_network_parameters.pkl', 'wb'))

    ## Compute final losses
    #final_losses = (losses['decoder'], losses['sindy_x'], losses['sindy_z'],
    #                             losses['sindy_regularization'])


    ## Compute the norm of the output dz or ddz
    if net_params['model_order'] == 1:
        sindy_predict_norm_z = torch.mean((network['dz'])**2)
    else:
        sindy_predict_norm_z = torch.mean((network['ddz'])**2)
    # Sindy coefficients (mask included)
    sindy_coefficients = network['sindy_coeffs']



    results_dict = {}
    results_dict['num_epochs'] = i
    results_dict['x_norm'] = x_norm
    results_dict['sindy_predict_norm_x'] = sindy_predict_norm_x
    results_dict['sindy_predict_norm_z'] = sindy_predict_norm_z
    results_dict['sindy_coefficients'] = sindy_coefficients
    results_dict['loss_decoder'] = losses['decoder'] #final_losses['decoder']
    results_dict['loss_decoder_sindy'] = losses['sindy_x'] #final_losses['sindy_x']
    results_dict['loss_sindy'] = losses['sindy_z'] #final_losses['sindy_z']
    results_dict['loss_sindy_regularization'] = losses['sindy_regularization'] #final_losses['sindy_regularization']

    # Convert CUDA tensors to NumPy arrays and store them in a new list
    validation_losses_np_array = np.zeros([len(validation_losses),len(validation_losses[0])])
    for i, tuple in enumerate(validation_losses):
        for j, tensor in enumerate(tuple):
            validation_losses_np_array[i,j] = tensor.cpu().detach().numpy()  # Convert tensor to NumPy array
    
    results_dict['validation_losses'] = validation_losses_np_array
    
    results_dict['sindy_model_terms'] = np.stack([tensor.cpu().detach().numpy() for tensor in sindy_model_terms], axis=0)

    return results_dict



def print_progress(i, loss, losses, val_loss, val_losses, x_norm, sindy_predict_norm):
    """
    Print loss function values to keep track of the training progress.

    Arguments:
        i - the training iteration
        loss - PyTorch tensor representing the total loss function used in training
        losses - dictionary of the individual losses that make up the total loss
        val_losses - PyTorch tensor of validation losses
        val_losses - dictionary of the validation individual losses that make up the total loss
        x_norm - float, the mean square value of the input
        sindy_predict_norm - float, the mean square value of the time derivatives of the input.
        Can be first or second order time derivatives depending on the model order.

    Returns:
        Tuple of losses calculated on the validation set.
    """
    training_loss_vals = (loss,) + tuple(losses.values())
    validation_loss_vals = (val_loss,) + tuple(val_losses.values())
    print("Epoch %d" % i)
    print("   training loss {0}, {1}".format(training_loss_vals[0],
                                             training_loss_vals[1:]))
    print("   validation loss {0}, {1}".format(validation_loss_vals[0],
                                               validation_loss_vals[1:]))
    decoder_losses = (val_losses['decoder'],val_losses['sindy_x'])
    loss_ratios = (decoder_losses[0] / x_norm, decoder_losses[1] / sindy_predict_norm)
    print("decoder loss ratio: %f, decoder SINDy loss  ratio: %f" % loss_ratios)
    return validation_loss_vals



def create_feed_dictionary(data, net_params, idxs=None):
    """
    Create the feed dictionary for passing into PyTorch.

    Arguments:
        data - Dictionary object containing the data to be passed in. Must contain input data x,
        along the first (and possibly second) order time derivatives dx (ddx).
        net_params - Dictionary object containing model and training parameters. The relevant
        parameters are model_order (which determines whether the SINDy model predicts first or
        second order time derivatives), sequential_thresholding (which indicates whether or not
        coefficient thresholding is performed), and learning rate (float that determines the learning rate).
        idxs - Optional array of indices that selects which examples from the dataset are passed
        in to PyTorch. If None, all examples are used.

    Returns:
        feed_dict - Dictionary object containing the relevant data to pass to PyTorch.
    """
    if idxs is None:
        idxs = torch.arange(data['x'].shape[0])
    feed_dict = {}
    feed_dict['x'] = torch.tensor(data['x'][idxs])
    feed_dict['dx'] = torch.tensor(data['dx'][idxs])
    feed_dict['ddx'] = torch.tensor(data['ddx'][idxs])
    #if net_params['model_order'] == 2:
        #feed_dict['ddx'] = torch.tensor(data['ddx'][idxs])
    feed_dict['learning_rate'] = torch.tensor(net_params['learning_rate'])
    return feed_dict
