import torch
import torch.nn as nn

from encoder_decoder import Encoder,Decoder
from SINDy import SINDy_order1, SINDy_order2

## Autoencoder SINDy (x_decoder + dz)
# Autoencoder: encoder + decoder
# Sindy: can be of first or second order. Therefore, will always take z
# as an input and if of higher order the dz will come into place. Note
# that from the input all the derivatives are known (x,dx,ddx), so it is
# straightforward to evaluate dz and ddz from the encoder and the adequate
# differentiation (see SI Appendix)

class AutoencoderSINDy(nn.Module):
    """
    Fully conected Autoencoder with non-linear activation functions except for
    the last encoder and decoder layers. The SINDy model is attached to the
    latent dimensions, and predicts first or second order derivatives depending
    on the model order.

    Args:
        net_params: tuple of network parameters including autoencoder parameters
        and SINDy parameters

    Return:
        network: tuple with essential variables that will be used along the
        training process

    """
    def __init__(self,net_params,device):
        super(AutoencoderSINDy, self).__init__()

        ## Unpack net_params list:
        # Layer dimensions
        input_dim = net_params['input_dim']
        hdims = net_params['hdims']
        latent_dim = net_params['latent_dim']

        # Activation function
        act_func = net_params['activation']

        # SINDy library
        model_order = net_params['model_order']
        poly_order = net_params['poly_order']
        if 'include_sine' in net_params.keys():
            include_sine = net_params['include_sine']
        else:
            include_sine = False
        library_dim = net_params['library_dim']

        # Dropout probability
        p_dropout = net_params['p_dropout']

        # Initialization of SINDy coeffs
        coeff_init = net_params['coefficient_initialization']

        # Sequential thresholding for SINDy coeffs (not used here for
        # this implementation)
        seq_threshold = net_params['sequential_thresholding']

        ## Empty list to store later important network parameters
        #network = {}

        ## Define placeholders (input tensors) in PyTorch
        x = torch.Tensor(net_params['n_samples'], input_dim) # Placeholder for x
        dx = torch.Tensor(net_params['n_samples'], input_dim)  # Placeholder for dx

        # Check for model_order to determine if ddx placeholder is needed
        if model_order == 2:
            ddx = torch.Tensor(net_params['n_samples'], input_dim) # Placeholder for ddx


        ## Define Autoencoder blocks
        # Encoder + Decoder
        self.encoder = Encoder(input_dim,hdims,latent_dim,act_func,p_dropout)
        self.decoder = Decoder(latent_dim,hdims,input_dim,act_func,p_dropout)

        ## Define the latent dimension tensors (z, dz, ddz)
        z = self.encoder.forward(x,act_func)
        if model_order == 1:
          dz = self.encoder.forward_order1(x, dx, act_func)
        else:
          dz,ddz = self.encoder.forward_order2(x, dx, ddx, act_func)

        ## Define SINDy block and initialize sindy coeffs
        if model_order == 1:
          self.SINDy = SINDy_order1(z,latent_dim,library_dim,poly_order,include_sine,device)
        else:
          self.SINDy = SINDy_order2(z,dz,latent_dim,library_dim,poly_order,include_sine,device)

        ## Define SINDy prediction
        sindy_predict = torch.matmul(self.SINDy.Theta, torch.mul(self.SINDy.coeff_mask,self.SINDy.Xi))

        ## Define x_hat, dx_hat and ddx_hat from SINDy prediction
        x_decoder = self.decoder.forward(z,act_func)
        if model_order == 1:
          dx_decoder = self.decoder.forward_order1(z, sindy_predict, act_func)
        else:
          dx_decoder,ddx_decoder = self.decoder.forward_order2(z, dz, sindy_predict, act_func)

        self.network = {}

        ## Define list of important network parameters that will be used
        self.network['x'] = x
        self.network['dx'] = dx
        self.network['z'] = z
        self.network['dz'] = dz
        self.network['x_decoder'] = x_decoder
        self.network['dx_decoder'] = dx_decoder
        self.network['Theta'] = self.SINDy.Theta
        self.network['coeff_mask'] = self.SINDy.coeff_mask
        self.network['Xi'] = self.SINDy.Xi
        self.network['sindy_coeffs'] = torch.mul(self.network['coeff_mask'],self.network['Xi'])

        if model_order == 1:
            self.network['dz_predict'] = sindy_predict
        else:
            self.network['ddz'] = ddz
            self.network['ddz_predict'] = sindy_predict
            self.network['ddx'] = ddx
            self.network['ddx_decoder'] = ddx_decoder

        #return network

    # Compute outputs from each block
    def forward(self, x, dx, ddx, network, net_params):
        """
        Forward pass for a fully conected Autoencoder with non-linear activation
        functions except for the last encoder and decoder layers and a SINDy
        model attached to the latent dimensions.

        Forward pass computes:
          - The latent features and reconstructed input features.
          - SINDy library and latent features derivatives (from input feat.)
          - SINDy prediction from SINDy library and coefficients
          - Reconstructed input derivatives

        Args:
            x: input features
            dx: derivatives of input features
            ddx: second order derivatives of input features
            network: tuple with essential variables that will be used along the
            training process
            net_params: tuple of network parameters including autoencoder parameters
            and SINDy parameters

        Return:
            network: tuple with essential variables that will be used along the
            training process and updated for the forward pass

        """
        # Encoder output - Latent dimension
        z = self.encoder(x,net_params['activation'])
        network['z'] = z

        # Decoder output - Reconstruction x_hat
        x_decoder = self.decoder(z,net_params['activation'])
        network['x_decoder'] = x_decoder

        # Compute dz and ddz based on the provided dx and ddx
        # Compute the SINDy library Theta
        if net_params['model_order'] == 1:
          dz = self.encoder.forward_order1(x, dx, net_params['activation'])
          network['Theta'] = self.SINDy.library_update(z)
          network['dz'] = dz
        else:
          dz,ddz = self.encoder.forward_order2(x, dx, ddx, net_params['activation'])
          network['Theta'] = self.SINDy.library_update(z,dz)
          network['dz'] = dz
          network['ddz'] = ddz

        # Compute the SINDy prediction for z_dot or z_dotdot
        network['sindy_coeffs'] = torch.mul(network['coeff_mask'], network['Xi'])
        sindy_predict = torch.matmul( network['Theta'], network['sindy_coeffs'] )

        # Compute dx_decoder and ddx_decoder based on the predicted dz and ddz
        if net_params['model_order'] == 1:
          dx_decoder = self.decoder.forward_order1(z, sindy_predict, net_params['activation'])
          network['dx_decoder'] = dx_decoder
          network['dz_predict'] = sindy_predict
        else:
          dx_decoder,ddx_decoder = self.decoder.forward_order2(z, dz, sindy_predict, net_params['activation'])
          network['dx_decoder'] = dx_decoder
          network['ddx_decoder'] = ddx_decoder
          network['ddz_predict'] = sindy_predict

        return network



# Defining the total loss function
def define_loss(network, net_params, device):
    """
    Create the loss functions:
      - Reconstruced loss: 'decoder'
      - SINDy loss in dx (or ddx): 'sindy_x'
      - SINDy loss in dz (or ddz): 'sindy_z'
      - SINDy regularization loss: 'sindy_regularization'

    Arguments:
        network: Dictionary object containing the elements of the network
        architecture.
        net_params: Dictionary of network parameters including autoencoder parameters
        and SINDy parameters

    Return:
        loss: total loss
        losses: tuple with for type of losses
        loss_refinement: total loss without SINDy regularization

    """

    ## Unpack network dictionary
    x = network['x']
    x_decoder = network['x_decoder']

    if net_params['model_order'] == 1:
        dz = network['dz']
        dz_predict = network['dz_predict']
        dx = network['dx']
        dx_decoder = network['dx_decoder']
    else:
        ddz = network['ddz']
        ddz_predict = network['ddz_predict']
        ddx = network['ddx']
        ddx_decoder = network['ddx_decoder']

    sindy_coefficients = torch.mul(network['coeff_mask'],network['Xi'])

    ## Create losses
    losses = {}

    # Reconstruction losses
    losses['decoder'] = torch.mean( (x - x_decoder)**2 ).to(device)

    # SINDy losses in x and z
    if net_params['model_order'] == 1:
        losses['sindy_z'] = torch.mean( (dz - dz_predict)**2 ).to(device)
        losses['sindy_x'] = torch.mean( (dx - dx_decoder )**2 ).to(device)
    else:
        losses['sindy_z'] = torch.mean( (ddz - ddz_predict)**2 ).to(device)
        losses['sindy_x'] = torch.mean( (ddx - ddx_decoder )**2 ).to(device)
    # Regularization SINDy losses
    losses['sindy_regularization'] = torch.mean(torch.abs(sindy_coefficients)).to(device)

    ## Total loss computation
    # Full training
    loss =   net_params['loss_weight_decoder'] * losses['decoder'] \
           + net_params['loss_weight_sindy_z'] * losses['sindy_z'] \
           + net_params['loss_weight_sindy_x'] * losses['sindy_x'] \
           + net_params['loss_weight_sindy_regularization'] * losses['sindy_regularization']

    # Refinement training: No regularization
    loss_refinement =   net_params['loss_weight_decoder'] * losses['decoder'] \
                      + net_params['loss_weight_sindy_z'] * losses['sindy_z'] \
                      + net_params['loss_weight_sindy_x'] * losses['sindy_x']

    return loss, losses, loss_refinement