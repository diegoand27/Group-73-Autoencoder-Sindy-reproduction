import torch
import torch.nn as nn
import torch.nn.init as init


## Encoder (Phi)
# Input: position field X at a time t
# Output: latent dimensions

class Encoder(nn.Module):
    """
    Fully conected encoder with non-linear activation functions except for
    the last network layer

    Args:
        input_dim: number of input features
        hdims: number of widths between the input and latent features
        latent_dim: number of latent features
        act_func: activation function that is used

    """
    # Define your encoder
    def __init__(self, input_dim, hdims, latent_dim, act_func, p_dropout):
        super(Encoder, self).__init__()

        # Encoder layers:
        # Input: input_dim
        # Hiden: hdims
        # Latent dimension: latent_dims
        # Non-linear functions are applied for all except for the last

        # Dropout probability
        self.p_dropout = p_dropout

        self.layers = nn.ModuleList()  # Container for encoder layers
        # Create linear layers based on hdims
        prev_dim = input_dim
        for hdim in hdims:
            self.layers.append(nn.Linear(prev_dim, hdim))
            prev_dim = hdim

        # Add final linear layer to map to latent_dim
        self.layers.append(nn.Linear(prev_dim, latent_dim))
        self.dropout = nn.Dropout(p=p_dropout)
        self.sigmoid = nn.Sigmoid()

        # Initialize weigths and biases
        self.init_params()

    # Initialization of weights and biases
    def init_params(self):
        """
        Initialize layer parameters. Sample weight from Xavier distribution
        and bias uniform zero distribution.

        """

        for layer in self.layers:
            # Xavier initialization for weights
            init.xavier_uniform_(layer.weight)
            # Zero initialization for biases
            init.zeros_(layer.bias)


    # Forward pass
    def forward(self, x, act_func):
        """
        Performs a forward pass on the encoder layers:  multiply input tensor
        by weights, add bias and use non-linear activation function.

        Args:
            x: input features
            act_func: activation function that is used

        Returns:
            z: latent features

        """

        # Iterate over layers and apply activation function
        for layer in self.layers[:-1]:  # Exclude the last linear layer
            x = self.dropout(self.sigmoid(layer(x)))
        z = self.dropout(self.layers[-1](x))  # Final linear layer without activation
        return z


    # Forward pass for first order model.
    # Compute the latent dim derivative
    def forward_order1(self, x, dx, act_func):
      """
      Performs a forward pass on the encoder layers on the input derivatives.
      Expression directly obtained from differentiating z with respect to time.
      Can be found on the SI Appendix

      Args:
          x: input features
          dx: derivatives of input features
          act_func: activation function that is used

      Returns:
          dz: derivatives of latent features

      """
      dz = dx
      input = x
      if act_func == 'sigmoid':
          for layer in self.layers[:-1]:  # Exclude the last layer (output layer)
              input = layer(input)
              input = self.sigmoid(input)
              dz = torch.mul(self.dropout(torch.mul(input, 1 - input)), torch.matmul(dz, layer.weight.T))
          # Last layer without non-linearity
          dz = self.dropout(torch.matmul(dz, self.layers[-1].weight.T))
      return dz


    # Forward pass for second order model.
    # Compute the latent dim first and second derivatives
    def forward_order2(self, x, dx, ddx, act_func):
      """
      Performs a forward pass on the encoder layers on the input derivatives.
      Expression directly obtained from differentiating dz with respect to time.
      Can be found on the SI Appendix

      Args:
          x: input features
          dx: derivatives of input features
          ddx: second order derivatives of input features
          act_func: activation function that is used

      Returns:
          dz: derivatives of latent features
          ddz: second order derivatives of latent features

      """

      dz = dx
      ddz = ddx
      input = x
      if act_func == 'sigmoid':
          for layer in self.layers[:-1]:  # Exclude the last layer (output layer)
              input = layer(input)
              input = self.sigmoid(input)
              dz_prev = torch.matmul(dz, layer.weight.T)
              sigmoid_derivative = torch.mul(input, 1 - input)
              sigmoid_derivative2 = torch.mul(sigmoid_derivative, 1 - 2 * input)
              dz  = self.dropout(torch.mul(sigmoid_derivative, dz_prev))
              ddz = self.dropout(torch.mul(sigmoid_derivative2, torch.square(dz_prev)) \
                    + torch.mul(sigmoid_derivative, torch.matmul(ddz, layer.weight.T)))
          # Last layer without non-linearity
          dz  = self.dropout(torch.matmul(dz, self.layers[-1].weight.T))
          ddz = self.dropout(torch.matmul(ddz, self.layers[-1].weight.T))

      return dz, ddz


## Decoder (Psi)
# Input: latent dimension
# Output: reconstructed input

class Decoder(nn.Module):
    """
    Fully conected decoder with non-linear activation functions except for
    the last network layer

    Args:
        latent_dim: number of latent features
        hdims: number of widths between the input and latent features
        input_dim: number of input features
        act_func: activation function that is used

    """
    # Define your decoder
    def __init__(self, latent_dim, hdims, input_dim, act_func, p_dropout):
        super(Decoder, self).__init__()

        # Decoder layers:
        # Input (latent_dim): latent_dims
        # Hiden: hdims[-1] ... hdims[0]
        # Output: input_dim
        # Non-linear functions are applied for all except for the last

        # Dropout probability
        self.p_dropout = p_dropout

        self.layers = nn.ModuleList()  # Container for encoder layers

        # Create linear layers based on hdims
        prev_dim = latent_dim
        for hdim in reversed(hdims):
            self.layers.append(nn.Linear(prev_dim, hdim))
            prev_dim = hdim

        # Add final linear layer to map to latent_dim
        self.layers.append(nn.Linear(prev_dim, input_dim))
        self.dropout = nn.Dropout(p=p_dropout)
        self.sigmoid = nn.Sigmoid()

        # Initialize weigths and biases
        self.init_params()


    def init_params(self):
        """
        Initialize layer parameters. Sample weight from Xavier distribution
        and bias uniform zero distribution.

        """

        for layer in self.layers:
            # Xavier initialization for weights
            init.xavier_uniform_(layer.weight)
            # Zero initialization for biases
            init.zeros_(layer.bias)


    def forward(self, z, act_func):
        """
        Performs a forward pass on the decoder layers:  multiply input tensor
        by weights, add bias and use non-linear activation function.

        Args:
            z: latent features
            act_func: activation function that is used

        Returns:
            x_decoder: reconstructed input features

        """

        # Iterate over layers and apply activation function
        for layer in self.layers[:-1]:  # Exclude the last linear layer
            z = self.dropout(self.sigmoid(layer(z)))
        x_decoder = self.dropout(self.layers[-1](z))  # Final linear layer without activation

        return x_decoder


    # Forward pass for first order model.
    # Compute the reconstructed derivative
    def forward_order1(self, z, dz_sindy, act_func):
      """
      Performs a forward pass on the encoder layers on the input derivatives.
      Expression directly obtained from differentiating x_decoder with respect
      to time. Can be found on the SI Appendix

      Args:
          z: latent features
          dz_sindy: SINDy prediction for first order latent derivatives
          act_func: activation function that is used

      Returns:
          dx_decoder: derivatives of reconstructed features

      """

      dx_decoder = dz_sindy
      input = z
      if act_func == 'sigmoid':
          for layer in self.layers[:-1]:  # Exclude the last layer (output layer)
              input = layer(input)
              input = self.sigmoid(input)
              dx_decoder = torch.mul(self.dropout(torch.mul(input, 1 - input)), torch.matmul(dx_decoder, layer.weight.T))
          # Last layer without non-linearity
          dx_decoder = self.dropout(torch.matmul(dx_decoder, self.layers[-1].weight.T))
      return dx_decoder


    # Forward pass for second order model.
    # Compute the reconstructed first and second derivatives
    def forward_order2(self, z, dz, ddz_sindy, act_func):
      """
      Performs a forward pass on the encoder layers on the input derivatives.
      Expression directly obtained from differentiating dx_decoder with respect
      to time. Can be found on the SI Appendix

      Args:
          z: latent features
          dz: derivative of latent features
          dz_sindy: SINDy prediction for second order latent derivatives
          act_func: activation function that is used

      Returns:
          dx_decoder: derivatives of reconstructed features
          ddx_decoder: second order derivatives of reconstructed features

      """
      dx_decoder = dz
      ddx_decoder = ddz_sindy
      input = z
      if act_func == 'sigmoid':
          for layer in self.layers[:-1]:  # Exclude the last layer (output layer)
              input = layer(input)
              input = self.sigmoid(input)
              dx_prev = torch.matmul(dx_decoder, layer.weight.T)
              sigmoid_derivative = torch.mul(input, 1 - input)
              sigmoid_derivative2 = torch.mul(sigmoid_derivative, 1 - 2 * input)
              dx_decoder  = self.dropout(torch.mul(sigmoid_derivative, dx_prev))
              ddx_decoder = self.dropout(torch.mul(sigmoid_derivative2, torch.square(dx_prev)) \
                    + torch.mul(sigmoid_derivative, torch.matmul(ddx_decoder, layer.weight.T)))
          # Last layer without non-linearity
          dx_decoder  = self.dropout(torch.matmul(dx_decoder, self.layers[-1].weight.T))
          ddx_decoder = self.dropout(torch.matmul(ddx_decoder, self.layers[-1].weight.T))

      return dx_decoder, ddx_decoder