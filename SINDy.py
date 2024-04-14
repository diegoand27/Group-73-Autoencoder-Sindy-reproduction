import torch
import torch.nn as nn
import torch.nn.init as init

import math

## SINDy - First order
# Input: latent dimension
# Output: reconstructed input

class SINDy_order1(nn.Module):
    """
    SINDy module for first order model. Includes the library, coefficient
    mask matrix and the SINDy coeff. matrix.

    Args:
        z: latent features
        latent_dim: latent dimension
        library_dim: size of the SINDy library
        poly_order: order of the polynomials
        include_sine: inlude sine or not

    """
    # Define your decoder
    def __init__(self, z, latent_dim, library_dim, poly_order, include_sine, device):
        super(SINDy_order1, self).__init__()

        # Create the library for the SINDy: Theta
        self.Theta = self.library_order1(z,latent_dim,poly_order,include_sine)

        ## Define mask coefficient matrix
        self.coeff_mask = torch.ones([library_dim,latent_dim], dtype=torch.float32)

        # Define trainaible parameter: Coefficient matrix Xi
        self.Xi = nn.Parameter(torch.Tensor(library_dim,latent_dim))

        # Save parameters to avoid repetition in other methods
        self.latent_dim = latent_dim
        self.library_dim = library_dim
        self.poly_order = poly_order
        self.include_sine = include_sine

        # Initialize Xi
        self.init_params()


    # Xi coeffs initialization
    def init_params(self):
        """
        Initialize SINDy coeffieints. Sample weight from Xavier distribution.

        """

        # Xavier initialization for sindy coefficients
        #init.xavier_uniform_(self.Xi)
        init.constant_(self.Xi, 1.0)


    # Update library
    def library_update(self,z):
        """
        Recompute the SINDy library without requiring all the inputs.

        Args:
            z: latent features

        Returns:
            SINDy library

        """

        return self.library_order1(z,self.latent_dim,self.poly_order,self.include_sine)


    # Build the SINDy library
    def library_order1(self, z, latent_dim, poly_order, include_sine=False):
        """
        Computes the SINDy library from all the defined parameters for a
        first order model.

        Args:
            z: latent features
            latent_dim: latent dimension
            poly_order: order of the polynomials
            include_sine: include sine or not

        Returns:
            First order SINDy library

        """

        # Initialize the library with a tensor of ones (n_samples,?)
        library = [torch.ones(z.size(0)).to(device=z.device)]

        for i in range(latent_dim):
            library.append(z[:, i])

        if poly_order > 1:
            for i in range(latent_dim):
                for j in range(i, latent_dim):
                    library.append(z[:, i] * z[:, j])

        if poly_order > 2:
            for i in range(latent_dim):
                for j in range(i, latent_dim):
                    for k in range(j, latent_dim):
                        library.append(z[:, i] * z[:, j] * z[:, k])

        if poly_order > 3:
            for i in range(latent_dim):
                for j in range(i, latent_dim):
                    for k in range(j, latent_dim):
                        for p in range(k, latent_dim):
                            library.append(z[:, i] * z[:, j] * z[:, k] * z[:, p])

        if poly_order > 4:
            for i in range(latent_dim):
                for j in range(i, latent_dim):
                    for k in range(j, latent_dim):
                        for p in range(k, latent_dim):
                            for q in range(p, latent_dim):
                                library.append(z[:, i] * z[:, j] * z[:, k] * z[:, p] * z[:, q])

        if include_sine:
            for i in range(latent_dim):
                library.append(torch.sin(z[:, i]))

        return torch.stack(library, dim=1)




## SINDy - Second order
# Input: latent dimension
# Output: reconstructed input

class SINDy_order2(nn.Module):
    """
    SINDy module for second order model. Includes the library, coefficient
    mask matrix and the SINDy coeff. matrix.

    Args:
        z: latent features
        dz: derivatives of latent features
        latent_dim: latent dimension
        library_dim: size of the SINDy library
        poly_order: order of the polynomials
        include_sine: inlude sine or not

    """
    # Define your decoder
    def __init__(self, z, dz, latent_dim, library_dim, poly_order, include_sine, device):
        super(SINDy_order2, self).__init__()

        # Create the library for the SINDy: Theta
        self.Theta = self.library_order2(z,dz,latent_dim,poly_order,include_sine)

        ## Define mask coefficient matrix
        self.coeff_mask = torch.ones([library_dim,latent_dim], dtype=torch.float32)

        # Define trainaible parameter: Coefficient matrix Xi
        self.Xi = nn.Parameter(torch.Tensor(library_dim,latent_dim))

        # Save parameters to avoid repetition in other methods
        self.latent_dim = latent_dim
        self.library_dim = library_dim
        self.poly_order = poly_order
        self.include_sine = include_sine

        # Initialize Xi
        self.init_params()


    # SINDy coefficients initalization
    def init_params(self):
        """
        Initialize SINDy coeffieints. Sample weight from Xavier distribution.

        """

        # Xavier initialization for sindy coefficients
        #init.xavier_uniform_(self.Xi)
        init.constant_(self.Xi, 1.0)


    # Update library
    def library_update(self,z,dz):
        """
        Recompute the SINDy library without requiring all the inputs.

        Args:
            z: latent features

        Returns:
            SINDy library

        """

        return self.library_order2(z,dz,self.latent_dim,self.poly_order,self.include_sine)


    # Build the SINDy library
    def library_order2(self, z, dz, latent_dim, poly_order, include_sine=False):
        """
        Computes the SINDy library from all the defined parameters for a
        second order model.

        Args:
            z: latent features
            dz: derivatives of latent features
            latent_dim: latent dimension
            poly_order: order of the polynomials
            include_sine: include sine or not

        Returns:
            Second order SINDy library

        """

        # Initialize the library with a tensor of ones (number of time points x ?)
        library = [torch.ones(z.size(0)).to(device=z.device)]

        # Concatenate z and dz along dimension 1
        z_combined = torch.cat([z, dz], dim=1).to(device=z.device)

        for i in range(2 * latent_dim):
            library.append(z_combined[:, i])

        if poly_order > 1:
            for i in range(2 * latent_dim):
                for j in range(i, 2 * latent_dim):
                    library.append(z_combined[:, i] * z_combined[:, j])

        if poly_order > 2:
            for i in range(2 * latent_dim):
                for j in range(i, 2 * latent_dim):
                    for k in range(j, 2 * latent_dim):
                        library.append(z_combined[:, i] * z_combined[:, j] * z_combined[:, k])

        if poly_order > 3:
            for i in range(2 * latent_dim):
                for j in range(i, 2 * latent_dim):
                    for k in range(j, 2 * latent_dim):
                        for p in range(k, 2 * latent_dim):
                            library.append(z_combined[:, i] * z_combined[:, j] * z_combined[:, k] * z_combined[:, p])

        if poly_order > 4:
            for i in range(2 * latent_dim):
                for j in range(i, 2 * latent_dim):
                    for k in range(j, 2 * latent_dim):
                        for p in range(k, 2 * latent_dim):
                            for q in range(p, 2 * latent_dim):
                                library.append(z_combined[:, i] * z_combined[:, j] * z_combined[:, k] *
                                              z_combined[:, p] * z_combined[:, q])

        if include_sine:
            for i in range(2 * latent_dim):
                library.append(torch.sin(z_combined[:, i]))

        return torch.stack(library, dim=1)

