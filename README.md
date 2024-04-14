Authors of the reproduction: - Diego Andía
                             - Éric Chillón
                             - Victor Cayetano Hernandez
                             - Matthijs Scheerden

This repository includes the code created by a group of master students of TU Delft for the course CS4240 Deep Learning. 
The objective of this project has been to reproduce the following paper: https://www.pnas.org/doi/suppl/10.1073/pnas.1906995116.
The original code from the authors is posted in: https://www.pnas.org/doi/suppl/10.1073/pnas.1906995116.
A blog-post showing the outcomes of the reproduction has been posted in:  

*Important notes: Check that the files created in Training.py are saved in the correct directory.
                  In order to run the rd (reaction diffusion) training and testing scripts you will need to run 
                  the reaction_diffusion.m file to generate the required data.

*All the scripts have been located together to avoid problems with PATH directions.

The structure of the repository is explained below:
  - autoencoder_SINDy.py : It includes the full Pythorch implementation of the autonder-Sindy model
                           that was made using TensorFlow by the authors.
  - encoder_decoder.py : Includes the Pythorch implementation of the encoder and decoder classes used
                         for the autoencoder-sindy model.
  - SINDy.py: File from the original authors repository. Since it does not use Tensorflow, it has been reused.
  - SINDY_utils.py: File from the original authors repository. Since it does not use Tensorflow, it has been reused.
  - training.py: Contains the full training and validation loop implementation for the autoencoder-Sindy model.
  - example_ [lorenz/pendulum/rd]: This files are the scripts used to generate the training data, these were directly obtained from
                                   the authors' repository, and includes small modifications in order to make it compatible with
                                   our implementation.
  - [lorenz/pendulum/rd]_train: This scripts are used to generate and train an autoencoder-sindy model,
                                      after finishing the execution 2 pickle files (net_params, parameters) and one pythorch model
                                      will be generated containing the required data to visualize the results.
  - [lorenz/pendulum/rd_model]_test:  This scripts are used to test the trained model and plot some results.
  



