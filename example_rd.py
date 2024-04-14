import numpy as np
import scipy.io as sio
import sys
import os

sys.path.append("./src")

print(os.getcwd())

def get_rd_data(random=True):
    data = sio.loadmat('reaction_diffusion.mat')

    n_samples = data['t'].size
    n = data['x'].size
    N = n*n

    data['uf'] += 1e-6*np.random.randn(data['uf'].shape[0], data['uf'].shape[1], data['uf'].shape[2])
    data['duf'] += 1e-6*np.random.randn(data['duf'].shape[0], data['duf'].shape[1], data['duf'].shape[2])

    if not random:
        # consecutive samples
        training_samples = np.arange(int(.8*n_samples))
        val_samples = np.arange(int(.8*n_samples), int(.9*n_samples))
        test_samples = np.arange(int(.9*n_samples), n_samples)
    else:
        # random samples
        perm = np.random.permutation(int(.9*n_samples))
        training_samples = perm[:int(.8*n_samples)]
        val_samples = perm[int(.8*n_samples):]

        test_samples = np.arange(int(.9*n_samples), n_samples)

    training_data = {'t': (data['t'][training_samples]).astype(np.float32),
                     'y1': (data['x'].T).astype(np.float32),
                     'y2': (data['y'].T).astype(np.float32),
                     'x': (data['uf'][:,:,training_samples].reshape((N,-1)).T).astype(np.float32),
                     'dx': (data['duf'][:,:,training_samples].reshape((N,-1)).T).astype(np.float32),}
    val_data = {'t': (data['t'][val_samples]).astype(np.float32),
                'y1': (data['x'].T).astype(np.float32),
                'y2': (data['y'].T).astype(np.float32),
                'x': (data['uf'][:,:,val_samples].reshape((N,-1)).T).astype(np.float32),
                'dx': (data['duf'][:,:,val_samples].reshape((N,-1)).T).astype(np.float32)}
    test_data = {'t': (data['t'][test_samples]).astype(np.float32),
                 'y1': (data['x'].T).astype(np.float32),
                 'y2': (data['y'].T).astype(np.float32),
                 'x': (data['uf'][:,:,test_samples].reshape((N,-1)).T).astype(np.float32),
                 'dx': (data['duf'][:,:,test_samples].reshape((N,-1)).T).astype(np.float32)}
    
    training_data['ddx'] = np.zeros_like(training_data['x'])
    val_data['ddx'] = np.zeros_like(training_data['x'])
    test_data['ddx'] = np.zeros_like(training_data['x'])
     
    return training_data, val_data, test_data
