import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from pid.optimizers.flow_pid import flow_pid
from pid.optimizers.tilde_pid import exact_gauss_tilde_pid



def prepare_data(data_dir, modalities):
    all_data = np.load(data_dir)
    dims = [256,256,256]
    
    dx = dims[modalities[0]]
    dy = dims[modalities[1]]
    dm = 1

    x_data = all_data[:, modalities[0]*256:(modalities[0]+1)*256]
    y_data = all_data[:, modalities[1]*256:(modalities[1]+1)*256]
    m_data = all_data[:, 256*3:]
    return x_data, y_data, m_data


if __name__ == '__main__':
    # assume features are already extracted
    enable_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
    print(f"Using device: {device}")

    x_data, y_data, m_data = prepare_data('./pretrained/mosi/mosi_features_mfm.npy', modalities = [0,2])

    scale = 1/2.0*0
    eps = np.random.rand(m_data.shape[0], 1) * scale
    m_data = m_data.reshape(-1, 1) + eps
    print(x_data.shape, y_data.shape, m_data.shape)

    mxy = np.vstack((m_data.T, x_data.T, y_data.T))
    cov = np.corrcoef(mxy)    # Shape: (dm+dx+dy, dm+dx+dy)
    print(f'cov.shape: {cov.shape} \n')

    ret = exact_gauss_tilde_pid(cov, m_data.shape[1], x_data.shape[1], y_data.shape[1])
    print(f'tilde pid: {ret[7]}, {ret[5]}, {ret[6]}, {ret[8]} \n')
    

    ret = flow_pid(m_data, x_data, y_data, n_flows=10, n_epochs=250, batch_size=1000, lr=2e-4, verbose=True, device=device)
    norm = ret[7] + ret[5] + ret[6] + ret[8]
    print(ret[7], ret[5], ret[6], ret[8])
    r, ux, uy, si = ret[7] / norm, ret[5] / norm, ret[6] / norm, ret[8] / norm
    print(f"flow pid, normalized I_mxy, R: {r}, UX: {ux}, UY: {uy}, S: {si}")



'''
RESULTS: (R, U1, U2, S)
Modalities [0,1]:
Tilde PID: 0.20379686851660017, 0.053254763683536266, 0.0, 0.2535796881356404 
Flow PID: 0.05576623098124661 0.2114928218412356 0.15105005756244833 0.07204369283668943

Modalities [0,2]:
Tilde PID:  0.2653343473418991, 0.0, 0.31809255374564577, 0.3332527738565385
Flow PID: 0.15876085700706666 0.1192304570791628 0.5369529864756843 0.203958725303927

Modalities [1,2]:
Tilde PID: 0.20806855249277045, 0.0, 0.3724716274514459, 0.2773892940970024
FLow PID: 0.12156292569568261 0.08563656425283594 0.5735329036826686 0.16373475130866866
'''

