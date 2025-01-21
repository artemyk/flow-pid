import numpy as np
from tqdm import tqdm

import torch

from ..utils.custom_flow import CartesianProductFlow
from .thin_pid import exact_gauss_thin_pid
from .tilde_pid import exact_gauss_tilde_pid

from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


def flow_pid(m,x,y, n_flows=3, n_epochs=250, batch_size=64, lr = 2e-4, verbose=False, ret_t_sigt=False, device='cpu'):
    trained_flow, trained_cov, training_losses = train_flow(m, x, y, n_flows=n_flows, n_epochs=n_epochs, batch_size=batch_size, lr=lr, verbose=verbose, device=device)

    trained_cov = covariance_to_correlation(trained_cov)
    ret = exact_gauss_thin_pid(trained_cov, m.shape[1], x.shape[1], y.shape[1], verbose=False, ret_t_sigt=ret_t_sigt)
    return ret

def covariance_to_correlation(covariance_matrix):
    covariance_matrix = np.array(covariance_matrix)
    
    if covariance_matrix.shape[0] != covariance_matrix.shape[1]:
        raise ValueError("Covariance matrix must be square")
    
    if not np.allclose(covariance_matrix, covariance_matrix.T):
        raise ValueError("Covariance matrix must be symmetric")
    
    std_devs = np.sqrt(np.diag(covariance_matrix))  
    outer_std = np.outer(std_devs, std_devs)
    correlation_matrix = covariance_matrix / outer_std 
    np.fill_diagonal(correlation_matrix, 1.0)
    
    return correlation_matrix

def standardize_data(data):
    # standardize data along columns
    return (data - data.mean(dim=0, keepdim=True)) / data.std(dim=0, keepdim=True)


def train_flow(m_data, x_data, y_data, n_flows, n_epochs=100, batch_size=64, lr=2e-4, encoder_path=None, verbose=False, device='cuda'):
    # Initialize model
    dim_x, dim_y, dim_m = x_data.shape[1], y_data.shape[1], m_data.shape[1]

    if encoder_path is not None:
        encoder = torch.load(encoder_path, weights_only=False).to(device)
    else:
        encoder = None
    flow = CartesianProductFlow(dim_m, dim_x, dim_y, n_flows).to(device)

    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    # Create and standardize dataset
    x_data = torch.tensor(x_data,dtype=torch.float32)
    x_data_standardized = standardize_data(x_data)

    y_data = torch.tensor(y_data,dtype=torch.float32)
    y_data_standardized = standardize_data(y_data)

    m_data = torch.tensor(m_data,dtype=torch.float32)
    m_data_standardized = standardize_data(m_data)

    dataset = TensorDataset(x_data_standardized, y_data_standardized, m_data_standardized)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    losses = []
    
    for epoch in tqdm(range(n_epochs)):
        epoch_losses = []
        
        for x_batch, y_batch, m_batch in dataloader:
            x_batch, y_batch, m_batch = x_batch.to(device), y_batch.to(device), m_batch.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            z_m, z_x, z_y, log_det = flow(m_batch, x_batch, y_batch)
            
            # Compute loss
            loss = flow.compute_loss(z_m, z_x, z_y, log_det)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        
        # with torch.no_grad():
        #     if (epoch + 1) % 50 == 0:
        #         if verbose:
        #             print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
        #
        #         # Print learned parameters
        #         # print(f"Learned mean: {flow.estimate_latent_mean(z_m, z_x, z_y).data}")
        #         # print(f"Learned covariance:\n{flow.estimate_latent_cov(z_m, z_x, z_y).data}")
        #
        #         if epoch > 40:
        #             cov = flow.estimate_latent_cov(m_data_standardized, x_data_standardized, y_data_standardized).detach().cpu().numpy()
        #             cov = covariance_to_correlation(cov)
        #             ret = exact_gauss_thin_pid(cov, dim_m, dim_x, dim_y)
        #
        #             if verbose:
        #                 print("Flow PID: ", ret[7], ret[5], ret[6], ret[8])
    
    # # Plot loss curve
    if verbose:
        final_loss = losses[-1]
        print(f"Final Loss: {final_loss:.4f}")

        # plt.figure(figsize=(10, 5))
        # plt.plot(losses)
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.title('Training Loss')
        # plt.show()

    z_combined = []
    for x_batch, y_batch, m_batch in dataloader:
        x_batch, y_batch, m_batch = x_batch.to(device), y_batch.to(device), m_batch.to(device)
        with torch.no_grad():
            z_m, z_x, z_y, log_det = flow(m_batch, x_batch, y_batch)
            z_combined.append(torch.cat([z_m, z_x, z_y], dim=-1))
    z_combined = torch.cat(z_combined, dim=0)
    cov = torch.cov(z_combined.T).cpu().numpy()
    
    return flow, cov, losses