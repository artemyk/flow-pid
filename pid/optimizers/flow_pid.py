import numpy as np
import math
import torch
from torch.special import log_ndtr

from ..models import CartesianProductFlow
from .thin_pid import exact_gauss_thin_pid

from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


def fit(model, dataloader, n_epochs, lr, device='cpu', verbose=False):
    nfm = fit_flows(model, dataloader, n_epochs, lr, device, verbose=verbose)
    ret = fit_pid(nfm, dataloader, device=device, verbose=verbose)

    return ret


def fit_flows(model, dataloader, n_epochs, lr, device='cpu', verbose=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    losses = []
    for epoch in range(n_epochs):
        epoch_losses = []
        progressbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for batch_n, (x_batch, y_batch, m_batch) in progressbar:
            x_batch, y_batch, m_batch = x_batch.to(device), y_batch.to(device), m_batch.to(device)

            optimizer.zero_grad()
            loss = model.learning_loss(m_batch, x_batch, y_batch)

            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()
                optimizer.step()

            epoch_losses.append(loss.item())
            progressbar.update()

        progressbar.close()
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)

        print('Train Epoch: {}/{} ({:.0f}%)\tLoss: {:.6f}'.format(
            epoch, n_epochs,
            100. * epoch / n_epochs,
            avg_loss))

    if verbose:
        print(f"Final Loss: {losses[-1]:.4f}")
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()

    return model


def fit_pid(model, dataloader, device='cpu', verbose=False, ret_t_sigt=False):
    z_mxy = []
    for x_batch, y_batch, m_batch in dataloader:
        x_batch, y_batch, m_batch = x_batch.to(device), y_batch.to(device), m_batch.to(device)
        with torch.no_grad():
            z_m, z_x, z_y = model(m_batch, x_batch, y_batch)
            z_mxy.append(torch.cat([z_m, z_x, z_y], dim=-1))

    z_mxy = torch.cat(z_mxy, dim=0)
    cov = torch.cov(z_mxy.T).cpu().numpy()

    trained_cov = covariance_to_correlation(cov)
    ret = exact_gauss_thin_pid(trained_cov, model.dm, model.dx, model.dy, verbose=False, ret_t_sigt=ret_t_sigt)

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


def train_flow(m_data, x_data, y_data, n_flows, n_epochs=100, batch_size=64, lr=2e-4, encoder=None, verbose=False, device='cuda'):
    # Initialize model
    dim_x, dim_y, dim_m = x_data.shape[1], y_data.shape[1], m_data.shape[1]

    encoder = encoder.to(device) if encoder is not None else None
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
            m_features, x_features, y_features = encoder(m_batch, x_batch, y_batch) if encoder is not None else (m_batch, x_batch, y_batch)
            z_m, z_x, z_y, log_det = flow(m_features, x_features, y_features)
            
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

    # save the encoder and flow
    torch.save(encoder, 'encoder.pt')
    torch.save(flow, 'flow.pt')

    z_combined = []
    for x_batch, y_batch, m_batch in dataloader:
        x_batch, y_batch, m_batch = x_batch.to(device), y_batch.to(device), m_batch.to(device)
        with torch.no_grad():
            m_batch, x_batch, y_batch = encoder(m_batch, x_batch, y_batch) if encoder is not None else (m_batch, x_batch, y_batch)
            z_m, z_x, z_y, log_det = flow(m_batch, x_batch, y_batch)
            z_combined.append(torch.cat([z_m, z_x, z_y], dim=-1))
    z_combined = torch.cat(z_combined, dim=0)
    cov = torch.cov(z_combined.T).cpu().numpy()
    
    return flow, cov, losses


def trained_covariance(m_data, x_data, y_data, flow_path, encoder_path=None, device='cuda'):
    encoder = torch.load(encoder_path) if encoder_path is not None else None
    flow = torch.load(flow_path)

    x_data = torch.tensor(x_data, dtype=torch.float)
    x_data_standardized = standardize_data(x_data)

    y_data = torch.tensor(y_data, dtype=torch.float)
    y_data_standardized = standardize_data(y_data)

    m_data = torch.tensor(m_data, dtype=torch.float)
    m_data_standardized = standardize_data(m_data)

    dataset = TensorDataset(x_data_standardized, y_data_standardized, m_data_standardized)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    z_combined = []
    for x_batch, y_batch, m_batch in dataloader:
        x_batch, y_batch, m_batch = x_batch.to(device), y_batch.to(device), m_batch.to(device)
        with torch.no_grad():
            m_batch, x_batch, y_batch = encoder(m_batch, x_batch, y_batch) if encoder is not None else (m_batch, x_batch, y_batch)
            z_m, z_x, z_y, log_det = flow(m_batch, x_batch, y_batch)
            z_combined.append(torch.cat([z_m, z_x, z_y], dim=-1))
    z_combined = torch.cat(z_combined, dim=0)
    cov = torch.cov(z_combined.T).cpu().numpy()

    return cov


def flow_pid(m,x,y, n_flows=3, n_epochs=250, batch_size=64, lr = 2e-4, encoder=None, verbose=False, ret_t_sigt=False, device='cpu'):
    trained_flow, trained_cov, training_losses = train_flow(m, x, y, n_flows=n_flows,
                                                            n_epochs=n_epochs, batch_size=batch_size, lr=lr,
                                                            encoder=encoder, verbose=verbose, device=device)

    trained_cov = covariance_to_correlation(trained_cov)
    ret = exact_gauss_thin_pid(trained_cov, m.shape[1], x.shape[1], y.shape[1], verbose=False, ret_t_sigt=ret_t_sigt)
    return ret