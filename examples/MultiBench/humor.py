import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from models.fusions import Concat, Sequential2
from models.lmi import AEMINE
from models.unimodels import LeNet, MLP, LeNetEncoder, DeLeNet
from pid.supervised_learning import train, single_test
from pid import flow_pid
from pid.tilde_pid import exact_gauss_tilde_pid
import pandas as pd
import os
from models import CartesianProductFlow


def prepare_data(data_dir, modalities):
    all_data = np.load(data_dir)
    dims = [64, 64, 64]
    # dims = [256,256,256]

    dx = dims[modalities[0]]
    dy = dims[modalities[1]]
    dm = 1

    x_data = all_data[:, modalities[0] * 64:(modalities[0] + 1) * 64]
    y_data = all_data[:, modalities[1] * 64:(modalities[1] + 1) * 64]
    m_data = all_data[:, 64 * 3:]
    return x_data, y_data, m_data


def learn_features_from_encoder(x_data, y_data, batch_size=1000, encoder_path=None, device=torch.device('cpu')):
    model = torch.load(encoder_path, weights_only=False).to(device)

    dataset = TensorDataset(
        torch.tensor(x_data, dtype=torch.float, device=device),
        torch.tensor(y_data, dtype=torch.float, device=device),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    ## Extract features
    x_features = []
    y_features = []
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            x_feature = model.encoders[0](x_batch)
            y_feature = model.encoders[1](y_batch)
            x_features.append(x_feature.cpu().numpy())
            y_features.append(y_feature.cpu().numpy())

    x_features = np.concatenate(x_features, axis=0)
    y_features = np.concatenate(y_features, axis=0)

    return x_features, y_features


def train_lmi(model, train_loader, lr=0.0001, epochs=300, device=torch.device('cpu'), verbose=False, save_path=None):
    """
    Train an LMI model.

    Args:
        model: The LMI model to train
        train_loader: DataLoader for training data
        lr (float): Learning rate
        epochs (int): Number of training epochs
        device (torch.device): Device to use for training
        verbose (bool): Whether to plot the loss curve
        save_path (str, optional): Path to save the trained model

    Returns:
        model: The trained model
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-07)

    # Training loop
    losses = []
    progress_bar = tqdm(range(epochs), desc="Training LMI model")
    for epoch in progress_bar:
        epoch_losses = []
        batch_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        for batch_n, (x_batch, y_batch, m_batch) in enumerate(batch_progress):
            x_batch, y_batch, m_batch = x_batch.to(device), y_batch.to(device), m_batch.to(device)

            # Ensure m_batch has the right shape for MINELoss
            if m_batch.dim() != x_batch.dim():
                m_batch = m_batch.view(m_batch.size(0), -1)

            model.train()
            optimizer.zero_grad()

            loss = model.learning_loss(x_batch, y_batch, m_batch)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            batch_progress.set_postfix({"loss": loss.item()})

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        progress_bar.set_postfix({"avg_loss": f"{avg_loss:.6f}"})

    if save_path:
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Created directory: {save_dir}")

        # Save the model
        torch.save(model, save_path)
        print(f'Model saved to {save_path}')

    return model


def extract_lmi_features(model, data_loader, device=torch.device('cpu')):
    """
    Extract features from a trained LMI model.

    Args:
        model: The trained LMI model
        data_loader: DataLoader for the data
        device: Device to use for inference

    Returns:
        tuple: Tuple of (x_features, y_features, m_features)
    """
    # Use the trained encoder to get the latent representations
    x_latent = []
    y_latent = []
    m_latent = []

    feature_progress = tqdm(data_loader, desc="Extracting features")

    for x_batch, y_batch, m_batch in feature_progress:
        model.eval()
        x_batch, y_batch, m_batch = x_batch.to(device), y_batch.to(device), m_batch.to(device)

        with torch.no_grad():
            Z_x, Z_y = model.encode(x_batch, y_batch)
        Z_m = m_batch

        x_latent.append(Z_x)
        y_latent.append(Z_y)
        m_latent.append(Z_m)

    x_latent = torch.cat(x_latent)
    y_latent = torch.cat(y_latent)
    m_latent = torch.cat(m_latent)

    # Convert to numpy arrays
    x_features = x_latent.cpu().numpy()
    y_features = y_latent.cpu().numpy()
    m_features = m_latent.cpu().numpy()

    return x_features, y_features, m_features


class FeatureExtractor(nn.Module):
    def __init__(self, encoder_list):
        super(FeatureExtractor, self).__init__()
        self.x_encoder = encoder_list[0]
        self.y_encoder = encoder_list[1]

    def forward(self, m, x, y):
        x = self.x_encoder(x)
        y = self.y_encoder(y)
        return m, x, y


def standardize_data(data):
    data = torch.from_numpy(np.nan_to_num((data - data.mean(axis=0)) / data.std(axis=0))).float()
    data = torch.clip(data, min=-10, max=10)
    return data


if __name__ == '__main__':
    # assume features are already extracted
    enable_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
    print(f"Using device: {device}")

    # x_data, y_data, m_data = prepare_data('./pretrained/humor/humor_features_mfm_64.npy', modalities = [1,2])
    data_dir = './examples/data/humor/humor.pkl'
    dataset = pd.read_pickle(data_dir)
    print(dataset['train'].keys())

    x_data, y_data, m_data = np.array(dataset['train']['vision']), np.array(dataset['train']['audio']), np.array(
        dataset['train']['labels'])
    print(x_data.shape, y_data.shape, m_data.shape)
    # Create dataset and dataloader
    # Ensure m_data has the right shape (N, 1) for proper concatenation in MINELoss
    m_data_reshaped = m_data.reshape(-1, 1) if len(m_data.shape) == 1 else m_data

    x_data_flat = x_data.reshape(x_data.shape[0], -1)
    y_data_flat = y_data.reshape(y_data.shape[0], -1)

    print(x_data_flat.shape, y_data_flat.shape, m_data_reshaped.shape)

    dataset = TensorDataset(
        standardize_data(torch.tensor(x_data_flat, dtype=torch.float, device=device)),
        standardize_data(torch.tensor(y_data_flat, dtype=torch.float, device=device)),
        standardize_data(torch.tensor(m_data_reshaped, dtype=torch.float, device=device))
    )
    train_loader = DataLoader(dataset, batch_size=1000, shuffle=True)

    # Define dimensions
    dx = x_data_flat.shape[1]
    dy = y_data_flat.shape[1]
    dm = 1  # Label dimension

    # Train the LMI model for dimension reduction
    latent_dim = 128
    save_path = './examples/pretrained/humor/lmi_model_av.pt'

    # Initialize the AEMINE model
    encoder = AEMINE(dx, dy, dm, latent_dim, alpha=1.0, lam=1.0).to(device)

    # Train the model
    lmi_encoder = train_lmi(
        encoder,
        train_loader,
        lr=2e-4,
        epochs=500,  # Reduced for testing, use 500 for full training
        device=device,
        verbose=False,
        save_path=save_path
    )

    lmi_encoder = torch.load(save_path)

    # Extract features
    x_lmi_features, y_lmi_features, m_lmi_features = extract_lmi_features(lmi_encoder, train_loader, device=device)

    # Use the extracted m_features directly from the model
    m_features = m_lmi_features

    # Run flow PID on the LMI features
    print("\nRunning flow PID on LMI features:")
    model = CartesianProductFlow(
        dm=dm,
        dx=latent_dim,
        dy=latent_dim,
        n_flows=2,
    ).to(device)

    x_lmi_features = standardize_data(x_lmi_features)
    y_lmi_features = standardize_data(y_lmi_features)
    m_lmi_features = standardize_data(m_lmi_features)

    new_dataset = TensorDataset(x_lmi_features, y_lmi_features, m_lmi_features)
    train_loader = DataLoader(new_dataset, batch_size=1000, shuffle=True)
    pid_flows = flow_pid.fit_flows(model, train_loader, 500, lr=1e-3, device=device, verbose=False)
    # save the flow model
    torch.save(pid_flows, './examples/pretrained/humor/lmi_flow_av.pt')
    ret_lmi = ret = flow_pid.fit_pid(pid_flows, train_loader, device=device, verbose=False)

    norm_lmi = ret_lmi[7] + ret_lmi[5] + ret_lmi[6] + ret_lmi[8]
    r_lmi, ux_lmi, uy_lmi, si_lmi = ret_lmi[7] / norm_lmi, ret_lmi[5] / norm_lmi, ret_lmi[6] / norm_lmi, ret_lmi[
        8] / norm_lmi
    print(f"flow pid (LMI), unnorm I_mxy, R: {ret_lmi[7]}, UX: {ret_lmi[5]}, UY: {ret_lmi[6]}, S: {ret_lmi[8]}")
    print(f"flow pid (LMI), normalized I_mxy, R: {r_lmi}, UX: {ux_lmi}, UY: {uy_lmi}, S: {si_lmi}")

    scale = 1 / 3.0
    # eps = np.random.rand(m_data_reshaped.shape[0], 1) * scale
    m_data_reshaped = m_data_reshaped.reshape(-1, 1)  # + eps
    print(x_lmi_features.shape, y_lmi_features.shape, m_lmi_features.shape)

    mxy = np.vstack((m_lmi_features.T, x_lmi_features.T, y_lmi_features.T))
    cov = np.corrcoef(mxy)  # Shape: (dm+dx+dy, dm+dx+dy)
    print(f'cov.shape: {cov.shape} \n')

    ret = exact_gauss_tilde_pid(cov, m_lmi_features.shape[1], x_lmi_features.shape[1], y_lmi_features.shape[1])
    print(f'tilde pid: {ret[7]}, {ret[5]}, {ret[6]}, {ret[8]} \n')
