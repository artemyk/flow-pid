"""Implements common unimodal encoders."""
import torch
import torchvision

from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import models as tmodels


class Linear(torch.nn.Module):
    """Linear Layer with Xavier Initialization, and 0 Bias."""

    def __init__(self, indim, outdim, xavier_init=False):
        """Initialize Linear Layer w/ Xavier Init.

        Args:
            indim (int): Input Dimension
            outdim (int): Output Dimension
            xavier_init (bool, optional): Whether to apply Xavier Initialization to Layer. Defaults to False.

        """
        super(Linear, self).__init__()
        self.fc = nn.Linear(indim, outdim)
        if xavier_init:
            nn.init.xavier_normal(self.fc.weight)
            self.fc.bias.data.fill_(0.0)

    def forward(self, x):
        """Apply Linear Layer to Input.

        Args:
            x (torch.Tensor): Input Tensor

        Returns:
            torch.Tensor: Output Tensor

        """
        return self.fc(x)


class MLP(torch.nn.Module):
    """Two layered perceptron."""

    def __init__(self, indim, hiddim, outdim, dropout=False, dropoutp=0.1, output_each_layer=False):
        """Initialize two-layered perceptron.

        Args:
            indim (int): Input dimension
            hiddim (int): Hidden layer dimension
            outdim (int): Output layer dimension
            dropout (bool, optional): Whether to apply dropout or not. Defaults to False.
            dropoutp (float, optional): Dropout probability. Defaults to 0.1.
            output_each_layer (bool, optional): Whether to return outputs of each layer as a list. Defaults to False.
        """
        super(MLP, self).__init__()
        self.fc = nn.Linear(indim, hiddim)
        self.fc2 = nn.Linear(hiddim, outdim)
        self.dropout_layer = torch.nn.Dropout(dropoutp)
        self.dropout = dropout
        self.output_each_layer = output_each_layer
        self.lklu = nn.LeakyReLU(0.2)

    def forward(self, x):
        """Apply MLP to Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        output = F.relu(self.fc(x))
        if self.dropout:
            output = self.dropout_layer(output)
        output2 = self.fc2(output)
        if self.dropout:
            output2 = self.dropout_layer(output)
        if self.output_each_layer:
            return [0, x, output, self.lklu(output2)]
        return output2


class LeNet(nn.Module):
    """Implements LeNet.

    Adapted from centralnet code https://github.com/slyviacassell/_MFAS/blob/master/models/central/avmnist.py.
    """

    def __init__(self, in_channels, args_channels, additional_layers, output_each_layer=False, linear=None,
                 squeeze_output=True):
        """Initialize LeNet.

        Args:
            in_channels (int): Input channel number.
            args_channels (int): Output channel number for block.
            additional_layers (int): Number of additional blocks for LeNet.
            output_each_layer (bool, optional): Whether to return the output of all layers. Defaults to False.
            linear (tuple, optional): Tuple of (input_dim, output_dim) for optional linear layer post-processing. Defaults to None.
            squeeze_output (bool, optional): Whether to squeeze output before returning. Defaults to True.
        """
        super(LeNet, self).__init__()
        self.output_each_layer = output_each_layer
        self.convs = [
            nn.Conv2d(in_channels, args_channels, kernel_size=5, padding=2, bias=False)]
        self.bns = [nn.BatchNorm2d(args_channels)]
        self.gps = [GlobalPooling2D()]
        for i in range(additional_layers):
            self.convs.append(nn.Conv2d((2 ** i) * args_channels, (2 ** (i + 1))
                                        * args_channels, kernel_size=3, padding=1, bias=False))
            self.bns.append(nn.BatchNorm2d(args_channels * (2 ** (i + 1))))
            self.gps.append(GlobalPooling2D())
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)
        self.gps = nn.ModuleList(self.gps)
        self.sq_out = squeeze_output
        self.linear = None
        if linear is not None:
            self.linear = nn.Linear(linear[0], linear[1])
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        """Apply LeNet to layer input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        tempouts = []
        out = x
        for i in range(len(self.convs)):
            out = F.relu(self.bns[i](self.convs[i](out)))
            out = F.max_pool2d(out, 2)
            gp = self.gps[i](out)
            tempouts.append(gp)

        if self.linear is not None:
            out = self.linear(out)
        tempouts.append(out)
        if self.output_each_layer:
            if self.sq_out:
                return [t.squeeze() for t in tempouts]
            return tempouts
        if self.sq_out:
            return out.squeeze()
        return out


class GlobalPooling2D(nn.Module):
    """Implements 2D Global Pooling."""

    def __init__(self):
        """Initializes GlobalPooling2D Module."""
        super(GlobalPooling2D, self).__init__()

    def forward(self, x):
        """Apply 2D Global Pooling to Layer Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        # apply global average pooling
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.mean(x, 2)
        x = x.view(x.size(0), -1)

        return x

class LeNetEncoder(nn.Module):
    """Implements a LeNet Encoder for MVAE."""
    
    def __init__(self, in_channels, arg_channels, additional_layers, latent, twooutput=True):
        """Instantiate LeNetEncoder Module

        Args:
            in_channels (int): Input Dimensions
            arg_channels (int): Arg channels dimension size
            additional_layers (int): Number of additional layers
            latent (int): Latent dimension size
            twooutput (bool, optional): Whether to output twice the size of the latent. Defaults to True.
        """
        super(LeNetEncoder, self).__init__()
        self.latent = latent
        self.lenet = LeNet(in_channels, arg_channels, additional_layers)
        if twooutput:
            self.linear = nn.Linear(
                arg_channels*(2**additional_layers), latent*2)
        else:
            self.linear = nn.Linear(
                arg_channels*(2**additional_layers), latent)

        self.twoout = twooutput

    def forward(self, x):
        """Apply LeNetEncoder to Layer Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        out = self.lenet(x)
        out = self.linear(out)
        if self.twoout:
            return out[:, :self.latent]# , out[:, self.latent:]
        return out

class DeLeNet(nn.Module):
    """Implements an image deconvolution decoder for MVAE."""
    
    def __init__(self, in_channels, arg_channels, additional_layers, latent):
        """Instantiate DeLeNet Module.

        Args:
            in_channels (int): Number of input channels
            arg_channels (int): Number of arg channels
            additional_layers (int): Number of additional layers.
            latent (int): Latent dimension size
        """
        super(DeLeNet, self).__init__()
        self.linear = nn.Linear(latent, arg_channels*(2**(additional_layers)))
        self.deconvs = []
        self.bns = []
        for i in range(additional_layers):
            self.deconvs.append(nn.ConvTranspose2d(arg_channels*(2**(additional_layers-i)), arg_channels*(
                2**(additional_layers-i-1)), kernel_size=4, stride=2, padding=1, bias=False))
            self.bns.append(nn.BatchNorm2d(
                arg_channels*(2**(additional_layers-i-1))))
        self.deconvs.append(nn.ConvTranspose2d(
            arg_channels, in_channels, kernel_size=8, stride=4, padding=1, bias=False))
        self.deconvs = nn.ModuleList(self.deconvs)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        """Apply DeLeNet to Layer Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        out = self.linear(x).unsqueeze(2).unsqueeze(3)
        for i in range(len(self.deconvs)):
            out = self.deconvs[i](out)
            
            if i < len(self.deconvs)-1:
                out = self.bns[i](out)
        return out


class TSEncoder(torch.nn.Module):
    """Implements a time series encoder for MVAE."""

    def __init__(self, indim, outdim, finaldim, timestep, returnvar=True, batch_first=False):
        """Instantiate TSEncoder Module.

        Args:
            indim (int): Input Dimension of GRU
            outdim (int): Output dimension of GRU
            finaldim (int): Output dimension of TSEncoder
            timestep (float): Number of timestamps
            returnvar (bool, optional): Whether to return the output split with the first encoded portion and the next or not. Defaults to True.
            batch_first (bool, optional): Whether the batching dimension is the first dimension of the input or not. Defaults to False.
        """
        super(TSEncoder, self).__init__()
        self.gru = nn.GRU(input_size=indim, hidden_size=outdim,
                          batch_first=batch_first)
        self.indim = indim
        self.ts = timestep
        self.finaldim = finaldim
        if returnvar:
            self.linear = nn.Linear(outdim * timestep, 2 * finaldim)
        else:
            self.linear = nn.Linear(outdim * timestep, finaldim)
        self.returnvar = returnvar

    def forward(self, x):
        """Apply TS Encoder to Layer Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        batch = len(x)
        input = x.reshape(batch, self.ts, self.indim).transpose(0, 1)
        output = self.gru(input)[0].transpose(0, 1)
        output = self.linear(output.flatten(start_dim=1))
        if self.returnvar:
            return output[:, :self.finaldim], output[:, self.finaldim:]
        return output


class TSDecoder(torch.nn.Module):
    """Implements a time-series decoder for MVAE."""

    def __init__(self, indim, outdim, finaldim, timestep):
        """Instantiate TSDecoder Module.

        Args:
            indim (int): Input dimension
            outdim (int): (unused) Output dimension
            finaldim (int): Hidden dimension
            timestep (int): Number of timesteps
        """
        super(TSDecoder, self).__init__()
        self.gru = nn.GRU(input_size=indim, hidden_size=indim)
        self.linear = nn.Linear(finaldim, indim)
        self.ts = timestep
        self.indim = indim

    def forward(self, x):
        """Apply TSDecoder to layer input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        hidden = self.linear(x).unsqueeze(0)
        next = torch.zeros(1, len(x), self.indim).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        nexts = []
        for i in range(self.ts):
            next, hidden = self.gru(next, hidden)
            nexts.append(next.squeeze(0))
        return torch.cat(nexts, 1)


class Identity(nn.Module):
    """Identity Module."""

    def __init__(self):
        """Initialize Identity Module."""
        super().__init__()

    def forward(self, x):
        """Apply Identity to Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        return x
