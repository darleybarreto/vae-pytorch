import torch
from torch import nn

from torch.autograd import Variable
from torch.nn import functional as F

def recons_loss(recon_x, x):
    return F.mse_loss(recon_x, x, reduction='sum')

def choose_vae(dataset):
    return MLPVAE if dataset == 'mnist' else ConvVAE

class MLPVAE(nn.Module):
    """
    Based on https://github.com/pytorch/examples/tree/master/vae

    :h_dim:             size of the hidden encoder/decoder representation
    :z_dim:             size of the latent space
    """
    def __init__(self, h_dim: int = 400, z_dim: int = 20, im_shape: tuple = (28,28)):
        super(MLPVAE, self).__init__()

        self.w,self.h  = im_shape

        self.fc1 = nn.Linear(self.w*self.h, h_dim)
        
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)
        # self.fc23 = nn.Linear(h_dim, z_dim)
        # self.fc24 = nn.Linear(h_dim, z_dim)

        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, self.w*self.h)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def _encode(self, x):
        h1 = self.relu(self.fc1(x))
        return h1

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))


class ConvVAE(nn.Module):
    def __init__(self, h_dim=512, z_dim=512):
        super(ConvVAE, self).__init__()
        
        # Encoder
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(8 * 8 * 16, h_dim)
        self.fc_bn1 = nn.BatchNorm1d(h_dim)
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)
        # self.fc43 = nn.Linear(h_dim, z_dim)
        # self.fc44 = nn.Linear(h_dim, z_dim)

        # Decoder
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc_bn3 = nn.BatchNorm1d(h_dim)
        self.fc4 = nn.Linear(h_dim, 8 * 8 * 16)
        self.fc_bn4 = nn.BatchNorm1d(8 * 8 * 16)

        self.conv5 = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(16)
        self.conv8 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def _encode(self, x):
        conv1 = self.relu(self.bn1(self.conv1(x)))
        conv2 = self.relu(self.bn2(self.conv2(conv1)))
        conv3 = self.relu(self.bn3(self.conv3(conv2)))
        conv4 = self.relu(self.bn4(self.conv4(conv3))).view(-1, 8 * 8 * 16)
        fc1 = self.relu(self.fc_bn1(self.fc1(conv4)))
        return fc1

    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3))).view(-1, 16, 8, 8)

        conv5 = self.relu(self.bn5(self.conv5(fc4)))
        conv6 = self.relu(self.bn6(self.conv6(conv5)))
        conv7 = self.relu(self.bn7(self.conv7(conv6)))
        conv8 = self.sigmoid(self.conv8(conv7))
        return conv8.view(-1, 3, 32, 32)
      