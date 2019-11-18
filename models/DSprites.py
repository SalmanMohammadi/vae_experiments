import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter 
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.utils import save_image

import numpy as np
import matplotlib.pyplot as plt
import dsprites_data as dsprites

import collections

DEVICE = torch.device("cuda")

class DSpritesVAE(nn.Module):
    def __str__(self):
        return "DSpritesVAE"

    def __init__(self, z_size=4, *args, **kwargs):
        super(DSpritesVAE, self).__init__()
        self.z_size = z_size
        # encoder
        self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, z_size) # mu
        self.fc3 = nn.Linear(2048, z_size) # logvar

        # decoder
        self.fc4 = nn.Linear(z_size, 2048)
        self.fc5 = nn.Linear(2048, 4096)

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        p_z = torch.rand_like(std)
        return mu + p_z*std

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        mu, logvar = self.fc2(h1), self.fc3(h1)
        return mu, logvar

    def decode(self, z):
        h3 = F.relu(self.fc4(z))
        x_ = torch.sigmoid(self.fc5(h3))
        return x_

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        x_ = self.decode(z)
        return x_, mu, logvar

    def compute_loss(self, x, x_, mu, logvar):
        """
        Standard ELBO
        """
        # expectation term/reconstruction loss
        r_loss = F.binary_cross_entropy(x_, x, reduction='sum')

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return r_loss + kl_loss, (r_loss, kl_loss)

    def batch_forward(self, data):
        data, _ = data
        data = data.to(DEVICE)
        x_, mu, logvar = self(data)
        return self.compute_loss(data, x_, mu, logvar)

class RotDSpritesVAE(DSpritesVAE):
    def __str__(self):
        return "RotDSpritesVAE"

    def __init__(self, z_size=4, classifier_size=1, include_loss=True, *args, **kwargs):
        super(RotDSpritesVAE, self).__init__(z_size=z_size)
        assert classifier_size <= self.z_size and classifier_size > 0

        self.include_loss = include_loss
        self.classifier_size = classifier_size
        self.fc6 = nn.Linear(self.classifier_size, 1)

    def predict_rotation(self, z):
        return self.fc6(z[:,:self.classifier_size])

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        rot_ = self.predict_rotation(z)
        x_ = self.decode(z)
        return x_, mu, logvar, rot_

    def compute_loss(self, x, x_, mu, logvar, rot, rot_):
        elbo, losses = super().compute_loss(x, x_, mu, logvar)

        mse = F.mse_loss(rot_, rot)
        if self.include_loss:
            return elbo + mse, losses + (mse,)
        return elbo, losses + (mse,)

    def batch_forward(self, data):
        data, t = data
        rot = t[:, 3]
        data, rot = data.to(DEVICE), rot.to(device=DEVICE)
        rot = rot.view(-1, 1)
        x_, mu, logvar, rot_ = self(data)
        return self.compute_loss(data, x_, mu, logvar, rot, rot_)

def train(model, dataset, epoch, optimizer, verbose=True, writer=None, log_interval=100):
    """
    Trains the model for a single 'epoch' on the data
    """
    model.train()
    train_loss = 0
    metrics_mean = []
    dataset_len = len(dataset) * dataset.batch_size
    for batch_id, data in enumerate(dataset):
        optimizer.zero_grad()
        loss, metrics = model.batch_forward(data)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        metrics_mean.append([x.item() for x in metrics])
        
        data_len = len(data[0])
        if batch_id % log_interval == 0 and verbose:
            print('Train Epoch: {}, batch: {}, loss: {}'.format(
                epoch, batch_id, loss.item() / data_len))
            metrics = [x.item()/data_len for x in metrics]
            print(metrics)

    metrics_mean = np.array(metrics_mean)
    metrics_mean = np.sum(metrics_mean, axis=0)/dataset_len

    if writer:
        writer.add_scalar('train/loss', train_loss /dataset_len, epoch)
        for label, metric in zip(['r_loss', 'kl_loss', 'mse'], metrics_mean):
            writer.add_scalar('train/'+label, metric, epoch)
        if len(metrics_mean) > 2:
            scaled_mse = np.sqrt(metrics_mean[-1])*45
            writer.add_scalar('train/scaled_mse', scaled_mse, epoch)
            writer.flush()
    if verbose:
        print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / dataset_len))
        

def test(model, dataset, verbose=True):
    """
    Evaluates the model
    """
    model.eval()
    test_loss = 0
    metrics_mean = []
    with torch.no_grad():
        for batch_id, data in enumerate(dataset):
            loss, metrics = model.batch_forward(data)
            metrics_mean.append([x.item() for x in metrics])
            test_loss += loss.item()

    test_loss /= len(dataset.dataset)
    metrics_mean = np.array(metrics_mean)
    metrics_mean = np.sum(metrics_mean, axis=0)/len(dataset.dataset)
    # metrics = [x.item()/len(dataset.dataset) for x in metrics]
    if verbose:
        print("Eval: ", test_loss, metrics_mean)
    return test_loss, metrics_mean

def get_dsprites(config):
    """
    Returns train and test DSprites dataset.
    """
    dataset = dsprites.DSpritesRaw(**config.dataset)
    train_data, test_data = dataset.get_train_test_datasets()

    train_loader = DataLoader(train_data, batch_size=config.model['batch_size'])
    test_loader = DataLoader(test_data, batch_size=config.model['batch_size'])

    return train_loader, test_loader

def setup(config):
    """
    Initializes experiment parameters from config.
    """
    model = config.model['model'](**config.hparams).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.model['lr'])
    
    train_data, test_data = get_dsprites(config)
    return train_data, test_data, model, optimizer

Config = collections.namedtuple(
    'Config',
    ['dataset', 'model', 'hparams']
)

config_ = Config(
    # dataset
    dataset={
        'test_index': 3
    },
    #model
    model={
        'model': RotDSpritesVAE,
        'epochs':20,
        'lr':0.001,
        'batch_size':256,
    },
    hparams={
        'z_size': 10,
        'classifier_size': 2,
        'include_loss': True
    }
)

if __name__ == "__main__":
    writer = SummaryWriter(log_dir='./tmp/rot/run2')
    train_data, test_data, model, opt = setup(config_)
    for epoch in range(config_.model['epochs']):
        train(model, train_data, epoch, opt, writer=writer, verbose=True)
    print(test(model, test_data, verbose=False))

    # View some predicitions from the model

    with torch.no_grad():
        sample, _ = next(iter(test_data))
        sample = sample[:9].to(DEVICE)
        sample, *_ = model(sample)
        sample = sample.cpu()
        fig, axes = plt.subplots(3, 3, figsize=(8, 8))
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, hspace=0.55)
        for idx, x in enumerate(sample):
            x = x.view((-1, 64, 64)).squeeze()
            np.ravel(axes)[idx].imshow(x, cmap="Greys")
        plt.show()