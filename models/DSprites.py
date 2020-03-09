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
        self.fc1 = nn.Linear(4096, 1200)
        self.fc2 = nn.Linear(1200, z_size) # mu
        self.fc3 = nn.Linear(1200, z_size) # logvar

        # decoder
        self.fc4 = nn.Linear(z_size, 1200)
        self.fc5 = nn.Linear(1200, 1200)
        self.fc6 = nn.Linear(1200, 4096)

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
        h4 = F.relu(self.fc5(h3))
        x_ = torch.sigmoid(self.fc6(h4))
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

class RegDSpritesVAE(DSpritesVAE):
    def __str__(self):
        return "RegDSpritesVAE"

    def __init__(self, z_size=4, classifier_size=1, include_loss=True, reg_index=-1, *args, **kwargs):
        super(RegDSpritesVAE, self).__init__(z_size=z_size)
        assert classifier_size <= self.z_size and classifier_size > 0

        self.include_loss = include_loss
        self.reg_index = reg_index
        self.classifier_size = classifier_size
        self.classifier = nn.Linear(self.classifier_size, 1)

    def regress(self, z):
        return self.classifier(z[:,:self.classifier_size])

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        y_ = self.regress(z)
        x_ = self.decode(z)
        return x_, mu, logvar, y_

    def compute_loss(self, x, x_, mu, logvar, y, y_):
        elbo, losses = super().compute_loss(x, x_, mu, logvar)

        mse = F.mse_loss(y_, y)
        if self.include_loss:
            return elbo + mse, losses + (mse,)
        return elbo, losses + (mse,)

    def batch_forward(self, data):
        data, t = data
        y = t[:, self.reg_index]
        data, y = data.to(DEVICE), y.to(device=DEVICE)
        y = y.view(-1, 1)
        x_, mu, logvar, y_ = self(data)
        return self.compute_loss(data, x_, mu, logvar, y, y_)

class FactorVAE(DSpritesVAE):
    def __str__(self):
        return "FactorVAE"

    def __init__(self, z_size=6, classifier_size=1, include_loss=True, *args, **kwargs):
        z_size = z_size - classifier_size
        super(FactorVAE, self).__init__()
        self.z_size = z_size
        self.fc1 = nn.Linear(4096, 1200)
        self.fc2 = nn.Linear(1200, z_size) # mu
        self.fc3 = nn.Linear(1200, z_size) # logvar

        # decoder
        self.fc4 = nn.Linear(z_size+classifier_size, 1200)
        self.fc5 = nn.Linear(1200, 1200)
        self.fc6 = nn.Linear(1200, 4096)
        assert classifier_size <= self.z_size and classifier_size > 0

        self.include_loss = include_loss
        self.classifier_size = classifier_size

        self.z_prime_mu = nn.Linear(1200, classifier_size) # mu
        self.z_prime_logvar = nn.Linear(1200, classifier_size) # logvar

        self.fc7 = nn.Linear(self.classifier_size, 6)

        self.ce_loss = nn.CrossEntropyLoss()

    def predict_latent(self, z_prime):
        return self.fc7(z_prime)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        mu, logvar = self.fc2(h1), self.fc3(h1)
        mu_prime, logvar_prime = self.z_prime_mu(h1), self.z_prime_logvar(h1)
        return mu, logvar, mu_prime, logvar_prime

    def sample(self, mu, logvar, mu_prime, logvar_prime):
        z_ = super().sample(mu, logvar)
        z_prime = super().sample(mu_prime, logvar_prime)
        return z_, z_prime
        
    def forward(self, x):
        mu, logvar, mu_prime, logvar_prime = self.encode(x)
        z_, z_prime = self.sample(mu, logvar, mu_prime, logvar_prime)
        z = torch.cat((z_, z_prime), dim=1)
        y_ = self.predict_latent(z_prime)
        x_ = self.decode(z)
        return x_, mu, logvar, mu_prime, logvar_prime, y_

    def compute_loss(self, x, x_, y, y_, mu, logvar, mu_prime, logvar_prime):
        elbo, losses = super().compute_loss(x, x_, mu, logvar)

        # KL divergence between z_prime and p
        kl_prime = -0.5 * torch.sum(1 + logvar_prime - mu_prime.pow(2) - logvar_prime.exp())

        # kl divergence between z_prime and z
        # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        kl_latent = 0

        # CE loss
        ce_loss = self.ce_loss(y_, y)

        return elbo + kl_prime + ce_loss - kl_latent, losses + (kl_prime, kl_latent, ce_loss,)

    def batch_forward(self, data, device=DEVICE):
        data, y = data
        data = data.to(device)
        y = y[:,2].to(device)
        x_, mu, logvar, mu_prime, logvar_prime, y_ = self(data)
        return self.compute_loss(data, x_, y, y_, mu, logvar, mu_prime, logvar_prime)
    
    def decode(self, z):
        h3 = F.tanh(self.fc4(z))
        h4 = F.tanh(self.fc5(h3))
        x_ = torch.sigmoid(self.fc6(h4))
        return x_

class FactorBernoulliVAE(DSpritesVAE):
    def __str__(self):
        return "FactorVAE"

    def __init__(self, z_size=6, classifier_size=1, include_loss=True, *args, **kwargs):
        z_size = z_size - classifier_size
        super(FactorVAE, self).__init__()
        self.z_size = z_size
        self.fc1 = nn.Linear(4096, 1200)
        self.fc2 = nn.Linear(1200, z_size) # mu
        self.fc3 = nn.Linear(1200, z_size) # logvar

        # decoder
        self.fc4 = nn.Linear(z_size+classifier_size, 1200)
        self.fc5 = nn.Linear(1200, 1200)
        self.fc6 = nn.Linear(1200, 4096)
        assert classifier_size <= self.z_size and classifier_size > 0

        self.include_loss = include_loss
        self.classifier_size = classifier_size

        self.z_prime_mu = nn.Linear(1200, classifier_size) # mu
        self.z_prime_logvar = nn.Linear(1200, classifier_size) # logvar

        self.fc7 = nn.Linear(self.classifier_size, 6)

        self.ce_loss = nn.CrossEntropyLoss()

    def predict_latent(self, z_prime):
        return self.fc7(z_prime)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        mu, logvar = self.fc2(h1), self.fc3(h1)
        mu_prime, logvar_prime = self.z_prime_mu(h1), self.z_prime_logvar(h1)
        return mu, logvar, mu_prime, logvar_prime

    def sample(self, mu, logvar, mu_prime, logvar_prime):
        z_ = super().sample(mu, logvar)
        z_prime = super().sample(mu_prime, logvar_prime)
        return z_, z_prime
        
    def forward(self, x):
        mu, logvar, mu_prime, logvar_prime = self.encode(x)
        z_, z_prime = self.sample(mu, logvar, mu_prime, logvar_prime)
        z = torch.cat((z_, z_prime), dim=1)
        y_ = self.predict_latent(z_prime)
        x_ = self.decode(z)
        return x_, mu, logvar, mu_prime, logvar_prime, y_

    def compute_loss(self, x, x_, y, y_, mu, logvar, mu_prime, logvar_prime):
        elbo, losses = super().compute_loss(x, x_, mu, logvar)

        # KL divergence between z_prime and p
        kl_loss = -0.5 * torch.sum(1 + logvar_prime - mu_prime.pow(2) - logvar_prime.exp())

        # CE loss
        ce_loss = self.ce_loss(y_, y)

        return elbo + kl_loss + ce_loss, losses + (kl_loss, ce_loss,)

    def batch_forward(self, data, device=DEVICE):
        data, y = data
        data = data.to(device)
        y = y[:,2].to(device)
        x_, mu, logvar, mu_prime, logvar_prime, y_ = self(data)
        return self.compute_loss(data, x_, y, y_, mu, logvar, mu_prime, logvar_prime) 

def train(model, dataset, epoch, optimizer, device=DEVICE, verbose=True, writer=None, log_interval=100, metrics_labels=None):
    """
    Trains the model for a single 'epoch' on the data
    """
    model.train()
    train_loss = 0
    metrics_mean = []
    dataset_len = len(dataset) * dataset.batch_size
    for batch_id, data in enumerate(dataset):
        optimizer.zero_grad()
        loss, metrics = model.batch_forward(data, device=DEVICE)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        metrics_mean.append([x.item() for x in metrics])
        
        data_len = len(data[0])
        if batch_id % log_interval == 0 and verbose:
            print('Train Epoch: {}, batch: {}, loss: {}'.format(
                epoch, batch_id, loss.item() / data_len))
            metrics = [x.item()/data_len for x in metrics]
            if metrics_labels:
                print(", ".join(list(map(lambda x: "%s: %.5f" % x, zip(metrics_labels, metrics)))))
            else:
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

def get_dsprites(config, dataset=None):
    """
    Returns train and test DSprites dataset.
    """
    if dataset is None:
        dataset = dsprites.DSpritesRaw(**config.dataset)
    train_data, test_data = dataset.get_train_test_datasets()

    train_loader = DataLoader(train_data, batch_size=config.model['batch_size'])
    test_loader = DataLoader(test_data, batch_size=config.model['batch_size'])

    return train_loader, test_loader

def setup(config, dataset=None, iid=False, device=DEVICE):
    """
    Initializes experiment parameters from config.
    """
    model = config.model['model'](**config.hparams).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.model['lr'])

    if iid:
        dsprites_loader = dsprites.DSpritesLoader()
        train_data = DataLoader(dsprites.DSPritesIID(size=config.dataset['train_size'], dsprites_loader=dsprites_loader),
                                batch_size=config.model['batch_size'], pin_memory=True)
        test_data = DataLoader(dsprites.DSPritesIID(size=config.dataset['test_size'], dsprites_loader=dsprites_loader),
                                batch_size=config.model['batch_size'])                    
    else:
        train_data, test_data = get_dsprites(config, dataset=dataset)
    return train_data, test_data, model, optimizer

Config = collections.namedtuple(
    'Config',
    ['dataset', 'model', 'hparams']
)

config_ = Config(
    # dataset
    dataset={
        'test_index': 3,
        'iid': True,
        'train_size': 300000,
        'test_size': 10000
    },
    #model
    model={
        'model': RegDSpritesVAE,
        'epochs':40,
        'lr':0.0001,
        'batch_size':512,
        'reg_index': 2
    },
    hparams={
        'z_size': 6,
        'classifier_size': 1,
        'include_loss': False
    }
)

if __name__ == "__main__":
    # writer = SummaryWriter(log_dir='./tmp/rot/run2')
    train_data, test_data, model, opt = setup(config_, iid=config_.dataset['iid'])
    # for epoch in range(config_.model['epochs']):
    #     train(model, train_data, epoch, opt, writer=None, verbose=True)
    print(test(model, test_data, verbose=False))

    # View some predicitions from the model

    with torch.no_grad():
        num_samples = 10
        results = []
        for i in range(config_.hparams['z_size']):
            samples = np.random.randn(num_samples, config_.hparams['z_size']).astype(np.float32)
            samples[:, i] = np.linspace(-1, 1, num=num_samples, dtype=np.float32)
            # print(samples)
            samples = model.decode(torch.tensor(samples).to(DEVICE))
            samples = samples.cpu()
            results.append(samples)

        fig, axes = plt.subplots(num_samples, config_.hparams['z_size'])
        plt.subplots_adjust(top=0.9, hspace=0.55)
        for i in range(num_samples):
            for j in range(config_.hparams['z_size']-1):
                print("i, j", i, j, config_.hparams['z_size'])
                print(len(results), len(results[i]))
                x = results[i][j].view((-1, 64, 64)).squeeze()
                axes[i, j].imshow(x, cmap="Greys")
                axes[i, j].axis('off')
        plt.show()
        plt.tight_layout()
        plt.axis('off')
        # sample = sample.cpu()
        # fig, axes = plt.subplots(3, 3, figsize=(8, 8))
       