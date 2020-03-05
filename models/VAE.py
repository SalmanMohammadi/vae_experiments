import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

import numpy as np
import matplotlib.pyplot as plt
import mnist_data as mnist

import collections

DEVICE = torch.device("cuda")

class MNISTVAE(nn.Module):

    def __str__(self):
        return "MNISTVAE"

    def __init__(self, z_size=4, *args, **kwargs):
        super(MNISTVAE, self).__init__()
        self.z_size = z_size
        # encoder
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, z_size) # mu
        self.fc3 = nn.Linear(512, z_size) # logvar

        # decoder
        self.fc4 = nn.Linear(z_size, 512)
        self.fc5 = nn.Linear(512, 784)

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

class RotVAE(MNISTVAE):
    def __str__(self):
        return "RotVAE"

    def __init__(self, z_size=4, classifier_size=1, include_loss=True, *args, **kwargs):
        super(RotVAE, self).__init__(z_size=z_size)
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
        rot = t[:, 1]
        data, rot = data.to(DEVICE), rot.to(device=DEVICE)
        rot = rot.view(-1, 1)
        x_, mu, logvar, rot_ = self(data)
        return self.compute_loss(data, x_, mu, logvar, rot, rot_)

class ScaleVAE(MNISTVAE):
    def __str__(self):
        return "ScaleVAE"

    def __init__(self, z_size=4, classifier_size=1, include_loss=True, *args, **kwargs):
        super(RotVAE, self).__init__(z_size=z_size)
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
        rot = t[:, 4]
        data, rot = data.to(DEVICE), rot.to(device=DEVICE)
        rot = rot.view(-1, 1)
        x_, mu, logvar, rot_ = self(data)
        return self.compute_loss(data, x_, mu, logvar, rot, rot_)

def train(model, dataset, epoch, optimizer, verbose=True, writer=None):
    """
    Trains the model for a single 'epoch' on the data
    """
    model.train()
    train_loss = 0
    metrics_mean = []
    for batch_id, data in enumerate(dataset):
        optimizer.zero_grad()
        loss, metrics = model.batch_forward(data)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        metrics_mean.append([x.item() for x in metrics])
        
        data_len = len(data[0])
        if batch_id % args.log_interval == 0 and verbose:
            print('Train Epoch: {}, batch: {}, loss: {}'.format(
                epoch, batch_id, loss.item() / data_len))
            metrics = [x.item()/data_len for x in metrics]
            print(metrics)

    metrics_mean = np.array(metrics_mean)
    metrics_mean = np.sum(metrics_mean, axis=0)/len(dataset.dataset)

    writer.add_scalar('train/loss', train_loss / len(dataset.dataset), epoch)
    for label, metric in zip(['r_loss', 'kl_loss', 'mse'], metrics_mean):
        writer.add_scalar('train/'+label, metric, epoch)
    if len(metrics_mean) > 2:
        scaled_mse = np.sqrt(metrics_mean[-1])*45
        writer.add_scalar('train/scaled_mse', scaled_mse, epoch)
    if verbose:
        print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(dataset.dataset)))
    writer.flush()
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

def get_mnist(config):
    # returns test and train data
    dataset = mnist.MNIST(**config.dataset)
    if config.transformations['trans_per_image'] > 0:
        dataset.create_transformations(**config.transformations)

    # construct pytorch DataLoader to wrap around MNIST Dataset
    data = DataLoader(dataset, config.model['batch_size'], shuffle=True)

    return data

def setup(config, dataset='dsprites'):
    """
    Initializes experiment parameters from config.
    """
    model = config.model['model'](**config.hparams).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.model['lr'])
    
    if dataset == 'mnist':
        train_data, test_data = get_mnist(config)
    elif dataset == 'dsprites':
        train_data, test_data = get_dsprites(config)
    return train_data, test_data, model, optimizer

parser = argparse.ArgumentParser(description="MNIST VAE Experiments")
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--run_dir', type=str, default='./')
parser.add_argument('--verbose', type=bool, default=True)
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

# Config file for expeirment hyperparameters
Config = collections.namedtuple(
    'Config',
    ['dataset', 'transformations', 'model', 'hparams']
)

## Eample MNIST Config
config = Config(
    # dataset
    dataset={
        'digits':None,
        'n_obs':60000, # max of ~6000 per image
        'csv_path': '../data/mnist_train.csv'
    },
    # transformations
    transformations={
        'trans_per_image':1,
        'max_angle':45,
        'max_brightness':0.,
        'max_noise':0.,
        'max_scale':1.
    },
    #model
    model={
        'model': RotVAE,
        'epochs':30,
        'lr':0.001,
        'batch_size':1024
    },
    hparams={
        'z_size': 8,
        'classifier_size': 2,
        'include_loss': True
    }
)

if __name__ == "__main__":
    assert args.mode in ['train', 'eval']
    # if args.mode == 'train':
    #     config.dataset['csv_path'] = "data/mnist_train.csv"
    # elif args.mode == 'eval':
    #     config.dataset ['csv_path'] = "data/mnist_test.csv"
    data, test_data, model, opt = setup(config)

    print(config)
    # print("*"*8)
    # print("Dataset has %d examples from %d digits with %d images per digit (%d original images) and %d transformations per image"
    #         % (config.dataset['n_obs'] * max(1, config.transformations['trans_per_image']),
    #          len(config.dataset['digits']), 
    #          config.dataset['n_obs'] // len(config.dataset['digits']),
    #          config.dataset['n_obs'],
    #          config.transformations['trans_per_image']))
    print("*"*8)
    d_x, t = next(iter(test_data))
    for epoch in range(1, config.model['epochs'] + 1):
        train(model, data, epoch, opt)
        # test(data, epoch) # todo load test dataset
    test_loss, metrics_mean = test(model, test_data, verbose=False)

    scaled_mse = np.sqrt(metrics_mean[-1]) * 45
    metrics = [test_loss] + list(metrics_mean) + [scaled_mse]
    [print(y, x) for x, y in zip(metrics, ["loss", "r_loss", "kl_loss", "mse", "scaled_ms"])]

    import matplotlib.pyplot as plt
    with torch.no_grad():
        d_x = d_x.to(DEVICE)
        t = t[:,1].cpu()
        mu, logvar = model.encode(d_x)
        z = model.sample(mu, logvar)
        z = z[:,:2]
        t = t*45
        plt.figure()
        plt.scatter(z[:,0].cpu(), z[:,1].cpu(), c=t, alpha=0.6, cmap='inferno')
        # for z_, t_ in zip(z, t):
        #     t_ = t_.item()
        #     t_ *= 45
        #     z_ = z_.cpu().numpy()
        #     print(z_, z_[0], z_[1], t_)
        #     plt.scatter([z_[0]], [z_[1]], c=t_, cmap='inferno')
        plt.colorbar()
        plt.show()
        # print(z.shape)
        # plt.plot(z[:,0], z[:, 1])
        exit()
        # n = 20  # figure with 15x15 digits
        # digit_size = 28
        # figure = np.zeros((digit_size * n, digit_size * n))
        # we will sample n points within [-15, 15] standard deviations
        grid_x = np.linspace(-100, 100, n)
        grid_y = np.linspace(-100, 100, n)

        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = z
                z_sample[0] = xi
                z_sample[1] = yi
                # z_sample = torch.tensor(np.array([[xi, yi, z[-2], z[-1]]], dtype=np.float32)).to(DEVICE)
                # print(z_sample,z_sample.shape)
                x_decoded = model.decode(z_sample).cpu()
                # print(x_decoded.shape)
                digit = x_decoded.reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(30, 30))
        plt.imshow(figure, cmap="Greys")
        plt.show()