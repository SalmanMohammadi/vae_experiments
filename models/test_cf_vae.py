import torch
import DSprites
from DSprites import FactorVAE
import numpy as np
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda")

config_ = DSprites.Config(
    # dataset
    dataset={
        'iid': True,
        'train_size': 300000,
        'test_size': 10000
    },
    #model
    model={
        'model': FactorVAE,
        'epochs':55,
        'lr':0.001,
        'batch_size':1024,
        'metrics_labels':['r_loss', 'kl', 'kl_prime', 'kl_latent', 'classification_ce']
    },
    hparams={
        'z_size': 10,
        'classifier_size': 1,
        'include_loss': False
    }
)

train_data, test_data, model, opt = DSprites.setup(config_, iid=config_.dataset['iid'])
for epoch in range(config_.model['epochs']):
    DSprites.train(model, train_data, epoch, opt, writer=None, verbose=True, metrics_labels=config_.model['metrics_labels'])

sample = next(iter(test_data))[0][0]
plt.imshow(sample.view(64, 64), cmap='Greys')
mu, _, mu_prime, _ = model.encode(sample.unsqueeze(dim=0).to(DEVICE))
z = torch.cat((mu, mu_prime), dim=1).detach().cpu()
num_samples = 11
results = []
for dimension in np.arange(config_.hparams['z_size']):
    values = np.linspace(-1., 1., num=num_samples)
    latent_traversal_vectors = np.tile(z.cpu(), [num_samples, 1])
    latent_traversal_vectors[:, dimension] += values
    images = model.decode(torch.tensor(latent_traversal_vectors).to(DEVICE)).view((-1, 64, 64)).detach().cpu().numpy()
    results.append(images)

results = np.array(results).swapaxes(0,1)
results = np.concatenate(results, 0)
print(results.shape)
fig=plt.figure(figsize=(8, 8))
fig.subplots_adjust(hspace=0, wspace=0) 
for i in range(1, num_samples*config_.hparams['z_size']+1):
    fig.add_subplot(config_.hparams['z_size'], num_samples, i)
    plt.imshow(1-results[i-1], cmap='Greys')
    # plt.axis('off')

sample = next(iter(test_data))[0][0]
plt.imshow(sample.view(64, 64), cmap='Greys')
mu, _, mu_prime, _ = model.encode(sample.unsqueeze(dim=0).to(DEVICE))
z = torch.cat((mu, mu_prime), dim=1).detach().cpu()
num_samples = 11
results = []
for dimension in np.arange(config_.hparams['z_size']):
    values = np.linspace(-1., 1., num=num_samples)
    latent_traversal_vectors = np.tile(z.cpu(), [num_samples, 1])
    latent_traversal_vectors[:, dimension] += values
    images = model.decode(torch.tensor(latent_traversal_vectors).to(DEVICE)).view((-1, 64, 64)).detach().cpu().numpy()
    results.append(images)

results = np.array(results).swapaxes(0,1)
results = np.concatenate(results, 0)
print(results.shape)
fig=plt.figure(figsize=(8, 8))
fig.subplots_adjust(hspace=0, wspace=0) 
for i in range(1, num_samples*config_.hparams['z_size']+1):
    fig.add_subplot(config_.hparams['z_size'], num_samples, i)
    plt.imshow(1-results[i-1], cmap='Greys')
    # plt.axis('off')

sample = next(iter(test_data))[0][0]
plt.imshow(sample.view(64, 64), cmap='Greys')
mu, _, mu_prime, _ = model.encode(sample.unsqueeze(dim=0).to(DEVICE))
z = torch.cat((mu, mu_prime), dim=1).detach().cpu()
num_samples = 11
results = []
for dimension in np.arange(config_.hparams['z_size']):
    values = np.linspace(-1., 1., num=num_samples)
    latent_traversal_vectors = np.tile(z.cpu(), [num_samples, 1])
    latent_traversal_vectors[:, dimension] += values
    images = model.decode(torch.tensor(latent_traversal_vectors).to(DEVICE)).view((-1, 64, 64)).detach().cpu().numpy()
    results.append(images)

results = np.array(results).swapaxes(0,1)
results = np.concatenate(results, 0)
print(results.shape)
fig=plt.figure(figsize=(8, 8))
fig.subplots_adjust(hspace=0, wspace=0) 
for i in range(1, num_samples*config_.hparams['z_size']+1):
    fig.add_subplot(config_.hparams['z_size'], num_samples, i)
    plt.imshow(1-results[i-1], cmap='Greys')
    # plt.axis('off')
plt.show()