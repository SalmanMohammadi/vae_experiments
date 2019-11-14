import sys
sys.path.append('../models')
import torch
import numpy as np
import matplotlib.pyplot as plt
import VAE
from VAE import RotVAE, MNISTVAE
from torch.utils.tensorboard import SummaryWriter

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import datetime

DEVICE = torch.device("cuda")

def logistic_regression(x, y):
    kf = KFold(n_splits=5, shuffle=True)
    acc = []
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #fit
        model = LogisticRegression(C=50., solver='saga', penalty='l1',
                          multi_class='multinomial', max_iter=500, tol=0.1)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        acc.append(accuracy_score(y_test, y_pred))
    
    return np.mean(acc), np.std(acc)


def plot_logistic_regression(latent_vectors, digits, z_sizes, config):
    """
    Latent_vectors shape (num_models, num_samples, z_size)
    digits shape (num_models, num_samples)
    """
    writer = SummaryWriter(log_dir=LOGDIR+'/log_reg_digits')
    fig = plt.figure(figsize=(10, 12))
    total_z = config.hparams['z_size']
    acc, acc_err = [], []
    cons_acc, cons_err = [], []
    uncons_acc, uncons_err = [], []
    for latent_vector, digit, z_size in zip(latent_vectors, digits, z_sizes):
        # all dimensions
        acc_, acc_err_ = logistic_regression(latent_vector, digit)
        acc.append(acc_)
        acc_err.append(acc_err_)
        
        if z_size == 0:
            # constrained dimensions
            acc_, acc_err_ = logistic_regression(latent_vector[:,:total_z], digit)
            cons_acc.append(acc_)
            cons_err.append(acc_err_)
            # unconstrained dimensions
            acc_, acc_err_ = logistic_regression(latent_vector[:,0:], digit)
            uncons_acc.append(acc_)
            uncons_err.append(acc_err_)
        else:
            # constrained dimensions
            acc_, acc_err_ = logistic_regression(latent_vector[:,:z_size], digit)
            cons_acc.append(acc_)
            cons_err.append(acc_err_)
            # unconstrained dimensions
            acc_, acc_err_ = logistic_regression(latent_vector[:,z_size:], digit)
            uncons_acc.append(acc_)
            uncons_err.append(acc_err_)

    plt.errorbar(z_sizes, acc, yerr=acc_err, marker="x", capsize=5, label="all")
    plt.errorbar(z_sizes, cons_acc, yerr=cons_err, marker="x", capsize=5, label="constrained")
    plt.errorbar(z_sizes, uncons_acc, yerr=uncons_err, marker="x", capsize=5, label="unconstrained")
    plt.xlabel("Number of classified dimension")
    plt.ylabel("Logistic Regression accuracy")
    plt.legend()
    plt.xticks(z_sizes)

    writer.add_figure("logistic_regression on digit number", fig)
    writer.flush()
    writer.close()


def get_encodings(model, test_data):
    # with torch.no_grad():
    latent_vectors, rotations, digits = [], [], []
    for batch_id, data in enumerate(test_data):
        data, t = data
        rot = t[:, 1]
        digit = t[:, 0]
        data = data.to(DEVICE)
        mu, logvar = model.encode(data)
        z = model.sample(mu, logvar)
        latent_vectors.extend(z.detach().cpu().numpy())
        rotations.extend(rot.cpu().numpy())
        digits.extend(digit.cpu().numpy())

    return latent_vectors, rotations, digits

def run_experiment(config):
    """
    Returns an array of [latents, rotations, digits]
    """

    train_data, test_data, model, opt = VAE.setup(config)
    print(config)
    if config.hparams['include_loss']:
        logdir='experiment_1_1.1/z_size='+str(config.hparams['z_size']) + \
                    '/model=' + str(model) + '/clasifier_size=' + \
                    str(config.hparams['classifier_size']) + \
                    '/date=' + str(datetime.datetime.now())
    else:
        logdir='experiment_1_1.1/z_size='+str(config.hparams['z_size']) + \
                    '/model=' + str(model) + '/clasifier_size=' + str(0) + \
                    '/date=' + str(datetime.datetime.now())
    writer = SummaryWriter(log_dir=logdir)

    for epoch in range(1, config.model['epochs'] + 1):
      VAE.train(model, train_data, epoch, opt, verbose=False, writer=writer)
    writer.flush()
    writer.close()
    test_loss, metrics_mean = VAE.test(model, test_data, verbose=False)

    scaled_mse = np.sqrt(metrics_mean[-1]) * 45
    metrics = [test_loss] + list(metrics_mean) + [scaled_mse]
    [print(y, x) for x, y in zip(metrics, ["loss", "r_loss", "kl_loss", "mse", "scaled_ms"])]

    latents, rotations, digits = get_encodings(model, test_data)
    return latents, rotations, digits


config_ = VAE.Config(
    # dataset
    dataset={
        'digits': None,
        'n_obs':30000, # max of ~6000 per image
        'csv_path': '../data/mnist_train.csv'
    },
    # transformations
    transformations={
        'trans_per_image':2,
        'max_angle':45,
        'max_brightness':0.,
        'max_noise':0.,
        'max_scale':1.
    },
    #model
    model={
        'model': RotVAE,
        'epochs':25,
        'lr':0.001,
        'batch_size':1024,
    },
    hparams={
        'z_size': 8,
        'classifier_size': 1,
        'include_loss': False
    }
)

LOGDIR = 'experiment_1_1.1/z_size' + str(config_.hparams['z_size'])

print(torch.cuda.get_device_name(torch.cuda.current_device()))
z_sizes = [1, 2, 3, 4]#, 4, 5, 6, 7, 8]
configs = []

configs.append(config_)
hparams = config_.hparams.copy()
hparams['include_loss'] = True
config_ = config_._replace(hparams=hparams)
for z in z_sizes:
    hparams = config_.hparams.copy()
    hparams["classifier_size"] = z
    configs.append(config_._replace(hparams=hparams))

z_sizes = [0] + z_sizes
latent_vectors, rotations, digits = map(np.array, zip(*[run_experiment(c) for c in configs]))

# Plot logistic regression against digits
plot_logistic_regression(latent_vectors, digits, z_sizes, config_)