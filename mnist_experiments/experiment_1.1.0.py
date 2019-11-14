import sys
sys.path.append('../models')
import torch
import numpy as np
import matplotlib.pyplot as plt
import VAE
from VAE import RotVAE, MNISTVAE
from torch.utils.tensorboard import SummaryWriter
import datetime
DEVICE = torch.device("cuda")

def plot_latents(latent_vectors, rotations, z_sizes, config_):
    """
    Visualises model latent space
    """
    logdir='experiment_1_1.0/z_size='+str(config_.hparams['z_size'])+'/latent_vis'
    writer = SummaryWriter(log_dir=logdir)
    fig, axes = plt.subplots(1, len(z_sizes), figsize=(12*len(z_sizes), 8))
    plt.tight_layout()
    for latents, rot, z_size, ax in zip(latent_vectors, rotations, 
                                            z_sizes, np.ravel(axes)):
        rot *= 45
        im = ax.scatter(latents[:,0], latents[:, 1], c=rot, alpha=0.6, cmap='inferno')
        ax.set(xlabel="latent dimension 0", ylabel="latent dimension 1")
        ax.set_title("Latent space for " + str(z_size) + " dimensions assigned to regressor")
        plt.colorbar(im, ax=ax)
    writer.add_figure("latent_spaces for "+ str(config_.hparams['z_size']), fig)
    writer.flush()
    writer.close()

def plot_metrics(results, z_sizes, config_):
    """
    Plots model metrics as a function of classifier z_sizes.
    """
    labels =["loss", "r_loss", "kl_loss", "mse", "scaled_ms"]
    logdir='experiment_1_1.0/z_size='+str(config_.hparams['z_size'])+'/metrics'
    writer = SummaryWriter(log_dir=logdir)
    plt.tight_layout()
    fig, axes = plt.subplots(len(labels),  figsize=(10, 5*len(labels)))
    for model_res, label, ax in zip(results, labels, np.ravel(axes)):
        ax.set_title("Test " + label + " for dimensions allocated to regressor")
        lines = ax.plot(z_sizes, model_res)
        ax.set_xticks(z_sizes)
        ax.legend(lines, z_sizes)
        ax.set(xlabel="Number of dimensions allocated", ylabel=label)
    writer.add_figure("test metrics for z_size " + str(config_.hparams['z_size']), fig)
    writer.flush()
    writer.close()

def input_sensitivites(sensitivities, z_sizes, config_):
    """
    Runs input sensitvity analysis on the models
    """
    logdir='experiment_1_1.0/z_size='+str(config_.hparams['z_size'])+'/inp_sens'

    writer = SummaryWriter(log_dir=logdir)
    fig, axes = plt.subplots(len(z_sizes),  figsize=(10, 5*len(z_sizes)))
    plt.tight_layout()
    # plt.subplots_adjust(top=0.9, hspace=0.55)
    for i, (sensitivity, label, ax) in enumerate(zip(sensitivites, z_sizes, np.ravel(axes))):
        ax.bar(range(config_.hparams['z_size']), sensitivity)
        if label > 0:
            ax.axvline(label-1, color='black')
        ax.set_title(str(label) + " dimenisons")
        ax.set_xticks(z_sizes)
        ax.set(xlabel="Dimension number", ylabel="Sensitivity")

    writer.add_figure("input sensitivity z_size="+str(label), fig)
    writer.flush()
    writer.close()


def get_encodings(model, test_data):
    """
    Returns latent encodings from the test set along with
    the input sensitivity for them
    """
    # with torch.no_grad():
    grads = np.zeros((784, model.z_size))
    z_s = []
    rotations = []
    for batch_id, data in enumerate(test_data):
        data, t = data
        rot = t[:, 1]
        data = data.to(DEVICE)
        mu, logvar = model.encode(data)
        z = model.sample(mu, logvar)
        z_s.extend(z.detach().cpu().numpy())
        rotations.extend(rot.cpu().numpy())
        for x in torch.split(z, 1):
            x = x.clone().detach()
            x = x.squeeze()
            x = x.repeat(784, 1)
            x.requires_grad_(True)
            out = model.decode(x)
            out.backward(torch.eye(784).to(DEVICE))
            grad = x.grad.data
            grads = np.add(grads, np.abs(grad.cpu().numpy()))

    # return np.sum(grads, axis=0)
    return np.linalg.norm(grads, axis=0), np.array(z_s), np.array(rotations)

def run_experiment(config):
    """
    Returns an array of [loss, metrics]
    """

    train_data, test_data, model, opt = VAE.setup(config)
    print(config)
    if config.hparams['include_loss']:
        logdir='experiment_1_1.0/z_size='+str(config.hparams['z_size']) + \
                    '/model=' + str(model) + '/clasifier_size=' + \
                    str(config.hparams['classifier_size']) + \
                    '/date=' + str(datetime.datetime.now())
    else:
        logdir='experiment_1_1.0/z_size='+str(config.hparams['z_size']) + \
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
    sensitivities, latents, rotations = get_encodings(model, test_data)
    return metrics, sensitivities, latents, rotations


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

# Log hparams 
logdir='experiment_1_1.0/z_size='+str(config_.hparams['z_size'])+'/hparams'
writer = SummaryWriter(log_dir=logdir)
dict_ = dict(config_._asdict())
new_dict = {}
for key, value in dict_.items():
    for val_key in value:
        new_dict[key+'-'+val_key] = str(dict_[key][val_key])
writer.add_hparams(new_dict, {})
writer.flush()
writer.close()

print(torch.cuda.get_device_name(torch.cuda.current_device()))
z_sizes = [1, 2]#, 3, 4, 5, 6, 7, 8]
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
results, sensitivites, latents, rotations = map(list, zip(*[run_experiment(c) for c in configs]))
results, latents, rotations = map(np.array, [results, latents, rotations])
# Plot input sensitivity graphs
input_sensitivites(sensitivites, z_sizes, config_)

# Plot model metrics as a fn of the experiment variable
plot_metrics(results.T, z_sizes, config_)

# Plot the model's latent space
plot_latents(latents, rotations, z_sizes, config_)


# Plot the model's decoded latent space
