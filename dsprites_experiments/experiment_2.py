import sys
sys.path.append('../models')
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import DSprites
from DSprites import DSpritesVAE, RotDSpritesVAE
from torch.utils.tensorboard import SummaryWriter
import datetime
import classifiers
import dsprites_data as dsprites

DEVICE = torch.device("cuda")

def get_encodings(model, test_data):
    """
    Returns latent encodings from the test set along with
    the input sensitivity for them
    """
    # with torch.no_grad():
    # grads = torch.zeros((4096, model.z_size)).to(DEVICE)
    z_s = []
    metadata = []
    with torch.no_grad():
        for batch_id, data in enumerate(test_data):
            data, t = data
            data = data.to(DEVICE)
            mu, logvar = model.encode(data)
            z = model.sample(mu, logvar)
            z_s.extend(z.detach().cpu().numpy())
            metadata.extend(t.cpu().numpy())
        # for x in torch.split(z, 1, dim=1):
        #     # x = x.clone().detach()
        #     # x = x.squeeze()
        #     print(x.shape)
        #     x = x.detach().squeeze().repeat(4096, 1)
        #     print(x.shape)
        #     x.requires_grad_(True)
        #     out = model.decode(x)
        #     out.backward(torch.eye(4096).to(DEVICE))
        #     grad = x.grad.data
        #     print(grad.shape)
        #     grads = torch.add(grads, torch.abs(grad))

    # grads = grads.cpu().numpy()
    return [], np.array(z_s), np.array(metadata)

def higgins_metric(latents, config, labels, classifier):
    z_size = config.hparams['classifier_size']
    # all dimensions
    all_score, *_ = classifier(latents, labels)
    # if config.hparams['include_loss']:
    #     # constrained dimensions
    #     constrained, *_ = classifier(latents[:,:z_size], labels)
    #     # unconstrained dimensions
    #     unconstrained, *_ = classifier(latents[:,z_size:], labels)
    # else:
    #     # constrained dimensions
    #     constrained, *_ = classifier(latents[:,:config.hparams['z_size']], labels)
    #     # unconstrained dimensions
    #     unconstrained, *_ = classifier(latents[:,0:], labels)

    return np.array(all_score, np.float32)

def run_experiment(config, dataset=None):

    # Train and evaluate the model
    train_data, test_data, model, opt = DSprites.setup(config, dataset=dataset)
    for epoch in range(1, config.model['epochs'] + 1):
      DSprites.train(model, train_data, epoch, opt, verbose=False, writer=None)

    test_loss, metrics_mean = DSprites.test(model, test_data, verbose=False)

    scaled_mse = np.sqrt(metrics_mean[-1]) * 2 *math.pi
    metrics = [test_loss] + list(metrics_mean) + [scaled_mse]
    [print(y, x) for x, y in zip(metrics, ["loss", "r_loss", "kl_loss", "mse", "scaled_ms"])]
    sensitivities, latents, metadata = get_encodings(model, test_data)
    # run experiments on latent space

    # classify shape
    # using KNN
    print("Shape KNN")
    shape_knn = higgins_metric(latents, config, 
        metadata[:, 1], classifiers.n_neighbours_classifier)
    # using Softmax regression
    print("Shape Softmax")
    shape_softmax = higgins_metric(latents, config, 
        metadata[:, 1], classifiers.softmax_regression)

    classification_results = np.array([shape_knn, shape_softmax])
    """
    regression results will be shape (2)
    """
    print("Classification results shape", classification_results.shape)

    # classify continous valued latents
    latent_factors = ['scale', 'orientation', 'posx', 'posy']
    all_regressors = [classifiers.n_neighbours,
                       classifiers.linear_regression,
                       classifiers.sigmoid_regression]
    """
    latent regressions will be
    [[[all_scale_knn], scale_linreg, scale_sigmoidreg],
     [orientation_knn, ...
      ...]]
    shape (4, 3) - (latent_factor, regressor, latent_portion)
    for every latent factor along rows and every classifier along columns
    """
    regression_results = []
    for i, latent_factor in enumerate(latent_factors, 2):
        results = []
        for regressor in all_regressors:
            print(latent_factor, regressor)
            results.append(higgins_metric(latents, config, metadata[:,i], regressor))
        regression_results.append(results)
    train_data, test_data = None, None
    regression_results = np.array(regression_results)
    print("Regression results shape", regression_results.shape)
    return [], [], np.array(regression_results), np.array(classification_results), metrics

def repeat_experiment(config_, num_repetitions, dataset=None, writer=None):
    print(config_)
    classification_results = np.zeros((num_repetitions, 2, 3), dtype=np.float32)
    regression_results = np.zeros((num_repetitions, 4, 3, 3), dtype=np.float32)
    metrics_results = []
    for exp_id in range(num_repetitions):
        print("Running experiment %d"%(exp_id))
        _, _, regressions, classifications, metrics = run_experiment(config_, dataset)
        regression_results[exp_id] = regressions
        classification_results[exp_id] = classifications
        # metrics = np.random.rand(5)
        metrics_results.append(metrics)
        # classification_results[exp_id] = np.random.rand(2,3)
        # regression_results[exp_id] = np.random.rand(4,3,3)

    print("****** RESULTS FROM EXPERIMENT *********")
    print(config_)
    print("regression results", np.mean(regression_results, axis=0))
    print("regression std", np.std(regression_results, axis=0))
    print("classification results", np.mean(classification_results, axis=0))
    print("classification std", np.std(classification_results, axis=0))
    print("metrics", np.mean(metrics_results, axis=0))
    print("metrics std", np.std(metrics_results, axis=0))
    # print(regression_results.shape)
    # print(np.mean(regression_results, axis=0).shape)
    # print("mean res", np.mean(regression_results, axis=0))
    # print("std res", np.std(regression_results, axis=0))
    # print("mean clasification", classification_results/num_repetitions)
    return (np.mean(regression_results, axis=0), 
            np.std(regression_results, axis=0),
            np.mean(classification_results, axis=0),
            np.std(classification_results, axis=0),
            np.mean(metrics_results, axis=0),
            np.std(metrics_results, axis=0))

def plot_results(classifier_sizes, results, num_repetitions):
    classification_labels = ["KNN", "Softmax"]
    regression_labels = ["KNNReg", "LinReg", "Sigmoid"]
    
    classification_latents = ["Shape"]
    regression_latents = ['Scale', 'Orientation', 'PosX', 'PosY']
    
    z_labels = ["all", "cons", "uncons"]
    results = tuple([np.array(x) for x in results])

    mu_regression, std_regression, mu_classification, std_classification, mu_metrics, std_metrics = results

    # Plot regressions
    mu_regression = np.swapaxes(mu_regression, 3, 2)
    std_regression = np.swapaxes(std_regression, 3, 2)

    fig, axes = plt.subplots(len(regression_labels), len(regression_latents),
                            sharex='col')
    for i, regressor in enumerate(regression_labels):
        for j, latent in enumerate(regression_latents):
            for x, label in enumerate(z_labels):
                axes[i, j].errorbar(classifier_sizes, mu_regression[:, j, x, i],
                            yerr=std_regression[:, j, x, i], label=label,
                            marker="x", capsize=5)
            axes[i, j].set_title(regressor + " regression on " + latent)
    plt.legend(loc='center left', bbox_to_anchor=(1, 1.7))
    fig.text(0.5, 0.04, 'Classified Dimensions', ha='center')
    fig.text(0.09, 0.5, 'Mean 4-Fold MSE', va='center', rotation='vertical')
    fig.tight_layout()
    # plt.show()

    # Plot classifications
    mu_classification = np.swapaxes(mu_classification, 2, 1)
    std_classification = np.swapaxes(std_classification, 2, 1)
    fig, axes = plt.subplots(len(classification_labels), len(classification_latents),
                            sharex='col', sharey='row')
    for i, classifier in enumerate(classification_labels):
        for x, label in enumerate(z_labels):
            axes[i].errorbar(classifier_sizes, mu_classification[:, x, i],
                        yerr=std_classification[:, x, i], label=label,
                        marker="x", capsize=5)
            axes[i].set_title(classifier + " classification on Shape")
    plt.legend(loc='center left', bbox_to_anchor=(1, 1))
    fig.text(0.5, 0.04, 'Classified Dimensions', ha='center')
    fig.text(0.04, 0.5, 'Mean 4-Fold Classification Accuracy', va='center', rotation='vertical')
    fig.tight_layout()
    
    # Plot metrics
    mu_metrics, std_metrics = np.array(mu_metrics).T, np.array(std_metrics).T
    metrics_labels = ["loss", "r_loss", "kl_loss", "mse", "scaled_ms"]
    fig, axes = plt.subplots(len(metrics_labels), 1, sharex='col')
    for i, (label, metric, std) in enumerate(zip(metrics_labels, mu_metrics, std_metrics)):
        axes[i].errorbar(classifier_sizes, metric, yerr=std, marker="x",
                         capsize=5)
        axes[i].set_title("Mean " + label + " across " + str(num_repetitions) + " models")
        axes[i].set(xlabel="", ylabel=label)
        axes[i].set_xticks(classifier_sizes)
    fig.text(0.5, 0.04, 'Classified Dimensions', ha='center')
    plt.show()


config_ = DSprites.Config(
    # dataset
    dataset={
        'test_index': 3
    },
    #model
    model={
        'model': RotDSpritesVAE,
        'epochs':20,
        'lr':0.001,
        'batch_size':512,
    },
    hparams={
        'z_size': 10,
        'classifier_size': 10,
        'include_loss': False
    }
)

dataset = dsprites.DSpritesRaw(**config_.dataset)
# dataset = None

z_sizes = [2, 4, 5, 6, 8, 16]
# classifier_sizes = [6, 8, 9]
configs = []
# configs.append(config_)
num_repetitions = 5

# results_all = [mu_regr, std_regr, mu_class, std_class, mu_metrics, std_metrics]
# results all = [(6,) x 6]
results_all = ([], [], [], [], [], [])
for z_size in z_size:
    hparams = config_.hparams.copy()
    hparams["z_size"] = z_size
    configs.append(config_._replace(hparams=hparams))

for config in configs:
    results = repeat_experiment(config, num_repetitions, dataset=dataset)
    for list_, elem in zip(results_all, results):
        list_.append(elem)

classifier_sizes = [0] + classifier_sizes
plot_results(classifier_sizes, results_all, num_repetitions)
