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
DEVICE = torch.device("cuda")

def get_encodings(model, test_data):
    """
    Returns latent encodings from the test set along with
    the input sensitivity for them
    """
    # with torch.no_grad():
    grads = np.zeros((4096, model.z_size))
    z_s = []
    metadata = []
    for batch_id, data in enumerate(test_data):
        data, t = data
        data = data.to(DEVICE)
        mu, logvar = model.encode(data)
        z = model.sample(mu, logvar)
        z_s.extend(z.detach().cpu().numpy())
        metadata.extend(t.cpu().numpy())
        # for x in torch.split(z, 1):
            # x = x.clone().detach()
        #     x = x.squeeze()
        #     x = x.repeat(4096, 1)
        #     x.requires_grad_(True)
        #     out = model.decode(x)
        #     out.backward(torch.eye(4096).to(DEVICE))
        #     grad = x.grad.data
        #     grads = np.add(grads, np.abs(grad.cpu().numpy()))

    # return np.sum(grads, axis=0)
    return np.linalg.norm(grads, axis=0), np.array(z_s), np.array(metadata)

def higgins_metric(latents, config, labels, classifier):
    z_size = config.hparams['classifier_size']
    # all dimensions
    all_score, *_ = classifier(latents, labels)
    if config.hparams['include_loss']:
        # constrained dimensions
        constrained, *_ = classifier(latents[:,:z_size], labels)
        # unconstrained dimensions
        unconstrained, *_ = classifier(latents[:,z_size:], labels)
    else:
        # constrained dimensions
        constrained, *_ = classifier(latents[:,:z_size], labels)
        # unconstrained dimensions
        unconstrained, *_ = classifier(latents[:,0:], labels)

    return [all_score, constrained, unconstrained]

def run_experiment(config):

    # Train and evaluate the model
    train_data, test_data, model, opt = DSprites.setup(config)
    for epoch in range(1, config.model['epochs'] + 1):
      DSprites.train(model, train_data, epoch, opt, verbose=True, writer=None)

    test_loss, metrics_mean = DSprites.test(model, test_data, verbose=False)

    scaled_mse = np.sqrt(metrics_mean[-1]) * 2 *math.pi
    metrics = [test_loss] + list(metrics_mean) + [scaled_mse]
    [print(y, x) for x, y in zip(metrics, ["loss", "r_loss", "kl_loss", "mse", "scaled_ms"])]
    sensitivities, latents, metadata = get_encodings(model, test_data)
    
    # run experiments on latent space

    # classify shape
    # using KNN
    shape_knn = higgins_metric(latents, config, 
        metadata[:, 1], classifiers.n_neighbours_classifier)

    # using Softmax regression
    shape_softmax = higgins_metric(latents, config, 
        metadata[:, 1], classifiers.softmax_regression)

    regression_results = [shape_knn, shape_softmax]
    # classify continous valued latents
    latent_factors = ['scale', 'orientation', 'posx', 'posy']
    all_classifiers = [classifiers.n_neighbours,
                       classifiers.linear_regression,
                       classifiers.sigmoid_regression]
    """
    latent results will be
    [[[all_scale_knn, cons, uncons], scale_linreg, scale_sigmoidreg],
     [orientation_knn, ...
      ...]]
    for every latent factor along rows and every classifier along columns
    """
    classification_results = []
    for i, latent_factor in enumerate(latent_factors, 2):
        results = []
        for classifier in all_classifiers:
            results.append(higgins_metric(latents, config, metadata[:,i], classifier))
        classification_results.append(results)

    return sensitivities, latents, np.array(regression_results), np.array(classification_results)


config_ = DSprites.Config(
    # dataset
    dataset={
        # latent_counts - ['color', 'shape', 'scale', 'orientation', 'posX', 'posY']
        # out of          [ 1         3         6       40              32     32  ]
        'latents_counts': [-1, -1, -1, -1, -1, -1]
    },
    #model
    model={
        'model': RotDSpritesVAE,
        'epochs':25,
        'lr':0.001,
        'batch_size':512,
    },
    hparams={
        'z_size': 8,
        'classifier_size': 1,
        'include_loss': True
    }
)


print([x.shape for x in run_experiment(config_)])