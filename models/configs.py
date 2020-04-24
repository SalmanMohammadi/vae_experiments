import DSprites
from DSprites import DSpritesVAE, FactorVAE, EarlyFactorVAE

vanilla_vae = DSprites.Config(
    # dataset
    dataset={
        'iid': True,
        'train_size': 500000,
        'test_size': 10000
    },
    #model
    model={
        'model': DSpritesVAE,
        'epochs':55,
        'lr':0.001,
        'batch_size':1024,
        'metrics_labels':['r_loss', 'kl', 'kl_prime', 'kl_latent', 'classification_ce']
    },
    hparams={
        'z_size': 10,
    }
)

isvae = DSprites.Config(
    # dataset
    dataset={
        'iid': True,
        'train_size': 500000,
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
        'include_loss': False,
        'gamma':1.
    }
)

lisvae = DSprites.Config(
    # dataset
    dataset={
        'iid': True,
        'train_size': 500000,
        'test_size': 10000
    },
    #model
    model={
        'model': EarlyFactorVAE,
        'epochs':55,
        'lr':0.001,
        'batch_size':1024,
        'metrics_labels':['r_loss', 'kl', 'kl_prime', 'kl_latent', 'classification_ce']
    },
    hparams={
        'z_size': 10,
        'classifier_size': 1,
        'include_loss': False,
        'gamma':1.
    }
)