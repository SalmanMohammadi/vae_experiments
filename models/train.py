import argparse
import configs
import torch
import DSprites
from torch.utils.tensorboard import SummaryWriter 

config_mappings = {
    'vae': configs.vanilla_vae,
    'isvae': configs.isvae,
    'lisvae': configs.lisvae,
}

def hparams_to_dict(kv):
    if kv:
        res = lambda kv: dict(list(map(lambda x: (x.split("=")[0], eval(x.split("=")[1])), kv.split(","))))
        return res(kv)
    else:
        return {}

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--experiment_id", type=int, default=0)
parser.add_argument("--experiment_name", type=str, default='')
parser.add_argument("--hparams", type=hparams_to_dict, default='')
args = parser.parse_args()

experiment_id = '/' + str(args.experiment_id)
experiment_name = '/' + args.experiment_name if args.experiment_name else ''
model_path = 'tmp/' + args.model + experiment_name + experiment_id


config = config_mappings[args.model]
config = config._replace(hparams={**config.hparams, **args.hparams})

print(config)
train_data, test_data, model, opt = DSprites.setup(config, iid=config.dataset['iid'])
writer = SummaryWriter(log_dir=model_path)
for epoch in range(config.model['epochs']):
    DSprites.train(model, train_data, epoch, opt, writer=writer, verbose=True, 
            metrics_labels=config.model['metrics_labels'])

_, metrics = DSprites.test(model, test_data, verbose=True, metrics_labels=config.model['metrics_labels'], writer=writer)
torch.save(model.state_dict(), model_path + ".pt")
metrics_labels = ['hparam/'+x for x in config.model['metrics_labels']]
writer.add_hparams(hparam_dict=config.hparams, metric_dict=dict(zip(metrics_labels, metrics)))