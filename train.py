import os, json, argparse
import torch
import models as module_arch
import evaluation.losses as module_loss
import evaluation.metrics as module_metric
import data_loader.dataloader as module_data

from utils.logger import Logger
from trainer.trainer import Trainer

def get_instance(module, name, config, *args):
	return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def main(config, resume):
	train_logger = Logger()

	model = get_instance(module_arch, 'arch', config)
	img_sz = config["train_loader"]["args"]["resize"]
	model.summary(input_shape=(3, img_sz, img_sz))

	train_loader = get_instance(module_data, 'train_loader', config).loader
	valid_loader = get_instance(module_data, 'valid_loader', config).loader

	loss = getattr(module_loss, config['loss'])
	metrics = [getattr(module_metric, met) for met in config['metrics']]

	trainable_params = filter(lambda p: p.requires_grad, model.parameters())
	optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
	lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer)

	trainer = Trainer(model, loss, metrics, optimizer, 
					  resume=resume,
					  config=config,
					  data_loader=train_loader,
					  valid_data_loader=valid_loader,
					  lr_scheduler=lr_scheduler,
					  train_logger=train_logger)
	trainer.train()

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Train model')
	parser.add_argument('-c', '--config', default=None, type=str, help='config file path')
	parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint')
	parser.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable')
	args = parser.parse_args()

	if args.config:
		config = json.load(open(args.config))
	elif args.resume:
		config = torch.load(args.resume)['config']
	else:
		raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

	if args.device:
		os.environ["CUDA_VISIBLE_DEVICES"] = args.device

	main(config, args.resume)