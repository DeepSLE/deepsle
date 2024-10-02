import os

import argparse
import datetime
import time
import numpy as np
import logging

from parser_argu import _seed_torch, get_args_parser
import torch
from torch.utils.tensorboard import SummaryWriter
import timm
assert timm.__version__ == "0.3.2"
from timm.models.layers import trunc_normal_
from timm.loss import LabelSmoothingCrossEntropy

import models_vit

import util.misc as misc
import util.lr_decay as lrd
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.datasets_sle import sle_dataset
from util.loss import LabelSmooth_CE_weight

from engine import train_one_epoch_bwce, evaluate_bwce

import warnings
warnings.filterwarnings("ignore", category=Warning)

def add_args_bwce(args):
	if args.task_type == 'SLE':
		args.cls_num_list = [3102, 1665]
	elif args.task_type == 'LN':
		args.cls_num_list = [1473, 192]
	elif args.task_type == 'LR':
		args.cls_num_list = [1342, 323]
	
	args.E1 = int(args.epochs * 0.2)
	args.E2 = int(args.epochs * 0.5)
	args.f_score_list = [1.0, 1.0]

	return args


def main(args):
	logging.basicConfig(filename=args.log_txt, level=logging.INFO, format='%(asctime)s - %(levelname)s -%(message)s')

	# print("{}".format(args).replace(', ', ',\n'))
	logging.info("{}".format(args).replace(', ', ',\n'))

	'''load dataset'''
	dataset_train = sle_dataset(mode='train', args=args)
	dataset_valid = sle_dataset(mode='valid', args=args)
	
	data_loader_train = torch.utils.data.DataLoader(
		dataset_train, batch_size=args.batch_size,
		num_workers=args.num_workers, pin_memory=args.pin_mem, shuffle=True
	)

	data_loader_valid = torch.utils.data.DataLoader(
		dataset_valid, batch_size=args.batch_size,
		num_workers=args.num_workers, pin_memory=args.pin_mem, shuffle=True
	)

	logging.info('============= Dataset Load Done =============')

	'''load models'''
	model = models_vit.__dict__['vit_large_patch16'](
		num_classes = args.nb_classes,
		drop_path_rate = args.drop_path,
		global_pool = args.global_pool
	)
	
	device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
	# if args.finetune and not 
	if not args.eval: # need finetune
		checkpoint = torch.load('checkpoint-800-wzy.pth', map_location='cpu')
		checkpoint_model = checkpoint['model']
		state_dict = model.state_dict()

		for k in ['head.weight', 'head.bias']:
			if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
				print(f"Removing key {k} from pretrained checkpoint")
				del checkpoint_model[k]

		# interpolate position embedding
		interpolate_pos_embed(model, checkpoint_model)

		# load pre-trained model
		msg = model.load_state_dict(checkpoint_model, strict=False)

		print('msg.missing_keys:==>', msg.missing_keys)
		if msg.missing_keys != []:
			if args.global_pool:
				assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
			else:
				assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

		# manually initialize fc layer
		trunc_normal_(model.head.weight, std=2e-5)

	model.to(device)
	n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

	logging.info("Model size:{:.5f} M".format(n_parameters/1.e6))
	logging.info('============= Model Init Done =============')

	'''load optimizer'''
	eff_batch_size = args.batch_size * args.accum_iter
	if args.lr is None: # only base_lr is specified
		args.lr = args.blr * eff_batch_size / 256
	logging.info("base lr: {:.2e}".format(args.lr*256/eff_batch_size))
	logging.info("actual lr: {:.2e}".format(args.lr))

	logging.info("accumulate grad iterations: {}".format(args.accum_iter))
	logging.info("effective batch size: {}".format(eff_batch_size))

	param_groups = lrd.param_groups_lrd(model, args.weight_decay, no_weight_decay_list = model.no_weight_decay(), layer_decay=args.layer_decay)

	optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
	loss_scaler = NativeScaler()

	'''load loss'''

	criterion = LabelSmooth_CE_weight(cls_num_list=args.cls_num_list, labelsmoothing_alpha=args.smoothing, E1=args.E1, E2=args.E2, E=args.epochs)
	
	logging.info("Criterion = {}".format(str(criterion)))
	
	'''train'''
	logging.info("Start training for {} epochs!".format(args.epochs))

	best_auc = 0.1
	# best_aupr = 0.1
	# best_F1 = 0.1

	start_time = time.time()
	log_writer = SummaryWriter(log_dir=args.tsbd_log)
	os.makedirs(args.model_dir, exist_ok=True)

	for epoch in range(args.start_epoch, args.epochs):
		train_stats, train_auroc, train_aupr = train_one_epoch_bwce(
			model, criterion, data_loader_train, optimizer, device, epoch, loss_scaler, args.clip_grad, log_writer=log_writer, args=args
		)
		logging.info("Epoch: {}, args.f1_score_list: {:.4f}, {:.4f}".format(epoch, args.f_score_list[0], args.f_score_list[1]))
		logging.info("Epoch: {}, train_lr: {:.6f}, train_loss: {:.4f}, train_auroc: {:.4f}, train_aupr: {:.4f} ".format(epoch, train_stats['lr'], train_stats['loss'],  train_auroc, train_aupr))

		valid_stats, valid_auroc, valid_aupr = evaluate_bwce(
			model, data_loader_valid, device, epoch, args=args
		)

		logging.info("Epoch: {}, valid_loss: {:.4f}, valid_auroc: {:.4f}, valid_aupr: {:.4f}".format(epoch, valid_stats['loss'], valid_auroc, valid_aupr))

		if best_auc < valid_auroc:
			best_auc = valid_auroc
			# save model
			misc.save_model(
                    args=args, model=model, model_without_ddp=model, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)

		if log_writer is not None:
			log_writer.add_scalar('perf_train/train_auroc', train_auroc, epoch)
			log_writer.add_scalar('perf_train/train_aupr', train_aupr, epoch)
			log_writer.add_scalar('perf_train/train_loss', train_stats['loss'], epoch)
			log_writer.add_scalar('perf_valid/valid_auroc', valid_auroc, epoch)
			log_writer.add_scalar('perf_valid/valid_aupr', valid_aupr, epoch)
			log_writer.add_scalar('perf_valid/valid_loss', valid_stats['loss'], epoch)


	total_time = time.time() - start_time
	total_time_str = str(datetime.timedelta(seconds=int(total_time)))
	logging.info("Training time {}".format(total_time_str))

if __name__=='__main__':
	args = get_args_parser()
	args = add_args_bwce(args)
	os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
	_seed_torch(args)
	main(args)
	logging.info("Done!")
