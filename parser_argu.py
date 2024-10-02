import os
import warnings
import argparse
from datetime import datetime
import numpy as np
import torch
import logging

def _seed_torch(args):
	r"""
	Sets custom seed for torch

	Args:
		- seed : Int

	Returns:
		- None

	"""
	import random
	seed = args.seed
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	if args.cuda:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		if device.type == 'cuda':
			torch.cuda.manual_seed(seed)
			torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
		else:
			raise EnvironmentError("GPU device not found")
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def get_args_parser():
	parser = argparse.ArgumentParser("SLE and related disease  fine-tuning", add_help=False)
	parser.add_argument('--batch_size', default=32, type=int, 
						help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus)')
	parser.add_argument('--accum_iter', default=1, type=int,
						help='Accumulate gradient iterations (for increasing the effective  batch size under memory constraints)')
	parser.add_argument('--epochs', default=50, type=int)

	# Model parameters
	parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
						help='Name of model to train')
	parser.add_argument('--input_size', default=224, type=int,
						help='image input size')
	parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
						help='Drop path rate (default: 0.1)')

	# Optimizer parameters
	parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
						help='Clip gradient norm (default: None, no clipping)')
	parser.add_argument('--weight_decay', type=float, default=0.05,
						help='weight decay (default: 0.05)')

	parser.add_argument('--lr', type=float, default=None, metavar='LR',
						help='learning rate (absolute lr)')
	parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
						help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
	parser.add_argument('--layer_decay', type=float, default=0.75,
						help='layer-wise lr decay from ELECTRA/BEiT')

	parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
						help='lower lr bound for cyclic schedulers that hit 0')

	parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
						help='epochs to warmup LR')

	# Augmentation parameters
	parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
						help='Color jitter factor (enabled only when not using Auto/RandAug)')
	parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
						help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
	parser.add_argument('--smoothing', type=float, default=0.1,
						help='Label smoothing (default: 0.1)')

	# * Random Erase params
	parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
					help='Random erase prob (default: 0.25)')
	parser.add_argument('--remode', type=str, default='pixel',
					help='Random erase mode (default: "pixel")')
	parser.add_argument('--recount', type=int, default=1,
					help='Random erase count (default: 1)')
	parser.add_argument('--resplit', action='store_true', default=False,
					help='Do not random erase first (clean) augmentation split')


	# * Mixup params
	parser.add_argument('--mixup', type=float, default=0,
						help='mixup alpha, mixup enabled if > 0.')
	parser.add_argument('--cutmix', type=float, default=0,
						help='cutmix alpha, cutmix enabled if > 0.')
	parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
						help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
	parser.add_argument('--mixup_prob', type=float, default=1.0,
						help='Probability of performing mixup or cutmix when either/both is enabled')
	parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
						help='Probability of switching to cutmix when both mixup and cutmix enabled')
	parser.add_argument('--mixup_mode', type=str, default='batch',
						help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

	# Dataset params
	parser.add_argument('--data_path', type=str, default='./label_csv/PUMCH_SLE_train.csv',help='the csv path of the data')
	parser.add_argument('--dataset_name', type=str,
						help='dataset name, PUMCH for internal, and others for external')
	parser.add_argument('--task_type', type=str, default='SLE', choices=['SLE', 'LN', 'LR'])
	parser.add_argument('--nb_classes', default=2, type=int, 
						help='number of the classification types')
						
	parser.add_argument('--global_pool', action='store_true')
	parser.set_defaults(global_pool=True)
	parser.add_argument('--cls_token', action='store_false', dest='global_pool',
						help='Use class token instead of global pool for classification')

	parser.add_argument('--cuda', type=bool, default=True, help='whether to use cuda')
	parser.add_argument('--gpu', type=str, default='3', 
						help='gpu device ids for CUDA_VISIBLE_DEVICES')
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--start_epoch', type=int, default=0, metavar='N',
						help='start epoch')
	parser.add_argument('--eval', action='store_true', 
						help='Perform evaluation only')
	parser.add_argument('--resume', default='', help='resume from checkpoint')
	parser.add_argument('--num_workers', default=4, type=int)
	parser.add_argument('--pin_mem', action='store_true', 
						help='Pin CPU memory  in DataLoader for more efficient(sometimes) transfer to GPU.')
	args = parser.parse_args()
	args = add_args(args)
	return args

def add_args(args):
	time_key = datetime.now().strftime('_%y-%m-%d_%H-%M-%S')
	args.store_name='_'.join([args.dataset_name, args.task_type, 'eval', str(args.eval), 'seed', str(args.seed), 'time', time_key])

	output_dir = './output_dir/'+args.dataset_name
	os.makedirs(output_dir, exist_ok=True)

	tensorboard_logdir = os.path.join(output_dir, 'tensorboad_log')
	os.makedirs(tensorboard_logdir, exist_ok=True)
	
	if args.dataset_name == 'PUMCH': # only internal set need train
		train_log_dir = os.path.join(output_dir, 'log_train')
		os.makedirs(train_log_dir, exist_ok=True)

		save_model_dir = os.path.join(output_dir, 'save_model')
		os.makedirs(save_model_dir, exist_ok=True)

	test_log_dir = os.path.join(output_dir, 'log_test')
	os.makedirs(test_log_dir, exist_ok=True)
	
	test_npz_dir = os.path.join(output_dir, 'test_npz')
	os.makedirs(test_npz_dir, exist_ok=True)

	if args.eval==True:
		test_csv = args.data_path.split('/')[-1].split('.')[0]
		args.log_txt = os.path.join(test_log_dir, args.store_name+'_'+test_csv+'.log')
		args.npz_save_path = os.path.join(test_npz_dir, args.store_name+'_'+test_csv+'_pred_results.npz')
	else:
		args.log_txt = os.path.join(train_log_dir, args.store_name+'.log')
		args.tsbd_log = os.path.join(tensorboard_logdir, args.store_name)
		args.model_dir = os.path.join(save_model_dir, args.store_name)

	return args
