import os
import argparse
import logging
import numpy as np
from datetime import datetime
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable

import timm
assert timm.__version__ == "0.3.2"

import models_vit

from util.datasets_sle import sle_dataset

import warnings
warnings.filterwarnings("ignore", category=Warning)

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
            # raise EnvironmentError("GPU device not found")
            print("CUDA not available. Using CPU")     
    else:
        device = torch.device("cpu") 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
      
def get_args_parser():
    parser = argparse.ArgumentParser("SLE and related disease external validation", add_help=False)
    parser.add_argument('--batch_size', default=32, type=int, 
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus)') 

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='image input size')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    # Dataset params
    parser.add_argument('--data_path', type=str, default='./data_input_example/external_test.csv', help='the csv path of the data')
    parser.add_argument('--dataset_name', type=str,  help='dataset name for  validation')
    parser.add_argument('--task_type', type=str, default='SLE', choices=['SLE', 'LN', 'LR'])
    parser.add_argument('--nb_classes', default=2, type=int, 
                        help='number of the classification types')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Other Settings 
    parser.add_argument('--cuda', type=bool, default=True, help='whether to use cuda')
    parser.add_argument('--gpu', type=str, default='0', 
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--seed', type=int, default=0)
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
    assert args.eval==True
    time_key = datetime.now().strftime('_%y-%m-%d_%H-%M')
    args.store_name='_'.join([args.dataset_name, args.task_type, 'eval', str(args.eval), 'seed', str(args.seed), 'time', time_key])
    test_csv = args.data_path.split('/')[-1].split('.')[0]
    args.log_txt = args.store_name+'_'+test_csv+'.log'
    args.npz_save_path = args.store_name+'_'+test_csv+'_pred_results.npz'

    return args

@torch.no_grad()
def evaluate_test(model: torch.nn.Module, data_loader: Iterable,
                device: torch.device, epoch: int, args=None):

    pred_softmax_list = []
    true_onehot_list = []

    model.eval()

    for batch_idx, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        true_label = F.one_hot(labels.to(torch.int64), num_classes=args.nb_classes)

        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            pred_softmax = nn.Softmax(dim=1)(outputs)

            pred_softmax_list.extend(pred_softmax.cpu().detach().numpy())
            true_onehot_list.extend(true_label.cpu().detach().numpy())

    true_onehot_list = np.asarray(true_onehot_list)
    pred_softmax_list = np.asarray(pred_softmax_list)

    return  pred_softmax_list, true_onehot_list

def test(args):
    logging.basicConfig(filename=args.log_txt, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("{}".format(args).replace(', ', ',\n'))

    '''load dataset'''
    dataset_test = sle_dataset(mode='test', args=args)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_mem, shuffle=False
    )
    logging.info("========== Test Dataset Load Done ===========")

    '''load model'''
    model = models_vit.__dict__['vit_large_patch16'](
        num_classes = args.nb_classes,
        drop_path_rate = args.drop_path,
        global_pool = args.global_pool
    )
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    assert args.eval

    checkpoint=torch.load(args.resume, map_location='cpu')
    print('checkpoint.keys():==>', checkpoint.keys())

    model.load_state_dict(checkpoint['model'])
    print('resume_epoch:==>', checkpoint['epoch'])
    logging.info("resume_epoch: {}".format(checkpoint['epoch']))

    model.to(device)
    logging.info("========= Model Resume Load Done ============")

    pred_softmax_list, true_onehot_list = evaluate_test(model, data_loader_test, device, 0, args)

    pred_softmax_list = np.asarray(pred_softmax_list)
    true_onehot_list = np.asarray(true_onehot_list)


    np.savez(args.npz_save_path, 
            true_onehot=true_onehot_list, 
            pred_softmax=pred_softmax_list)

    return true_onehot_list, pred_softmax_list    

if __name__ == '__main__':
    args = get_args_parser()

    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    _seed_torch(args)
    true_onehot_list, pred_softmax_list = test(args)

    logging.info("Done!")