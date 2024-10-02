import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import util.misc as misc
import util.lr_sched as lr_sched

from typing import Iterable

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

def train_one_epoch_bwce(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, 
                    loss_scaler, max_norm: float=0,
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    pred_softmax_list = []
    true_onehot_list = []

    for batch_idx, (inputs, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # use a per iteration (instead of per epoch) lr scheduler
        if batch_idx % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, batch_idx/len(data_loader)+epoch, args)
        
        inputs = inputs.to(device)
        labels = labels.to(device)

        true_label = F.one_hot(labels.to(torch.int64), num_classes = args.nb_classes)

        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            # print('outputs.shape:==>', outputs.shape)
            # print('labels.shape:===>', labels.shape)
            loss = criterion(outputs, labels, epoch+1, args.f_score_list)

            pred_softmax = nn.Softmax(dim=1)(outputs)
            # _, pred_decode = torch.max(pred_softmax, 1)
            # _, true_decode = torch.max(true_label, 1)

            pred_softmax_list.extend(pred_softmax.cpu().detach().numpy())
            true_onehot_list.extend(true_label.cpu().detach().numpy())

        loss_value = loss.item()

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False, update_grad=(batch_idx+1)%1==0)
        
        if (batch_idx+1) % accum_iter == 0:
            optimizer.zero_grad() 
        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        
        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None and (batch_idx+1)%accum_iter==0:
            """ Use epoch_1000x as the x-axis in tensorboard. This calibrates different curves when batch size changes. """
            epoch_1000x = int((batch_idx/len(data_loader)+epoch)*1000)
            log_writer.add_scalar('loss-lr/loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('loss-lr/lr', max_lr, epoch_1000x)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Average stats:", metric_logger)

    # print('true_onehot_list:==>', true_onehot_list)
    # print('pred_softmax_list:==>', pred_softmax_list)

    au_roc = roc_auc_score(true_onehot_list, pred_softmax_list)
    au_pr = average_precision_score(true_onehot_list, pred_softmax_list) 

    print('Metrics: auroc-{:.4f}, aupr-{:.4f}'.format(au_roc, au_pr))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, au_roc, au_pr

@torch.no_grad()
def evaluate_bwce(model: torch.nn.Module, data_loader: Iterable, 
            device: torch.device, epoch: int, args=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter=" ")
    header = 'Test:'
    print_freq = 10

    pred_softmax_list = []
    true_onehot_list = []

    # switch to evaluation mode
    model.eval()

    for batch_idx, (inputs, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        inputs = inputs.to(device)
        labels = labels.to(device)

        true_label = F.one_hot(labels.to(torch.int64), num_classes=args.nb_classes)

        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            pred_softmax = nn.Softmax(dim=1)(outputs)

            pred_softmax_list.extend(pred_softmax.cpu().detach().numpy())
            true_onehot_list.extend(true_label.cpu().detach().numpy())
        
        metric_logger.update(loss=loss.item())
    metric_logger.synchronize_between_processes()

    true_onehot_list = np.asarray(true_onehot_list)
    pred_softmax_list = np.asarray(pred_softmax_list)

    # calculate f1-score of class c on validation set after epoch e
    true_0 = true_onehot_list[:,0]
    pred_0 = (pred_softmax_list[:, 0]>=0.5).astype(int)

    true_1 = true_onehot_list[:,1]
    pred_1 = (pred_softmax_list[:, 1]>=0.5).astype(int)

    f1_class_0 = f1_score(true_0, pred_0)
    f1_class_1 = f1_score(true_1, pred_1)
    args.f_score_list = [f1_class_0, f1_class_1]

    if args.nb_classes == 2:
        au_roc = roc_auc_score(true_onehot_list[:,-1], pred_softmax_list[:,-1])
        au_pr = average_precision_score(true_onehot_list[:,-1], pred_softmax_list[:,-1])
    else:
        au_roc = roc_auc_score(true_onehot_list, pred_softmax_list, multi_class='ovr', average='macro')
        au_pr = average_precision_score(true_onehot_list, pred_softmax_list, average='macro')
    
    print('Metrics: auroc-{:.4f}, aupr-{:.4f}'.format(au_roc, au_pr))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, au_roc, au_pr


def misc_measures(confusion_matrix):
    
    acc = []
    sensitivity = []
    specificity = []
    precision = []
    G = []
    F1_score_2 = []
    mcc_ = []
    print('confusion_matrix.shape:==>', confusion_matrix.shape)
    # revised by lty
    if confusion_matrix.shape[0]==2:
        start_ = 1
    else:
        start_ = 0
    #  source code here is 1
    for i in range(start_, confusion_matrix.shape[0]):
        cm1=confusion_matrix[i]
        acc.append(1.*(cm1[0,0]+cm1[1,1])/np.sum(cm1))
        sensitivity_ = 1.*cm1[1,1]/(cm1[1,0]+cm1[1,1])
        sensitivity.append(sensitivity_)
        specificity_ = 1.*cm1[0,0]/(cm1[0,1]+cm1[0,0])
        specificity.append(specificity_)
        precision_ = 1.*cm1[1,1]/(cm1[1,1]+cm1[0,1])
        precision.append(precision_)
        G.append(np.sqrt(sensitivity_*specificity_))
        F1_score_2.append(2*precision_*sensitivity_/(precision_+sensitivity_))
        mcc = (cm1[0,0]*cm1[1,1]-cm1[0,1]*cm1[1,0])/np.sqrt((cm1[0,0]+cm1[0,1])*(cm1[0,0]+cm1[1,0])*(cm1[1,1]+cm1[1,0])*(cm1[1,1]+cm1[0,1]))
        mcc_.append(mcc)
        
    acc = np.array(acc).mean()
    sensitivity = np.array(sensitivity).mean()
    specificity = np.array(specificity).mean()
    precision = np.array(precision).mean()
    G = np.array(G).mean()
    F1_score_2 = np.array(F1_score_2).mean()
    mcc_ = np.array(mcc_).mean()
    
    return acc, sensitivity, specificity, precision, G, F1_score_2, mcc_