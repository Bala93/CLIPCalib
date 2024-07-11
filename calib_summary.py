import torch
from matplotlib import pyplot as plt
import numpy as np
from torch.nn.functional import softmax
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from metrics import expected_calibration_error, maximum_calibration_error, AdaptiveECELoss
from utils import reliability_curve
import pandas as pd
#import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import os 
import sys



def post_process(dataset_type, adapter_type, nshots):


    target_path = '/export/livia/home/vision/Bmurugesan/CLIPCalib/output/FINAL/{}/ZS/vitb16/target.pt'.format(dataset_type) 

    #zs_path = '/export/livia/home/vision/Bmurugesan/CLIPCalib/output/FINAL/{}/ZS/vitb16/logits.pt'.format(dataset_type)
    #adapt_path = '/export/livia/home/vision/Bmurugesan/CLIPCalib/output/FINAL/{}/ADAPTER/vitb16/SGD_lr1e-1_B256_ep300_{}Init_noneConstraint_{}shots/seed1/logits.pt'.format(dataset_type, adapter_type, nshots)
    #adaptpre_path ='/export/livia/home/vision/Bmurugesan/CLIPCalib/output/FINAL/{}/ADAPTER_ZSNORM/vitb16/SGD_lr1e-1_B256_ep300_{}Init_noneConstraint_{}shots/seed1/logits.pt'.format(dataset_type, adapter_type, nshots)
    #adaptpen_path = '/export/livia/home/vision/Bmurugesan/CLIPCalib/output/FINAL/{}/ADAPTER_ZSPEN/vitb16/SGD_lr1e-1_B256_ep300_{}Init_noneConstraint_{}shots/seed1/logits.pt'.format(dataset_type, adapter_type, nshots)

    zs_path = '/export/livia/home/vision/Bmurugesan/CLIPCalib/output/FINAL/{}/ZS/rn50/logits.pt'.format(dataset_type)
    #adapt_path = '/export/livia/home/vision/Bmurugesan/CLIPCalib/output/FINAL/{}/ADAPTER/rn50/SGD_lr1e-1_B256_ep300_{}Init_noneConstraint_{}shots/seed1/logits.pt'.format(dataset_type, adapter_type, nshots)
    adapt_path = '/export/livia/home/vision/Bmurugesan/CLIPCalib/output/FINAL/{}/ADAPTER/rn50/SGD_lr1e-1_B256_ep300_{}Init_noneConstraint_{}shots_ls/seed1/logits.pt'.format(dataset_type, adapter_type, nshots)
    adaptpre_path ='/export/livia/home/vision/Bmurugesan/CLIPCalib/output/FINAL/{}/ADAPTER_ZSNORM/rn50/SGD_lr1e-1_B256_ep300_{}Init_noneConstraint_{}shots/seed1/logits.pt'.format(dataset_type, adapter_type, nshots)
    adaptpen_path = '/export/livia/home/vision/Bmurugesan/CLIPCalib/output/FINAL/{}/ADAPTER_ZSPEN/rn50/SGD_lr1e-1_B256_ep300_{}Init_noneConstraint_{}shots/seed1/logits.pt'.format(dataset_type, adapter_type, nshots)
 

    print (zs_path, os.path.isfile(zs_path))
    print (adapt_path, os.path.isfile(adapt_path))
    print (adaptpre_path, os.path.isfile(adaptpre_path))
    print (adaptpen_path, os.path.isfile(adaptpen_path))
    print (target_path, os.path.isfile(target_path))
    
    logits_zs = torch.load(zs_path, map_location='cpu').float()
    logits_adapt = torch.load(adapt_path, map_location='cpu').float() 
    logits_adapt_pren = torch.load(adaptpre_path,map_location='cpu').float()
    logits_adapt_pen = torch.load(adaptpen_path,map_location='cpu').float()
    target = torch.load(target_path, map_location='cpu')
    
    min_logits_zs, max_logits_zs = torch.min(logits_zs,1)[0].unsqueeze(1), torch.max(logits_zs,1)[0].unsqueeze(1) 
    min_logits_adapt, max_logits_adapt = torch.min(logits_adapt,1)[0].unsqueeze(1), torch.max(logits_adapt,1)[0].unsqueeze(1)
    logits_adapt_norm = (logits_adapt - min_logits_adapt)/ (max_logits_adapt - min_logits_adapt)
    logits_adapt_postn = logits_adapt_norm * (max_logits_zs - min_logits_zs) + min_logits_zs
   
    #zs
    sftmx_zs = softmax(logits_zs,1)
    conf_zs, pred_zs = torch.max(sftmx_zs,1)
    acc_zs = round(accuracy_score(target,pred_zs) * 100,2)
    ece_zs = round(expected_calibration_error(conf_zs,pred_zs,target).item()*100,2)

    #adapt
    sftmx_adapt = softmax(logits_adapt,1)
    conf_adapt, pred_adapt = torch.max(sftmx_adapt,1)
    acc_adapt = round(accuracy_score(target, pred_adapt) * 100,2)
    ece_adapt = round(expected_calibration_error(conf_adapt,pred_adapt,target).item()*100,2)

    #adaptpost
    sftmx_adapt_postn = softmax(logits_adapt_postn,1)
    conf_adapt_postn, pred_adapt_postn = torch.max(sftmx_adapt_postn,1)
    acc_adapt_postn = round(accuracy_score(target, pred_adapt_postn) * 100,2)
    ece_adapt_postn  = round(expected_calibration_error(conf_adapt_postn,pred_adapt_postn,target).item()*100,2)

    ##adaptpre
    sftmx_adapt_pren = softmax(logits_adapt_pren,1)
    conf_adapt_pren, pred_adapt_pren = torch.max(sftmx_adapt_pren,1)
    acc_adapt_pren = round(accuracy_score(target, pred_adapt_pren) * 100,2)
    ece_adapt_pren = round(expected_calibration_error(conf_adapt_pren,pred_adapt_pren,target).item()*100,2)

    ## adaptpen
    sftmx_adapt_pen = softmax(logits_adapt_pen,1)
    conf_adapt_pen, pred_adapt_pen = torch.max(sftmx_adapt_pen,1)
    acc_adapt_pen = round(accuracy_score(target, pred_adapt_pen) * 100,2)
    ece_adapt_pen = round(expected_calibration_error(conf_adapt_pen,pred_adapt_pen,target).item()*100,2)


    return acc_zs, acc_adapt, acc_adapt_pren, acc_adapt_postn, acc_adapt_pen, ece_zs, ece_adapt, ece_adapt_pren, ece_adapt_postn, ece_adapt_pen

info = {'dset':[],'method':[],'acc':[], 'ece':[]}
adapter_type = 'TR'
nshots = 16 #sys.argv[1]

for dset in tqdm(['imagenetv2','imagenet_sketch', 'imagenet_a', 'imagenet_r']):

    try:
        acc_zs, acc_adapt, acc_adapt_pren, acc_adapt_postn, acc_adapt_pen, ece_zs, ece_adapt, ece_adapt_pren, ece_adapt_postn, ece_adapt_pen = post_process(dset, adapter_type, nshots)
    except Exception as error:
        print (dset, adapter_type, nshots, error)
        continue

    info['dset'] += [dset]*5

    info['acc'].append(acc_zs)
    info['ece'].append(ece_zs)
    info['method'].append('zs')
    
    info['acc'].append(acc_adapt)
    info['ece'].append(ece_adapt)
    info['method'].append('adapt')

    info['acc'].append(acc_adapt_postn)
    info['ece'].append(ece_adapt_postn)
    info['method'].append('adapt_posn')

    info['acc'].append(acc_adapt_pren)
    info['ece'].append(ece_adapt_pren)
    info['method'].append('adapt_pren')

    info['acc'].append(acc_adapt_pen)
    info['ece'].append(ece_adapt_pen)
    info['method'].append('adapt_pen')
 
    df = pd.DataFrame(info)

    #break
    
print (df)
