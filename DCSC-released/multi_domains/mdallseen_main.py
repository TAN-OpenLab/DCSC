# -*-coding:utf-8-*-
"""
@Project    : StR-GRD
@Time       : 2022/11/24 22:38
@Author     : Danke Wu
@File       : mdallseen_main.py
For multi-domain validation,all domains are unknown for training, testing on each domain.
Experiments setting is the same as MDFEND,M3FEND
"""

import torch
import random
import numpy as np


# from multi_domains.Net_EANN import Net
# from multi_domains.Net_MDFEND import Net
# from multi_domains.EDDFN_NET import Net
# from multi_domains.Net_M3FEND import Net
# from multi_domains.Net_MDDA import Net
# from multi_domains.DCSC_MD import Net
from baselines.Net_woDCSC import Net
# from baselines.NET import Net

import argparse
import os,re
import sys
from string import digits

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
##CH-9
torch.manual_seed(6)

def run(dataset,  all_datasets, start_epoch,num_domain, seed):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=dataset,
                        choices=['weibo', '4charliehebdo', '4ferguson', '4germanwings-crash', '4ottawashooting',
                                 '4sydneysiege'],
                        help='The name of dataset')
    parser.add_argument('--epochs', type=int, default=200, help='The number of epochs to run')
    parser.add_argument('--start_epoch', type=int, default=start_epoch, help='Continue to train')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--num_posts', type=int, default=50)
    parser.add_argument('--num_style', type=int, default=8)
    parser.add_argument('--margin', type=tuple, default=0.5,
                        help='the margin in recovered loss')
    parser.add_argument('--domain_num', type=int, default=num_domain,
                        help='number of domains')
    parser.add_argument('--text_embedding', type=tuple, default=(768, 200),
                        help=' reduce the dimension of the text vector')
    parser.add_argument('--encoder_pars', type=tuple, default=(1, 1, 200, 100, 0.2),
                        help='num_layers, n_head, f_in,f_hid, dropout')
    parser.add_argument('--CNN_pars', type=tuple, default=(num_domain, 5, 0.2),
                        help='domain_num, filter_num, dropout')
    parser.add_argument('--mdfend_pars', type=tuple, default=(num_domain, 5, 0.2),
                        help='domain_num, num_expert, dropout')
    parser.add_argument('--eddfn_paras', type=tuple, default=(768, [200], 0.0),
                        help=' c_in, c_hid, dropout')
    parser.add_argument('--TextCNN_pars', type=tuple, default=(1, 50),
                        help='num_fliters, h_hid')

    parser.add_argument('--lr', type=tuple, default=1e-2, help='lr')
    parser.add_argument('--weight_decay', type=float, default=10, help='weight_decay')
    parser.add_argument('--dropout', type=float, default=0.0, help='attention dropout in GAT')
    parser.add_argument('--decay_start_epoch', type=float, default=10, help='decay_start_epoch')
    parser.add_argument('--patience', type=int, default=30, help='early stopping')
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--save_dir', type=str, default=r'E:\WDK_workshop\SRGRD\results\T-TEST\5fold',
                        help='Directory name to save the model')
    parser.add_argument('--model_name', type=str, default=r'EDDFN batch=64 1e-2 ch-9_'+ str(seed),
                        help='Directory name to save the GAN')
    parser.add_argument('--data_dir', default=r'E:\WDK_workshop\SRGRD\data\ch-9\data_withdomain_91',type=str)
    parser.add_argument('--target_domain', default=r'E:\WDK_workshop\SRGRD\data\ch-9\data_withdomain_91\test', type=str)
    # parser.add_argument('--data_dir', default=r'E:\WDK_workshop\SRGRD\data\data_withdomain_51', type=str)
    # parser.add_argument('--target_domain', default=r'E:\WDK_workshop\SRGRD\data\data_withdomain_51\test', type=str)
    parser.add_argument('--data_eval', default='', type=str)

    args = parser.parse_args()
    print(args)

    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Net(args, device)

    if not os.path.exists(os.path.join(args.save_dir, args.model_name)):
        os.mkdir(os.path.join(args.save_dir, args.model_name))
    if not os.path.exists(os.path.join(args.save_dir, args.model_name, args.dataset)):
        os.mkdir(os.path.join(args.save_dir, args.model_name, args.dataset))


    if os.path.exists(os.path.join(args.save_dir, args.model_name, args.dataset,
                                   str(args.start_epoch) + '_model_states.pkl')):
        model_path = os.path.join(args.save_dir, args.model_name, args.dataset)
        start_epoch = model.load(model_path, args.start_epoch)
        start_epoch = start_epoch + 1
    else:
        start_epoch = 0
        print("start from epoch {}".format(start_epoch))
        argsDict = args.__dict__
        with open(os.path.join(args.save_dir, args.model_name, args.dataset, 'setting.txt'), 'w') as f:
            f.writelines('------------------ start ------------------' + '\n')
            for eachArg, value in argsDict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')

    domain_dict = {'社会生活':0, '政治':1,'文体娱乐':2,'财经商业':3,'医药健康':4,'军事':5,'灾难事故':6,'教育考试':7,'科技':8}
    # domain_dict = {'charliehebdo': 0, 'ferguson': 1, 'germanwings-crash': 2, 'ottawashooting': 3, 'sydneysiege': 4}
    model.train_epoch(args.data_dir, start_epoch, domain_dict)
    for domain in domain_dict.keys():
        model.test(args.target_domain,domain,domain_dict)

    model.all_test(args.target_domain, domain_dict.keys(), domain_dict)


    print(" [*] Training finished!")


if __name__ == '__main__':

    # datasets = ['data_withdomain_51']
    # all_datasets = ['4charliehebdo', '4ferguson', '4germanwings-crash', '4ottawashooting', '4sydneysiege']
    #
    all_datasets = ['8财经商业','8教育考试','8军事', '8科技', '8社会生活', '8文体娱乐', '8医药健康', '8灾难事故', '8政治']
    datasets= [ 'data_withdomain_91']
    start_epoch = 71
    seeds = [ 1024 ]#[1024,2048,3090,4587,5964,6782, 7431, 8761, 9014]
    num_domain = len(all_datasets)
    for seed in seeds:
        for dataset in datasets:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print(dataset)
            run(dataset, all_datasets, start_epoch,num_domain, seed)