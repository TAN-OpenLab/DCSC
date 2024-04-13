# -*-coding:utf-8-*-
"""
@Project    : StR-GRD
@Time       : 2022/11/13 10:30
@Author     : Danke Wu
@File       : multi_domain_main.py
For multi-domain cross validation,one domain is unseen for training
"""


import torch
import random
import numpy as np

from multi_domains.DCSC_MD import Net
# from Net_EANN import Net
# from EDDFN_NET import Net
# from Net_MDFEND import Net
# from Net_MDDA import Net
from Net_MDDA_DCSC import Net
# from multi_domains.Net_M3FEND import Net


import argparse
import os,re
import sys
from string import digits

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def run(dataset, all_datasets, start_epoch,num_domain, seed):
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
    parser.add_argument('--num_domain', type=int, default=num_domain)
    parser.add_argument('--margin', type=tuple, default=0.5,
                        help='the margin in recovered loss')
    parser.add_argument('--num_style', type=int, default=8,
                        help='the number of styles ,default=8')
    parser.add_argument('--text_embedding', type=tuple, default=(768, 200),
                        help=' reduce the dimension of the text vector')
    parser.add_argument('--encoder_pars', type=tuple, default=(1, 1, 200, 100, 0.2),
                        help='num_layers, n_head, f_in,f_hid, dropout')
    parser.add_argument('--CNN_pars', type=tuple, default=(num_domain, 5, 0.2),
                        help='domain_num, filter_num, dropout')
    parser.add_argument('--mdfend_pars', type=tuple, default=(num_domain, num_domain-1, 0.2),
                        help='domain_num, num_expert, dropout')
    parser.add_argument('--eddfn_paras', type=tuple, default=(768, [200], 0.0),
                        help=' c_in, c_hid, dropout')

    parser.add_argument('--lr', type=tuple, default= 1e-3, help='lr')
    parser.add_argument('--weight_decay', type=float, default=10, help='weight_decay')
    parser.add_argument('--dropout', type=float, default=0.0, help='attention dropout in GAT')
    parser.add_argument('--decay_start_epoch', type=float, default=10, help='decay_start_epoch')
    parser.add_argument('--patience', type=int, default=30, help='early stopping')
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--save_dir', type=str, default=r'E:\WDK_workshop\SRGRD\results\exp9 mc_loss ablation',
                        help='Directory name to save the model')
    parser.add_argument('--model_name', type=str, default=r'MDDA_stymix_domaindetection batch=64 1e-3 ch-9-1_'+str(seed),
                        help='Directory name to save the GAN')
    parser.add_argument('--modelname', type=str, default= 'domaindetection')
    parser.add_argument('--data_dir', default=os.path.join(r'E:\WDK_workshop\SRGRD\data\ch-9\data_withdomain', dataset),
                        type=str)
    # parser.add_argument('--data_dir', default=os.path.join(r'E:\WDK_workshop\SRGRD\data\ch-6\data_withdomain', dataset),
    #                     type=str)
    # parser.add_argument('--data_dir', default=os.path.join(r'E:\WDK_workshop\SRGRD\data\data_withdomain', dataset),
    #                     type=str)
    # parser.add_argument('--data_dir', default=os.path.join(r'E:\WDK_workshop\SRGRD\data\en_3\date_withdomain', dataset),
    #                     type=str)
    if dataset =='weibo':
        target_data = 'weibo2021'
    elif dataset =='weibo2021':
        target_data = 'weibo'
    else:
        target_data =dataset.strip(digits)
    parser.add_argument('--target_domain', default=os.path.join(r'E:\WDK_workshop\SRGRD\data\ch-9\raw_data_withdomain',
                                                                target_data ), type=str)
    # parser.add_argument('--target_domain', default=os.path.join(r'E:\WDK_workshop\SRGRD\data\raw_data_withdomain',
    #                                                             target_data), type=str)
    # parser.add_argument('--target_domain', default=os.path.join(r'E:\WDK_workshop\SRGRD\data\en_3\raw_date_withdomain', target_data),
    #                     type=str)
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


    domains = set(all_datasets) - set([args.dataset])
    idx = 0
    #source domian train
    domian2idx_dict = {}
    for domain in domains:
        d = domain.strip(digits)
        domian2idx_dict[d] = idx
        idx += 1
    model.train_epoch(args.data_dir, start_epoch, domian2idx_dict)
    d = args.dataset.strip(digits)
    domian2idx_dict[d] = len(all_datasets)
    model.test(args.target_domain, '',domian2idx_dict)
    # domain_dict = {'社会生活': 0, '政治': 1, '文体娱乐': 2, '财经商业': 3, '医药健康': 4, '军事': 5, '灾难事故': 6, '教育考试': 7, '科技': 8}
    # domain_dict = { '政治': 0, '医药健康': 1, '军事': 2, '灾难事故': 3, '教育考试': 4, '科技': 5}
    # domain_dict = { 'charliehebdo':0,  'ferguson':1,  'germanwings-crash':2,  'ottawashooting':3,  'sydneysiege':4}
    # model.train_epoch(args.data_dir, start_epoch, domain_dict)
    # model.test(args.target_domain, '', domain_dict)
    print(" [*] Training finished!")

##CH-9
if __name__ == '__main__':

    # datasets = [  '4charliehebdo', '4ferguson','4germanwings-crash', '4ottawashooting', '4sydneysiege']
    # all_datasets = ['4charliehebdo', '4ferguson', '4germanwings-crash', '4ottawashooting', '4sydneysiege']
    # datasets = [ '5教育考试','5科技', '5医药健康', '5灾难事故', '5政治','5教育考试']#
    # all_datasets= [ '5教育考试','5军事', '5科技', '5医药健康', '5灾难事故', '5政治']
    # all_datasets = ['3财经商业', '3社会生活', '3文体娱乐']
    datasets = [ '8财经商业']
    all_datasets= ['8财经商业','8教育考试','8军事', '8科技', '8社会生活', '8文体娱乐', '8医药健康', '8灾难事故', '8政治']
    # datasets = [  '2politic', '2gossip', '2health']
    # all_datasets = [ '2politic', '2gossip', '2health']

    start_epoch = 26
    seeds = [1024]#[1024,2048,3090,4587,5964, 6782, 7431, 8761, 9014]
    num_domain = len(all_datasets)-1
    # lrs = [0.1, 5e-2
    # , 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5]
    for seed in seeds:
        for dataset in datasets:
            # random.seed(seed)
            # np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print(dataset)
            run(dataset, all_datasets, start_epoch, num_domain, seed)







