# -*-coding:utf-8-*-
"""
@Project    : StR-GRD
@Time       : 2022/10/31 10:27
@Author     : Danke Wu
@File       : main.py
"""
import torch

# from baselines.NET import Net
from baselines.Net_woDCSC import Net
# from DCSC import Net

import argparse
import os
import sys
from string import digits

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def run(dataset, start_epoch,domain_num,seed):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=dataset,
                        choices=['weibo', '4charliehebdo', '4ferguson', '4germanwings-crash', '4ottawashooting',
                                 '4sydneysiege'],
                        help='The name of dataset')
    parser.add_argument('--epochs', type=int, default=200, help='The number of epochs to run')
    parser.add_argument('--start_epoch', type=int, default=start_epoch, help='Continue to train')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--num_posts', type=int, default= 50,choices=[50,140, 114])
    parser.add_argument('--margin', type=float, default=0.5,
                        help='the margin in recovered loss')
    parser.add_argument('--domain_num', type=float, default=domain_num,
                        help='the domain_num')
    parser.add_argument('--text_embedding', type=tuple, default=(768, 200),
                        help=' reduce the dimension of the text vector')
    parser.add_argument('--encoder_pars', type=tuple, default=(1, 1, 200, 100, 0.2),
                        help='num_layers, n_head, f_in, f_hid, dropout')
    parser.add_argument('--GCN_pars', type=tuple, default=(200,200),
                        help=' h_GCN)')
    parser.add_argument('--Claim_pars', type=tuple, default=(1),
                        help=' num_heads')
    parser.add_argument('--TextCNN_pars', type=tuple, default=(1,200),
                        help='num_fliters, h_hid')
    parser.add_argument('--lr', type=tuple, default=1e-3, help='lr')
    parser.add_argument('--dropout', type=float, default=0.0, help='attention dropout in GAT')
    parser.add_argument('--decay_start_epoch', type=float, default=10, help='decay_start_epoch')
    parser.add_argument('--patience', type=int, default=30, help='early stopping')
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--save_dir', type=str, default=r'E:\WDK_workshop\SRGRD\results\T-TEST',
                        help='Directory name to save the model')
    parser.add_argument('--model_name', type=str, default=r'TransE wostymix batch=64 1e-3 ch-9_'+ str(seed),
                        help='Directory name to save the GAN')
    parser.add_argument('--data_dir', default=os.path.join(r'E:\WDK_workshop\SRGRD\data\ch-9\data_withdomain', dataset),
                        type=str)
    # parser.add_argument('--data_dir', default=os.path.join(r'E:\WDK_workshop\SRGRD\data\ch-6\data_withdomain', dataset),
    #                     type=str)
    # parser.add_argument('--data_dir', default=os.path.join(r'E:\WDK_workshop\SRGRD\data\data_withdomain', dataset),
    #                     type=str)
    # parser.add_argument('--data_dir', default=os.path.join(r'E:\WDK_workshop\SRGRD\data\ch-9\data_claim', dataset),
    #                     type=str)
    # parser.add_argument('--data_dir', default=os.path.join(r'E:\WDK_workshop\SRGRD\data\data_claim', dataset),
    #                     type=str)
    if dataset =='weibo':
        target_data = 'weibo2021'
    elif dataset =='weibo2021':
        target_data = 'weibo'
    else:
        target_data =dataset.strip(digits)
    parser.add_argument('--target_domain', default=r'E:\WDK_workshop\SRGRD\data\ch-9\raw_data_withdomain',type=str)
    # parser.add_argument('--target_domain', default=r'E:\WDK_workshop\SRGRD\data\raw_data_withdomain',type=str)
    # parser.add_argument('--target_domain', default=r'E:\WDK_workshop\SRGRD\data\ch-9\raw_data_claim',type=str)
    # parser.add_argument('--target_domain', default=r'E:\WDK_workshop\SRGRD\data\raw_data_claim', type=str)
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
            f.writelines('-------------------  end -------------------')

    domains = set(all_datasets) - set([args.dataset])
    idx = 0
    # source domian train
    domian2idx_dict = {}
    for domain in domains:
        d = domain.strip(digits)
        domian2idx_dict[d] = idx
        idx += 1
    model.train_epoch(args.data_dir, start_epoch, domian2idx_dict)
    d = args.dataset.strip(digits)
    domian2idx_dict[d] = len(all_datasets)
    model.test(args.target_domain, target_data, domian2idx_dict)
    print(" [*] Training finished!")



if __name__ == '__main__':
    datasets =  [ '8社会生活', '8政治', '8文体娱乐', '8财经商业', '8医药健康', '8军事', '8灾难事故', '8教育考试', '8科技']
    all_datasets = ['8社会生活', '8政治', '8文体娱乐', '8财经商业', '8医药健康', '8军事', '8灾难事故', '8教育考试', '8科技']
    # datasets = [  '4charliehebdo', '4ferguson', '4germanwings-crash', '4ottawashooting', '4sydneysiege']
    # all_datasets = ['4charliehebdo', '4ferguson', '4germanwings-crash', '4ottawashooting', '4sydneysiege']
    # datasets = [ '5教育考试','5军事', '5科技', '5医药健康', '5灾难事故', '5政治']#
    # all_datasets= [ '5教育考试','5军事', '5科技', '5医药健康', '5灾难事故', '5政治']

    # datasets = ['4charliehebdo', '4ferguson','4germanwings-crash','4ottawashooting', '4sydneysiege','weibo','weibo2021']
    start_epoch = 77
    seeds = [1024,2048,3090,4587,5964]
    domain_num = len(all_datasets)
    for seed in seeds:
        for dataset in datasets:
            torch.manual_seed(seed)
            print(dataset)
            run(dataset, start_epoch, domain_num, seed)











