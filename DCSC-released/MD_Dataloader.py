# -*-coding:utf-8-*-
"""
@Project    : StR-GRD
@Time       : 2022/11/14 10:13
@Author     : Danke Wu
@File       : MD_Dataloader.py
"""
import os
from MD_Dataset import MyDataset_withdomains
from torch.utils.data import DataLoader

def seperate_dataloader_withdomains(datapath, dataset, batch, num_worker, num_nodes, domian_dict):

    nonrumor_path = os.path.join(datapath, dataset, 'nonrumor')
    nonrumor_files = os.listdir(nonrumor_path)
    nonrumor_dataset = MyDataset_withdomains(nonrumor_path, nonrumor_files, num_nodes, domian_dict)
    rumor_path = os.path.join(datapath, dataset, 'rumor')
    rumor_files = os.listdir(rumor_path)
    rumor_dataset = MyDataset_withdomains(rumor_path, rumor_files, num_nodes, domian_dict)
    #DDT
    assert batch % 2 ==0 or batch % 4 ==0
    # rvsn = len(rumor_files) / len(nonrumor_files)
    # if rvsn >1 :
    #     n_rumor = int(batch / 2)
    # elif rvsn >= 0.5:
    #     n_rumor = round(rvsn * batch / 2)
    # elif rvsn < 0.5:
    #     n_rumor = int(batch / 4)
    # n_nonrumor = int(batch / 2)

    n_nonrumor, n_rumor = int(batch / 2), int(batch / 2)

    # #InF
    # rvsn = len(rumor_files) / (len(nonrumor_files) + len(rumor_files))
    # n_nonrumor,n_rumor = batch - math.ceil(batch * rvsn), math.ceil(batch * rvsn)

    nonrumor_loader = DataLoader(nonrumor_dataset,
                              batch_size= n_nonrumor,
                              shuffle=True,
                              num_workers=num_worker,
                              drop_last= True,
                              pin_memory=True)
    rumor_loader = DataLoader(rumor_dataset,
                              batch_size= n_rumor,
                              shuffle=True,
                              num_workers=num_worker,
                              drop_last=True,
                              pin_memory=True)

    return nonrumor_loader, rumor_loader

def normal_dataloader_withdomains(datapath, dataset, batch_size, num_worker, num_nodes, domain_dict):

    path = os.path.join(datapath, dataset)
    files = os.listdir(path)
    dataset = MyDataset_withdomains(path, files, num_nodes, domain_dict)
    if dataset =='':
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_worker,
                                drop_last=False,
                                pin_memory=True)
    else:
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_worker,
                                drop_last=False,
                                pin_memory=True)

    return dataloader