# -*-coding:utf-8-*-
"""
@Project    : StR-GRD
@Time       : 2022/11/22 9:34
@Author     : Danke Wu
@File       : Net_woDCSC.py
"""

import os, sys
import time
from MD_Dataloader import seperate_dataloader_withdomains, normal_dataloader_withdomains
from utils import *
from metrics import *
from criterions import *
from baselines.BIGRU import BIGRU


class Net(object):
    def __init__(self, args, device):
        # parameters

        self.epochs = args.epochs
        self.start_epoch = args.start_epoch
        self.batch_size = args.batch_size
        self.num_posts = args.num_posts
        self.margin = args.margin
        self.c_in,self.c_hid = args.text_embedding
        self.num_layers_trans, self.n_head_trans, _,self.h_hid_trans, self.dropout = args.encoder_pars
        self.n_filters, self.cnn_hid = args.TextCNN_pars
        self.num_worker = args.num_worker
        self.lr = args.lr
        self.decay_start_epoch = args.decay_start_epoch
        self.device = device
        self.b1, self.b2 = args.b1, args.b2
        self.checkpoint = args.start_epoch
        self.patience = args.patience
        self.dataset = args.dataset
        self.model_path = os.path.join(args.save_dir, args.model_name, args.dataset)


        #=====================================load rumor_detection model================================================
        # self.net = VAE(self.c_in,self.c_hid,num_layers=1,bidirection=False).to(self.device)
        self.net = BIGRU(self.c_in, self.c_hid, num_layers=1, bidirection=False).to(self.device)
        # self.net = TextCNN(self.n_filters, self.c_in, self.c_hid, self.num_posts, self.dropout).to(self.device)
        # self.net = TransE(self.c_in, self.c_hid, self.num_layers_trans, self.n_head_trans, self.h_hid_trans,
        #                   self.dropout).to(self.device)

        print(self.net)


        # =====================================load loss function================================================
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.SGD(
            [{'params': self.net.parameters(), 'lr': self.lr, 'momentum': 0.9, 'weight_decay': 1e-4}])


        self.lr_lambda = LambdaLR(self.epochs, self.start_epoch, decay_start_epoch=self.decay_start_epoch)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self.lr_lambda.step)

        torch.autograd.set_detect_anomaly(True)

    def train_epoch(self, datapath, start_epoch,domain_dict):

        nonrumor_loader, rumor_loader = seperate_dataloader_withdomains(datapath, 'train', self.batch_size,
                                                                        self.num_worker,
                                                                        self.num_posts, domain_dict)
        val_loader = normal_dataloader_withdomains(datapath, 'val', self.batch_size, self.num_worker, self.num_posts,
                                                   domain_dict)
        # ==================================== train and val dataGAN with model=========================================
        acc_check = 0
        loss_check = 100
        self.train_hist = {}
        self.train_hist['train_loss'] = []
        self.train_hist['train_acc'] = []
        self.train_hist['test_acc'] = []
        self.train_hist['acc_1'] = []
        self.train_hist['pre_1'] = []
        self.train_hist['recall_1'] = []
        self.train_hist['f1_1'] = []
        self.train_hist['acc_2'] = []
        self.train_hist['pre_2'] = []
        self.train_hist['recall_2'] = []
        self.train_hist['f1_2'] = []

        start_time = time.clock()
        patience = self.patience
        for epoch in range(start_epoch, self.epochs):

            train_loss, train_acc = self.train_batch(epoch,nonrumor_loader,rumor_loader)

            self.lr_scheduler.step()


            print(train_loss, train_acc)
            with torch.no_grad():
                val_loss, Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2= self.evaluation(val_loader,epoch)
            end_time = time.clock()
            self.train_hist['train_loss'].append(train_loss)
            self.train_hist['train_acc'].append(train_acc)
            self.train_hist['test_acc'].append(Acc_all)
            self.train_hist['acc_1'].append(Acc1)
            self.train_hist['pre_1'].append(Prec1)
            self.train_hist['recall_1'].append(Recll1)
            self.train_hist['f1_1'].append(F1)
            self.train_hist['acc_2'].append(Acc2)
            self.train_hist['pre_2'].append(Prec2)
            self.train_hist['recall_2'].append(Recll2)
            self.train_hist['f1_2'].append(F2)

            if Acc_all > acc_check or (acc_check == Acc_all and loss_check > val_loss):  # or
                acc_check = Acc_all
                loss_check = val_loss
                self.checkpoint = epoch
                self.save(self.model_path, epoch)
                patience = self.patience
                print(acc_check, loss_check)

            # self.checkpoint = epoch
            # self.save(self.model_path, epoch)

            patience -= 1
            if not patience or train_loss >6:
                break

        # # ==================================== test model with model================================================
        # with torch.no_grad():
        #     test_loader = normal_dataloader(datapath, 'test',self.batch_size, self.num_worker, self.num_posts)
        #
        #     start_epoch = self.load(self.model_path, self.checkpoint)
        #
        #     test_loss, Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2 = self.evaluation(test_loader, start_epoch)
        #
        #     with open(os.path.join(self.model_path, 'predict.txt'), 'w') as f:
        #         hist = [str(k) + ':' + str(self.train_hist[k]) for k in self.train_hist.keys()]
        #         f.write('\n'.join(hist) + '\n')
        #
        #     with open(os.path.join(self.model_path, 'predict.txt'), 'a') as f:
        #         hist = [Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2]
        #         hist = list(map(str, hist))
        #         f.write(
        #             'source domain' + str(start_epoch) + '\n' + ' Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2' + '\n' + '\t'.join(
        #                 hist) + '\n')
        #         f.close()

    def train_batch(self, epoch, nonrumor_loader, rumor_loader):

        train_loss_value,acc_all = 0,0
        iterloader = zip(nonrumor_loader, rumor_loader)

        for iter, (Nonrumors, Rumors) in enumerate(iterloader):
            xn, yn, An = Nonrumors
            xr, yr, Ar = Rumors
            xn = xn.to(self.device)
            yn = yn.to(self.device)
            xr = xr.to(self.device)
            yr = yr.to(self.device)
            Ar = Ar.to(self.device)
            An = An.to(self.device)

            x = torch.cat((xn, xr) ,dim=0)
            y = torch.cat((yn, yr), dim=0)
            A = torch.cat((An, Ar), dim=0)


            # ====================================train Model============================================
            self.net.train()

            # Step A: orginal data
            pred = self.net(x)

            loss = self.criterion(pred,y)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            acc = binary_accuracy(pred, y)

            train_loss_value += loss.item()
            acc_all += acc

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f][acc: %f] "
                % (
                    epoch,
                    self.epochs,
                    iter,
                    len(nonrumor_loader),
                    loss.item(),
                    acc

                )
            )

        train_loss_value = round( train_loss_value / (iter+1), 4)
        acc_all_value = round(acc_all.item() / (iter+1), 4)

        return train_loss_value,  acc_all_value


    def evaluation(self, dataloader, epoch):

        self.net.eval()

        total_loss, total_Acc = 0,0
        TP1, FP1, FN1, TN1 = 0, 0, 0, 0
        TP2, FP2, FN2, TN2 = 0, 0, 0, 0
        num_sample =0

        for iter, sample in enumerate(dataloader):
            x, y, A = sample
            x = x.to(self.device)
            A = A.to(self.device)
            y = y.to(self.device)

            # Step A: orginal data
            pred = self.net(x)
            loss= self.criterion(pred,y)

            total_loss += loss.item()
            pred = pred.data.max(1)[1].cpu()
            # pred = pred_cont_A.data.max(1)[1].cpu()
            tp1, fn1, fp1, tn1, tp2, fn2, fp2, tn2 = count_2class(pred, y.cpu())

            TP1 += tp1
            FP1 += fp1
            FN1 += fn1
            TN1 += tn1
            TP2 += tp2
            FP2 += fp2
            FN2 += fn2
            TN2 += tn2
            num_sample += len(y)

        (Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2) = \
            evaluationclass(TP1, FN1, FP1, TN1, TP2, FN2, FP2, TN2, num_sample)
        test_loss_value = total_loss / (iter+1)

        print('eval_loss:%0.4f,acc:%0.4f' % (test_loss_value, Acc_all))
        print('acc_1:%0.4f, pre1:%0.4f, recall1:%0.4f, f1_1:%0.4f' % (Acc1, Prec1, Recll1, F1))
        print('acc_2:%0.4f, pre2:%0.4f, recall2:%0.4f, f1_2:%0.4f' % (Acc2, Prec2, Recll2, F2))

        return test_loss_value, Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2



    def test(self, datapath,target_data, domain_dict):

        datafile = os.listdir(datapath)

        test_loader = normal_dataloader_withdomains(datapath, target_data, self.batch_size, self.num_worker,
                                                    self.num_posts, domain_dict)
        with torch.no_grad():
            start_epoch = self.load(self.model_path, self.checkpoint)
            test_loss_value, Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2 = self.evaluation(test_loader, start_epoch)
            with open(os.path.join(self.model_path, 'predict.txt'), 'a') as f:
                hist = [Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2]
                hist = list(map(str, hist))
                f.write(
                    'target domian:'  + '\t' + str(target_data)+ '\t'+ str(start_epoch) + '\n' + ' Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2' + '\n' + '\t'.join(
                        hist) + '\n')
                f.close()

            print(Acc_all)

    def all_test(self, datapath, datasets, domain_dict):
        Conter_Item = ['TP1', 'FP1', 'FN1', 'TN1', 'TP2', 'FP2', 'FN2', 'TN2']
        Conter = {}
        for item in Conter_Item:
            Conter[item] = 0
        num_sample = 0

        for dataset in datasets:
            test_loader = normal_dataloader_withdomains(datapath, dataset, self.batch_size, self.num_worker, self.num_posts,
                                            domain_dict)
            _ = self.load(self.model_path, self.checkpoint)
            with torch.no_grad():
                conter, num = self.all_transfer_test(test_loader)
            for item in Conter_Item:
                Conter[item] += conter[item]
            num_sample+= num
        acc_test_dict = evaluationclass_dict(Conter, num_sample)
        print(acc_test_dict.items())
        with open(os.path.join(self.model_path, 'predict.txt'), 'a') as f:
            f.write('all doamin:' +'\t' + str(self.checkpoint) + '\n' +
                    '\t'.join(list(acc_test_dict.keys())) + '\n' + '\t'.join(
                 map(str, list(acc_test_dict.values()))) + '\n')

    def all_transfer_test(self, dataloader):

        Conter_Item = ['TP1', 'FP1', 'FN1', 'TN1', 'TP2', 'FP2', 'FN2', 'TN2']
        Conter = {}
        for item in Conter_Item:
            Conter[item] = 0
        num_sample = 0
        self.net.eval()

        for iter, sample in enumerate(dataloader):
            x, y, D = sample
            x = x.to(self.device)
            y_ = y.to(self.device)
            D = D.to(self.device)

            preds = self.net(x)
            preds = preds.data.max(1)[1].cpu()

            (tp1, fn1, fp1, tn1, tp2, fn2, fp2, tn2) = count_2class(preds, y_)
            Conter['TP1'] += tp1
            Conter['FN1'] += fn1
            Conter['FP1'] += fp1
            Conter['TN1'] += tn1
            Conter['TP2'] += tp2
            Conter['FN2'] += fn2
            Conter['FP2'] += fp2
            Conter['TN2'] += tn2

            num_sample += len(y)
        return Conter, num_sample


    def save(self, model_path, epoch):
        save_states = {'net': self.net.state_dict(),
                       'optimizer': self.optimizer.state_dict(),
                       'checkpoint': epoch,
                       # 'rumor_anchor': self.r_anchor,
                       # 'nonrumor_anchor': self.n_anchor

        }
        torch.save(save_states, os.path.join(model_path, str(epoch) + '_model_states.pkl'))
        print('save classifer : %d epoch' % epoch)

    def load(self, model_path, checkpoint):
        states_dicts = torch.load( os.path.join(model_path, str(checkpoint) + '_model_states.pkl'))

        self.net.load_state_dict(states_dicts['net'])
        self.optimizer.load_state_dict(states_dicts['optimizer'])
        start_epoch = states_dicts['checkpoint']
        print("load epoch {} success!".format(start_epoch))

        return start_epoch


