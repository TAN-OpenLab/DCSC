# -*-coding:utf-8-*-
"""
@Project    : StR-GRD
@Time       : 2022/10/19 11:15
@Author     : Danke Wu
@File       : metrics.py
"""
# -*-coding:utf-8-*-
"""
@Project    : DMCD
@Time       : 2022/10/4 16:04
@Author     : Danke Wu
@File       : metrics.py
"""
# -*-coding:utf-8-*-
import torch

def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = torch.max(output,dim=1)[1]
        correct = pred.eq(target).float().sum()
        correct.mul_(100. / batch_size)
        return correct

def count_2class(prediction, y):
    TP1, FP1, FN1, TN1 = 0, 0, 0, 0
    TP2, FP2, FN2, TN2 = 0, 0, 0, 0
    for i in range(len(y)):
        Act, Pre = y[i], prediction[i]

        ## for class 1
        if Act == 0 and Pre == 0: TP1 += 1
        if Act == 0 and Pre != 0: FN1 += 1
        if Act != 0 and Pre == 0: FP1 += 1
        if Act != 0 and Pre != 0: TN1 += 1
        ## for class 2
        if Act == 1 and Pre == 1: TP2 += 1
        if Act == 1 and Pre != 1: FN2 += 1
        if Act != 1 and Pre == 1: FP2 += 1
        if Act != 1 and Pre != 1: TN2 += 1

    return TP1, FN1, FP1, TN1, TP2, FN2, FP2, TN2


def evaluationclass(TP1, FN1, FP1, TN1, TP2, FN2, FP2, TN2, num_sample):  # 2 dim


    ## print result
    Acc_all = round(float(TP1 + TP2) / float(num_sample), 4)
    Acc1 = round(float(TP1 + TN1) / float(TP1 + TN1 + FN1 + FP1), 4)
    if (TP1 + FP1)==0:
        Prec1 =0
    else:
        Prec1 = round(float(TP1) / float(TP1 + FP1), 4)
    if (TP1 + FN1 )==0:
        Recll1 =0
    else:
        Recll1 = round(float(TP1) / float(TP1 + FN1 ), 4)
    if (Prec1 + Recll1 )==0:
        F1 =0
    else:
        F1 = round(2 * Prec1 * Recll1 / (Prec1 + Recll1 ), 4)

    Acc2 = round(float(TP2 + TN2) / float(TP2 + TN2 + FN2 + FP2), 4)
    if (TP2 + FP2)==0:
        Prec2 =0
    else:
        Prec2 = round(float(TP2) / float(TP2 + FP2), 4)
    if (TP2 + FN2 )==0:
        Recll2 =0
    else:
        Recll2 = round(float(TP2) / float(TP2 + FN2 ), 4)
    if (Prec2 + Recll2 )==0:
        F2 =0
    else:
        F2 = round(2 * Prec2 * Recll2 / (Prec2 + Recll2 ), 4)

    return Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2


def evaluationclass_dict( conter_dict, num_sample):  # 2 dim

    ACC_dict = {}

    ## print result
    ACC_dict['Acc_all'] = round(float(conter_dict['TP1'] + conter_dict['TP2'] ) / float(num_sample), 4)
    ACC_dict['Acc1'] = round(float(conter_dict['TP1']  + conter_dict['TN1'] ) / float(conter_dict['TP1'] + conter_dict['TN1'] + conter_dict['FN1'] + conter_dict['FP1'] ), 4)
    if (conter_dict['TP1']  + conter_dict['FP1'] )==0:
        ACC_dict['Prec1'] =0
    else:
        ACC_dict['Prec1'] = round(float(conter_dict['TP1'] ) / float(conter_dict['TP1']  + conter_dict['FP1']), 4)
    if (conter_dict['TP1']  + conter_dict['FN1']  )==0:
        ACC_dict['Recll1'] =0
    else:
        ACC_dict['Recll1'] = round(float(conter_dict['TP1'] ) / float(conter_dict['TP1']+ conter_dict['FN1']), 4)
    if (ACC_dict['Prec1'] + ACC_dict['Recll1'] )==0:
        ACC_dict['F1'] =0
    else:
        ACC_dict['F1'] = round(2 * ACC_dict['Prec1'] * ACC_dict['Recll1'] / (ACC_dict['Prec1'] + ACC_dict['Recll1'] ), 4)

    ACC_dict['Acc2'] = round(float(conter_dict['TP2']  + conter_dict['TN2']) / float(conter_dict['TP2'] + conter_dict['TN2'] + conter_dict['FN2'] + conter_dict['FP2'] ), 4)
    if (conter_dict['TP2']  + conter_dict['FP2'] )==0:
        ACC_dict['Prec2'] =0
    else:
        ACC_dict['Prec2'] = round(float(conter_dict['TP2'] ) / float(conter_dict['TP2'] + conter_dict['FP2'] ), 4)
    if (conter_dict['TP2']  + conter_dict['FN2'] )==0:
        ACC_dict['Recll2'] =0
    else:
        ACC_dict['Recll2'] = round(float(conter_dict['TP2'] ) / float(conter_dict['TP2'] + conter_dict['FN2'] ), 4)
    if (ACC_dict['Prec2'] + ACC_dict['Recll2'] )==0:
        ACC_dict['F2'] =0
    else:
        ACC_dict['F2'] = round(2 * ACC_dict['Prec2'] * ACC_dict['Recll2'] / (ACC_dict['Prec2'] + ACC_dict['Recll2'] ), 4)

    return ACC_dict
