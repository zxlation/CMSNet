import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve
import torch
from torch.nn import functional as F


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


class FusionMatrix(object):
    def __init__(self, num_classes, class_name):
        self.num_classes = num_classes
        self.class_name = class_name
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)

    def update(self, output, label):
        length = output.shape[0]
        for i in range(length):
            self.matrix[int(output[i]), int(label[i])] += 1

    def get_fusion_matrix(self):
        return self.matrix

    def get_rec_per_class(self):
        rec = np.array(
            [
                self.matrix[i, i] / self.matrix[:, i].sum()
                for i in range(self.num_classes)
            ]
        )
        rec[np.isnan(rec)] = 0
        return rec

    def get_pre_per_class(self):
        pre = np.array(
            [
                self.matrix[i, i] / self.matrix[i, :].sum()
                for i in range(self.num_classes)
            ]
        )
        pre[np.isnan(pre)] = 0
        return pre

    def get_spec_per_class(self):
        spec_list = []
        for i in range(self.num_classes):
            TN = self.matrix.sum() - self.matrix[i, :].sum() - self.matrix[:, i].sum() \
                 + self.matrix[i, i]
            FP = self.matrix[:, i].sum() - self.matrix[i, i]
            spec = TN / (TN + FP)
            spec_list.append(spec)
        SP = np.array(spec_list)
        SP[np.isnan(SP)] = 0
        return SP

    def get_accuracy(self):
        acc = (
            np.sum([self.matrix[i, i] for i in range(self.num_classes)])
            / self.matrix.sum()
        )
        return acc

    def get_acc_per_class(self):
        acc = np.array(
            [
                self.matrix[i, i] / self.matrix[:, i].sum()
                for i in range(self.num_classes)
            ]
        )
        acc[np.isnan(acc)] = 0
        return acc

    # convert multi-classification into binary classification tasks, and calculate the accuracy.
    def get_binary_accuracy(self, class_no):
        class_no = int(class_no)
        if class_no >= self.num_classes or class_no < 0:
            raise AttributeError(
                "Parameter class_no must less than the number of class.")

        error_num = 0
        for i in range(self.num_classes):
            error_num += self.matrix[i, class_no]
            error_num += self.matrix[class_no, i]
        error_num -= 2 * self.matrix[class_no, class_no]
        ok_num = self.matrix.sum() - error_num
        acc = ok_num / self.matrix.sum()
        return acc

    def get_binary_average_pre(self, class_no):
        class_no = int(class_no)
        if class_no >= self.num_classes or class_no < 0:
            raise AttributeError(
                "Parameter class_no must less than the number of class.")
        pos_pre = self.matrix[class_no, class_no] / self.matrix[class_no, :].sum()
        TN = self.matrix.sum() - self.matrix[class_no,:].sum() - self.matrix[:, class_no].sum() \
                  +  self.matrix[class_no, class_no]
        FN = self.matrix[:, class_no].sum() - self.matrix[class_no, class_no]
        neg_pre = TN / (TN + FN)
        AP = (pos_pre + neg_pre) / 2
        return AP


    def get_balance_accuracy(self):
        rec = self.get_rec_per_class()
        bacc = np.mean(rec)
        return bacc

    def get_f1_score(self):
        recall = self.get_rec_per_class()
        precision = self.get_pre_per_class()
        f1_score = 2 * (precision * recall) / (precision + recall)
        f1_score[np.isnan(f1_score)] = 0
        return np.mean(f1_score)

    def plot_confusion_matrix(self, normalize = False, cmap=plt.cm.Blues, save_path='混淆矩阵.pdf'):
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = self.matrix.T

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=self.class_name, yticklabels=self.class_name,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.savefig(save_path)
        plt.show()

    # 2017，保存为二分类混淆矩阵
    def plot_binary_confusion_matrix(self, normalize=False, cmap=plt.cm.Blues, save_path='混淆矩阵.pdf', class_no=0):
        class_no = int(class_no)
        if class_no >= self.num_classes or class_no < 0:
            raise AttributeError(
                "Parameter class_no must less than the number of class.")
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = self.matrix.T
        TP = cm[class_no, class_no]
        FN = cm[class_no, :].sum() - cm[class_no, class_no]
        FP = cm[:, class_no].sum() - cm[class_no, class_no]
        TN = cm.sum() - TP - FN - FP
        binary_matrix = np.zeros((2, 2), dtype=int)
        binary_matrix[0, 0] = TP
        binary_matrix[0, 1] = FN
        binary_matrix[1, 0] = FP
        binary_matrix[1, 1] = TN
        class_name = []
        class_name.append(self.class_name[class_no])
        others = 'Non-' + self.class_name[class_no]
        class_name.append(others)

        fig, ax = plt.subplots()
        im = ax.imshow(binary_matrix, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(binary_matrix.shape[1]),
               yticks=np.arange(binary_matrix.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=class_name, yticklabels=class_name,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = binary_matrix.max() / 2.
        for i in range(binary_matrix.shape[0]):
            for j in range(binary_matrix.shape[1]):
                ax.text(j, i, format(binary_matrix[i, j], fmt),
                        ha="center", va="center",
                        color="white" if binary_matrix[i, j] > thresh else "black")
        fig.tight_layout()
        plt.savefig(save_path)
        plt.show()



def accuracy(output, label):
    cnt = label.shape[0]
    true_count = (output == label).sum()
    now_accuracy = true_count / cnt
    return now_accuracy, cnt


def get_roc_auc(all_preds, all_labels):
    one_hot = label_to_one_hot(all_labels, all_preds)

    fpr = {}
    tpr = {}
    roc_auc = np.zeros([all_preds.shape[1]])
    for i in range(all_preds.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(one_hot[:, i], all_preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return roc_auc

def label_to_one_hot(label, num_class):
    one_hot = F.one_hot(torch.from_numpy(label).long(), num_class).float()
    one_hot = one_hot.numpy()

    return one_hot