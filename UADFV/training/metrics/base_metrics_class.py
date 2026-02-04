import numpy as np
from sklearn import metrics
from collections import defaultdict
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import label_binarize

def get_accracy(output, label):
    _, prediction = torch.max(output, 1)    # argmax
    correct = (prediction == label).sum().item()
    accuracy = correct / prediction.size(0)
    return accuracy


def get_prediction(output, label):
    prob = nn.functional.softmax(output, dim=1)[:, 1]
    prob = prob.view(prob.size(0), 1)
    label = label.view(label.size(0), 1)
    #print(prob.size(), label.size())
    datas = torch.cat((prob, label.float()), dim=1)
    return datas


def calculate_metrics_for_train(label, pred):
    """
    计算训练指标，完整支持二分类、多分类及单类别场景，增强异常处理
    """
    # 转换为numpy数组
    y_true = label.cpu().numpy()
    y_pred = pred.cpu().numpy()

    # 自动检测任务类型
    is_multiclass = False
    if y_pred.ndim > 1:
        num_classes = y_pred.shape[1]
        is_multiclass = num_classes > 2
    else:
        unique_classes = np.unique(y_true)
        is_multiclass = len(unique_classes) > 2

    print(f"[TRAIN METRICS] 任务类型: {'多分类' if is_multiclass else '二分类'}")
    print(f"[DEBUG] 标签分布: {np.bincount(y_true)}")

    if not is_multiclass:
        # 二分类情况
        if y_pred.ndim > 1:
            y_pred = y_pred[:, 1]  # 提取正类概率

        pred_class = (y_pred > 0.5).astype(int)
        acc = metrics.accuracy_score(y_true, pred_class)

        # 处理单类别情况
        unique_y_true = np.unique(y_true)
        if len(unique_y_true) <= 1:
            print("[WARNING] 二分类场景中只有一个类别，AUC/EER/AP设为0.5")
            auc = 0.5
            eer = 0.5
            ap = 0.5
        else:
            try:
                auc = metrics.roc_auc_score(y_true, y_pred)
                fpr, tpr, _ = metrics.roc_curve(y_true, y_pred, pos_label=1)
                fnr = 1 - tpr
                eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
                ap = metrics.average_precision_score(y_true, y_pred)
            except ValueError as e:
                if "Only one class present" in str(e):
                    print("[WARNING] 二分类AUC计算时发现单类别，设为0.5")
                    auc = 0.5
                    eer = 0.5
                    ap = 0.5
                else:
                    raise

    else:
        # 多分类情况
        num_classes = y_pred.shape[1]
        valid_classes = np.arange(num_classes)
        y_true_clamped = np.clip(y_true, 0, num_classes - 1)
        y_true_bin = label_binarize(y_true_clamped, classes=valid_classes)

        pred_class = np.argmax(y_pred, axis=1)
        acc = metrics.accuracy_score(y_true, pred_class)

        eer = 0.0  # 多分类不计算EER

        # 处理单类别情况
        unique_y_true = np.unique(y_true)
        if len(unique_y_true) <= 1:
            print("[WARNING] 多分类场景中只有一个类别，AUC/AP设为0.5")
            auc = 0.5
            ap = 0.5
        else:
            # 确保y_true_bin和y_pred的形状匹配
            if y_true_bin.shape[1] != y_pred.shape[1]:
                raise ValueError(f"类别数不匹配: y_true_bin有{y_true_bin.shape[1]}类, y_pred有{y_pred.shape[1]}类")

            try:
                auc = metrics.roc_auc_score(y_true_bin, y_pred, multi_class='ovr')
            except ValueError as e:
                if "Only one class present" in str(e):
                    print("[WARNING] 多分类AUC计算时发现单类别，设为0.5")
                    auc = 0.5
                else:
                    raise

            # 计算AP并处理单类别子问题
            ap_scores = []
            for c in range(y_pred.shape[1]):
                y_true_c = y_true_bin[:, c]
                y_pred_c = y_pred[:, c]
                if np.sum(y_true_c) > 0 and np.sum(1 - y_true_c) > 0:
                    try:
                        ap = metrics.average_precision_score(y_true_c, y_pred_c)
                        ap_scores.append(ap)
                    except ValueError as e:
                        if "Only one class present" in str(e):
                            print(f"[WARNING] 类别{c}只有单类别，AP设为0.5")
                            ap_scores.append(0.5)
                        else:
                            raise
                else:
                    print(f"[WARNING] 类别{c}样本不均衡，AP设为0.5")
                    ap_scores.append(0.5)

            ap = np.mean(ap_scores)

    return auc, eer, acc, ap
# ------------ compute average metrics of batches---------------------
class Metrics_batch():
    def __init__(self):
        self.tprs = []
        self.mean_fpr = np.linspace(0, 1, 100)
        self.aucs = []
        self.eers = []
        self.aps = []

        self.correct = 0
        self.total = 0
        self.losses = []

    def update(self, label, output):
        acc = self._update_acc(label, output)
        if output.size(1) == 2:
            prob = torch.softmax(output, dim=1)[:, 1]
        else:
            prob = output
        #label = 1-label
        #prob = torch.softmax(output, dim=1)[:, 1]
        auc, eer = self._update_auc(label, prob)
        ap = self._update_ap(label, prob)

        return acc, auc, eer, ap

    def _update_auc(self, lab, prob):
        fpr, tpr, thresholds = metrics.roc_curve(lab.squeeze().cpu().numpy(),
                                                 prob.squeeze().cpu().numpy(),
                                                 pos_label=1)
        if np.isnan(fpr[0]) or np.isnan(tpr[0]):
            return -1, -1

        auc = metrics.auc(fpr, tpr)
        interp_tpr = np.interp(self.mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        self.tprs.append(interp_tpr)
        self.aucs.append(auc)

        # return auc

        # EER
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        self.eers.append(eer)

        return auc, eer

    def _update_acc(self, lab, output):
        _, prediction = torch.max(output, 1)    # argmax
        correct = (prediction == lab).sum().item()
        accuracy = correct / prediction.size(0)
        # self.accs.append(accuracy)
        self.correct = self.correct+correct
        self.total = self.total+lab.size(0)
        return accuracy

    def _update_ap(self, label, prob):
        y_true = label.cpu().detach().numpy()
        y_pred = prob.cpu().detach().numpy()
        ap = metrics.average_precision_score(y_true,y_pred)
        self.aps.append(ap)

        return np.mean(ap)

    def get_mean_metrics(self):
        mean_acc, std_acc = self.correct/self.total, 0
        mean_auc, std_auc = self._mean_auc()
        mean_err, std_err = np.mean(self.eers), np.std(self.eers)
        mean_ap, std_ap = np.mean(self.aps), np.std(self.aps)
        
        return {'acc':mean_acc, 'auc':mean_auc, 'eer':mean_err, 'ap':mean_ap}

    def _mean_auc(self):
        mean_tpr = np.mean(self.tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(self.mean_fpr, mean_tpr)
        std_auc = np.std(self.aucs)
        return mean_auc, std_auc

    def clear(self):
        self.tprs.clear()
        self.aucs.clear()
        # self.accs.clear()
        self.correct=0
        self.total=0
        self.eers.clear()
        self.aps.clear()
        self.losses.clear()


# ------------ compute average metrics of all data ---------------------
class Metrics_all():
    def __init__(self):
        self.probs = []
        self.labels = []
        self.correct = 0
        self.total = 0

    def store(self, label, output):
        prob = torch.softmax(output, dim=1)[:, 1]
        _, prediction = torch.max(output, 1)    # argmax
        correct = (prediction == label).sum().item()
        self.correct += correct
        self.total += label.size(0)
        self.labels.append(label.squeeze().cpu().numpy())
        self.probs.append(prob.squeeze().cpu().numpy())

    def get_metrics(self):
        y_pred = np.concatenate(self.probs)
        y_true = np.concatenate(self.labels)
        # auc
        fpr, tpr, thresholds = metrics.roc_curve(y_true,y_pred,pos_label=1)
        auc = metrics.auc(fpr, tpr)
        # eer
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        # ap
        ap = metrics.average_precision_score(y_true,y_pred)
        # acc
        acc = self.correct / self.total
        return {'acc':acc, 'auc':auc, 'eer':eer, 'ap':ap}

    def clear(self):
        self.probs.clear()
        self.labels.clear()
        self.correct = 0
        self.total = 0


# only used to record a series of scalar value
class Recorder:
    def __init__(self):
        self.sum = 0
        self.num = 0
    def update(self, item, num=1):
        if item is not None:
            self.sum += item * num
            self.num += num
    def average(self):
        if self.num == 0:
            return None
        return self.sum/self.num
    def clear(self):
        self.sum = 0
        self.num = 0
