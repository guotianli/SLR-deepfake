import numpy as np
from sklearn import metrics
from sklearn.preprocessing import label_binarize


def parse_metric_for_print(metric_dict):
    if metric_dict is None:
        return "\n"
    str = "\n"
    str += "================================ Each dataset best metric ================================ \n"
    for key, value in metric_dict.items():
        if key != 'avg':
            str = str + f"| {key}: "
            for k, v in value.items():
                # 格式化数值保留4位小数
                if isinstance(v, (int, float)):
                    v = f"{v:.4f}" if isinstance(v, float) else str(v)
                str = str + f" {k}={v} "
            str = str + "| \n"
        else:
            str += "============================================================================================= \n"
            str += "================================== Average best metric ====================================== \n"
            avg_dict = value
            for avg_key, avg_value in avg_dict.items():
                if avg_key == 'dataset_dict':
                    for key, value in avg_value.items():
                        str = str + f"| {key}: {value} | \n"
                else:
                    if isinstance(avg_value, (int, float)):
                        avg_value = f"{avg_value:.4f}" if isinstance(avg_value, float) else str(avg_value)
                    str = str + f"| avg {avg_key}: {avg_value} | \n"
    str += "============================================================================================="
    return str


def get_test_metrics(y_pred, y_true, img_names=None, is_multiclass=None):
    """
    计算测试指标，支持二分类和多分类情况

    Args:
        y_pred: 预测分数数组或概率矩阵
        y_true: 真实标签数组
        img_names: 图像路径列表（用于视频级别指标计算）
        is_multiclass: 是否为多分类任务（None表示自动检测）
    """
    y_pred = np.array(y_pred).squeeze()
    y_true = np.array(y_true).squeeze()
    results = {}

    # 自动检测任务类型
    if is_multiclass is None:
        unique_classes = np.unique(y_true)
        is_multiclass = (len(unique_classes) > 2) or (y_pred.ndim > 1 and y_pred.shape[1] > 2)

    print(f"[INFO] 计算指标: 任务类型 = {'多分类' if is_multiclass else '二分类'}")

    # 二分类情况
    if not is_multiclass:
        # 确保y_pred是概率分数
        if y_pred.ndim > 1:
            y_pred = y_pred[:, 1]  # 假设第二列是正类的概率

        # 计算二分类指标
        prediction_class = (y_pred > 0.5).astype(int)
        acc = metrics.accuracy_score(y_true, prediction_class)
        f1 = metrics.f1_score(y_true, prediction_class, average='binary')

        # 计算AUC和EER
        if len(np.unique(y_true)) > 1:
            auc = metrics.roc_auc_score(y_true, y_pred)
            # 计算EER
            fpr, tpr, _ = metrics.roc_curve(y_true, y_pred, pos_label=1)
            fnr = 1 - tpr
            eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
        else:
            auc = 0.5
            eer = 0.5

        # 计算AP
        ap = metrics.average_precision_score(y_true, y_pred)

        results.update({'acc': acc, 'f1': f1, 'auc': auc, 'eer': eer, 'ap': ap})

        # 视频级别指标
        if img_names and isinstance(img_names[0], list):
            video_auc, video_eer = _get_video_metrics_binary_correct_path(img_names, y_pred, y_true)
            results.update({'video_auc': video_auc, 'video_eer': video_eer})

        results.update({'pred': y_pred, 'label': y_true, 'is_multiclass': False})

    # 多分类情况
    else:
        # 确保y_pred是概率矩阵
        if y_pred.ndim == 1:
            unique_classes = np.unique(y_true)
            y_pred = label_binarize(y_pred, classes=unique_classes)

        # 多分类指标计算
        prediction_class = np.argmax(y_pred, axis=1)
        acc = metrics.accuracy_score(y_true, prediction_class)
        macro_f1 = metrics.f1_score(y_true, prediction_class, average='macro')

        # 多分类AP计算
        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
        ap_scores = []
        for c in range(y_pred.shape[1]):
            if np.sum(y_true_bin[:, c]) > 0:
                ap = metrics.average_precision_score(y_true_bin[:, c], y_pred[:, c])
                ap_scores.append(ap)
        ap = np.mean(ap_scores) if ap_scores else 0.0

        results.update({'acc': acc, 'macro_f1': macro_f1, 'ap': ap})

        # 视频级别指标
        if img_names and isinstance(img_names[0], list):
            video_metrics = _get_video_metrics_multiclass(img_names, prediction_class, y_true)
            results.update(video_metrics)

        results.update({'pred': y_pred, 'label': y_true, 'is_multiclass': True})

    return results


def _get_video_metrics_multiclass(image_paths, prediction_class, y_true):
    '''处理多分类视频级别指标'''
    # 创建视频名称到预测和真实标签的映射
    video_to_preds = {}
    video_to_labels = {}

    for path, pred, label in zip(image_paths, prediction_class, y_true):
        # 提取视频名称（根据实际路径格式调整）
        parts = path.split('/') if '/' in path else path.split('\\')
        video_name = parts[-2] if len(parts) > 1 else parts[-1]  # 通常取倒数第二部分作为视频名

        if video_name not in video_to_preds:
            video_to_preds[video_name] = []
            video_to_labels[video_name] = []

        video_to_preds[video_name].append(pred)
        video_to_labels[video_name].append(label)

    # 计算每个视频的指标
    video_metrics = {}
    for video_name in video_to_preds:
        preds = np.array(video_to_preds[video_name])
        labels = np.array(video_to_labels[video_name])

        # 多数表决确定视频级别预测
        video_pred = np.bincount(preds).argmax()
        # 假设所有帧的标签相同，取第一个
        video_true = labels[0]

        # 计算视频级别指标
        video_acc = 1.0 if video_pred == video_true else 0.0
        # 注意：视频级别F1计算需要多个视频样本，这里简化处理
        video_macro_f1 = metrics.f1_score(labels, preds, average='macro') if len(np.unique(labels)) > 1 else 1.0

        video_metrics[video_name] = {
            'video_acc': video_acc,
            'video_macro_f1': video_macro_f1
        }

    # 计算所有视频的平均指标
    avg_video_acc = np.mean([m['video_acc'] for m in video_metrics.values()])
    avg_video_macro_f1 = np.mean([m['video_macro_f1'] for m in video_metrics.values()])

    return {
        'video_acc': avg_video_acc,
        'video_macro_f1': avg_video_macro_f1
    }


def _get_video_metrics_binary_correct_path(image_paths, pred_scores, labels):
    '''处理二分类视频级别指标'''
    # 创建视频名称到预测分数和真实标签的映射
    video_to_scores = {}
    video_to_labels = {}

    for path, score, label in zip(image_paths, pred_scores, labels):
        # 提取视频名称（根据实际路径格式调整）
        parts = path.split('/') if '/' in path else path.split('\\')
        video_name = parts[-2] if len(parts) > 1 else parts[-1]  # 通常取倒数第二部分作为视频名

        if video_name not in video_to_scores:
            video_to_scores[video_name] = []
            video_to_labels[video_name] = []

        video_to_scores[video_name].append(score)
        video_to_labels[video_name].append(label)

    # 计算每个视频的指标
    video_metrics = {}
    for video_name in video_to_scores:
        scores = np.array(video_to_scores[video_name])
        labels = np.array(video_to_labels[video_name])

        # 视频级别预测（平均分数）
        video_score = np.mean(scores)
        # 视频真实标签（假设所有帧标签相同）
        video_label = labels[0]

        # 计算视频级别AUC
        if len(np.unique(labels)) > 1:
            video_auc = metrics.roc_auc_score(labels, scores)
        else:
            video_auc = 0.5  # 只有一类样本时AUC为0.5

        # 计算视频级别EER
        if len(np.unique(labels)) > 1:
            # 使用sklearn 1.2+中的metrics.eer函数
            try:
                video_eer = metrics.eer(labels, scores)[0]
            except:
                # 旧版本sklearn没有eer函数，使用自定义实现
                fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
                fnr = 1 - tpr
                video_eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
        else:
            video_eer = 0.5

        video_metrics[video_name] = {
            'video_auc': video_auc,
            'video_eer': video_eer
        }

    # 计算所有视频的平均指标
    avg_video_auc = np.mean([m['video_auc'] for m in video_metrics.values()])
    avg_video_eer = np.mean([m['video_eer'] for m in video_metrics.values()])

    return avg_video_auc, avg_video_eer