
import os
import sys

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

import pickle
import datetime
import logging
import numpy as np
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from metrics.base_metrics_class import Recorder
from torch.optim.swa_utils import AveragedModel, SWALR
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn import metrics
from metrics.utils import get_test_metrics

FFpp_pool = ['FaceForensics++', 'FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer(object):
    def __init__(
            self,
            config,
            model,
            optimizer,
            scheduler,
            logger,
            metric_scoring='auc',
            time_now=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
            swa_model=None
    ):
        if config is None or model is None or optimizer is None or logger is None:
            raise ValueError("config, model, optimizier, logger, and tensorboard writer must be implemented")

        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.swa_model = swa_model
        self.writers = {}
        self.logger = logger
        self.metric_scoring = metric_scoring
        self.best_metrics_all_time = defaultdict(
            lambda: defaultdict(lambda: float('-inf') if self.metric_scoring != 'eer' else float('inf'))
        )
        self.speed_up()
        self.timenow = time_now

        # 核心修正：将全局变量FFpp_pool赋值为实例属性（必须在__init__中尽早定义）
        self.FFpp_pool = FFpp_pool  # 确保self.FFpp_pool存在

        # 判断是否为多分类任务（基于配置文件中的类别数）
        self.is_multiclass = self.config.get('num_classes', 2) > 2

        if 'task_target' not in config:
            self.log_dir = os.path.join(
                self.config['log_dir'],
                self.config['model_name'] + '_' + self.timenow
            )
        else:
            task_str = f"_{config['task_target']}" if config['task_target'] is not None else ""
            self.log_dir = os.path.join(
                self.config['log_dir'],
                self.config['model_name'] + task_str + '_' + self.timenow
            )
        os.makedirs(self.log_dir, exist_ok=True)

    def get_writer(self, phase, dataset_key, metric_key):
        writer_key = f"{phase}-{dataset_key}-{metric_key}"
        if writer_key not in self.writers:
            writer_path = os.path.join(
                self.log_dir, phase, dataset_key, metric_key, "metric_board"
            )
            os.makedirs(writer_path, exist_ok=True)
            self.writers[writer_key] = SummaryWriter(writer_path)
        return self.writers[writer_key]

    def speed_up(self):
        self.model.to(device)
        self.model.device = device
        if self.config['ddp']:
            num_gpus = torch.cuda.device_count()
            print(f'avai gpus: {num_gpus}')
            self.model = DDP(
                self.model,
                device_ids=[self.config['local_rank']],
                find_unused_parameters=True,
                output_device=self.config['local_rank']
            )

    def setTrain(self):
        self.model.train()
        self.train = True

    def setEval(self):
        self.model.eval()
        self.train = False

    def load_ckpt(self, model_path):
        if os.path.isfile(model_path):
            saved = torch.load(model_path, map_location='cpu')
            suffix = model_path.split('.')[-1]
            if suffix == 'p':
                self.model.load_state_dict(saved.state_dict())
            else:
                self.model.load_state_dict(saved)
            self.logger.info(f'Model found in {model_path}')
        else:
            raise NotImplementedError(f"=> no model found at '{model_path}'")

    def save_ckpt(self, phase, dataset_key, ckpt_info=None):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"ckpt_best.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        if self.config['ddp']:
            torch.save(self.model.state_dict(), save_path)
        else:
            if 'svdd' in self.config['model_name']:
                torch.save({
                    'R': self.model.R,
                    'c': self.model.c,
                    'state_dict': self.model.state_dict(),
                }, save_path)
            else:
                torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"Checkpoint saved to {save_path}, current ckpt is {ckpt_info}")

    def save_swa_ckpt(self):
        save_dir = self.log_dir
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"swa.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        torch.save(self.swa_model.state_dict(), save_path)
        self.logger.info(f"SWA Checkpoint saved to {save_path}")

    def save_feat(self, phase, fea, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        feat_name = f"feat_best.npy"
        save_path = os.path.join(save_dir, feat_name)
        np.save(save_path, fea)
        self.logger.info(f"Feature saved to {save_path}")

    def save_data_dict(self, phase, data_dict, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'data_dict_{phase}.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(data_dict, file)
        self.logger.info(f"data_dict saved to {file_path}")

    def save_metrics(self, phase, metric_one_dataset, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, 'metric_dict_best.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(metric_one_dataset, file)
        self.logger.info(f"Metrics saved to {file_path}")

    def train_step(self, data_dict):
        if self.config['optimizer']['type'] == 'sam':
            for i in range(2):
                predictions = self.model(data_dict)
                losses = self.model.get_losses(data_dict, predictions)
                if i == 0:
                    pred_first = predictions
                    losses_first = losses
                self.optimizer.zero_grad()
                losses['overall'].backward()
                if i == 0:
                    self.optimizer.first_step(zero_grad=True)
                else:
                    self.optimizer.second_step(zero_grad=True)
            return losses_first, pred_first
        else:
            predictions = self.model(data_dict)
            if type(self.model) is DDP:
                losses = self.model.module.get_losses(data_dict, predictions)
            else:
                losses = self.model.get_losses(data_dict, predictions)
            self.optimizer.zero_grad()
            losses['overall'].backward()
            self.optimizer.step()
            return losses, predictions

    def train_epoch(
            self,
            epoch,
            train_data_loader,
            test_data_loaders=None,
    ):
        self.logger.info(f"===> Epoch[{epoch}] start!")
        if epoch >= 1:
            times_per_epoch = 2
        else:
            times_per_epoch = 1

        test_step = len(train_data_loader) // times_per_epoch
        step_cnt = epoch * len(train_data_loader)

        data_dict = train_data_loader.dataset.data_dict
        self.save_data_dict('train', data_dict, ','.join(self.config['train_dataset']))

        train_recorder_loss = defaultdict(Recorder)
        train_recorder_metric = defaultdict(Recorder)

        for iteration, data_dict in tqdm(enumerate(train_data_loader), total=len(train_data_loader)):
            self.setTrain()
            tensor_keys = ['image', 'label', 'landmark', 'mask', 'feat']  # 根据数据集实际键调整
            for key in data_dict.keys():
                if key in tensor_keys and data_dict[key] is not None:
                    if isinstance(data_dict[key], torch.Tensor):
                        data_dict[key] = data_dict[key].to(device)
                    elif isinstance(data_dict[key], list) and all(isinstance(x, torch.Tensor) for x in data_dict[key]):
                        data_dict[key] = torch.stack(data_dict[key]).to(device)
                    else:
                        self.logger.warning(f"Skipping key {key} as it is not a tensor or list of tensors")

            losses, predictions = self.train_step(data_dict)

            if 'SWA' in self.config and self.config['SWA'] and epoch > self.config['swa_start']:
                self.swa_model.update_parameters(self.model)

            if type(self.model) is DDP:
                batch_metrics = self.model.module.get_train_metrics(data_dict, predictions)
            else:
                batch_metrics = self.model.get_train_metrics(data_dict, predictions)

            for name, value in batch_metrics.items():
                train_recorder_metric[name].update(value)
            for name, value in losses.items():
                train_recorder_loss[name].update(value)

            if iteration % 300 == 0 and self.config['local_rank'] == 0:
                if self.config['SWA'] and (epoch > self.config['swa_start'] or self.config['dry_run']):
                    self.scheduler.step()

                loss_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_loss.items():
                    v_avg = v.average()
                    if v_avg is None:
                        loss_str += f"training-loss, {k}: not calculated    "
                        continue
                    loss_str += f"training-loss, {k}: {v_avg:.4f}    "
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_loss/{k}', v_avg, global_step=step_cnt)
                self.logger.info(loss_str)

                metric_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_metric.items():
                    v_avg = v.average()
                    if v_avg is None:
                        metric_str += f"training-metric, {k}: not calculated    "
                        continue
                    metric_str += f"training-metric, {k}: {v_avg:.4f}    "
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_metric/{k}', v_avg, global_step=step_cnt)
                self.logger.info(metric_str)

                # 清空 recorder（仅记录当前 300 样本的平均值）
                for recorder in train_recorder_loss.values():
                    recorder.clear()
                for recorder in train_recorder_metric.values():
                    recorder.clear()

            if (step_cnt + 1) % test_step == 0:
                if test_data_loaders is not None:
                    if not self.config['ddp'] or (self.config['ddp'] and dist.get_rank() == 0):
                        self.logger.info("===> Test start!")
                        self.test_epoch(epoch, iteration, test_data_loaders, step_cnt)
            step_cnt += 1
        return

    def test_one_dataset(self, data_loader):
        test_recorder_loss = defaultdict(Recorder)
        prediction_lists = []
        feature_lists = []
        label_lists = []
        for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
            if 'label_spe' in data_dict:
                data_dict.pop('label_spe')  # 移除特定标签（如二分类的辅助标签）

            tensor_keys = ['image', 'label', 'landmark', 'mask', 'feat']
            for key in data_dict.keys():
                if key in tensor_keys and data_dict[key] is not None:
                    if isinstance(data_dict[key], torch.Tensor):
                        data_dict[key] = data_dict[key].to(device)
                    elif isinstance(data_dict[key], list) and all(isinstance(x, torch.Tensor) for x in data_dict[key]):
                        data_dict[key] = torch.stack(data_dict[key]).to(device)
                    else:
                        self.logger.warning(f"Skipping key {key} as it is not a tensor or list of tensors")

            with torch.no_grad():
                predictions = self.model(data_dict, inference=True)

            if self.is_multiclass:
                pred_probs = predictions['prob'].cpu().detach().numpy()
                prediction_lists.extend(pred_probs)
            else:
                # 二分类：预测为单个概率值
                pred_scores = predictions['prob'].cpu().detach().numpy()
                prediction_lists.extend(pred_scores)

            label_lists.extend(data_dict['label'].cpu().detach().numpy())
            feature_lists.extend(predictions['feat'].cpu().detach().numpy())

            if type(self.model) is not AveragedModel:
                if type(self.model) is DDP:
                    losses = self.model.module.get_losses(data_dict, predictions)
                else:
                    losses = self.model.get_losses(data_dict, predictions)
                for name, value in losses.items():
                    test_recorder_loss[name].update(value)

        return test_recorder_loss, np.array(prediction_lists), np.array(label_lists), np.array(feature_lists)

    def save_best(self, epoch, iteration, step, losses_one_dataset_recorder, key, metric_one_dataset):
        # 获取配置中的多分类标志（需在Trainer类初始化时从config中读取）
        is_multiclass = self.config.get('is_multiclass', False)  # 假设配置中有该字段，如多分类设为True

        # 根据任务类型定义有效指标
        if is_multiclass:
            valid_metrics = ['acc', 'macro_f1', 'micro_f1', 'weighted_f1', 'ap']  # 扩展多分类指标
            metric_type = "多分类"
        else:
            valid_metrics = ['acc', 'auc', 'eer', 'ap', 'video_auc', 'video_eer']  # 保留二分类指标
            metric_type = "二分类"

        # 校验指标是否符合当前任务类型
        if self.metric_scoring not in valid_metrics:
            raise ValueError(f"{metric_type}场景下 metric_scoring 必须为 {valid_metrics} 之一")

        # 获取当前指标的最佳值（处理eer等需要最小值的指标）
        if self.metric_scoring == 'eer':
            best_metric = self.best_metrics_all_time[key].get(self.metric_scoring, float('inf'))
            improved = metric_one_dataset[self.metric_scoring] < best_metric
        else:
            best_metric = self.best_metrics_all_time[key].get(self.metric_scoring, float('-inf'))
            improved = metric_one_dataset[self.metric_scoring] > best_metric

        # 打印核心指标
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"[指标监控] 数据集: {key} | 阶段: Epoch[{epoch}]-Iter[{iteration}]")
        self.logger.info(f"当前最佳 {self.metric_scoring}: {best_metric:.4f}")
        self.logger.info(f"当前 {self.metric_scoring}: {metric_one_dataset[self.metric_scoring]:.4f}")
        self.logger.info(f"{'='*50}")

        if improved:
            # 更新最佳指标记录
            old_best = best_metric
            self.best_metrics_all_time[key][self.metric_scoring] = metric_one_dataset[self.metric_scoring]
            if key == 'avg':
                self.best_metrics_all_time[key]['dataset_dict'] = metric_one_dataset.get('dataset_dict', {})

            # 打印指标提升信息
            improvement = abs(old_best - metric_one_dataset[self.metric_scoring])
            self.logger.info(f"[指标提升] {key} 数据集的 {self.metric_scoring} 提升了 {improvement:.4f}")
            self.logger.info(f"[新最佳值] {self.metric_scoring}: {metric_one_dataset[self.metric_scoring]:.4f}")

            # 保存检查点（排除特定数据集，如FFpp_pool）
            if self.config['save_ckpt'] and key not in self.FFpp_pool:
                ckpt_name = f"best_epoch{epoch}_iter{iteration}"
                self.save_ckpt('test', key, ckpt_name)
                self.logger.info(f"[检查点保存] 已保存最佳模型至 {ckpt_name}")

            # 保存指标到文件或日志
            self.save_metrics('test', metric_one_dataset, key)
            self.logger.info(f"[指标保存] 已保存完整指标到 {key} 文件夹")

        # 记录损失和指标到日志及TensorBoard
        if losses_one_dataset_recorder is not None:
            loss_str = f"[损失指标] Dataset: {key} | Step: {step} | "
            for loss_name, loss_recorder in losses_one_dataset_recorder.items():
                avg_loss = loss_recorder.average()
                if avg_loss is not None:
                    loss_str += f"{loss_name}: {avg_loss:.4f} | "
                    writer = self.get_writer('test', key, loss_name)
                    writer.add_scalar(f'test_losses/{loss_name}', avg_loss, global_step=step)
            self.logger.info(loss_str.rstrip(' | '))

        # 打印所有可用指标（按重要性排序）
        metric_str = f"[评估指标] Dataset: {key} | Step: {step} | "
        priority_metrics = ['acc', 'auc', 'eer', 'ap', 'macro_f1', 'micro_f1', 'weighted_f1']
        for metric_name in priority_metrics:
            if metric_name in metric_one_dataset:
                metric_str += f"{metric_name}: {metric_one_dataset[metric_name]:.4f} | "
                writer = self.get_writer('test', key, metric_name)
                writer.add_scalar(f'test_metrics/{metric_name}', metric_one_dataset[metric_name], global_step=step)

        # 打印其他指标
        for metric_name, metric_value in metric_one_dataset.items():
            if metric_name not in priority_metrics and metric_name not in ['pred', 'label', 'dataset_dict']:
                metric_str += f"{metric_name}: {metric_value:.4f} | "
                writer = self.get_writer('test', key, metric_name)
                writer.add_scalar(f'test_metrics/{metric_name}', metric_value, global_step=step)

        self.logger.info(metric_str.rstrip(' | '))
        self.logger.info(f"{'='*50}\n")
    def test_epoch(self, epoch, iteration, test_data_loaders, step):
        self.setEval()

        # 根据是否为多分类任务初始化不同的平均指标字典
        if self.is_multiclass:
            avg_metric = {
                'acc': 0,
                'macro_f1': 0,
                'ap': 0,  # 多分类平均精度
                'video_acc': 0,  # 多分类视频准确率
                'video_macro_f1': 0,  # 多分类视频宏平均F1
                'dataset_dict': {}
            }
        else:
            # 二分类支持的指标
            avg_metric = {'acc': 0, 'auc': 0, 'eer': 0, 'ap': 0, 'video_auc': 0, 'video_eer': 0, 'dataset_dict': {}}

        keys = test_data_loaders.keys()

        for key in keys:
            data_dict = test_data_loaders[key].dataset.data_dict
            self.save_data_dict('test', data_dict, key)

            losses_one_dataset_recorder, predictions_nps, label_nps, feature_nps = self.test_one_dataset(
                test_data_loaders[key]
            )

            # 计算指标，传递is_multiclass参数
            metric_one_dataset = get_test_metrics(
                y_pred=predictions_nps,
                y_true=label_nps,
                img_names=data_dict.get('image', []),
                is_multiclass=self.is_multiclass  # 传递多分类标志
            )

            # 更新平均指标
            for metric_name, value in metric_one_dataset.items():
                if metric_name not in avg_metric or metric_name == 'dataset_dict':
                    continue

                # 确保只累加当前任务类型支持的指标
                if self.is_multiclass:
                    if metric_name.startswith('video_') and 'auc' in metric_name:
                        continue  # 跳过多分类中不存在的 video_auc
                else:
                    if metric_name.startswith('video_') and ('acc' in metric_name or 'macro_f1' in metric_name):
                        continue  # 跳过二分类中不存在的 video_acc/video_macro_f1

                avg_metric[metric_name] += value

            # 保存最佳模型和指标
            if type(self.model) is not AveragedModel:
                self.save_best(epoch, iteration, step, losses_one_dataset_recorder, key, metric_one_dataset)
            if self.config.get('save_feat', False):
                self.save_feat('test', feature_nps, key)

        # 计算平均指标（宏平均）
        if len(keys) > 0 and self.config.get('save_avg', False):
            for key in avg_metric:
                if key != 'dataset_dict':
                    avg_metric[key] /= len(keys)
            self.save_best(epoch, iteration, step, None, 'avg', avg_metric)

        self.logger.info('===> Test Done!')
        return self.best_metrics_all_time

    @torch.no_grad()
    def inference(self, data_dict):
        # 确保数据在正确的设备上
        tensor_keys = ['image', 'label', 'landmark', 'mask', 'feat']
        for key in data_dict.keys():
            if key in tensor_keys and data_dict[key] is not None:
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].to(device)
                elif isinstance(data_dict[key], list) and all(isinstance(x, torch.Tensor) for x in data_dict[key]):
                    data_dict[key] = torch.stack(data_dict[key]).to(device)

        self.setEval()
        predictions = self.model(data_dict, inference=True)
        # 处理多分类输出
        if self.is_multiclass:
            # 返回每个类别的概率和预测类别
            probs = predictions['prob'].cpu().detach()
            pred_classes = torch.argmax(probs, dim=1)
            predictions['pred_class'] = pred_classes
            predictions['prob'] = probs
        return predictions