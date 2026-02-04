import os
import numpy as np
import cv2
import random
import datetime
import time
import yaml
import pickle
from tqdm import tqdm
from copy import deepcopy
from PIL import Image as pil_image
from metrics.utils import get_test_metrics
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from dataset.ff_blend import FFBlendDataset
from dataset.fwa_blend import FWABlendDataset
from dataset.pair_dataset import pairDataset

from trainer.trainer import Trainer
from detectors import DETECTOR
from metrics.base_metrics_class import Recorder
from collections import defaultdict
import pandas as pd
import argparse
from logger import create_logger
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str,
                    default='/data/Btask/DeepfakeBench/training/config/detector/srm.yaml',
                    help='path to detector YAML file')
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--weights_path', type=str,
                    default='/data/Btask/DeepfakeBench/training/weights/srm_best.pth')
#parser.add_argument("--lmdb", action='store_true', default=False)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

on_2060 = "2060" in torch.cuda.get_device_name()


def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        test_set = DeepfakeAbstractBaseDataset(
            config=config,
            mode='test',
        )
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=config['test_batchSize'],
                shuffle=False,
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last=False
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def generate_batch_data(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()  # 确保模型处于评估模式

    for batch_idx, data_dict in enumerate(data_loader):
        # 初始化存储列表
        prediction_list = []
        feature_list = []
        label_list = []
        image_paths_list = []

        # 直接获取图像路径（无需遍历张量）
        image_paths = data_dict['image_path']
        # 处理路径格式（例如多帧情况）
        for paths in image_paths:
            if isinstance(paths, list):
                image_paths_list.append(paths[0])  # 取第一个路径（按需调整）
            else:
                image_paths_list.append(paths)

        # 数据移动到 GPU
        data_dict['image'] = data_dict['image'].to(device)
        data_dict['label'] = data_dict['label'].to(device)
        if 'landmark' in data_dict and data_dict['landmark'] is not None:
            data_dict['landmark'] = data_dict['landmark'].to(device)
        if 'mask' in data_dict and data_dict['mask'] is not None:
            data_dict['mask'] = data_dict['mask'].to(device)

        # 模型推理（无梯度计算）
        with torch.no_grad():
            predictions = model(data_dict)

        # 收集结果
        label_list.extend(data_dict['label'].cpu().numpy())
        prediction_list.extend(predictions['prob'].cpu().numpy())
        feature_list.extend(predictions['feat'].cpu().numpy())

        # 保存批次数据（包含路径）
        batch_data = {
            'predictions': np.array(prediction_list),
            'labels': np.array(label_list),
            'features': np.array(feature_list),
            'image_paths': np.array(image_paths_list, dtype=object)  # 允许保存字符串列表
        }
        np.save(f'npy/0501_srm/batch_{batch_idx}.npy', batch_data)

        yield batch_data['predictions'], batch_data['labels'], batch_data['features']



def test_one_dataset(model, data_loader):
    for batch_data in generate_batch_data(model, data_loader):
        yield batch_data


def test_epoch(model, test_data_loaders):
    model.eval()
    metrics_all_datasets = {}

    for key, data_loader in test_data_loaders.items():
        data_dict = data_loader.dataset.data_dict

        prediction_lists = []
        feature_lists = []
        label_lists = []

        # Iterate through batches
        for predictions_nps, label_nps, feat_nps in generate_batch_data(model, data_loader):
            prediction_lists.extend(predictions_nps)
            label_lists.extend(label_nps)
            feature_lists.extend(feat_nps)

        # Convert lists to numpy arrays
        prediction_arrays = np.asarray(prediction_lists)
        label_arrays = np.asarray(label_lists)

        # Compute metrics for the dataset
        metric_one_dataset = get_test_metrics(y_pred=prediction_arrays, y_true=label_arrays, img_names=data_dict['image'])

        # Save DataFrame to CSV file
        df = pd.DataFrame({
            'y_pred': prediction_lists,
            'y_true': label_lists,
            'img_names': data_dict['image']
        })
        df.to_csv(f'npy/0501_srm/{key}_metrics_srm.csv', index=False)

        metrics_all_datasets[key] = metric_one_dataset

        # Print dataset metrics
        tqdm.write(f"Dataset: {key}")
        for k, v in metric_one_dataset.items():
            tqdm.write(f"{k}: {v}")

    return metrics_all_datasets


@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions


def main():
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    if on_2060:
        config['lmdb_dir'] = r'I:\transform_2_lmdb'
        config['train_batchSize'] = 10
        config['workers'] = 0
    else:
        config['workers'] = 8
        config['lmdb_dir'] = r'./data/LMDBs'
    weights_path = None
    # If arguments are provided, they will overwrite the yaml settings
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    if args.weights_path:
        config['weights_path'] = args.weights_path
        weights_path = args.weights_path

    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)

    # prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    epoch = 0
    if weights_path:
        try:
            epoch = int(weights_path.split('/')[-1].split('.')[0].split('_')[2])
        except:
            epoch = 0
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt, strict=True)
        print('===> Load checkpoint done!')
    else:
        print('Fail to load the pre-trained weights')

    # start testing
    test_epoch(model, test_data_loaders)
    print('===> Test Done!')


if __name__ == '__main__':
    main()