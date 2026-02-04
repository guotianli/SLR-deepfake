import sys
import lmdb
import os
import math
import yaml
import glob
import json
import numpy as np
from copy import deepcopy
import cv2
import random
from PIL import Image
from collections import defaultdict
import torch
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms as T
import albumentations as A
import re
from .albu import IsotropicResize
from tqdm import tqdm  # 添加这行导入tqdm库
FFpp_pool = ['FaceForensics++', 'FaceShifter', 'DeepFakeDetection', 'FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT']


def all_in_pool(inputs, pool):
    for each in inputs:
        if each not in pool:
            return False
    return True


# abstract_dataset.py 中BalancedSampler的改进版本
class BalancedSampler(torch.utils.data.Sampler):
    """类别均衡采样器，确保每个类别至少有指定数量的样本，改进了错误处理和性能"""

    def __init__(self, data_source, num_classes, min_samples_per_class=5, replacement=True):
        self.data_source = data_source
        self.num_classes = num_classes
        self.min_samples_per_class = min_samples_per_class
        self.replacement = replacement
        self.class_indices = [[] for _ in range(num_classes)]

        # 添加进度显示，便于观察数据处理进度
        print("正在统计类别分布...")
        for idx, item in enumerate(tqdm(data_source)):
            try:
                # 从元组中获取标签（假设标签位于元组的第2个位置）
                if isinstance(item, (list, tuple)) and len(item) > 2:
                    label = item[2].item()  # 获取标签值
                else:
                    # 回退到字典方式（如果数据项是字典）
                    label = item['label'] if isinstance(item, dict) else 0
                    print(f"警告：数据项格式异常，索引{idx}，使用默认标签处理方式")

                self.class_indices[label].append(idx)
            except Exception as e:
                print(f"处理样本索引{idx}时出错: {e}")
                continue

        # 对少数类进行过采样
        print("正在进行类别均衡处理...")
        for i in range(num_classes):
            if len(self.class_indices[i]) < min_samples_per_class and replacement:
                repeats = min_samples_per_class // len(self.class_indices[i]) + 1
                self.class_indices[i] = self.class_indices[i] * repeats
                self.class_indices[i] = self.class_indices[i][:min_samples_per_class]
        print("类别均衡处理完成")

    def __iter__(self):
        indices = []
        for i in range(self.num_classes):
            indices.extend(random.sample(self.class_indices[i], self.min_samples_per_class))
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.num_classes * self.min_samples_per_class

class DeepfakeAbstractBaseDataset(data.Dataset):
    """
    Abstract base class for all deepfake datasets with balanced sampling support.
    """

    def __init__(self, config=None, mode='train'):
        """Initializes the dataset object with balanced sampling capabilities."""
        self.config = config
        self.mode = mode
        self.compression = config['compression']
        self.frame_num = config['frame_num'][mode]
        self.video_level = config.get('video_mode', False)
        self.clip_size = config.get('clip_size', None)
        self.lmdb = config.get('lmdb', False)
        self.image_list = []
        self.label_list = []

        if mode == 'train':
            dataset_list = config['train_dataset']
            image_list, label_list = [], []
            for one_data in dataset_list:
                tmp_image, tmp_label, tmp_name = self.collect_img_and_label_for_one_dataset(one_data)
                image_list.extend(tmp_image)
                label_list.extend(tmp_label)
            if self.lmdb:
                if len(dataset_list) > 1:
                    if all_in_pool(dataset_list, FFpp_pool):
                        lmdb_path = os.path.join(config['lmdb_dir'], f"FaceForensics++_lmdb")
                        self.env = lmdb.open(lmdb_path, create=False, subdir=True, readonly=True, lock=False)
                    else:
                        raise ValueError('Training with multiple dataset and lmdb is not implemented yet.')
                else:
                    lmdb_path = os.path.join(config['lmdb_dir'],
                                             f"{dataset_list[0] if dataset_list[0] not in FFpp_pool else 'FaceForensics++'}_lmdb")
                    self.env = lmdb.open(lmdb_path, create=False, subdir=True, readonly=True, lock=False)
        elif mode in ['test', 'val']:
            dataset_key = 'test_dataset' if mode == 'test' else 'val_dataset'
            one_data = config[dataset_key]
            if isinstance(one_data, list) and len(one_data) > 0:
                one_data = one_data[0]
            assert isinstance(one_data, str), f"Invalid {dataset_key}: {one_data}"
            print(f"Loading {mode} dataset: {one_data}")
            image_list, label_list, name_list = self.collect_img_and_label_for_one_dataset(one_data)
            print(f"Collected {len(image_list)} samples for {mode} dataset: {one_data}")
            if self.lmdb:
                lmdb_path = os.path.join(config['lmdb_dir'],
                                         f"{one_data}_lmdb" if one_data not in FFpp_pool else 'FaceForensics++_lmdb')
                self.env = lmdb.open(lmdb_path, create=False, subdir=True, readonly=True, lock=False)
        else:
            raise NotImplementedError('Only train, test, and val modes are supported.')

        assert len(image_list) != 0 and len(label_list) != 0, f"Collect nothing for {mode} mode!"
        self.image_list, self.label_list = image_list, label_list
        self.data_dict = {'image': self.image_list, 'label': self.label_list}
        self.transform = self.init_data_aug_method()
        self.class_frequencies = self.calculate_class_frequencies()

    def calculate_class_frequencies(self):
        """计算每个类别的样本频率，用于类别权重计算"""
        if not hasattr(self, 'label_list') or len(self.label_list) == 0:
            return None

        num_classes = self.config['backbone_config']['num_classes']
        class_counts = np.zeros(num_classes, dtype=int)

        for label in self.label_list:
            class_counts[label] += 1

        total_samples = len(self.label_list)
        # 拉普拉斯平滑处理，避免除零错误
        frequencies = (class_counts + 1) / (total_samples + num_classes)
        return frequencies.tolist()

    def init_data_aug_method(self):
        """初始化增强策略，增强数据多样性"""
        shear_limit = self.config['data_aug'].get('shear_limit', [-10, 10])
        shear_prob = self.config['data_aug'].get('shear_prob', 0.5)
        zoom_prob = self.config['data_aug'].get('zoom_prob', 0.5)

        trans = A.Compose([
            A.HorizontalFlip(p=self.config['data_aug']['flip_prob']),
            A.Rotate(limit=self.config['data_aug']['rotate_limit'], p=self.config['data_aug']['rotate_prob']),
            A.GaussianBlur(blur_limit=self.config['data_aug']['blur_limit'], p=self.config['data_aug']['blur_prob']),
            A.OneOf([
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA,
                                interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA,
                                interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_LINEAR,
                                interpolation_up=cv2.INTER_LINEAR),
            ], p=0 if self.config['with_landmark'] else 1),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=self.config['data_aug']['brightness_limit'],
                                           contrast_limit=self.config['data_aug']['contrast_limit']),
                A.FancyPCA(),
                A.HueSaturationValue()
            ], p=0.7),  # 提高数据增强应用概率
            A.ImageCompression(quality_lower=self.config['data_aug']['quality_lower'],
                               quality_upper=self.config['data_aug']['quality_upper'], p=0.5),
            # 修复：使用Affine变换替代Shear
            A.Affine(
                shear=shear_limit,
                p=shear_prob,
                mode=cv2.BORDER_CONSTANT,
                cval=0
            ),
            # 修复：使用兼容的ZoomBlur参数
            A.ZoomBlur(
                max_factor=1.5,  # 最大缩放因子
                step_factor=(0.05, 0.1),  # 缩放步长
                p=zoom_prob
            )
        ], keypoint_params=A.KeypointParams(format='xy') if self.config['with_landmark'] else None)
        return trans

    def collect_img_and_label_for_one_dataset(self, dataset_name: str):
        """收集单个数据集的图像和标签，确保类别完整"""
        label_list = []
        frame_path_list = []
        video_name_list = []

        if not os.path.exists(self.config['dataset_json_folder']):
            self.config['dataset_json_folder'] = '/data/Btask/DeepfakeBench/preprocessing/dataset_json'
        try:
            with open(os.path.join(self.config['dataset_json_folder'], dataset_name + '.json'), 'r') as f:
                dataset_info = json.load(f)
        except Exception as e:
            print(e)
            raise ValueError(f'dataset {dataset_name} not exist!')

        cp = None
        if dataset_name == 'FaceForensics++_c40':
            dataset_name = 'FaceForensics++'
            cp = 'c40'
        elif dataset_name == 'FF-DF_c40':
            dataset_name = 'FF-DF'
            cp = 'c40'
        elif dataset_name == 'FF-F2F_c40':
            dataset_name = 'FF-F2F'
            cp = 'c40'
        elif dataset_name == 'FF-FS_c40':
            dataset_name = 'FF-FS'
            cp = 'c40'
        elif dataset_name == 'FF-NT_c40':
            dataset_name = 'FF-NT'
            cp = 'c40'
        elif dataset_name == 'Roop':
            dataset_name = 'Roop'
            cp = 'c40'
        elif dataset_name == 'Race':
            dataset_name = 'Race'
            cp = 'c40'
        elif dataset_name == 'UADFV':
            dataset_name = 'UADFV'
            cp = None

        for label in dataset_info[dataset_name]:
            sub_dataset_info = dataset_info[dataset_name][label][self.mode]
            if dataset_name == 'UADFV':
                pass
            elif cp is None and dataset_name in ['FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT', 'FaceForensics++',
                                                 'DeepFakeDetection', 'FaceShifter']:
                sub_dataset_info = sub_dataset_info[self.compression]
            elif cp == 'c40' and dataset_name in ['FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT', 'FaceForensics++',
                                                  'DeepFakeDetection', 'FaceShifter']:
                sub_dataset_info = sub_dataset_info['c40']

            for video_name, video_info in sub_dataset_info.items():
                unique_video_name = video_info['label'] + '_' + video_name
                if video_info['label'] not in self.config['label_dict']:
                    raise ValueError(f'Label {video_info["label"]} is not found in the configuration file.')
                label_idx = self.config['label_dict'][video_info['label']]
                frame_paths = video_info['frames']

                # 统一帧排序方式，确保按帧号排序
                def extract_frame_number(path):
                    filename = os.path.basename(path)
                    match = re.search(r'\d+', filename)
                    if match:
                        return int(match.group())
                    print(f"警告：无法从文件名 '{filename}' 中提取数字，路径：{path}，使用默认排序")
                    return 0

                frame_paths = sorted(frame_paths, key=lambda x: extract_frame_number(x))
                total_frames = len(frame_paths)

                if self.frame_num < total_frames:
                    total_frames = self.frame_num
                    if self.video_level:
                        start_frame = random.randint(0, total_frames - self.frame_num)
                        frame_paths = frame_paths[start_frame:start_frame + self.frame_num]
                    else:
                        step = total_frames // self.frame_num
                        frame_paths = [frame_paths[i] for i in range(0, total_frames, step)][:self.frame_num]

                if self.video_level:
                    if self.clip_size is None:
                        raise ValueError('clip_size must be specified when video_level is True.')
                    if total_frames >= self.clip_size:
                        selected_clips = []
                        num_clips = total_frames // self.clip_size
                        if num_clips > 1:
                            clip_step = (total_frames - self.clip_size) // (num_clips - 1)
                            for i in range(num_clips):
                                start_frame = random.randrange(i * clip_step, min((i + 1) * clip_step,
                                                                                  total_frames - self.clip_size + 1))
                                continuous_frames = frame_paths[start_frame:start_frame + self.clip_size]
                                selected_clips.append(continuous_frames)
                        else:
                            start_frame = random.randrange(0, total_frames - self.clip_size + 1)
                            continuous_frames = frame_paths[start_frame:start_frame + self.clip_size]
                            selected_clips.append(continuous_frames)
                        label_list.extend([label_idx] * len(selected_clips))
                        frame_path_list.extend(selected_clips)
                        video_name_list.extend([unique_video_name] * len(selected_clips))
                    else:
                        print(
                            f"Skipping video {unique_video_name} because it has less than clip_size ({self.clip_size}) frames ({total_frames}).")
                else:
                    label_list.extend([label_idx] * total_frames)
                    frame_path_list.extend(frame_paths)
                    video_name_list.extend([unique_video_name] * len(frame_paths))

        shuffled = list(zip(label_list, frame_path_list, video_name_list))
        random.shuffle(shuffled)
        label_list, frame_path_list, video_name_list = zip(*shuffled)
        return frame_path_list, label_list, video_name_list

    def load_rgb(self, file_path):
        size = self.config['resolution']
        if not self.lmdb:
            assert os.path.exists(file_path), f"{file_path} does not exist"
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError(f'Loaded image is None: {file_path}')
        else:
            with self.env.begin(write=False) as txn:
                if file_path[0] == '.':
                    file_path = file_path.replace('./datasets\\', '')
                image_bin = txn.get(file_path.encode())
                image_buf = np.frombuffer(image_bin, dtype=np.uint8)
                img = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(np.array(img, dtype=np.uint8))

    def load_mask(self, file_path):
        size = self.config['resolution']
        if file_path is None:
            return np.zeros((size, size, 1))
        if not self.lmdb:
            if os.path.exists(file_path):
                mask = cv2.imread(file_path, 0)
                if mask is None:
                    mask = np.zeros((size, size))
            else:
                return np.zeros((size, size, 1))
        else:
            with self.env.begin(write=False) as txn:
                if file_path[0] == '.':
                    file_path = file_path.replace('./datasets\\', '')
                image_bin = txn.get(file_path.encode())
                image_buf = np.frombuffer(image_bin, dtype=np.uint8)
                mask = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
        mask = cv2.resize(mask, (size, size)) / 255
        mask = np.expand_dims(mask, axis=2)
        return np.float32(mask)

    def load_landmark(self, file_path):
        if file_path is None:
            return np.zeros((81, 2))
        if not self.lmdb:
            if os.path.exists(file_path):
                landmark = np.load(file_path)
            else:
                return np.zeros((81, 2))
        else:
            with self.env.begin(write=False) as txn:
                if file_path[0] == '.':
                    file_path = file_path.replace('./datasets\\', '')
                binary = txn.get(file_path.encode())
                landmark = np.frombuffer(binary, dtype=np.uint32).reshape((81, 2))
        return np.float32(landmark)

    def to_tensor(self, img):
        return T.ToTensor()(img)

    def normalize(self, img):
        mean = self.config['mean']
        std = self.config['std']
        normalize = T.Normalize(mean=mean, std=std)
        return normalize(img)

    def data_aug(self, img, landmark=None, mask=None, augmentation_seed=None):
        if augmentation_seed is not None:
            random.seed(augmentation_seed)
            np.random.seed(augmentation_seed)

        kwargs = {'image': img}
        if landmark is not None:
            kwargs['keypoints'] = landmark
            kwargs['keypoint_params'] = A.KeypointParams(format='xy')
        if mask is not None:
            kwargs['mask'] = mask

        transformed = self.transform(**kwargs)
        augmented_img = transformed['image']
        augmented_landmark = transformed.get('keypoints')
        augmented_mask = transformed.get('mask')

        if augmented_landmark is not None:
            augmented_landmark = np.array(augmented_landmark)

        if augmentation_seed is not None:
            random.seed()
            np.random.seed()

        return augmented_img, augmented_landmark, augmented_mask

    def __getitem__(self, index, no_norm=False):
        image_paths = self.data_dict['image'][index]
        label = self.data_dict['label'][index]

        if not isinstance(image_paths, list):
            image_paths = [image_paths]

        image_tensors = []
        landmark_tensors = []
        mask_tensors = []
        augmentation_seed = None

        for image_path in image_paths:
            if self.video_level and image_path == image_paths[0]:
                augmentation_seed = random.randint(0, 2 ** 32 - 1)

            try:
                image = self.load_rgb(image_path)
            except Exception as e:
                print(f"Error loading image at index {index}: {e}")
                return self.__getitem__(0)
            image = np.array(image)

            mask = self.load_mask(image_path.replace('frames', 'masks')) if self.config['with_mask'] else None
            landmarks = self.load_landmark(image_path.replace('frames', 'landmarks').replace('.png', '.npy')) if \
            self.config['with_landmark'] else None

            if self.mode == 'train' and self.config['use_data_augmentation']:
                image_trans, lm_trans, mk_trans = self.data_aug(image, landmarks, mask, augmentation_seed)
            else:
                image_trans, lm_trans, mk_trans = deepcopy(image), deepcopy(landmarks), deepcopy(mask)

            if not no_norm:
                image_tensor = self.normalize(self.to_tensor(image_trans))
            else:
                image_tensor = self.to_tensor(image_trans)

            landmark_tensor = torch.from_numpy(lm_trans) if self.config[
                                                                'with_landmark'] and lm_trans is not None else None
            mask_tensor = torch.from_numpy(mk_trans) if self.config['with_mask'] and mk_trans is not None else None

            image_tensors.append(image_tensor)
            landmark_tensors.append(landmark_tensor)
            mask_tensors.append(mask_tensor)

        if self.video_level:
            images = torch.stack(image_tensors, dim=0)
            landmarks = torch.stack([lt for lt in landmark_tensors if lt is not None], dim=0) if self.config[
                'with_landmark'] else None
            masks = torch.stack([mt for mt in mask_tensors if mt is not None], dim=0) if self.config[
                'with_mask'] else None
        else:
            images = image_tensors[0]
            landmarks = landmark_tensors[0] if self.config['with_landmark'] else None
            masks = mask_tensors[0] if self.config['with_mask'] else None

        primary_image_path = image_paths[0]
        directory_part = primary_image_path.split('\\')[-2] if '\\' in primary_image_path else \
        primary_image_path.split('/')[-2]
        try:
            if "DeepFakeDetection" in primary_image_path:
                id_index = int(directory_part.split('__')[0].split('_')[1])
            else:
                id_index = int(directory_part.split('_')[-1])
        except:
            id_index = -1

        return (
            torch.tensor(id_index, dtype=torch.long),
            images,
            torch.tensor(label, dtype=torch.long),
            landmarks,
            masks,
            image_paths
        )

    @staticmethod
    def collate_fn(batch):
        id_indices, images, labels, landmarks, masks, paths = zip(*batch)
        flat_image_paths = []
        for sample_paths in paths:
            flat_image_paths.extend(sample_paths)

        id_indices = torch.stack(id_indices)

        if len(images[0].shape) == 3:
            images = torch.stack(images, dim=0)
        else:
            images = torch.stack(images, dim=0)

        labels = torch.stack(labels) if isinstance(labels[0], torch.Tensor) else torch.tensor(labels)
        landmarks = torch.stack(landmarks) if all(lm is not None for lm in landmarks) else None
        masks = torch.stack(masks) if all(m is not None for m in masks) else None

        return {
            'id_index': id_indices,
            'image': images,
            'label': labels,
            'landmark': landmarks,
            'mask': masks,
            'image_path': flat_image_paths
        }

    def __len__(self):
        assert len(self.image_list) == len(self.label_list), 'Number of images and labels are not equal'
        return len(self.image_list)


if __name__ == "__main__":
    with open('/data/home/zhiyuanyan/DeepfakeBench/training/config/detector/video_baseline.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train_set = DeepfakeAbstractBaseDataset(
        config=config,
        mode='train',
    )
    # 使用平衡采样器
    sampler = BalancedSampler(
        train_set,
        num_classes=config['backbone_config']['num_classes'],
        min_samples_per_class=config.get('balanced_sampler', {}).get('min_samples_per_class', 5)
    )
    train_data_loader = \
        torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config['train_batchSize'],
            sampler=sampler,
            num_workers=0,
            collate_fn=train_set.collate_fn,
        )
    from tqdm import tqdm

    for iteration, batch in enumerate(tqdm(train_data_loader)):
        pass