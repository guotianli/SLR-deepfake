import torch
import torch.nn as nn
import torch.nn.functional as F
from .abstract_loss_func import AbstractLossClass
from metrics.registry import LOSSFUNC


@LOSSFUNC.register_module(module_name="cross_entropy")
class CrossEntropyLoss(AbstractLossClass):
    """
    改进的交叉熵损失函数，支持类别权重、标签平滑和损失缩放。
    """

    def __init__(self,
                 weight=None,
                 ignore_index=-100,
                 reduction='mean',
                 label_smoothing=0.0,
                 loss_scale=1.0,
                 use_class_balancing=False,
                 dataset=None):
        """
        初始化交叉熵损失函数参数。

        Args:
            weight: 各类别的权重张量，用于处理样本不均衡问题。
            ignore_index: 忽略的目标值索引，不参与损失计算。
            reduction: 损失 reduction 方法，可选 'mean', 'sum', 'none'。
            label_smoothing: 标签平滑系数，范围 [0.0, 1.0)。
            loss_scale: 损失缩放因子，用于调整损失大小。
            use_class_balancing: 是否使用类别平衡权重。
            dataset: 数据集对象，用于获取类别频率。
        """
        super().__init__()

        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.loss_scale = loss_scale
        self.use_class_balancing = use_class_balancing

        # 初始化类别权重
        if use_class_balancing and dataset is not None:
            self._init_class_weights(dataset)

        # 创建损失函数
        self._create_loss_function()

        # 跟踪训练统计信息
        self.reset_stats()

    def _init_class_weights(self, dataset):
        """
        从数据集初始化类别权重，用于处理样本不均衡问题。
        """
        if hasattr(dataset, 'calculate_class_frequencies'):
            frequencies = dataset.calculate_class_frequencies()
            if frequencies is not None:
                # 使用频率的倒数作为权重
                self.weight = torch.tensor([1.0 / freq for freq in frequencies],
                                           dtype=torch.float32)
                print(f"类别平衡权重已启用，权重范围: [{min(self.weight):.4f}, {max(self.weight):.4f}]")
            else:
                print("警告: 无法获取类别频率，类别平衡权重未启用")
        else:
            print("警告: 数据集不支持类别频率计算，类别平衡权重未启用")

    def _create_loss_function(self):
        """
        根据配置创建适当的损失函数。
        """
        if self.label_smoothing > 0:
            # 使用自定义标签平滑实现
            self._loss_fn = self._label_smoothing_cross_entropy
        else:
            # 使用PyTorch内置实现
            self._loss_fn = nn.CrossEntropyLoss(
                weight=self.weight,
                ignore_index=self.ignore_index,
                reduction=self.reduction
            )

    def _label_smoothing_cross_entropy(self, inputs, targets):
        """
        实现带标签平滑的交叉熵损失。
        """
        # 确保标签平滑在有效范围内
        epsilon = min(max(self.label_smoothing, 0.0), 0.999)

        # 计算类别数
        num_classes = inputs.size(-1)

        # 创建平滑后的标签分布
        one_hot = torch.zeros_like(inputs).scatter(1, targets.unsqueeze(1), 1)
        smooth_labels = one_hot * (1 - epsilon) + (1 - one_hot) * epsilon / (num_classes - 1)

        # 计算对数概率
        log_probs = F.log_softmax(inputs, dim=-1)

        # 计算损失
        loss = -torch.sum(smooth_labels * log_probs, dim=-1)

        # 应用权重（如果有）
        if self.weight is not None:
            loss = loss * self.weight[targets]

        # 应用reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

    def reset_stats(self):
        """重置训练统计信息"""
        self.batch_count = 0
        self.total_loss = 0.0
        self.class_losses = {}

    def forward(self, inputs, targets):
        """
        计算交叉熵损失。

        Args:
            inputs: 预测分数的PyTorch张量，形状 (batch_size, num_classes)。
            targets: 真实类别索引的PyTorch张量，形状 (batch_size)。

        Returns:
            表示交叉熵损失的标量张量。
        """
        # 验证输入形状
        if inputs.dim() != 2 or targets.dim() != 1:
            raise ValueError(f"输入形状错误: inputs {inputs.shape}, targets {targets.shape}")

        if inputs.size(0) != targets.size(0):
            raise ValueError(f"批量大小不匹配: inputs {inputs.size(0)}, targets {targets.size(0)}")

        # 计算损失
        loss = self._loss_fn(inputs, targets)

        # 应用损失缩放
        if self.loss_scale != 1.0:
            loss = loss * self.loss_scale

        # 更新统计信息
        self._update_stats(loss, inputs, targets)

        return loss

    def _update_stats(self, loss, inputs, targets):
        """更新训练统计信息"""
        self.batch_count += 1
        self.total_loss += loss.item()

        # 计算每类损失
        if self.reduction == 'none':
            for c in torch.unique(targets):
                if c == self.ignore_index:
                    continue
                mask = targets == c
                class_loss = loss[mask].mean().item()
                if c.item() in self.class_losses:
                    self.class_losses[c.item()].append(class_loss)
                else:
                    self.class_losses[c.item()] = [class_loss]

    def get_mean_loss(self):
        """获取平均损失"""
        if self.batch_count == 0:
            return 0.0
        return self.total_loss / self.batch_count

    def get_class_losses(self):
        """获取每类的平均损失"""
        return {c: sum(losses) / len(losses) for c, losses in self.class_losses.items()}

    def get_loss_stats(self):
        """获取损失统计信息"""
        stats = {
            'batch_count': self.batch_count,
            'mean_loss': self.get_mean_loss(),
            'class_losses': self.get_class_losses()
        }
        return stats


@LOSSFUNC.register_module(module_name="focal_loss")
class FocalLoss(AbstractLossClass):
    """
    Focal Loss - 解决类别不平衡问题的损失函数。
    """

    def __init__(self, alpha=1, gamma=2, reduction='mean', ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # 计算交叉熵
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)

        # 计算pt
        pt = torch.exp(-ce_loss)

        # 计算focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        # 应用reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss