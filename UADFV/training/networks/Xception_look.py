import torch
import torchviz
from .xception import Xception  # 根据实际的模型类名导入

# 加载你的模型
model = Xception(num_classes=2)  # 假设有10个类别

# 创建一个示例输入
example_input = torch.rand(1, 3, 224, 224)  # 假设输入尺寸是 3x224x224，第一个批次大小为1

# 将模型输入设置为第一个批次的输入
model_input = model(example_input)

# 生成神经网络结构图（使用第一个批次的输入）
dot = torchviz.make_dot(model_input, params=dict(model.named_parameters()))

# 渲染结构图到文件
dot.render("model_structure_first_batch", format="png")
