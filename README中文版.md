# SLR-deepfake
本项目包含论文《Score-based Likelihood Ratios for Deepfake Image Evidence 
Using Deep Learning Features》的代码、数据集及完整处理流程，用于复现实验结果。

## 1. 数据集来自UADFV，是公开数据集
### 1.1 原始数据
- 来源：数据集来自UADFV，是公开数据集路径为：https://docs.google.com/forms/d/e/1FAIpQLScKPoOv15TIZ9Mn0nGScIVgKRM9tFWOmjh9eHKx57Yp-XcnxA/viewform
- 存储路径：`UADFV/datasets`
- 数据规模：real和fake各有49个文件，每个文件有此人32张图像。
- 标注规则：real和fake类

### 1.2 清洗/处理后数据
- 存储路径：`UADFV/preprocessing/dataset_json
- 数据内容：json格式文件
- 处理目的：包括人脸图像路径和标签

## 2. 完整数据处理流程 
1. 视频处理为图像：
   - 步骤：读取原始图像 → 用81个特征点裁剪获得人脸→ 尺寸统一为 256×256 → 保存至 `UADFV/datasets`
   - 对应代码： UADFV/preprocessing/preprocess.py
2. 保存为json格式，便于深度学习，在保存时，注意区分train\val\test子集，防止数据泄露：
   - 步骤：将处理完毕的图像存储其路径和标签
   - 对应代码：UADFV/preprocessing/rearrange.py
3. 训练网络：
   - 步骤：训练在验证集上表现好的网络，有良好的二分类性能，这样提取的npy文件可以区分real和fake类的特征，训练结果保存为pth文件保存在UADFV/weights文件夹下
   - 对应代码：UADFV/training/train.py
4. 测试网络，结果保存到UADFV/training/npy文件夹中
   - 步骤：利用pth测试测试集
   - 对应代码：test_0501_srm_img.py
5.查看npy文件的形状 
   -对应代码：0npyShap.py
6.对npy的real和fake类的特征降维查看，便于直观观察
    -对应代码：1TSNE.py
7.记录网络性能，绘制ROC曲线
   -对应代码：2ROC.py
8.计算类中心与真、伪图像的npy的相似度
   -对应代码：3Metric.py
9.绘制概率密度曲线
   -对应代码：4PDF.py
10.计算似然比
   -对应代码：5LR.py
11.绘制Tippett曲线，判断LR模型性能
   -对应代码：6Tippett.py
12.绘制DET曲线，判断LR模型性能
   -对应代码：7Det.py
13.绘制ECE曲线，判断LR模型性能
  -对应代码：8ECE.py
14.绘制ELUB曲线,解决因为概率密度拟合造成的拖尾效应
  -对应代码：9ELUB.py

## 3. 复现步骤
1. 克隆仓库：`git clone https://github.com/guotianli/SLR-deepfake.git`
2. 安装依赖：`pip install -r requirements.txt`
3. 按顺序运行脚本，有写死的文件路径根据本地路径修改
4. 结果输出