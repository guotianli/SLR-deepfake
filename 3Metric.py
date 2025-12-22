import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# 配置参数
detector = 'srm'
input_base = './npyPaper2'  # npy文件所在的根目录
metric_dir = './3METRIC'    # 相似度指标保存目录
centroid_save_path = './3METRIC/train_real_centroid.npy'  # 训练集real类中心保存路径
chunk_size = 1000           # 每批次处理的样本数
sample_ratio = 0.1          # 采样比例，当不使用类中心计算时有效
use_centroid = True         # 是否使用类中心计算相似度（固定为True，因需求依赖类中心）

# 确保输出目录存在
os.makedirs(metric_dir, exist_ok=True)


def load_npy_files(folder_path):
    """加载npy文件中的特征向量、标签和路径"""
    all_features = []
    all_labels = []
    all_paths = []

    npy_files = glob.glob(os.path.join(folder_path, '*.npy'))
    if not npy_files:
        print(f"警告: {folder_path} 中没有找到npy文件")
        return None, None, None

    print(f"正在从 {folder_path} 加载 {len(npy_files)} 个npy文件...")
    for npy_file in tqdm(npy_files, desc=f"处理 {os.path.basename(folder_path)}"):
        try:
            data = np.load(npy_file, allow_pickle=True).item()
            all_features.extend(data['features'])
            all_labels.extend(data['labels'])
            all_paths.extend(data['image_paths'])
        except Exception as e:
            print(f"加载 {npy_file} 时出错: {e}")

    if not all_features or not all_labels or not all_paths:
        print(f"警告: {folder_path} 中没有有效的数据")
        return None, None, None

    return np.array(all_features), np.array(all_labels), all_paths


def flatten_features(features):
    """将多维特征向量展平为二维数组（样本数×特征维度）"""
    original_shape = features.shape
    # 保持第一维（样本数）不变，将其余维度展平为一维
    flattened = features.reshape(original_shape[0], -1)
    print(f"将特征从形状 {original_shape} 展平为 {flattened.shape}")
    print(f"单条数据/类中心维度：{flattened.shape[1]} 维")  # 新增：打印特征维度，验证匹配性
    return flattened


def calculate_and_save_train_real_centroid(class0_features, save_path):
    """计算训练集real类（类0）的类中心并保存"""
    class0_features_flat = flatten_features(class0_features)
    train_real_centroid = np.mean(class0_features_flat, axis=0, keepdims=True)  # 1×特征维度
    np.save(save_path, train_real_centroid)
    print(f"训练集real类中心已保存至: {save_path}")
    print(f"训练集real类中心形状: {train_real_centroid.shape}")  # 验证中心维度
    return train_real_centroid


def load_train_real_centroid(load_path):
    """加载训练集real类中心"""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"训练集real类中心文件不存在: {load_path}，请先运行训练集处理流程")
    train_real_centroid = np.load(load_path)
    print(f"成功加载训练集real类中心，形状: {train_real_centroid.shape}")
    return train_real_centroid


def process_similarity_chunks_optimized(class0_features, class1_features, dataset_name, train_real_centroid=None):
    """
    优化的相似度计算函数：
    - 训练集：计算real类中心并保存，同时计算real样本与自身中心的相似度（real-real）
    - 验证集/测试集：使用训练集real类中心，计算自身real/fake样本与该中心的相似度（real-real/real-fake）
    """
    # 展平所有输入特征（确保维度统一）
    class0_features_flat = flatten_features(class0_features)  # 当前数据集的real样本（类0）
    class1_features_flat = flatten_features(class1_features)  # 当前数据集的fake样本（类1）
    n_class0 = len(class0_features_flat)
    n_class1 = len(class1_features_flat)

    # 创建保存文件（统一命名规则，区分数据集）
    csv_real_real = os.path.join(metric_dir, f'{dataset_name}_{detector}_real-real.csv')  # 当前real vs 训练集real中心
    csv_real_fake = os.path.join(metric_dir, f'{dataset_name}_{detector}_real-fake.csv')  # 当前fake vs 训练集real中心
    # 初始化CSV表头
    pd.DataFrame(columns=['sample_idx', 'centroid_type', 'similarity']).to_csv(csv_real_real, index=False)
    pd.DataFrame(columns=['sample_idx', 'centroid_type', 'similarity']).to_csv(csv_real_fake, index=False)

    print(f"计算 {dataset_name} 的相似度:")

    # --------------------------
    # 1. 训练集处理逻辑：计算并保存real类中心，同时生成real-real结果
    # --------------------------
    if dataset_name == 'srm_train':
        # 计算训练集real类中心
        train_real_centroid = calculate_and_save_train_real_centroid(class0_features, centroid_save_path)
        # 计算训练集real样本与自身中心的相似度（real-real）
        print(f"  计算训练集real样本与自身类中心的相似度（共 {n_class0} 个样本）...")
        sim_real_real = cosine_similarity(class0_features_flat, train_real_centroid).flatten()
        # 保存real-real结果（sample_idx：样本索引；centroid_type：中心类型）
        df_real_real = pd.DataFrame({
            'sample_idx': range(n_class0),
            'centroid_type': ['train_real_centroid'] * n_class0,
            'similarity': sim_real_real
        })
        df_real_real.to_csv(csv_real_real, mode='a', header=False, index=False)
        print(f"    训练集real-real结果已保存至: {csv_real_real}")

        # 训练集fake样本与自身real中心的相似度（real-fake，可选计算，按需保留）
        print(f"  计算训练集fake样本与自身real类中心的相似度（共 {n_class1} 个样本）...")
        sim_real_fake = cosine_similarity(class1_features_flat, train_real_centroid).flatten()
        df_real_fake = pd.DataFrame({
            'sample_idx': range(n_class1),
            'centroid_type': ['train_real_centroid'] * n_class1,
            'similarity': sim_real_fake
        })
        df_real_fake.to_csv(csv_real_fake, mode='a', header=False, index=False)
        print(f"    训练集real-fake结果已保存至: {csv_real_fake}")

    # --------------------------
    # 2. 验证集/测试集处理逻辑：加载训练集real中心，计算两类样本相似度
    # --------------------------
    else:
        # 加载训练集real类中心（确保维度匹配）
        train_real_centroid = load_train_real_centroid(centroid_save_path)
        # 验证维度匹配性（若不匹配则报错）
        if train_real_centroid.shape[1] != class0_features_flat.shape[1]:
            raise ValueError(
                f"类中心与样本维度不匹配！训练集real中心维度: {train_real_centroid.shape[1]}, "
                f"{dataset_name} real样本维度: {class0_features_flat.shape[1]}"
            )

        # 2.1 计算当前数据集real样本与训练集real中心的相似度（real-real）
        print(f"  计算 {dataset_name} real样本与训练集real类中心的相似度（共 {n_class0} 个样本）...")
        sim_real_real = cosine_similarity(class0_features_flat, train_real_centroid).flatten()
        df_real_real = pd.DataFrame({
            'sample_idx': range(n_class0),
            'centroid_type': ['train_real_centroid'] * n_class0,
            'similarity': sim_real_real
        })
        df_real_real.to_csv(csv_real_real, mode='a', header=False, index=False)
        print(f"    {dataset_name} real-real结果已保存至: {csv_real_real}")

        # 2.2 计算当前数据集fake样本与训练集real中心的相似度（real-fake）
        print(f"  计算 {dataset_name} fake样本与训练集real类中心的相似度（共 {n_class1} 个样本）...")
        sim_real_fake = cosine_similarity(class1_features_flat, train_real_centroid).flatten()
        df_real_fake = pd.DataFrame({
            'sample_idx': range(n_class1),
            'centroid_type': ['train_real_centroid'] * n_class1,
            'similarity': sim_real_fake
        })
        df_real_fake.to_csv(csv_real_fake, mode='a', header=False, index=False)
        print(f"    {dataset_name} real-fake结果已保存至: {csv_real_fake}")

    print(f"  {dataset_name} 相似度计算完成！")


def process_dataset(folder_name):
    """处理单个数据集文件夹（训练集/验证集/测试集）"""
    print(f"\n==== 处理数据集: {folder_name} ====")
    folder_path = os.path.join(input_base, folder_name)

    # 加载当前数据集的特征、标签、路径
    features, labels, paths = load_npy_files(folder_path)
    if features is None:
        return

    # 分离当前数据集的real类（0）和fake类（1）
    class0_mask = labels == 0  # real类
    class1_mask = labels == 1  # fake类
    class0_features = features[class0_mask]
    class1_features = features[class1_mask]

    print(f"当前数据集类别分布 - real类（0）: {len(class0_features)} 个样本, fake类（1）: {len(class1_features)} 个样本")

    # 计算相似度（训练集需传None，验证集/测试集自动加载训练集中心）
    process_similarity_chunks_optimized(class0_features, class1_features, folder_name)

    # 打印当前数据集处理总结
    print(f"{folder_name} 处理完成，样本总数: {len(labels)}")


def main():
    """主函数：按训练集→验证集→测试集顺序处理（确保训练集先计算中心）"""
    print(f"开始处理检测器: {detector}")
    print(f"输入数据路径: {input_base}")
    print(f"相似度指标保存路径: {metric_dir}")
    print(f"训练集real类中心保存路径: {centroid_save_path}")
    print(f"是否使用训练集real类中心计算: {use_centroid}")

    # 数据集处理顺序：必须先处理训练集（生成类中心），再处理验证集/测试集
    dataset_folders = ['srm_train', 'srm_val', 'srm_test']

    for folder in dataset_folders:
        process_dataset(folder)

    print("\n所有数据集处理已完成!")


if __name__ == '__main__':
    main()