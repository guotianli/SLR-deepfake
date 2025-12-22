import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
from tqdm import tqdm

# ====================== 爱思唯尔SCI论文图表规范配置 ======================
plt.rcParams.update({
    # 字体设置（强制无衬线字体）
    "font.family": ["Arial", "Helvetica", "sans-serif"],
    "font.sans-serif": ["Arial"],
    "font.size": 10,  # 基础字体大小

    # 坐标轴设置
    "axes.linewidth": 1.0,  # 坐标轴线条粗细（1pt）
    "axes.spines.top": True,  # 保留顶部边框
    "axes.spines.right": True,  # 保留右侧边框

    # 刻度设置（如果显示刻度）
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.direction": "in",  # 刻度向内
    "ytick.direction": "in",

    # 图例设置
    "legend.fontsize": 8,  # 图例字号
    "legend.title_fontsize": 9,  # 图例标题字号
    "legend.frameon": True,  # 图例带边框
    "legend.framealpha": 1.0,  # 边框不透明
    "legend.edgecolor": "black",  # 图例边框黑色
    "legend.loc": "best",  # 自动选择最佳位置
    "legend.borderaxespad": 0.5,
    "legend.labelspacing": 0.3,
})

# 颜色配置（Elsevier印刷友好色，高对比度）
COLOR_REAL = '#4DB748'  # blue（真实样本）
COLOR_FAKE = '#FF69B4'  # orange（伪造样本）
color_map = [COLOR_REAL, COLOR_FAKE]  # 与类别0/1对应

# 标签映射（简洁明确）
label_dict = {
    0: 'Real',
    1: 'Fake',
}


def tsne_draw(x_transformed, numerical_labels, ax, folder_name=None):
    """绘制符合Elsevier规范的t-SNE散点图"""
    # 分离真实和伪造样本（便于控制绘制顺序，避免遮挡）
    real_mask = numerical_labels == 0
    fake_mask = numerical_labels == 1

    # 绘制伪造样本（先绘制，避免遮挡真实样本）
    ax.scatter(
        x_transformed[fake_mask, 0], x_transformed[fake_mask, 1],
        color=COLOR_FAKE, s=15, alpha=0.6, marker='o',  # 小圆点
        edgecolor='none'  # 去除边缘线，避免视觉杂乱
    )

    # 绘制真实样本（后绘制，星形标记突出显示）
    ax.scatter(
        x_transformed[real_mask, 0], x_transformed[real_mask, 1],
        color=COLOR_REAL, s=20, alpha=0.7, marker='*',  # 星形标记
        edgecolor='none'
    )

    # 设置标题（简洁，10pt）
    if folder_name:
        dataset_type = folder_name.split('_')[-1].capitalize()  # 提取train/val/test并大写
        ax.set_title(f't-SNE Visualization of UADFV', fontsize=10, pad=10)

    # 隐藏坐标轴（t-SNE通常不需要显示具体坐标值）
    ax.axis('off')

    return ax


def load_npy_files(folder_path):
    """加载指定文件夹中的所有npy文件"""
    all_features = []
    all_labels = []

    npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    if not npy_files:
        print(f"Warning: No .npy files found in {folder_path}")
        return None, None

    print(f"Loading {len(npy_files)} .npy files from {folder_path}...")
    for file in tqdm(npy_files, desc=f"Loading {os.path.basename(folder_path)}"):
        file_path = os.path.join(folder_path, file)
        try:
            data = np.load(file_path, allow_pickle=True).item()
            if 'features' not in data or 'labels' not in data:
                print(f"Warning: Missing 'features' or 'labels' in {file}")
                continue

            features = data['features']
            labels = data['labels']

            # 展平多维特征
            if features.ndim > 2:
                features = features.reshape(features.shape[0], -1)
            all_features.append(features)
            all_labels.extend(labels)

        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue

    if not all_features:
        print(f"Warning: No valid feature data in {folder_path}")
        return None, None

    return np.vstack(all_features), np.array(all_labels)


def process_folder(input_folder, output_folder, perplexity=30, sample_size=5000):
    """处理单个文件夹并生成符合规范的t-SNE可视化"""
    os.makedirs(output_folder, exist_ok=True)

    # 加载数据
    features, labels = load_npy_files(input_folder)
    if features is None or labels is None:
        print(f"Skipping {input_folder} due to no valid data")
        return

    # 采样（控制样本量，避免点过于密集）
    if len(features) > sample_size:
        print(f"Sampling {sample_size} from {len(features)} samples")
        indices = np.random.choice(len(features), sample_size, replace=False)
        features = features[indices]
        labels = labels[indices]

    # 执行t-SNE降维
    print(f"Performing t-SNE (perplexity={perplexity})")
    tsne = TSNE(
        n_components=2, perplexity=perplexity,
        random_state=42, n_iter=1000,
        learning_rate=200  # 优化学习率，提高聚类效果
    )
    features_tsne = tsne.fit_transform(features)

    # 创建图形（单栏图尺寸：8cm宽，8cm高，符合Elsevier布局）
    plt.figure(figsize=(3.15, 3.15))  # 1英寸≈2.54cm，3.15英寸≈8cm
    ax = plt.gca()

    # 绘制t-SNE图
    folder_name = os.path.basename(input_folder)
    tsne_draw(features_tsne, labels, ax, folder_name)

    # 创建图例（黑色边框，紧凑布局）
    handles = [
        plt.Line2D([0], [0], marker='*', color='w',
                   markerfacecolor=COLOR_REAL, markersize=6, label=label_dict[0]),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=COLOR_FAKE, markersize=5, label=label_dict[1])
    ]
    ax.legend(
        handles=handles,
        title="Class",  # 图例标题
        title_fontsize=8,
        fontsize=7,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),  # 图例放在图右侧
        borderaxespad=0,
        frameon=True,
        edgecolor='black',
        framealpha=1.0,
        labelspacing=0.5,
        handletextpad=0.5
    )

    # 紧凑布局
    plt.tight_layout()

    # 保存为600dpi TIFF格式（Elsevier推荐）
    dataset_type = folder_name.split('_')[-1]
    output_path = os.path.join(output_folder, f'tsne_{dataset_type}.tiff')
    plt.savefig(
        output_path,
        dpi=600,
        format='tiff',
        bbox_inches='tight',  # 确保图例完整保存
        pil_kwargs={"compression": "tiff_lzw"}
    )
    plt.close()

    print(f"t-SNE visualization saved to: {output_path}")


def main():
    input_base = './npyPaper2'
    output_base = './1TSNE'
    perplexity = 30
    sample_size = 5000  # 控制样本量，避免点重叠

    subfolders = [f for f in os.listdir(input_base)
                  if os.path.isdir(os.path.join(input_base, f))]

    if not subfolders:
        print(f"Error: No subfolders found in {input_base}")
        return

    print(f"Found {len(subfolders)} subfolders: {subfolders}")

    for folder in subfolders:
        folder_path = os.path.join(input_base, folder)
        print(f"\n==== Processing folder: {folder} ====")
        process_folder(folder_path, output_base, perplexity, sample_size)

    print("\nAll t-SNE visualizations completed!")


if __name__ == '__main__':
    main()