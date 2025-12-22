import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# 配置参数
metric_dir = './3METRIC'  # 相似度指标保存目录
output_fig_dir = './4PDF'  # 图表保存目录
plt.rcParams['font.sans-serif'] = ['Arial']  # 设置英文字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
plt.rcParams['lines.linewidth'] = 0.8  # 统一线条宽度
plt.rcParams['axes.labelsize'] = 12  # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 10  # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 10  # y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 11  # 图例字体大小

# 确保输出目录存在
os.makedirs(output_fig_dir, exist_ok=True)


def get_dataset_type(filename):
    if 'srm_train' in filename:
        return 'Train'
    elif 'srm_val' in filename:
        return 'Val'
    elif 'srm_test' in filename:
        return 'Test'
    return None


def plot_dataset_comparison(dataset_type, class0_class0_file, class0_class1_file):
    """绘制单个数据集的Real-Real与Real-Fake相似度对比小提琴图（无图题、无直方图，图例左移至坐标系中间偏左位置）"""
    try:
        # 读取两类数据
        df_real_real = pd.read_csv(class0_class0_file)
        df_real_fake = pd.read_csv(class0_class1_file)

        if 'similarity' not in df_real_real.columns or 'similarity' not in df_real_fake.columns:
            print(f"Warning: 'similarity' column missing in {dataset_type} data")
            return

        # 提取相似度数据
        sim_real_real = df_real_real['similarity'].values
        sim_real_fake = df_real_fake['similarity'].values

        # 创建图表（仅保留小提琴图，调整画布尺寸适配单图）
        plt.figure(figsize=(10, 6))
        gs = GridSpec(1, 1)  # 单图布局

        # 颜色配置（保持原配色方案，确保视觉一致性）
        colors = {'Real-Real': '#7E99F4', 'Real-Fake': '#CC7C71'}

        # 绘制小提琴图（不显示内部线条，仅保留分布形态）
        ax1 = plt.subplot(gs[0])
        data = [sim_real_fake, sim_real_real]  # 左侧Real-Fake，右侧Real-Real
        labels = ['Real-Fake', 'Real-Real']

        # 绘制小提琴图（隐藏均值、中位数、极值线，聚焦分布形态）
        violin = ax1.violinplot(
            data,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            points=50,  # 控制点数量，平衡平滑度与计算效率
            widths=0.5  # 调整小提琴宽度，避免过于拥挤
        )

        # 设置小提琴图颜色与透明度
        for i, pc in enumerate(violin['bodies']):
            pc.set_facecolor(colors[labels[i]])
            pc.set_alpha(0.8)  # 适当提高透明度，增强视觉层次
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)

        # 坐标轴配置（无图题，仅保留必要标签）
        ax1.set_ylabel('Similarity', fontsize=12)
        ax1.set_xticks([1, 2])
        ax1.set_xticklabels(labels)
        ax1.grid(True, linestyle='--', alpha=0.3)  # 网格线弱化，不干扰分布显示

        # 计算统计信息（用于图例旁补充说明，增强信息完整性）
        stats_real_real = f'N={len(sim_real_real)}, Mean={np.mean(sim_real_real):.4f}, Std={np.std(sim_real_real):.4f}'
        stats_real_fake = f'N={len(sim_real_fake)}, Mean={np.mean(sim_real_fake):.4f}, Std={np.std(sim_real_fake):.4f}'

        # 图例置于坐标系中间偏左位置（通过bbox_to_anchor的x参数左移，x=0.3更靠左，结合垂直居中）
        # 先获取y轴数据范围，确保图例垂直居中
        y_min = min(np.min(sim_real_real), np.min(sim_real_fake))
        y_max = max(np.max(sim_real_real), np.max(sim_real_fake))
        y_mid = (y_min + y_max) / 2

        # 创建带统计信息的图例项
        legend_elements = [
            plt.Line2D([0], [0], color=colors['Real-Real'], lw=4, label=f'Real-Real ({stats_real_real})'),
            plt.Line2D([0], [0], color=colors['Real-Fake'], lw=4, label=f'Real-Fake ({stats_real_fake})')
        ]

        # 调整bbox_to_anchor的x值为0.3（原0.5），实现图例左移，避免遮挡左侧Real-Fake分布
        ax1.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.3, 0.5),
                   frameon=True, fancybox=False, edgecolor='black', framealpha=0.8)

        # 调整布局（确保图例不溢出，坐标轴标签完整显示）
        plt.tight_layout()
        # 保存图片（PNG格式，300DPI满足学术图表要求）
        fig_filename = os.path.join(output_fig_dir, f'UADFV_{dataset_type}_violin.png')
        plt.savefig(fig_filename, dpi=300, bbox_inches='tight', pad_inches=0.3)
        print(f"Successfully saved {dataset_type} violin plot: {fig_filename}")
        plt.close()

    except Exception as e:
        print(f"Error plotting {dataset_type} violin plot: {e}")


def main():
    print("Starting similarity comparison violin plot generation...")
    print(f"Data directory: {metric_dir}")
    print(f"Output directory: {output_fig_dir}")

    # 查找所有CSV文件（验证数据目录下是否存在目标文件）
    csv_files = glob.glob(os.path.join(metric_dir, '*_class0_class0.csv')) + \
                glob.glob(os.path.join(metric_dir, '*_class0_class1.csv'))

    if not csv_files:
        print(f"Warning: No CSV files found in {metric_dir}")
        return

    # 打印找到的CSV文件路径（便于排查文件路径问题）
    print("Found CSV files:")
    for file in csv_files:
        print(f"- {os.path.basename(file)}")

    # 按数据集类型分组文件（使用绝对路径拼接，避免相对路径错误）
    dataset_files = {
        'Train': {
            'real_real': os.path.join(metric_dir, 'srm_train_srm_real-real.csv'),
            'real_fake': os.path.join(metric_dir, 'srm_train_srm_real-fake.csv')
        },
        'Val': {
            'real_real': os.path.join(metric_dir, 'srm_val_srm_real-real.csv'),
            'real_fake': os.path.join(metric_dir, 'srm_val_srm_real-fake.csv')
        },
        'Test': {
            'real_real': os.path.join(metric_dir, 'srm_test_srm_real-real.csv'),
            'real_fake': os.path.join(metric_dir, 'srm_test_srm_real-fake.csv')
        }
    }

    # 验证分组文件是否存在（避免文件缺失导致的报错）
    print("\nVerifying dataset files:")
    for dataset_type, files in dataset_files.items():
        real_real_exists = os.path.exists(files['real_real'])
        real_fake_exists = os.path.exists(files['real_fake'])
        print(
            f"- {dataset_type}: Real-Real={'Exists' if real_real_exists else 'Missing'}, Real-Fake={'Exists' if real_fake_exists else 'Missing'}")
        # 更新文件路径为None（若文件不存在）
        if not real_real_exists:
            files['real_real'] = None
        if not real_fake_exists:
            files['real_fake'] = None

    # 绘制每个数据集的小提琴图
    for dataset_type, files in dataset_files.items():
        if files['real_real'] and files['real_fake']:
            print(f"\nPlotting {dataset_type} dataset violin plot...")
            plot_dataset_comparison(
                dataset_type,
                files['real_real'],
                files['real_fake']
            )
        else:
            print(f"Warning: Missing files for {dataset_type} dataset, skipping plotting...")

    print("\nAll violin plots generated successfully!")


if __name__ == '__main__':
    main()