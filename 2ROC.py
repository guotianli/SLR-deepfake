import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# ====================== 爱思唯尔SCI论文图表规范配置 ======================
plt.rcParams.update({
    # 字体设置（强制无衬线字体，Elsevier要求）
    "font.family": ["Arial", "Helvetica", "sans-serif"],
    "font.sans-serif": ["Arial"],  # 优先使用Arial
    "font.size": 10,  # 基础字体大小

    # 坐标轴设置
    "axes.labelsize": 10,  # 坐标轴标签字号
    "axes.titlesize": 12,  # 图表标题字号
    "axes.linewidth": 1.0,  # 坐标轴线条粗细（1pt）
    "axes.grid": False,  # 去除网格线（Elsevier不推荐）
    "axes.spines.top": True,  # 保留顶部边框
    "axes.spines.right": True,  # 保留右侧边框

    # 刻度设置
    "xtick.labelsize": 8,  # x轴刻度字号
    "ytick.labelsize": 8,  # y轴刻度字号
    "xtick.major.width": 1.0,  # 刻度线粗细
    "ytick.major.width": 1.0,
    "xtick.major.size": 4,  # 刻度长度
    "ytick.major.size": 4,

    # 线条设置
    "lines.linewidth": 1.5,  # 曲线线条粗细（1.5pt）
    "lines.markersize": 5,  # 标记大小

    # 图例设置
    "legend.fontsize": 8,  # 图例字号
    "legend.frameon": True,  # 图例带边框
    "legend.framealpha": 1.0,  # 边框不透明
    "legend.edgecolor": "black",  # 图例边框颜色
    "legend.loc": "lower right",  # ROC曲线图例位置
})

# 颜色配置（符合Elsevier印刷友好规范）
COLOR_ROC = '#003366'  # 深蓝色（ROC曲线）
COLOR_PR = '#CC3333'  # 深红色（PR曲线）
COLOR_BASELINE = '#666666'  # 灰色（基准线）

# 配置路径
detector = 'srm'
input_base = './npyPaper2'
output_dir = './2ROC'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'csv'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)


def load_npy_files(folder_path):
    all_prob = []
    all_labels = []
    all_paths = []
    npy_files = glob.glob(os.path.join(folder_path, '*.npy'))
    if not npy_files:
        print(f"警告: {folder_path} 中无npy文件")
        return None, None, None
    print(f"从 {folder_path} 加载 {len(npy_files)} 个文件...")
    for npy_file in tqdm(npy_files, desc=f"处理 {os.path.basename(folder_path)}"):
        try:
            data = np.load(npy_file, allow_pickle=True).item()
            if 'predictions' in data and 'labels' in data and 'image_paths' in data:
                all_prob.extend(data['predictions'])
                all_labels.extend(data['labels'])
                all_paths.extend(data['image_paths'])
            else:
                print(f"警告: {npy_file} 缺少必要键")
        except Exception as e:
            print(f"加载 {npy_file} 出错: {e}")
    if not all_prob or not all_labels:
        print(f"警告: {folder_path} 无有效数据")
        return None, None, None
    return all_prob, all_labels, all_paths


def calculate_metrics(prob, labels):
    fpr, tpr, _ = roc_curve(labels, prob)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(labels, prob)
    average_precision = average_precision_score(labels, prob)
    return fpr, tpr, roc_auc, precision, recall, average_precision


def plot_metrics_curves(fpr, tpr, roc_auc, precision, recall, average_precision, dataset_name):
    """绘制符合Elsevier规范的ROC和PR曲线组合图"""
    # 图表尺寸：双栏图宽度17cm（约6.7英寸），高度8cm（约3.15英寸）
    plt.figure(figsize=(6.7, 3.15))  # 符合Elsevier双栏布局
    gs = plt.GridSpec(1, 2, wspace=0.3)  # 两图间距

    # 左侧：ROC曲线
    ax1 = plt.subplot(gs[0, 0])
    # 保留6位小数显示
    ax1.plot(fpr, tpr, color=COLOR_ROC, lw=1.5,
             label=f'AUC = {roc_auc:.6f}')  # 不四舍五入，保留6位小数
    ax1.plot([0, 1], [0, 1], color=COLOR_BASELINE, lw=1.0, linestyle='--')  # 基准线细于数据曲线
    ax1.set_xlim(-0.01, 1.01)  # 略微扩展范围，避免曲线贴边
    ax1.set_ylim(-0.01, 1.01)
    ax1.set_xlabel('False Positive Rate', fontsize=10)
    ax1.set_ylabel('True Positive Rate', fontsize=10)
    # 统一ROC曲线标题
    ax1.set_title('ROC Curve - UADFV Dataset', fontsize=10)
    ax1.legend(frameon=True, edgecolor='black')
    # 确保坐标轴四周边框完整
    for spine in ax1.spines.values():
        spine.set_linewidth(1.0)

    # 右侧：PR曲线
    ax2 = plt.subplot(gs[0, 1])
    # 保留6位小数显示
    ax2.step(recall, precision, where='post', color=COLOR_PR, lw=1.5,
             label=f'AP = {average_precision:.6f}')  # 不四舍五入，保留6位小数
    ax2.set_xlim(-0.01, 1.01)
    ax2.set_ylim(-0.01, 1.01)
    ax2.set_xlabel('Recall', fontsize=10)
    ax2.set_ylabel('Precision', fontsize=10)
    # 统一PR曲线标题
    ax2.set_title('Precision-Recall Curve - UADFV Dataset', fontsize=10)
    ax2.legend(frameon=True, edgecolor='black')
    # 确保坐标轴四周边框完整
    for spine in ax2.spines.values():
        spine.set_linewidth(1.0)

    plt.tight_layout()
    # 保存为600dpi TIFF格式（Elsevier推荐）
    plot_path = os.path.join(output_dir, 'plots', f'uadfv_{dataset_name}_metrics.tiff')
    plt.savefig(plot_path, dpi=600, format='tiff', bbox_inches='tight',
                pil_kwargs={"compression": "tiff_lzw"})  # LZW压缩不损失质量
    plt.close()
    print(f"组合曲线已保存至: {plot_path}")


def save_to_csv(prob, labels, paths, dataset_name):
    data = {
        'detector': detector,
        'dataset': f'UADFV_{dataset_name}',
        'probability': prob,
        'label': labels,
        'image_path': paths
    }
    df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, 'csv', f'uadfv_{dataset_name}_{detector}_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"数据已保存至: {csv_path}")


def process_dataset(folder_name):
    """处理单个数据集并保留数据集名称"""
    print(f"\n==== 处理 {folder_name} ====")
    folder_path = os.path.join(input_base, folder_name)
    prob, labels, paths = load_npy_files(folder_path)
    if prob is None: return

    # 提取数据集类型（train/val/test）
    dataset_type = folder_name.split('_')[-1]  # 从srm_train提取train

    fpr, tpr, roc_auc, precision, recall, average_precision = calculate_metrics(prob, labels)
    plot_metrics_curves(fpr, tpr, roc_auc, precision, recall, average_precision, dataset_type)
    save_to_csv(prob, labels, paths, dataset_type)

    # 控制台输出也保留6位小数
    print(f"UADFV {dataset_type} 数据集指标:")
    print(f"  AUC: {roc_auc:.6f}, AP: {average_precision:.6f}")
    print(f"  样本总数: {len(labels)}")


def main():
    print(f"开始分析检测器: {detector}")
    print(f"输入路径: {input_base}, 输出路径: {output_dir}")
    dataset_folders = ['srm_train', 'srm_val', 'srm_test']

    for folder in dataset_folders:
        process_dataset(folder)
    print("\n所有处理完成")


if __name__ == '__main__':
    main()