import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm
import os


# ----------------------------
# 全局设置：符合Elsevier/SCI图表规范
# ----------------------------
plt.rcParams.update({
    # 字体：无衬线Arial（期刊首选）
    "font.family": ["Arial", "Helvetica", "sans-serif"],
    "font.size": 10,  # 基础字号
    "axes.labelsize": 10,  # 坐标轴标签字号
    "axes.titlesize": 10,  # 图表标题字号
    "legend.fontsize": 8,  # 图例字号
    "xtick.labelsize": 8,  # x轴刻度字号
    "ytick.labelsize": 8,  # y轴刻度字号
    # 线条与刻度设置
    "axes.linewidth": 1.0,  # 坐标轴线条粗细（1pt）
    "lines.linewidth": 1.5,  # 曲线线条粗细
    "lines.markersize": 4,  # 标记点大小
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
})


def load_data(file_path):
    """
    加载数据（核心修正：正确映射Hp/Hd）
    - Hp（同一来源，真实匹配）：class == 'real_real'
    - Hd（不同来源，虚假匹配）：class == 'real_fake'
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    try:
        df = pd.read_csv(file_path)
        required_columns = ['LogLR', 'class']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"CSV缺少必要列: {', '.join(missing_columns)}")

        # 正确划分Hp和Hd样本（修正原映射错误）
        hp_scores = df[df['class'] == 'real_real']['LogLR'].values  # Hp：real-real（同一来源）
        hd_scores = df[df['class'] == 'real_fake']['LogLR'].values  # Hd：real-fake（不同来源）

        if len(hp_scores) == 0 or len(hd_scores) == 0:
            class_distribution = df['class'].value_counts().to_dict()
            raise ValueError(f"缺少Hp/Hd样本，当前类别分布: {class_distribution}")

        print(f"加载数据完成:")
        print(f"- Hp（real-real，同一来源）样本数: {len(hp_scores)}")
        print(f"- Hd（real-fake，不同来源）样本数: {len(hd_scores)}")
        return hp_scores, hd_scores
    except Exception as e:
        print(f"数据加载失败: {e}")
        raise


def compute_det_curve(hp_scores, hd_scores, num_thresholds=1000):
    """
    计算DET曲线数据（基于Hp/Hd正确定义）
    - FAP（假接受率）：错误接受Hd样本的概率 → Hd样本LogLR ≥ 阈值
    - FRP（假拒绝率）：错误拒绝Hp样本的概率 → Hp样本LogLR < 阈值
    """
    # 确定阈值范围（覆盖所有LogLR值）
    min_val = min(np.min(hp_scores), np.min(hd_scores))
    max_val = max(np.max(hp_scores), np.max(hd_scores))
    thresholds = np.linspace(min_val, max_val, num_thresholds)

    # 向量化计算FAP和FRP（提升效率）
    fap = np.mean(hd_scores >= thresholds[:, np.newaxis], axis=1)  # Hd≥阈值→错误接受
    frp = np.mean(hp_scores < thresholds[:, np.newaxis], axis=1)   # Hp<阈值→错误拒绝
    return fap, frp, thresholds


def gaussian_warp(probabilities):
    """高斯变形：将概率转换为标准正态分布分位数（避免无穷大）"""
    return norm.ppf(np.clip(probabilities, 1e-8, 1 - 1e-8))


def find_eer(fap, frp, thresholds):
    """寻找等错误率(EER)（补充边界提示，提升精度）"""
    abs_diff = np.abs(fap - frp)
    idx = np.nanargmin(abs_diff)  # FAP与FRP最接近的索引

    # 边界值处理（避免极端阈值导致的偏差）
    if idx == 0 or idx == len(fap) - 1:
        print(f"警告：EER落在阈值边界（索引{idx}），结果可能存在轻微偏差")
        eer = (fap[idx] + frp[idx]) / 2  # 取平均值减少偏差
        return eer, eer, thresholds[idx]

    # 线性插值计算精确EER
    t_low, t_high = thresholds[idx - 1], thresholds[idx + 1]
    fap_low, fap_high = fap[idx - 1], fap[idx + 1]
    frp_low, frp_high = frp[idx - 1], frp[idx + 1]

    # 避免分母为0（防止插值失效）
    slope_diff = (frp_high - frp_low) - (fap_high - fap_low)
    if slope_diff == 0:
        print(f"警告：插值斜率为0，使用近似EER")
        eer = (fap[idx] + frp[idx]) / 2
        return eer, eer, thresholds[idx]

    # 求解FAP=FRP时的阈值和EER
    t_interp = (frp_low - fap_low) / slope_diff
    eer = fap_low + t_interp * (fap_high - fap_low)
    eer_threshold = t_low + t_interp * (t_high - t_low)
    return eer, eer, eer_threshold


def plot_linearized_det(fap, frp, eer, output_dir, dataset_name):
    """绘制符合Elsevier规范的线性化DET曲线（补充Hp/Hd标注）"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset_name}_linearized_det_curve.tiff")

    # 高斯变形（线性化DET曲线）
    warped_fap = gaussian_warp(fap)
    warped_frp = gaussian_warp(frp)
    warped_eer = gaussian_warp(eer)

    # 创建图表（单栏排版尺寸：8×6英寸）
    fig, ax = plt.subplots(figsize=(8, 6))

    # 1. 绘制DET曲线（橙色主曲线）
    ax.plot(warped_fap, warped_frp, color='orange', linewidth=1.5, label='DET Curve', zorder=2)

    # 2. 绘制FAP=FRP参考线（灰色虚线）
    min_warp = min(np.min(warped_fap), np.min(warped_frp)) - 1
    max_warp = max(np.max(warped_fap), np.max(warped_frp)) + 1
    ax.plot(
        [min_warp, max_warp], [min_warp, max_warp],
        color='#B4C5D9', linestyle='--', linewidth=1.0,
        label='FAP = FRP (Reference Line)', zorder=1
    )

    # 3. 标注EER点（粉色实心点+箭头注释）
    ax.scatter(warped_eer, warped_eer, color='#E9B5C1', s=60, edgecolors='black', linewidth=0.5, zorder=3)
    ax.annotate(
        f'EER: {eer:.4f}',
        xy=(warped_eer, warped_eer),
        xytext=(warped_eer + 0.5, warped_eer + 0.5),  # 位置调整避免遮挡
        fontsize=8,
        arrowprops=dict(facecolor='#E9B5C1', edgecolor='none', shrink=0.05, width=0.5, headwidth=5),
        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.8)
    )

    # 4. 坐标轴设置（概率→正态分位数映射，标签明确）
    prob_ticks = [0.001, 0.01, 0.05, 0.2, 0.4, 0.6]  # 原始概率刻度
    warp_ticks = gaussian_warp(np.array(prob_ticks))  # 转换为正态分位数
    ax.set_xticks(warp_ticks)
    ax.set_xticklabels([f"{p:.1%}" for p in prob_ticks], rotation=45, ha='right')  # 旋转避免重叠
    ax.set_yticks(warp_ticks)
    ax.set_yticklabels([f"{p:.1%}" for p in prob_ticks], ha='right')

    # 5. 坐标轴标签（补充Hp/Hd含义，符合期刊可解释性要求）
    ax.set_xlabel('False Acceptance Rate', fontsize=10)
    ax.set_ylabel('False Rejection Rate', fontsize=10)

    # 6. 图例与网格（图例右上角，网格淡化不干扰数据）
    ax.legend(loc='upper right', frameon=True, framealpha=0.8, edgecolor='gray', fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.2, linewidth=0.5, zorder=0)


    # 紧凑布局（预留底部标题空间，避免裁剪）
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # 保存图表（600dpi高分辨率，TIFF-LZW无损压缩）
    plt.savefig(
        output_path,
        dpi=600,
        bbox_inches='tight',
        format='tiff',
        pil_kwargs={"compression": "tiff_lzw"}
    )
    print(f"{dataset_name} DET曲线已保存至: {output_path}")
    plt.close()


def process_dataset(input_path, output_subdir, dataset_name):
    """处理单个数据集：加载Hp/Hd→计算DET→找EER→绘图→保存结果"""
    print(f"\n===== 开始处理数据集: {dataset_name} =====")
    print(f"输入路径: {input_path}")
    print(f"输出目录: {output_subdir}")

    os.makedirs(output_subdir, exist_ok=True)
    try:
        # 1. 加载修正后的Hp/Hd分数
        hp_scores, hd_scores = load_data(input_path)

        # 2. 计算DET曲线数据
        fap, frp, thresholds = compute_det_curve(hp_scores, hd_scores)

        # 3. 计算EER
        eer, _, eer_threshold = find_eer(fap, frp, thresholds)
        print(f"{dataset_name} EER计算完成: EER = {eer:.6f}, 对应阈值 = {eer_threshold:.6f}")

        # 4. 绘制并保存DET曲线
        plot_linearized_det(fap, frp, eer, output_subdir, dataset_name)

        # 5. 保存EER详细结果（含样本数，便于后续分析）
        eer_output_path = os.path.join(output_subdir, f"{dataset_name}_eer_result.txt")
        with open(eer_output_path, 'w') as f:
            f.write(f"Dataset Name: {dataset_name}\n")
            f.write(f"Processing Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Hp (real-real, Same Source) Sample Count: {len(hp_scores)}\n")
            f.write(f"Hd (real-fake, Different Source) Sample Count: {len(hd_scores)}\n")
            f.write(f"Equal Error Rate (EER): {eer:.6f}\n")
            f.write(f"EER Corresponding Threshold (LogLR): {eer_threshold:.6f}\n")
        print(f"{dataset_name} EER结果已保存至: {eer_output_path}")

        return eer, eer_threshold
    except Exception as e:
        print(f"处理数据集 {dataset_name} 时出错: {str(e)}")
        return None, None


def main():
    """主函数：批量处理数据集+汇总结果"""
    # 定义数据集列表（Train/Val/Test，路径与原代码一致）
    datasets = [
        {
            'name': 'Train',
            'input_path': './5LR/csv/Train/Train_likelihood_ratios.csv',
            'output_subdir': './7DET/Train'
        },
        {
            'name': 'Val',
            'input_path': './5LR/csv/Val/Val_likelihood_ratios.csv',
            'output_subdir': './7DET/Val'
        },
        {
            'name': 'Test',
            'input_path': './5LR/csv/Test/Test_likelihood_ratios.csv',
            'output_subdir': './7DET/Test'
        }
    ]

    # 创建主输出目录
    os.makedirs('./7DET', exist_ok=True)

    # 批量处理每个数据集并记录结果
    results = {}
    for dataset in datasets:
        eer, threshold = process_dataset(
            input_path=dataset['input_path'],
            output_subdir=dataset['output_subdir'],
            dataset_name=dataset['name']
        )
        if eer is not None:
            results[dataset['name']] = {
                'eer': eer,
                'threshold': threshold,
                'hp_samples': len(load_data(dataset['input_path'])[0]),
                'hd_samples': len(load_data(dataset['input_path'])[1])
            }

    # 打印汇总结果（含Hp/Hd样本数）
    print("\n===== 所有数据集处理完成 - 汇总结果 =====")
    print("核心定义：Hp=real-real（同一来源），Hd=real-fake（不同来源）\n")
    for name, res in results.items():
        print(f"{name} Dataset:")
        print(f"  - Hp样本数: {res['hp_samples']}, Hd样本数: {res['hd_samples']}")
        print(f"  - EER: {res['eer']:.6f}")
        print(f"  - EER对应阈值（LogLR）: {res['threshold']:.6f}\n")

    # 保存汇总结果（补充时间戳，便于追溯）
    summary_path = './7DET/all_datasets_eer_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("===== DET Curve & EER Summary for All Datasets =====\n")
        f.write(f"Summary Generated Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Core Definition: Hp=real-real (Same Source), Hd=real-fake (Different Source)\n\n")
        for name, res in results.items():
            f.write(f"1. {name} Dataset:\n")
            f.write(f"   - Hp Sample Count: {res['hp_samples']}\n")
            f.write(f"   - Hd Sample Count: {res['hd_samples']}\n")
            f.write(f"   - Equal Error Rate (EER): {res['eer']:.6f}\n")
            f.write(f"   - EER Corresponding Threshold (LogLR): {res['threshold']:.6f}\n\n")
    print(f"汇总结果已保存至: {summary_path}")


if __name__ == "__main__":
    main()