import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os
import matplotlib.gridspec as gridspec

# 设置符合SCI论文的字体和格式
plt.rcParams["font.family"] = ["Times New Roman", "Arial"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号
plt.rcParams["font.size"] = 10  # 基础字体大小
plt.rcParams["axes.labelsize"] = 12  # 坐标轴标签字体大小
plt.rcParams["axes.titlesize"] = 14  # 标题字体大小
plt.rcParams["legend.fontsize"] = 10  # 图例字体大小
plt.rcParams["xtick.labelsize"] = 10  # x轴刻度字体大小
plt.rcParams["ytick.labelsize"] = 10  # y轴刻度字体大小
plt.rcParams["lines.linewidth"] = 1.5  # 线条宽度
plt.rcParams["axes.linewidth"] = 1.0  # 坐标轴边框宽度
plt.rcParams["xtick.direction"] = "in"  # x轴刻度朝内
plt.rcParams["ytick.direction"] = "in"  # y轴刻度朝内
plt.rcParams["xtick.major.size"] = 4  # x轴主刻度长度
plt.rcParams["ytick.major.size"] = 4  # y轴主刻度长度
plt.rcParams["xtick.minor.size"] = 2  # x轴次刻度长度
plt.rcParams["ytick.minor.size"] = 2  # y轴次刻度长度


def load_data(file_path):
    """加载LR数据并检查必要字段"""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")

    required_cols = ['sample_idx', 'centroid_type', 'similarity', 'LR', 'LogLR', 'class']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    # 提取real-real和real-fake类数据
    df_real_real = df[df['class'] == 'real_real'].copy()
    df_real_fake = df[df['class'] == 'real_fake'].copy()

    print(f"Data loaded - real-real samples: {len(df_real_real)}, real-fake samples: {len(df_real_fake)}")
    return df, df_real_real, df_real_fake


def calculate_elub(df_real_real, df_real_fake, n_cmlr=1):
    """计算ELUB-LR安全边界，修正NBE计算逻辑"""
    # 提取LogLR数据（使用全量数据）
    rr_loglr = df_real_real['LogLR'].values
    rf_loglr = df_real_fake['LogLR'].values

    # 原始LR系统范围标记（仅用于可视化）
    rr_original_range = [-2, 5]  # real-real类原始LR系统范围
    rf_original_range = [-10, 2.5]  # real-fake类原始LR系统范围

    # 识别原始范围内的数据（仅用于统计，不用于NBE计算）
    rr_in_range = rr_loglr[(rr_loglr >= rr_original_range[0]) & (rr_loglr <= rr_original_range[1])]
    rf_in_range = rf_loglr[(rf_loglr >= rf_original_range[0]) & (rf_loglr <= rf_original_range[1])]
    print(f"原始LR系统范围内样本数 - real-real: {len(rr_in_range)}, real-fake: {len(rf_in_range)}")

    # 添加超尾部CMLR（基于全量数据的极端值）
    max_observed_rr = np.max(rr_loglr) if len(rr_loglr) > 0 else 21
    min_observed_rf = np.min(rf_loglr) if len(rf_loglr) > 0 else -21

    # 关键修正：将CMLR添加到全量数据，确保分子分母数据范围一致
    rr_with_cmlr = np.concatenate([rr_loglr, [max_observed_rr] * n_cmlr])
    rf_with_cmlr = np.concatenate([rf_loglr, [min_observed_rf] * n_cmlr])

    n_rr = len(rr_with_cmlr)
    n_rf = len(rf_with_cmlr)

    # 生成用于计算NBE的LR阈值范围
    log10_lr_th_range = np.linspace(-15, 15, 300)
    nbe_values = []

    for log10_lr_th in log10_lr_th_range:
        lr_th = 10 ** log10_lr_th

        # --------------------------
        # 修正1：正确计算中性系统的期望效用（EU_neutral）
        # 中性系统特征：所有样本LR=1，决策基于1与LR阈值的比较
        # --------------------------
        if 1 <= lr_th:  # LR=1 ≤ LRₜₕ → 对Hp判定为"不支持"，对Hd判定为"支持"
            p_neutral_rr_le = 1.0  # P(LR≤LRₜₕ | Hp) = 1
            p_neutral_rf_gt = 0.0  # P(LR>LRₜₕ | Hd) = 0
        else:  # LR=1 > LRₜₕ → 对Hp判定为"支持"，对Hd判定为"不支持"
            p_neutral_rr_le = 0.0
            p_neutral_rf_gt = 1.0
        eu_neutral = p_neutral_rr_le + lr_th * p_neutral_rf_gt

        # --------------------------
        # 修正2：LR系统的期望效用（EU_LR）使用与中性系统相同的数据范围
        # --------------------------
        n_rr_le = np.sum(rr_with_cmlr <= log10_lr_th)  # Hp中LR≤LRₜₕ的数量
        n_rf_gt = np.sum(rf_with_cmlr > log10_lr_th)  # Hd中LR>LRₜₕ的数量

        p_lr_rr_le = n_rr_le / n_rr  # P(LR≤LRₜₕ | Hp)
        p_lr_rf_gt = n_rf_gt / n_rf  # P(LR>LRₜₕ | Hd)
        eu_lr = p_lr_rr_le + lr_th * p_lr_rf_gt

        # --------------------------
        # 计算NBE（避免除零和极端值）
        # --------------------------
        if eu_lr == 0:
            nbe = np.inf
        else:
            nbe = eu_neutral / eu_lr  # 正确的NBE比值：EU_neutral / EU_LR

        # 限制NBE范围，避免对数计算溢出
        nbe = np.clip(nbe, 1e-5, 1e5)
        nbe_values.append(nbe)

    # 计算log10(NBE)
    log10_nbe = np.log10(nbe_values)

    # 找到NBE=0的临界点（使用线性插值）
    target = 0  # 寻找log10(NBE)=0的点

    # 计算real-real类上限（log10_lr_max）
    high_range_mask = log10_lr_th_range >= 0
    high_log10_lr = log10_lr_th_range[high_range_mask]
    high_log10_nbe = np.array(log10_nbe)[high_range_mask]

    cross_idx = np.where(np.diff(np.sign(high_log10_nbe - target)))[0]
    log10_lr_max = 6.5  # 基于原始范围的初始估计
    if len(cross_idx) > 0:
        idx = cross_idx[0]
        x1, x2 = high_log10_lr[idx], high_log10_lr[idx + 1]
        y1, y2 = high_log10_nbe[idx], high_log10_nbe[idx + 1]
        log10_lr_max = x1 + (target - y1) * (x2 - x1) / (y2 - y1)

    # 计算real-fake类下限（log10_lr_min）
    low_range_mask = log10_lr_th_range <= 0
    low_log10_lr = log10_lr_th_range[low_range_mask]
    low_log10_nbe = np.array(log10_nbe)[low_range_mask]

    cross_idx = np.where(np.diff(np.sign(low_log10_nbe - target)))[0]
    log10_lr_min = -12.0  # 基于原始范围的初始估计
    if len(cross_idx) > 0:
        idx = cross_idx[-1]
        x1, x2 = low_log10_lr[idx], low_log10_lr[idx + 1]
        y1, y2 = low_log10_nbe[idx], low_log10_nbe[idx + 1]
        log10_lr_min = x1 + (target - y1) * (x2 - x1) / (y2 - y1)

    # 转换为LR值
    lr_max = 10 ** log10_lr_max
    lr_min = 10 ** log10_lr_min

    return lr_max, lr_min, log10_lr_max, log10_lr_min, log10_lr_th_range, log10_nbe, rr_original_range, rf_original_range


def adjust_extreme_lr(df, lr_max, lr_min, log10_lr_max, log10_lr_min):
    """调整数据中的极端LR值"""
    df_adjusted = df.copy()

    # 调整real-real类的极端高LR
    rr_mask = (df_adjusted['class'] == 'real_real') & (df_adjusted['LogLR'] > log10_lr_max)
    df_adjusted.loc[rr_mask, 'LogLR'] = log10_lr_max
    df_adjusted.loc[rr_mask, 'LR'] = lr_max

    # 调整real-fake类的极端低LR
    rf_mask = (df_adjusted['class'] == 'real_fake') & (df_adjusted['LogLR'] < log10_lr_min)
    df_adjusted.loc[rf_mask, 'LogLR'] = log10_lr_min
    df_adjusted.loc[rf_mask, 'LR'] = lr_min

    return df_adjusted, rr_mask.sum(), rf_mask.sum()


def plot_kde(data, label, color, ax, linestyle='-'):
    """绘制核密度估计曲线"""
    kde = gaussian_kde(data)
    x_range = np.linspace(min(data) - 1, max(data) + 1, 200)
    ax.plot(x_range, kde(x_range), color=color, label=label, linestyle=linestyle)
    ax.fill_between(x_range, kde(x_range), alpha=0.2, color=color)
    return ax


def create_sci_figures(original_df, adjusted_df, log10_lr_max, log10_lr_min,
                       log10_lr_th_range, log10_nbe, rr_original_range, rf_original_range,
                       output_dir="9ELUB"):
    """创建符合SCI论文标准的图表"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 1. NBE曲线 (Figure 1)
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(log10_lr_th_range, log10_nbe, 'k-', label='NBE curve')
    ax.axhline(y=0, color='r', linestyle='--', label='NBE = 0 (neutral system)')
    ax.axvline(x=log10_lr_max, color='b', linestyle='-.', label=f'real-real upper bound: {log10_lr_max:.1f}')
    ax.axvline(x=log10_lr_min, color='g', linestyle='-.', label=f'real-fake lower bound: {log10_lr_min:.1f}')

    # 添加原始LR系统范围标记
    ax.axvspan(rr_original_range[0], rr_original_range[1], color='blue', alpha=0.1,
               label=f'real-real original range: [{rr_original_range[0]}, {rr_original_range[1]}]')
    ax.axvspan(rf_original_range[0], rf_original_range[1], color='green', alpha=0.1,
               label=f'real-fake original range: [{rf_original_range[0]}, {rf_original_range[1]}]')

    ax.set_xlabel(r'$\log_{10}$(LR threshold)')
    ax.set_ylabel(r'$\log_{10}$(NBE)')
    ax.set_title('Normalized Bayesian Error Rate Curve')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_xlim([-15, 15])
    ax.set_ylim([-5, 5])
    ax.minorticks_on()

    # 将图例放在坐标系外右侧
    ax.legend(frameon=True, edgecolor='k', fancybox=False, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'Figure_1_NBE_Curve.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'Figure_1_NBE_Curve.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 调整前后的LogLR分布对比 (Figure 2) - 纵向排列
    fig = plt.figure(figsize=(7, 10))
    gs = gridspec.GridSpec(2, 1, hspace=0.4)

    # 原始数据分布（上图）
    ax1 = fig.add_subplot(gs[0, 0])
    rr_original = original_df[original_df['class'] == 'real_real']['LogLR'].values
    rf_original = original_df[original_df['class'] == 'real_fake']['LogLR'].values

    ax1 = plot_kde(rr_original, 'real-real', 'b', ax1)
    ax1 = plot_kde(rf_original, 'real-fake', 'r', ax1)

    # 标记原始LR系统范围
    ax1.axvline(x=rr_original_range[1], color='b', linestyle='--',
                label=f'real-real upper limit: {rr_original_range[1]}')
    ax1.axvline(x=rf_original_range[0], color='r', linestyle='--',
                label=f'real-fake lower limit: {rf_original_range[0]}')
    ax1.axvspan(rr_original_range[0], rr_original_range[1], color='blue', alpha=0.1)
    ax1.axvspan(rf_original_range[0], rf_original_range[1], color='red', alpha=0.1)

    ax1.set_xlabel(r'$\log_{10}$(LR)')
    ax1.set_ylabel('Density')
    ax1.set_title('Before ELUB Adjustment')
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.set_xlim([-25, 25])
    ax1.minorticks_on()
    ax1.legend(frameon=True, edgecolor='k', fancybox=False, loc='center left', bbox_to_anchor=(1, 0.5))

    # 调整后的数据分布（下图）
    ax2 = fig.add_subplot(gs[1, 0])
    rr_adjusted = adjusted_df[adjusted_df['class'] == 'real_real']['LogLR'].values
    rf_adjusted = adjusted_df[adjusted_df['class'] == 'real_fake']['LogLR'].values

    ax2 = plot_kde(rr_adjusted, 'real-real', 'b', ax2)
    ax2 = plot_kde(rf_adjusted, 'real-fake', 'r', ax2)

    # 标记ELUB边界
    ax2.axvline(x=log10_lr_max, color='b', linestyle='-.', label=f'real-real upper bound: {log10_lr_max:.1f}')
    ax2.axvline(x=log10_lr_min, color='g', linestyle='-.', label=f'real-fake lower bound: {log10_lr_min:.1f}')
    ax2.axvspan(rr_original_range[0], rr_original_range[1], color='blue', alpha=0.1)
    ax2.axvspan(rf_original_range[0], rf_original_range[1], color='red', alpha=0.1)

    ax2.set_xlabel(r'$\log_{10}$(LR)')
    ax2.set_ylabel('Density')
    ax2.set_title('After ELUB Adjustment')
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.set_xlim([-25, 25])
    ax2.minorticks_on()
    ax2.legend(frameon=True, edgecolor='k', fancybox=False, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'Figure_2_LR_Distribution_Comparison.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'Figure_2_LR_Distribution_Comparison.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 极端值调整的放大视图 (Figure 3)
    fig = plt.figure(figsize=(7, 10))
    gs = gridspec.GridSpec(2, 1, hspace=0.4)

    # real-real极端值区域放大（上图）
    ax1 = fig.add_subplot(gs[0, 0])
    ax1 = plot_kde(rr_original, 'Original real-real', 'b', ax1)
    ax1 = plot_kde(rr_adjusted, 'Adjusted real-real', 'b', ax1, linestyle='--')

    ax1.axvline(x=rr_original_range[1], color='gray', linestyle='--',
                label=f'Original upper limit: {rr_original_range[1]}')
    ax1.axvline(x=log10_lr_max, color='b', linestyle='-.', label=f'ELUB upper bound: {log10_lr_max:.1f}')
    ax1.axvspan(rr_original_range[0], rr_original_range[1], color='blue', alpha=0.1)

    ax1.set_xlabel(r'$\log_{10}$(LR)')
    ax1.set_ylabel('Density')
    ax1.set_title('real-real Extreme Values')
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.set_xlim([rr_original_range[1] - 1, 22])
    ax1.minorticks_on()
    ax1.legend(frameon=True, edgecolor='k', fancybox=False, loc='center left', bbox_to_anchor=(1, 0.5))

    # real-fake极端值区域放大（下图）
    ax2 = fig.add_subplot(gs[1, 0])
    ax2 = plot_kde(rf_original, 'Original real-fake', 'r', ax2)
    ax2 = plot_kde(rf_adjusted, 'Adjusted real-fake', 'r', ax2, linestyle='--')

    ax2.axvline(x=rf_original_range[0], color='gray', linestyle='--',
                label=f'Original lower limit: {rf_original_range[0]}')
    ax2.axvline(x=log10_lr_min, color='g', linestyle='-.', label=f'ELUB lower bound: {log10_lr_min:.1f}')
    ax2.axvspan(rf_original_range[0], rf_original_range[1], color='red', alpha=0.1)

    ax2.set_xlabel(r'$\log_{10}$(LR)')
    ax2.set_ylabel('Density')
    ax2.set_title('real-fake Extreme Values')
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.set_xlim([-22, rf_original_range[0] + 1])
    ax2.minorticks_on()
    ax2.legend(frameon=True, edgecolor='k', fancybox=False, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'Figure_3_Extreme_Value_Detail.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'Figure_3_Extreme_Value_Detail.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"All figures saved to {output_dir} directory")


def main():
    # 数据文件路径（请替换为您的实际路径）
    file_path = r"E:\学习目录\修正论文2_图对\5LR\csv\Test\Test_likelihood_ratios.csv"

    # 加载数据
    df, df_real_real, df_real_fake = load_data(file_path)

    # 计算ELUB边界
    lr_max, lr_min, log10_lr_max, log10_lr_min, log10_lr_th_range, log10_nbe, rr_original_range, rf_original_range = calculate_elub(
        df_real_real, df_real_fake, n_cmlr=1)

    # 调整极端LR值
    df_adjusted, rr_adj_count, rf_adj_count = adjust_extreme_lr(df, lr_max, lr_min, log10_lr_max, log10_lr_min)
    print(f"Adjustment complete - real-real extreme values: {rr_adj_count}, real-fake extreme values: {rf_adj_count}")

    # 保存调整后的数据
    output_dir = "9ELUB2"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "Adjusted_Test_likelihood_ratios.csv")
    df_adjusted.to_csv(output_path, index=False)
    print(f"Adjusted data saved to: {output_path}")

    # 创建符合SCI标准的图表
    create_sci_figures(df, df_adjusted, log10_lr_max, log10_lr_min, log10_lr_th_range, log10_nbe,
                       rr_original_range, rf_original_range, output_dir)


if __name__ == "__main__":
    main()
