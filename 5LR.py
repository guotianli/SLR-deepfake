import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import KFold
import multiprocessing
from scipy import stats
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

# ====================== 全局配置（固定数据路径 + Elsevier规范） ======================
metric_dir = r"E:\学习目录\修正论文2_图对\3METRIC"
base_output_fig_dir = './5LR/figures'
base_output_csv_dir = './5LR/csv'

# 爱思唯尔期刊字体规范
plt.rcParams.update({
    "font.family": ["Arial", "Helvetica", "sans-serif"],
    "font.sans-serif": ["Arial"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.labelweight": "normal",
    "axes.titlesize": 12,
    "legend.fontsize": 8,
    "legend.title_fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "axes.unicode_minus": False,
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.edgecolor": "black",
})

# 颜色方案（印刷友好）
COLOR_REAL = '#7E99F4'  # Light blue (Real-Real)
COLOR_FAKE = '#CC7C71'  # Light red (Real-Fake)
COLOR_BANDWIDTH = '#663399'  # Dark purple (Bandwidth optimization)

# 确保输出目录存在
for dataset_type in ['Train', 'Val', 'Test']:
    os.makedirs(os.path.join(base_output_fig_dir, dataset_type), exist_ok=True)
    os.makedirs(os.path.join(base_output_csv_dir, dataset_type), exist_ok=True)

n_cpus = max(1, multiprocessing.cpu_count() - 1)
LOG_EPS = np.log10(np.finfo(float).eps)
LOG_MAX = np.log10(np.finfo(float).max)
# 全局存储训练集带宽（供验证集/测试集复用）
train_bandwidths = {'real_real': None, 'real_fake': None}


# ====================== 工具函数 ======================
def silverman_bandwidth(data):
    n = len(data)
    if n < 2:
        return 0.001
    sigma = min(np.std(data, ddof=1), (np.percentile(data, 75) - np.percentile(data, 25)) / 1.34)
    return 0.9 * sigma * (n ** (-1 / 5))


def interquartile_range(data):
    return np.subtract(*np.percentile(data, [75, 25]))


def estimate_density_bins(data):
    n = len(data)
    if n < 10:
        return 5
    iqr = interquartile_range(data)
    if iqr == 0:
        iqr = np.std(data, ddof=1)
    bin_width = 2 * iqr * (n ** (-1 / 3))
    return int(np.ceil((data.max() - data.min()) / (bin_width + 1e-10)))


def detect_multimodal(data):
    n_bins = estimate_density_bins(data)
    hist, edges = np.histogram(data, bins=n_bins, density=True)
    peaks, _ = find_peaks(hist, distance=0.1 * n_bins)
    return len(peaks) >= 2, peaks


def get_bandwidth_range(data, initial_bw):
    n = len(data)
    if n < 2:
        return 0.001, 0.1
    sigma = np.std(data, ddof=1)
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    is_multimodal = detect_multimodal(data)[0]
    data_range = np.max(data) - np.min(data)

    min_bw_factor = 0.2 if not is_multimodal else 0.1
    max_bw_factor = 3.0 if not is_multimodal else 5.0
    if abs(skewness) > 1.5:
        max_bw_factor = 8.0
    if kurtosis > 3:
        min_bw_factor = 0.3
    if data_range < 0.2:
        max_bw_factor = max(2.0, max_bw_factor)

    return max(0.001, initial_bw * min_bw_factor), initial_bw * max_bw_factor


def adaptive_bandwidth_regularization(data, bandwidth, min_bw=0.005):
    n = len(data)
    iqr = interquartile_range(data)
    if n < 100 or iqr < 0.1:
        min_bw = max(min_bw, 0.01)
    if abs(stats.skew(data)) > 1.0:
        bandwidth = max(bandwidth * 1.2, min_bw)
    if detect_multimodal(data)[0]:
        bandwidth = max(bandwidth * 1.5, min_bw)
    return max(min_bw, bandwidth)


def find_optimal_bandwidth_cv(data, min_bw=None, max_bw=None, n_bins=30, cv=10):
    """固定使用10折交叉验证的带宽优化函数"""
    n = len(data)
    if n < 2:
        return 0.001, [], []
    initial_bw = silverman_bandwidth(data)
    if min_bw is None or max_bw is None:
        min_bw, max_bw = get_bandwidth_range(data, initial_bw)
    if min_bw >= max_bw:
        min_bw = max_bw / 2.0
    bandwidths = np.logspace(np.log10(min_bw), np.log10(max_bw), n_bins)

    # 强制设置为10折交叉验证
    cv = 10
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    scores = []
    for bw in bandwidths:
        cv_scores = []
        for train_idx, test_idx in kf.split(data):
            kde = KernelDensity(bandwidth=bw).fit(data[train_idx].reshape(-1, 1))
            cv_scores.append(kde.score(data[test_idx].reshape(-1, 1)))
        scores.append(np.mean(cv_scores))

    best_idx = np.argmax(scores)
    best_bw = bandwidths[best_idx]
    return adaptive_bandwidth_regularization(data, best_bw), bandwidths, scores


def fit_kde_with_optimal_bandwidth(data, dataset_type, data_type, min_bw=None, max_bw=None, n_bins=30, cv=10):
    """训练集算带宽，验证集/测试集复用，明确传递10折参数"""
    if dataset_type == 'Train':
        # 训练集：计算并保存最优带宽（使用10折交叉验证）
        bw, history, scores = find_optimal_bandwidth_cv(data, min_bw, max_bw, n_bins, cv)
        train_bandwidths[data_type] = bw
        print(f"✅ Train set {data_type} optimal bandwidth: {bw:.6f} (10-fold CV)")
    else:
        # 验证集/测试集：强制复用训练集带宽
        bw = train_bandwidths.get(data_type, 0.001)
        if bw is None:
            raise ValueError(f"❌ Train set {data_type} bandwidth not found! Process Train set first.")
        # 生成带宽历史（仅用于绘图）
        initial = silverman_bandwidth(data)
        history = np.logspace(np.log10(initial * 0.2), np.log10(initial * 5), 30)
        scores = np.zeros_like(history) - 1e10
        scores[np.argmin(np.abs(history - bw))] = 0  # 标记训练集带宽位置
        print(f"✅ {dataset_type} {data_type} reuses Train set bandwidth: {bw:.6f}")

    return KernelDensity(bandwidth=bw).fit(data.reshape(-1, 1)), bw, history, scores


def calculate_likelihood_ratios(sim_real_real, sim_real_fake, kde_real, kde_fake, eps=1e-20):
    """修正LR计算逻辑，并删除极端值裁剪处理"""
    x_min = min(np.min(sim_real_real), np.min(sim_real_fake)) - 0.1
    x_max = max(np.max(sim_real_real), np.max(sim_real_fake)) + 0.1
    x_range = np.linspace(x_min, x_max, 1000)

    # 计算概率密度并归一化
    real_density = np.exp(kde_real.score_samples(x_range.reshape(-1, 1)))
    fake_density = np.exp(kde_fake.score_samples(x_range.reshape(-1, 1)))
    real_density = np.maximum(real_density, eps) / np.trapz(real_density, x_range)
    fake_density = np.maximum(fake_density, eps) / np.trapz(fake_density, x_range)

    # 插值计算似然比
    real_interp = interp1d(x_range, real_density, bounds_error=False, fill_value=(real_density[0], real_density[-1]))
    fake_interp = interp1d(x_range, fake_density, bounds_error=False, fill_value=(fake_density[0], fake_density[-1]))

    # 核心修正：LogLR = log10(P(E|Hp)/P(E|Hd)) = log10(real_interp / fake_interp)
    log_lr_real = (np.log(real_interp(sim_real_real)) - np.log(fake_interp(sim_real_real))) / np.log(10)
    log_lr_fake = (np.log(real_interp(sim_real_fake)) - np.log(fake_interp(sim_real_fake))) / np.log(10)

    # 删除极端值处理：不再进行分位数裁剪，保留原始计算结果
    return {
        'lr_real': 10 ** log_lr_real,
        'lr_fake': 10 ** log_lr_fake,
        'log_lr_real': log_lr_real,  # 未裁剪的原始值
        'log_lr_fake': log_lr_fake,  # 未裁剪的原始值
        'x_range': x_range,
        'real_density': real_density,
        'fake_density': fake_density
    }


# ====================== 绘图函数（完整实现） ======================
def plot_similarity_distribution(dataset_type, sim_real_real, sim_real_fake, output_dir):
    try:
        fig = plt.figure(figsize=(8, 6))
        gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)

        # 1. 直方图子图
        ax1 = fig.add_subplot(gs[0])
        bins_real = estimate_density_bins(sim_real_real)
        bins_fake = estimate_density_bins(sim_real_fake)

        ax1.hist(sim_real_real, bins=bins_real, density=False,
                 color=COLOR_REAL, edgecolor='black', linewidth=0.5,
                 alpha=0.7, label='Real-Real')
        ax1.hist(sim_real_fake, bins=bins_fake, density=False,
                 color=COLOR_FAKE, edgecolor='black', linewidth=0.5,
                 alpha=0.7, label='Real-Fake')

        # 统计文本（英文）
        stats_real = (f"Real-Real (N={len(sim_real_real)}):\n"
                      f"Mean={np.mean(sim_real_real):.4f}, Std={np.std(sim_real_real):.4f}")
        stats_fake = (f"Real-Fake (N={len(sim_real_fake)}):\n"
                      f"Mean={np.mean(sim_real_fake):.4f}, Std={np.std(sim_real_fake):.4f}")

        ax1.text(0.02, 0.98, stats_real, transform=ax1.transAxes,
                 va='top', fontsize=8,
                 bbox=dict(facecolor='white', edgecolor='black', alpha=1.0, boxstyle='round,pad=0.3'))
        ax1.text(0.52, 0.98, stats_fake, transform=ax1.transAxes,
                 va='top', fontsize=8,
                 bbox=dict(facecolor='white', edgecolor='black', alpha=1.0, boxstyle='round,pad=0.3'))

        # 图例：顶部中央，两列布局
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2,
                  frameon=True, framealpha=1.0, edgecolor='black', fontsize=8)
        ax1.set_xlabel('Similarity', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.grid(False)

        # 2. 密度曲子图（带填充）
        ax2 = fig.add_subplot(gs[1])
        x_range = np.linspace(min(sim_real_real.min(), sim_real_fake.min()) - 0.1,
                              max(sim_real_real.max(), sim_real_fake.max()) + 0.1, 1000)

        kde_real = KernelDensity(bandwidth=silverman_bandwidth(sim_real_real)).fit(sim_real_real.reshape(-1, 1))
        kde_fake = KernelDensity(bandwidth=silverman_bandwidth(sim_real_fake)).fit(sim_real_fake.reshape(-1, 1))

        real_curve = np.exp(kde_real.score_samples(x_range.reshape(-1, 1)))
        fake_curve = np.exp(kde_fake.score_samples(x_range.reshape(-1, 1)))

        ax2.plot(x_range, real_curve, color=COLOR_REAL, linewidth=1.5, label='Real-Real')
        ax2.fill_between(x_range, real_curve, color=COLOR_REAL, alpha=0.3)
        ax2.plot(x_range, fake_curve, color=COLOR_FAKE, linewidth=1.5, label='Real-Fake')
        ax2.fill_between(x_range, fake_curve, color=COLOR_FAKE, alpha=0.3)

        # 图例：顶部中央，两列布局
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2,
                  frameon=True, framealpha=1.0, edgecolor='black', fontsize=8)
        ax2.set_xlabel('Similarity', fontsize=10)
        ax2.set_ylabel('Density', fontsize=10)
        ax2.grid(False)

        # 移除图注，调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig_path = os.path.join(output_dir, f'{dataset_type}_similarity_distribution.tiff')
        fig.savefig(fig_path, dpi=600, format='tiff', bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
        plt.close()

        return {'stats_real_real': stats_real, 'stats_real_fake': stats_fake}
    except Exception as e:
        print(f"❌ Similarity distribution plot failed: {e}")
        return None


def plot_kde_with_peak_labels(dataset_type, x_range, real_density, fake_density, bw_real, bw_fake, output_dir):
    try:
        fig, ax = plt.subplots(figsize=(8, 6))

        # 绘制KDE曲线与填充
        ax.plot(x_range, real_density, color=COLOR_REAL, linewidth=1.5, label='Real-Real')
        ax.fill_between(x_range, real_density, color=COLOR_REAL, alpha=0.3)
        ax.plot(x_range, fake_density, color=COLOR_FAKE, linewidth=1.5, label='Real-Fake')
        ax.fill_between(x_range, fake_density, color=COLOR_FAKE, alpha=0.3)

        # 峰值标注
        real_peak_idx = np.argmax(real_density)
        fake_peak_idx = np.argmax(fake_density)
        real_peak_x, real_peak_y = x_range[real_peak_idx], real_density[real_peak_idx]
        fake_peak_x, fake_peak_y = x_range[fake_peak_idx], fake_density[fake_peak_idx]

        ax.scatter(real_peak_x, real_peak_y, color='black', s=30, zorder=3)
        ax.scatter(fake_peak_x, fake_peak_y, color='black', s=30, zorder=3)

        ax.annotate(f'Peak: {real_peak_x:.4f}',
                    xy=(real_peak_x, real_peak_y),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, bbox=dict(facecolor='white', edgecolor='black', alpha=1.0, boxstyle='round,pad=0.2'))
        ax.annotate(f'Peak: {fake_peak_x:.4f}',
                    xy=(fake_peak_x, fake_peak_y),
                    xytext=(5, -15), textcoords='offset points',
                    fontsize=8, bbox=dict(facecolor='white', edgecolor='black', alpha=1.0, boxstyle='round,pad=0.2'))

        # 图例：顶部中央，两列布局
        custom_lines = [
            Line2D([0], [0], color=COLOR_REAL, lw=1.5),
            Line2D([0], [0], color=COLOR_FAKE, lw=1.5)
        ]
        legend_labels = [
            f'Real-Real (BW: {bw_real:.6f})',
            f'Real-Fake (BW: {bw_fake:.6f})'
        ]

        ax.legend(custom_lines, legend_labels,
                  loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2,
                  frameon=True, framealpha=1.0, edgecolor='black', fontsize=8)

        ax.set_xlabel('Similarity', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.grid(False)

        # 移除图注，调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig_path = os.path.join(output_dir, f'{dataset_type}_kde_comparison.tiff')
        fig.savefig(fig_path, dpi=600, format='tiff', bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
        plt.close()
    except Exception as e:
        print(f"❌ KDE comparison plot failed: {e}")


def plot_log_likelihood_ratios(dataset_type, lr_results, output_dir):
    try:
        fig, ax = plt.subplots(figsize=(8, 6))

        # 绘制直方图
        ax.hist(lr_results['log_lr_real'], bins=50, density=True,
                color=COLOR_REAL, edgecolor='black', linewidth=0.5,
                alpha=0.7, label='Real-Real')
        ax.hist(lr_results['log_lr_fake'], bins=50, density=True,
                color=COLOR_FAKE, edgecolor='black', linewidth=0.5,
                alpha=0.7, label='Real-Fake')

        # 参考线
        ax.axvline(0, color='black', linestyle='--', linewidth=0.75, alpha=0.7)

        # 统计文本（英文，增加极端值信息）
        stats_text = (
            f"Real-Real: Mean={np.mean(lr_results['log_lr_real']):.4f} ± {np.std(lr_results['log_lr_real']):.4f}\n"
            f"Range: [{np.min(lr_results['log_lr_real']):.2f}, {np.max(lr_results['log_lr_real']):.2f}]\n\n"
            f"Real-Fake: Mean={np.mean(lr_results['log_lr_fake']):.4f} ± {np.std(lr_results['log_lr_fake']):.4f}\n"
            f"Range: [{np.min(lr_results['log_lr_fake']):.2f}, {np.max(lr_results['log_lr_fake']):.2f}]")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                va='top', fontsize=8,
                bbox=dict(facecolor='white', edgecolor='black', alpha=1.0, boxstyle='round,pad=0.3'))

        # 图例：顶部中央，两列布局
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2,
                  frameon=True, framealpha=1.0, edgecolor='black', fontsize=8)
        ax.set_xlabel('Log10 Likelihood Ratio', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.grid(False)

        # 移除图注，调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig_path = os.path.join(output_dir, f'{dataset_type}_log_likelihood_ratio.tiff')
        fig.savefig(fig_path, dpi=600, format='tiff', bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
        plt.close()
    except Exception as e:
        print(f"❌ Likelihood ratio plot failed: {e}")


def plot_bandwidth_optimization(bandwidth_type, bandwidths, scores, best_bw, dataset_type, output_dir):
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        # 紫色曲线与填充
        ax.plot(bandwidths, scores, 'o-', color=COLOR_BANDWIDTH, alpha=0.7, markersize=5, linewidth=1.5)
        ax.fill_between(bandwidths, scores, color=COLOR_BANDWIDTH, alpha=0.3)
        ax.axvline(best_bw, color=COLOR_BANDWIDTH, linestyle='--', linewidth=0.75,
                   label=f'Optimal: {best_bw:.6f} (10-fold CV)')
        ax.set_xscale('log')

        # 图例：顶部中央，两列布局
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2,
                  frameon=True, framealpha=1.0, edgecolor='black', fontsize=8)
        ax.set_xlabel('Bandwidth', fontsize=10)
        ax.set_ylabel('Cross-Validation Log Likelihood', fontsize=10)
        ax.grid(False)

        # 移除图注，调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # 保存
        fig_path = os.path.join(output_dir, f'{dataset_type}_{bandwidth_type}_bandwidth_optimization.tiff')
        fig.savefig(fig_path, dpi=600, format='tiff', bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
        plt.close()
    except Exception as e:
        print(f"❌ Bandwidth optimization plot failed: {e}")


# ====================== 数据保存函数 ======================
def save_likelihood_ratios_to_csv(dataset_type, real_file, fake_file, lr_results, output_dir):
    try:
        df_real = pd.read_csv(real_file)
        df_fake = pd.read_csv(fake_file)
        df_real['LR'] = lr_results['lr_real']
        df_real['LogLR'] = lr_results['log_lr_real']  # 保存未裁剪的原始值
        df_real['class'] = 'real_real'
        df_fake['LR'] = lr_results['lr_fake']
        df_fake['LogLR'] = lr_results['log_lr_fake']  # 保存未裁剪的原始值
        df_fake['class'] = 'real_fake'

        output_path = os.path.join(output_dir, f'{dataset_type}_likelihood_ratios.csv')
        pd.concat([df_real, df_fake], ignore_index=True).to_csv(output_path, index=False)
        print(f"✅ Likelihood ratio results saved to: {output_path}")
    except Exception as e:
        print(f"❌ Failed to save likelihood ratio CSV: {e}")


# ====================== 主处理函数 ======================
def process_dataset(dataset_type, real_real_file, real_fake_file, min_bw=0.001, max_bw=0.1, n_bins=30, cv=10):
    """处理数据集时明确指定10折交叉验证"""
    try:
        output_fig_dir = os.path.join(base_output_fig_dir, dataset_type)
        output_csv_dir = os.path.join(base_output_csv_dir, dataset_type)
        os.makedirs(output_fig_dir, exist_ok=True)

        # 验证文件存在性
        for file in [real_real_file, real_fake_file]:
            if not os.path.exists(file):
                raise FileNotFoundError(f"❌ File not found: {file}")

        # 加载数据
        df_real = pd.read_csv(real_real_file)
        df_fake = pd.read_csv(real_fake_file)
        sim_real = df_real['similarity'].values
        sim_fake = df_fake['similarity'].values

        print(f"\n===== Processing {dataset_type} dataset =====")
        print(f"Real-Real samples: {len(sim_real)}, Real-Fake samples: {len(sim_fake)} (using 10-fold CV)")

        # 绘制相似度分布
        sim_stats = plot_similarity_distribution(dataset_type, sim_real, sim_fake, output_fig_dir)
        if not sim_stats:
            return None

        # 拟合KDE模型 - Real-Real（传递10折参数）
        kde_real, bw_real, bw_hist_real, scores_real = fit_kde_with_optimal_bandwidth(
            sim_real, dataset_type, 'real_real', min_bw, max_bw, n_bins, cv
        )
        plot_bandwidth_optimization('real_real', bw_hist_real, scores_real, bw_real, dataset_type, output_fig_dir)

        # 拟合KDE模型 - Real-Fake（传递10折参数）
        kde_fake, bw_fake, bw_hist_fake, scores_fake = fit_kde_with_optimal_bandwidth(
            sim_fake, dataset_type, 'real_fake', min_bw, max_bw, n_bins, cv
        )
        plot_bandwidth_optimization('real_fake', bw_hist_fake, scores_fake, bw_fake, dataset_type, output_fig_dir)

        # 计算似然比（已修正逻辑，无极端值处理）
        lr_results = calculate_likelihood_ratios(sim_real, sim_fake, kde_real, kde_fake)

        # 绘制KDE对比图
        plot_kde_with_peak_labels(
            dataset_type,
            lr_results['x_range'],
            lr_results['real_density'],
            lr_results['fake_density'],
            bw_real,
            bw_fake,
            output_fig_dir
        )

        # 后续处理
        plot_log_likelihood_ratios(dataset_type, lr_results, output_fig_dir)
        save_likelihood_ratios_to_csv(dataset_type, real_real_file, real_fake_file, lr_results, output_csv_dir)

        return {
            'dataset_type': dataset_type,
            'stats_real_real': sim_stats['stats_real_real'],
            'stats_real_fake': sim_stats['stats_real_fake'],
            'mean_log_lr_real': np.mean(lr_results['log_lr_real']),
            'mean_log_lr_fake': np.mean(lr_results['log_lr_fake']),
            'min_log_lr_real': np.min(lr_results['log_lr_real']),  # 新增：记录最小值
            'max_log_lr_real': np.max(lr_results['log_lr_real']),  # 新增：记录最大值
            'min_log_lr_fake': np.min(lr_results['log_lr_fake']),
            'max_log_lr_fake': np.max(lr_results['log_lr_fake']),
            'optimal_bandwidth_real': bw_real,
            'optimal_bandwidth_fake': bw_fake,
            'real_real_size': len(sim_real),
            'real_fake_size': len(sim_fake)
        }
    except Exception as e:
        print(f"❌ {dataset_type} processing failed: {e}")
        return None


def main():
    print("===== Full Dataset Likelihood Ratio Analysis Tool (Revised) =====")
    print(f"Data path: {metric_dir}")
    print(f"Output figures path: {base_output_fig_dir}")
    print(f"Output CSV path: {base_output_csv_dir}")

    # 定义6个文件的完整路径（按文件名精确匹配）
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

    # 验证所有文件是否存在
    missing_files = []
    for ds_type, files in dataset_files.items():
        for file_type, file_path in files.items():
            if not os.path.exists(file_path):
                missing_files.append(f"{ds_type} {file_type}: {file_path}")

    if missing_files:
        print("❌ Missing files, cannot continue processing:")
        for file in missing_files:
            print(f"  - {file}")
        return

    # 处理参数（明确设置cv=10）
    min_bw, max_bw = 0.001, 0.1
    n_bins = 30
    cv = 10  # Fixed to 10-fold globally

    # 按顺序处理数据集（训练集→验证集→测试集）
    results = []

    # 1. 处理训练集（计算带宽，使用10折交叉验证）
    train_result = process_dataset(
        'Train',
        dataset_files['Train']['real_real'],
        dataset_files['Train']['real_fake'],
        min_bw, max_bw, n_bins, cv
    )
    if train_result:
        results.append(train_result)
    else:
        print("❌ Train set processing failed, terminating workflow")
        return

    # 2. 处理验证集（复用训练集带宽）
    val_result = process_dataset(
        'Val',
        dataset_files['Val']['real_real'],
        dataset_files['Val']['real_fake'],
        min_bw, max_bw, n_bins, cv
    )
    if val_result:
        results.append(val_result)

    # 3. 处理测试集（复用训练集带宽）
    test_result = process_dataset(
        'Test',
        dataset_files['Test']['real_real'],
        dataset_files['Test']['real_fake'],
        min_bw, max_bw, n_bins, cv
    )
    if test_result:
        results.append(test_result)

    # 输出汇总报告（增加极端值信息）
    if results:
        print("\n===== Processing Results Summary =====")
        for res in results:
            print(f"\n{res['dataset_type']} dataset:")
            print(f"  Real-Real samples: {res['real_real_size']}, Optimal bandwidth: {res['optimal_bandwidth_real']:.6f}")
            print(f"  Real-Fake samples: {res['real_fake_size']}, Optimal bandwidth: {res['optimal_bandwidth_fake']:.6f}")
            print(
                f"  Real samples LogLR: Mean={res['mean_log_lr_real']:.4f}, Range=[{res['min_log_lr_real']:.2f}, {res['max_log_lr_real']:.2f}]")
            print(
                f"  Fake samples LogLR: Mean={res['mean_log_lr_fake']:.4f}, Range=[{res['min_log_lr_fake']:.2f}, {res['max_log_lr_fake']:.2f}]")
            print(f"  Cross-validation folds: 10-fold")

    print("\n===== All datasets processed =====")


if __name__ == '__main__':
    main()
