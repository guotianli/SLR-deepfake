import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import roc_auc_score
import os
import joblib
import json

# ----------------------------
# 全局设置：符合Elsevier/SCI图表规范
# ----------------------------
plt.rcParams.update({
    "font.family": ["Arial", "Helvetica", "sans-serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
})


def load_data(file_path, dataset_type="test"):
    """加载数据集并提取关键信息"""
    try:
        data = pd.read_csv(file_path)
        print(f"Successfully loaded {dataset_type} data: {file_path}")
        print(f"Data shape: {data.shape}")
        print(f"Data columns: {list(data.columns)}")

        # 检查必要列
        required_cols = ['LR', 'class']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"{dataset_type} data missing required columns: {required_cols}")

        # 标签映射（与LR=Hp/Hd定义对齐）
        y_true = np.zeros(len(data), dtype=int)
        y_true[data['class'] == 'real_real'] = 1  # Hp（real-real）→ 正样本（1）
        y_true[data['class'] == 'real_fake'] = 0  # Hd（real-fake）→ 负样本（0）

        # 统计样本数量
        n_hp = np.sum(y_true == 1)
        n_hd = np.sum(y_true == 0)
        print(f"Hp (real-real) samples: {n_hp}, Hd (real-fake) samples: {n_hd}")

        # 计算对数似然比（log10(LR)）
        uncalibrated_lr = data['LR'].values
        log_uncalibrated_lr = np.log10(uncalibrated_lr)

        # 验证LR与标签的关联性
        lr_hp = uncalibrated_lr[y_true == 1]  # Hp样本的LR
        lr_hd = uncalibrated_lr[y_true == 0]  # Hd样本的LR
        print(f"\nLR distribution validation (LR>1 supports Hp):")
        print(f"Hp LR: mean={np.mean(lr_hp):.4f}, median={np.median(lr_hp):.4f}, min={np.min(lr_hp):.4e}")
        print(f"Hd LR: mean={np.mean(lr_hd):.4f}, median={np.median(lr_hd):.4f}, max={np.max(lr_hd):.4e}")
        print(f"Hp samples with LR>1: {np.mean(lr_hp > 1):.2%}, Hd samples with LR<1: {np.mean(lr_hd < 1):.2%}")

        # 警告：若LR分布不符合定义，提示检查LR计算逻辑
        if np.mean(lr_hp > 1) < 0.5 or np.mean(lr_hd < 1) < 0.5:
            print("⚠️ Warning: LR distribution does not match 'LR>1 supports Hp'! Check LR calculation.")

        print(f"\nlog(LR) range: min={np.min(log_uncalibrated_lr):.4f}, max={np.max(log_uncalibrated_lr):.4f}")
        return y_true, log_uncalibrated_lr, uncalibrated_lr, data.copy()

    except Exception as e:
        print(f"Error loading {dataset_type} data: {e}")
        return None, None, None, None


def calculate_ece(log_likelihood_ratios, y_true, pi_range):
    """计算经验交叉熵(ECE)"""
    ece_values = []
    for pi in pi_range:
        # 处理极端值，避免除零错误（保留先验优势比的真实趋势）
        denominator = 1 - pi
        if abs(denominator) < 1e-10:  # 避免分母接近0
            prior_odds = 1e10  # 合理大值
        elif pi <= 0:  # pi为负时，先验优势比为负（体现先验偏向Hd）
            prior_odds = pi / denominator
        else:
            prior_odds = pi / (1 - pi)  # 正常先验优势比

        # 裁剪prior_odds避免极端值影响ECE计算
        prior_odds_clamped = np.clip(prior_odds, -1e10, 1e10)
        posterior_odds = 10 ** log_likelihood_ratios * prior_odds_clamped  # 后验优势比
        posterior_prob_hp = posterior_odds / (1 + posterior_odds)  # P(Hp|特征)

        ece = 0.0
        # 正样本（Hp）的交叉熵：-log2(P(Hp|特征))
        if np.sum(y_true == 1) > 0:
            ece += pi * np.mean(-np.log2(np.clip(posterior_prob_hp[y_true == 1], 1e-10, 1 - 1e-10)))
        # 负样本（Hd）的交叉熵：-log2(P(Hd|特征)) = -log2(1-P(Hp|特征))
        if np.sum(y_true == 0) > 0:
            ece += (1 - pi) * np.mean(-np.log2(np.clip(1 - posterior_prob_hp[y_true == 0], 1e-10, 1 - 1e-10)))

        ece_values.append(np.min([ece, 1.0]))  # 截断ECE上限为1.0（符合行业惯例）
    return np.array(ece_values)


def calculate_cllr(log_likelihood_ratios, y_true):
    """计算似然比代价(Cllr)"""
    # 分离Hp和Hd样本的logLR
    loglr_hp = log_likelihood_ratios[y_true == 1]
    loglr_hd = log_likelihood_ratios[y_true == 0]

    # 转换为LR并裁剪极端值（避免数值溢出）
    lr_hp = np.clip(10 ** loglr_hp, 1e-10, 1e10)
    lr_hd = np.clip(10 ** loglr_hd, 1e-10, 1e10)

    n_hp, n_hd = len(lr_hp), len(lr_hd)
    if n_hp == 0 or n_hd == 0:
        print(f"Warning: Insufficient samples for Cllr (Hp={n_hp}, Hd={n_hd})")
        return float('inf')

    # Cllr计算逻辑
    cllr_pos = np.mean(np.log2(1 + 1 / lr_hp))
    cllr_neg = np.mean(np.log2(1 + lr_hd))
    cllr = 0.5 * (cllr_pos + cllr_neg)

    return np.min([cllr, 1.0])  # 截断上限为1.0


def calibrate_with_pav(uncalibrated_log_lr, y_true):
    """使用PAV算法（Isotonic Regression）校准"""
    # 转换为P(Hp|特征)（与LR定义匹配）
    prob_hp = 1 / (1 + 10 ** (-uncalibrated_log_lr))  # LR>1 → prob_hp>0.5
    prob_hp = np.clip(prob_hp, 1e-10, 1 - 1e-10)  # 避免极端值影响校准

    # 初始化PAV校准器
    pav_calibrator = IsotonicRegression(
        out_of_bounds='clip',
        y_min=1e-10,
        y_max=1 - 1e-10
    )
    pav_calibrator.fit(prob_hp.reshape(-1, 1), y_true)

    # 计算校准后的P(Hp|特征)
    calibrated_prob_hp = pav_calibrator.predict(prob_hp.reshape(-1, 1))
    calibrated_prob_hp = np.clip(calibrated_prob_hp, 1e-10, 1 - 1e-10)

    # 转换回logLR（log10(Hp/Hd)）
    calibrated_log_lr = np.log10(calibrated_prob_hp / (1 - calibrated_prob_hp))
    calibrated_log_lr = np.clip(calibrated_log_lr, -10, 10)  # 裁剪极端值

    return calibrated_log_lr, pav_calibrator


def save_calibrated_results(original_data, uncalibrated_lr, calibrated_log_lr,
                            output_dir="./8ECE/calibrated_results"):
    """保存校准结果"""
    os.makedirs(output_dir, exist_ok=True)
    result_df = original_data.copy()

    # 补充关键指标
    result_df['Original LR (Hp/Hd)'] = uncalibrated_lr
    result_df['Original logLR (log10(Hp/Hd))'] = np.log10(uncalibrated_lr)
    result_df['Calibrated logLR (log10(Hp/Hd))'] = calibrated_log_lr
    result_df['Calibrated LR (Hp/Hd)'] = 10 ** calibrated_log_lr
    result_df['P(Hp|feature)'] = 1 / (1 + 10 ** (-calibrated_log_lr))
    result_df['Label (1=Hp, 0=Hd)'] = np.where(result_df['class'] == 'real_real', 1, 0)
    result_df['Calibration Method'] = 'PAV Algorithm (Isotonic Regression)'

    # 保存CSV文件
    output_file = os.path.join(output_dir, "calibrated_pav_algorithm.csv")
    result_df.to_csv(output_file, index=False)
    print(f"Calibrated results saved to: {output_file}")
    return output_file


def save_calibration_model(calibrator, model_dir="./8ECE/models"):
    """保存校准模型"""
    os.makedirs(model_dir, exist_ok=True)
    model_data = {
        "calibrator": calibrator,
        "method": "PAV Algorithm (Isotonic Regression)",
        "input_definition": "logLR = log10(Hp/Hd), Hp=real-real, Hd=real-fake",
        "output_definition": "calibrated_logLR = log10(Hp/Hd)",
        "saved_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # 保存模型
    model_file = os.path.join(model_dir, "calibrator_pav_algorithm.joblib")
    joblib.dump(model_data, model_file)
    print(f"Calibration model saved to: {model_file}")
    return model_file


def plot_ece_comparison_with_vertical_zoom(log_uncalibrated_lr, calibrated_log_lr, y_true, save_path=None):
    """
    绘制【原始ECE图 + 纵向放大图】纵向并列子图
    核心优化：1. 保持曲线原始取值不变；2. x轴显示范围限制在[-4,4]
    """
    if save_path is None:
        save_path = "./8ECE/plots/ECE_PAV_Algorithm_with_Vertical_Zoom.tiff"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # ----------------------------
    # 1. 构建先验概率范围（确保包含pi=0.5）
    # ----------------------------
    pi_main = np.linspace(0.01, 0.99, 100)  # 主先验范围（0.01~0.99）
    # 确保包含pi=0.5的精确值
    if 0.5 not in pi_main:
        pi_main = np.append(pi_main, 0.5)
        pi_main = np.sort(pi_main)

    pi_left = np.linspace(-0.001, 0.01, 20, endpoint=False)  # 左扩展（-0.001~0.01）
    pi_right = np.linspace(0.999, 1.0001, 20, endpoint=False)  # 右扩展（0.999~1.0001）
    pi_full = np.concatenate([pi_left, pi_main, pi_right])  # 完整pi范围

    # ----------------------------
    # 2. 计算log10(prior_odds)：确保包含0点
    # ----------------------------
    log10_prior_odds_full = []
    for pi in pi_full:
        denominator = 1 - pi

        # 处理分母接近0的情况（pi接近1）
        if abs(denominator) < 1e-10:
            log10_po = 10.0  # 设定合理最大值，避免无限大

        # 处理负pi值（先验偏向Hd）
        elif pi <= 0:
            prior_odds_abs = abs(pi / denominator)
            # 避免对0取对数，设置下限
            prior_odds_abs_clamped = np.clip(prior_odds_abs, 1e-10, 1e10)
            log10_po = -np.log10(prior_odds_abs_clamped)  # 负号表示偏向Hd

        # 精确处理pi=0.5（先验均等）
        elif abs(pi - 0.5) < 1e-10:
            log10_po = 0.0  # 确保此处为0点

        # 正常pi范围（0 < pi < 1）
        else:
            prior_odds = pi / denominator
            # 裁剪极端值，避免log10溢出
            prior_odds_clamped = np.clip(prior_odds, 1e-10, 1e10)
            log10_po = np.log10(prior_odds_clamped)

        log10_prior_odds_full.append(log10_po)

    # 转换为数组并最终清理可能的异常值
    log10_prior_odds_full = np.array(log10_prior_odds_full)
    log10_prior_odds_full = np.nan_to_num(
        log10_prior_odds_full,
        nan=0.0,  # 替换NaN为0
        posinf=10.0,  # 替换正无穷为10
        neginf=-10.0  # 替换负无穷为-10
    )

    # 打印x轴数据范围（验证）
    print(f"x轴（log10(prior_odds)）数据范围：")
    print(f"  最小值：{log10_prior_odds_full.min():.4f}")
    print(f"  最大值：{log10_prior_odds_full.max():.4f}")
    print(f"  包含0点？：{0 in np.round(log10_prior_odds_full, 4)}（对应pi=0.5）")

    # ----------------------------
    # 3. 计算ECE（使用完整数据，不做筛选）
    # ----------------------------
    ece_original_full = calculate_ece(log_uncalibrated_lr, y_true, pi_full)
    ece_calibrated_full = calculate_ece(calibrated_log_lr, y_true, pi_full)
    ece_lr1_full = calculate_ece(np.zeros_like(log_uncalibrated_lr), y_true, pi_full)  # LR=1

    # ----------------------------
    # 4. 计算性能指标（基于主范围ECE）
    # ----------------------------
    ece_original_main = calculate_ece(log_uncalibrated_lr, y_true, pi_main)
    ece_calibrated_main = calculate_ece(calibrated_log_lr, y_true, pi_main)
    cllr_original = calculate_cllr(log_uncalibrated_lr, y_true)
    cllr_calibrated = calculate_cllr(calibrated_log_lr, y_true)
    prob_original = 1 / (1 + 10 ** (-log_uncalibrated_lr))
    prob_calibrated = 1 / (1 + 10 ** (-calibrated_log_lr))
    auc_original = roc_auc_score(y_true, prob_original)
    auc_calibrated = roc_auc_score(y_true, prob_calibrated)
    ece_improvement = np.mean(ece_original_main - ece_calibrated_main)

    # ----------------------------
    # 5. 创建纵向并列子图（x轴显示范围限制在[-4,4]）
    # ----------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)  # 共享x轴确保对齐
    fig.subplots_adjust(hspace=0.15)

    # ----------------------------
    # 上图：完整ECE对比图
    # ----------------------------
    ax1.plot(log10_prior_odds_full, ece_lr1_full, '-.', color='#FF7F0E',
             label='LR=1 (No Discrimination)', linewidth=1.5, zorder=1)
    ax1.plot(log10_prior_odds_full, ece_original_full, '--', color='#1F77B4',
             label=f'Uncalibrated (Cllr={cllr_original:.4f})',
             linewidth=1.5, zorder=2)
    ax1.plot(log10_prior_odds_full, ece_calibrated_full, '-', color='#2CA02C',
             label=f'PAV Calibrated (Cllr={cllr_calibrated:.4f})',
             linewidth=1.5, zorder=3)

    # 参考线与坐标轴设置（x轴显示范围限制在[-4,4]）
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.0, zorder=0)  # 标记0点
    ax1.set_ylim(0, 1.05)  # 完整范围
    ax1.set_xlim(-4, 4)  # 仅限制显示范围，不改变数据
    ax1.set_ylabel('Empirical Cross-Entropy (ECE)', fontsize=10)
    ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, zorder=0)
    ax1.legend(loc='upper left', frameon=False, fontsize=8)
    ax1.yaxis.set_major_locator(MaxNLocator(5))
    ax1.set_title('(a) Full Range of ECE Comparison', fontsize=10, pad=10)

    # ----------------------------
    # 下图：纵向放大图（固定纵轴0-0.025）
    # ----------------------------
    ax2.plot(log10_prior_odds_full, ece_original_full, '--', color='#1F77B4',
             label=f'Uncalibrated (Cllr={cllr_original:.4f})',
             linewidth=1.5, zorder=2)
    ax2.plot(log10_prior_odds_full, ece_calibrated_full, '-', color='#2CA02C',
             label=f'PAV Calibrated (Cllr={cllr_calibrated:.4f})',
             linewidth=1.5, zorder=3)

    # 参考线与坐标轴设置（x轴显示范围限制在[-4,4]）
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.0, zorder=0)  # 标记0点
    ax2.set_ylim(0, 0.025)  # 固定放大范围
    ax2.set_xlabel(r'Prior log$_{10}$(Odds)', fontsize=10)
    ax2.set_ylabel('Empirical Cross-Entropy (ECE)', fontsize=10)
    ax2.set_xlim(-4, 4)  # 仅限制显示范围，不改变数据
    ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, zorder=0)
    ax2.legend(loc='upper left', frameon=False, fontsize=8)

    # 确保x轴刻度合理分布在[-4,4]范围内
    ax2.xaxis.set_major_locator(MaxNLocator(6))
    ax2.yaxis.set_major_locator(MaxNLocator(5))  # 自动分配5个刻度
    ax2.set_title('(b) Zoomed View (0 to 0.025)', fontsize=10, pad=10)

    # 紧凑布局
    plt.tight_layout()

    # 保存图表
    plt.savefig(
        save_path,
        dpi=600,
        bbox_inches='tight',
        format='tiff',
        pil_kwargs={"compression": "tiff_lzw"}
    )
    print(f"ECE comparison with vertical zoom plot saved to: {save_path}")
    plt.close()

    # 返回评估指标
    return {
        "ece_improvement": ece_improvement,
        "cllr": cllr_calibrated,
        "auc": auc_calibrated,
        "cllr_original": cllr_original,
        "auc_original": auc_original
    }


def main():
    # 测试集数据路径（根据实际情况修改）
    test_path = r"E:\学习目录\修正论文2_图对\5LR\csv\Test\Test_likelihood_ratios.csv"

    print("=== Loading test data ===")
    test_y_true, test_log_uncalibrated_lr, test_uncalibrated_lr, test_data = load_data(test_path, "test")
    if test_y_true is None:
        print("Test data loading failed, program terminated")
        return

    # 使用PAV算法进行校准
    print("\n=== Executing PAV Algorithm Calibration ===")
    try:
        calibrated_log_lr, calibrator = calibrate_with_pav(
            test_log_uncalibrated_lr, test_y_true
        )

        # 绘制并保存带纵向放大图的ECE对比图
        plot_path = "./8ECE/plots/ECE_PAV_Algorithm_with_Vertical_Zoom.tiff"
        metrics = plot_ece_comparison_with_vertical_zoom(
            test_log_uncalibrated_lr, calibrated_log_lr, test_y_true, plot_path
        )

        # 保存校准结果
        results_path = save_calibrated_results(
            test_data, test_uncalibrated_lr, calibrated_log_lr
        )

        # 保存校准模型
        model_path = save_calibration_model(calibrator)

        # 输出评估结果
        print("\n=== PAV Algorithm Evaluation Results ===")
        print(f"  Original Cllr: {metrics['cllr_original']:.4f}")
        print(f"  Calibrated Cllr: {metrics['cllr']:.4f}")
        print(f"  Cllr Improvement: {metrics['cllr_original'] - metrics['cllr']:.4f}")
        print(f"  Original AUC: {metrics['auc_original']:.4f}")
        print(f"  Calibrated AUC: {metrics['auc']:.4f}")
        print(f"  Mean ECE Improvement: {metrics['ece_improvement']:.4f}")

        # 保存总结信息
        summary = {
            "method": "PAV Algorithm (Isotonic Regression)",
            "lr_definition": "LR = Hp/Hd, where Hp=real-real, Hd=real-fake",
            "metrics": {k: float(v) for k, v in metrics.items()},
            "sample_counts": {
                "Hp (real-real)": int(np.sum(test_y_true == 1)),
                "Hd (real-fake)": int(np.sum(test_y_true == 0))
            },
            "paths": {
                "results": results_path,
                "model": model_path,
                "plot": plot_path
            }
        }

        summary_path = "./8ECE/summary_pav.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"\nSummary results saved to: {summary_path}")

    except Exception as e:
        print(f"Error executing PAV Algorithm: {e}")

    print("\n=== PAV calibration pipeline completed ===")


if __name__ == "__main__":
    main()
