import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
import os

# ----------------------------
# 全局设置：符合期刊规范（全英文）
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
    "axes.unicode_minus": False,
})


# ----------------------------
# 数据加载函数
# ----------------------------
def load_data(input_path):
    """Load data and separate Real-Real and Real-Fake groups"""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    df = pd.read_csv(input_path)
    required_columns = ['LogLR', 'class']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV must contain columns: {', '.join(required_columns)}")

    # Separate groups
    real_real_data = df[df['class'] == 'real_real']['LogLR'].values  # Real-Real group
    real_fake_data = df[df['class'] == 'real_fake']['LogLR'].values  # Real-Fake group

    print(f"Data separation results:")
    print(f"- Real-Real samples: {len(real_real_data)}")
    print(f"- Real-Fake samples: {len(real_fake_data)}")

    if len(real_real_data) == 0 or len(real_fake_data) == 0:
        raise ValueError(f"Missing class samples: Real-Real={len(real_real_data)}, Real-Fake={len(real_fake_data)}")

    return real_real_data, real_fake_data


# ----------------------------
# 分布计算函数（核心修改：直接平滑1-CDF）
# ----------------------------
def compute_empirical_functions(real_real_data, real_fake_data):
    """
    Compute empirical functions:
    - Real-Real curve: CDF = P(LogLR ≤ x)
    - Real-Fake curve: 1-CDF = P(LogLR ≥ x) (directly calculated)
    """
    # Real-Real data (sorted x increasing)
    rr_sorted = np.sort(real_real_data)
    n_rr = len(rr_sorted)
    rr_cdf = np.arange(1, n_rr + 1) / n_rr  # CDF for Real-Real

    # Real-Fake data (sorted x increasing)
    rf_sorted = np.sort(real_fake_data)
    n_rf = len(rf_sorted)
    # Directly calculate empirical 1-CDF for Real-Fake
    rf_empirical_1cdf = np.array([np.sum(rf_sorted >= x) / n_rf for x in rf_sorted])

    return rr_sorted, rr_cdf, rf_sorted, rf_empirical_1cdf


def find_optimal_bandwidth_for_cdf(data):
    """Optimal bandwidth for Real-Real CDF using LOO cross-validation"""
    if len(data) < 5:
        return 1.0  # Default for small samples

    bandwidths = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    mse_scores = []
    loo = LeaveOneOut()

    for bw in bandwidths:
        y_true = []
        y_pred = []

        for train_idx, test_idx in loo.split(data):
            train_data = data[train_idx]
            test_point = data[test_idx][0]

            kde = stats.gaussian_kde(train_data, bw_method=bw)
            # Calculate smoothed CDF: P(LogLR ≤ x)
            pred_smoothed_cdf = kde.integrate_box_1d(-np.inf, test_point)
            # Empirical CDF as true value
            true_empirical_cdf = np.mean(train_data <= test_point)

            y_true.append(true_empirical_cdf)
            y_pred.append(pred_smoothed_cdf)

        mse = mean_squared_error(y_true, y_pred)
        mse_scores.append(mse)

    best_bw = bandwidths[np.argmin(mse_scores)]
    print(f"Optimal bandwidth for Real-Real CDF: {best_bw} (MSE: {min(mse_scores):.6f})")
    return best_bw


def find_optimal_bandwidth_for_1cdf(data):
    """Optimal bandwidth for Real-Fake 1-CDF using LOO cross-validation (direct smoothing)"""
    if len(data) < 5:
        return 1.0  # Default for small samples

    bandwidths = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    mse_scores = []
    loo = LeaveOneOut()

    for bw in bandwidths:
        y_true = []
        y_pred = []

        for train_idx, test_idx in loo.split(data):
            train_data = data[train_idx]
            test_point = data[test_idx][0]

            kde = stats.gaussian_kde(train_data, bw_method=bw)
            # Directly calculate smoothed 1-CDF: P(LogLR ≥ x)
            pred_smoothed_1cdf = kde.integrate_box_1d(test_point, np.inf)
            # Empirical 1-CDF as true value
            true_empirical_1cdf = np.mean(train_data >= test_point)

            y_true.append(true_empirical_1cdf)
            y_pred.append(pred_smoothed_1cdf)

        mse = mean_squared_error(y_true, y_pred)
        mse_scores.append(mse)

    best_bw = bandwidths[np.argmin(mse_scores)]
    print(f"Optimal bandwidth for Real-Fake 1-CDF: {best_bw} (MSE: {min(mse_scores):.6f})")
    return best_bw


def compute_smoothed_functions(real_real_data, real_fake_data, points=3000):
    """Compute smoothed functions with direct 1-CDF smoothing for Real-Fake group"""
    # Find optimal bandwidths for each function type
    bw_rr = find_optimal_bandwidth_for_cdf(real_real_data)  # Real-Real CDF
    bw_rf = find_optimal_bandwidth_for_1cdf(real_fake_data)  # Real-Fake 1-CDF (direct)

    # Determine x range with 10% extension
    all_data = np.concatenate([real_real_data, real_fake_data])
    x_min, x_max = all_data.min(), all_data.max()
    range_ext = (x_max - x_min) * 0.1
    x_range = np.linspace(x_min - range_ext, x_max + range_ext, points)

    # Smooth Real-Real CDF
    kde_rr = stats.gaussian_kde(real_real_data, bw_method=bw_rr)
    rr_cdf_smoothed = np.array([kde_rr.integrate_box_1d(-np.inf, x) for x in x_range])
    rr_cdf_smoothed = np.maximum.accumulate(rr_cdf_smoothed)  # Ensure non-decreasing

    # Smooth Real-Fake 1-CDF directly (core modification)
    kde_rf = stats.gaussian_kde(real_fake_data, bw_method=bw_rf)
    rf_1cdf_smoothed = np.array([kde_rf.integrate_box_1d(x, np.inf) for x in x_range])
    rf_1cdf_smoothed = np.minimum.accumulate(rf_1cdf_smoothed)  # Ensure non-increasing

    return x_range, rr_cdf_smoothed, rf_1cdf_smoothed, bw_rr, bw_rf


# ----------------------------
# 绘图函数（保持期刊规范）
# ----------------------------
def plot_raw_tippett(rr_x, rr_cdf, rf_x, rf_1cdf, output_path, dataset_name):
    """Plot raw Tippett plot with legend in TOP-RIGHT of axes (inside) and no title"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Red curve: Real-Real CDF
    ax.plot(rr_x, rr_cdf, 'r-', linewidth=1.5,
            label='LRs of Real-Real class')

    # Blue curve: Real-Fake 1-CDF
    ax.plot(rf_x, rf_1cdf, 'b-', linewidth=1.5,
            label='LRs of Real-Fake class')

    # Reference lines
    ax.axvline(x=0, color='green', linestyle='--', alpha=0.7,
               label='LogLR=0 (No discrimination)')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

    # Axis settings
    ax.set_xlabel('log$_{10}$ LR')
    ax.set_ylabel('Cumulative proportion')
    ax.set_ylim(-0.05, 1.05)

    # Combine x ranges
    all_x = np.concatenate([rr_x, rf_x])
    ax.set_xlim(all_x.min() - 0.5, all_x.max() + 0.5)

    # Legend in TOP-RIGHT (inside axes)
    ax.legend(
        loc='upper right',
        bbox_to_anchor=(0.95, 0.95),
        frameon=True,
        framealpha=0.9
    )
    ax.grid(alpha=0.2, linestyle='--')

    # Save without title
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, format='tiff',
                bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
    print(f"Raw Tippett plot saved to: {output_path}")
    plt.close()


def plot_fitted_tippett(x_range, rr_cdf_smoothed, rf_1cdf_smoothed, bw_rr, bw_rf,
                        output_path, dataset_name):
    """Plot fitted Tippett plot with legend in TOP-RIGHT of axes (inside) and no title"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Red curve: Real-Real CDF
    ax.plot(x_range, rr_cdf_smoothed, 'r-', linewidth=1.5,
            label=f'LRs of Real-Real class')

    # Blue curve: Real-Fake 1-CDF
    ax.plot(x_range, rf_1cdf_smoothed, 'b-', linewidth=1.5,
            label=f'LRs of Real-Fake class')


    # Reference lines
    ax.axvline(x=0, color='green', linestyle='--', alpha=0.7,
               label='LogLR=0 (No discrimination)')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

    # Axis settings
    ax.set_xlabel('log$_{10}$ LR')
    ax.set_ylabel('Cumulative proportion')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(x_range.min(), x_range.max())

    # Legend in TOP-RIGHT (inside axes)
    # Legend in TOP-RIGHT (inside axes)
    ax.legend(
        loc='upper right',
        bbox_to_anchor=(0.9, 0.9),
        frameon=True,
        framealpha=0.9
    )
    ax.grid(alpha=0.2, linestyle='--')

    # Save without title
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, format='tiff',
                bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
    print(f"Fitted Tippett plot saved to: {output_path}")
    plt.close()


# ----------------------------
# 主处理函数
# ----------------------------
def process_dataset(input_path, output_dir, dataset_name):
    """Process single dataset with direct 1-CDF smoothing for Real-Fake group"""
    print(f"\nProcessing dataset: {dataset_name}")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load data
    real_real_data, real_fake_data = load_data(input_path)

    # 2. Compute empirical functions and plot raw data
    rr_x, rr_cdf, rf_x, rf_1cdf = compute_empirical_functions(real_real_data, real_fake_data)
    raw_path = os.path.join(output_dir, f'{dataset_name}_raw_tippett.tiff')
    plot_raw_tippett(rr_x, rr_cdf, rf_x, rf_1cdf, raw_path, dataset_name)

    # 3. Compute smoothed functions with direct 1-CDF smoothing
    x_range, rr_cdf_smoothed, rf_1cdf_smoothed, bw_rr, bw_rf = compute_smoothed_functions(
        real_real_data, real_fake_data
    )

    # 4. Plot fitted data
    fitted_path = os.path.join(output_dir, f'{dataset_name}_fitted_tippett.tiff')
    plot_fitted_tippett(x_range, rr_cdf_smoothed, rf_1cdf_smoothed, bw_rr, bw_rf, fitted_path, dataset_name)

    print(f"{dataset_name} processing completed")


def main():
    """Main function to process all datasets"""
    datasets = [
        {
            "name": "Train",
            "input_path": "./5LR/csv/Train/Train_likelihood_ratios.csv",
            "output_dir": "./6Tippett/Train"
        },
        {
            "name": "Val",
            "input_path": "./5LR/csv/Val/Val_likelihood_ratios.csv",
            "output_dir": "./6Tippett/Val"
        },
        {
            "name": "Test",
            "input_path": "./5LR/csv/Test/Test_likelihood_ratios.csv",
            "output_dir": "./6Tippett/Test"
        }
    ]

    for dataset in datasets:
        try:
            process_dataset(
                input_path=dataset["input_path"],
                output_dir=dataset["output_dir"],
                dataset_name=dataset["name"]
            )
        except Exception as e:
            print(f"Failed to process {dataset['name']}: {str(e)}")

    print("\nAll Tippett plots generated successfully")


if __name__ == "__main__":
    main()