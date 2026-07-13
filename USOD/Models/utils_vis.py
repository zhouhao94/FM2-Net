import os
import re
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


EPS = 1e-8


def _normalize_single_to_01_gamma(
    x,
    lower_percentile=1.0,
    upper_percentile=97.0,
    gamma=0.6
):
    """
    单图归一化 + gamma 增强。

    gamma < 1:
        会提升中等响应，使蓝绿区域更容易变成黄红。

    upper_percentile 从 99 降到 97:
        可以避免极少数异常强响应把整体压暗。
    """
    if torch.is_tensor(x):
        x_np = x.detach().float().cpu().numpy()
    else:
        x_np = np.asarray(x, dtype=np.float32)

    vmin = np.percentile(x_np, lower_percentile)
    vmax = np.percentile(x_np, upper_percentile)

    x_np = np.clip(x_np, vmin, vmax)
    x_np = (x_np - vmin) / (vmax - vmin + EPS)

    # gamma correction
    x_np = np.power(x_np, gamma)

    return torch.from_numpy(x_np).float()


def _safe_name(name):
    name = str(name)
    name = os.path.basename(name)
    name = os.path.splitext(name)[0]
    name = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)
    return name


def _normalize_with_fixed_range(x, vmin, vmax):
    if torch.is_tensor(x):
        x_np = x.detach().float().cpu().numpy()
    else:
        x_np = np.asarray(x, dtype=np.float32)

    x_np = np.clip(x_np, vmin, vmax)
    x_np = (x_np - vmin) / (vmax - vmin + EPS)

    return torch.from_numpy(x_np).float()


def compute_pair_heatmap_ranges_per_sample(
    x_spatial_low,
    x_spatial_high,
    mode="abs_mean",
    lower_percentile=1.0,
    upper_percentile=99.0
):
    """
    复现 visualize_high_low_frequency_pair_no_pca 中
    _normalize_pair_to_01(low_map[i], high_map[i]) 的 per-sample 归一化范围。

    返回:
        ranges: list of dict
        [
            {"vmin": ..., "vmax": ...},
            {"vmin": ..., "vmax": ...},
            ...
        ]
    """
    x_low_cpu = x_spatial_low.detach().float().cpu()
    x_high_cpu = x_spatial_high.detach().float().cpu()

    low_map = _to_spatial_feature_map(x_low_cpu, mode=mode)    # [B, H, W]
    high_map = _to_spatial_feature_map(x_high_cpu, mode=mode)  # [B, H, W]

    batch_size = low_map.shape[0]

    ranges = []

    for i in range(batch_size):
        low_np = low_map[i].numpy()
        high_np = high_map[i].numpy()

        all_value = np.concatenate(
            [
                low_np.reshape(-1),
                high_np.reshape(-1)
            ],
            axis=0
        )

        vmin = np.percentile(all_value, lower_percentile)
        vmax = np.percentile(all_value, upper_percentile)

        ranges.append(
            {
                "vmin": float(vmin),
                "vmax": float(vmax)
            }
        )

    return ranges


def _normalize_pair_to_01(
    x1,
    x2,
    lower_percentile=1.0,
    upper_percentile=99.0
):
    if torch.is_tensor(x1):
        x1_np = x1.detach().float().cpu().numpy()
    else:
        x1_np = np.asarray(x1, dtype=np.float32)

    if torch.is_tensor(x2):
        x2_np = x2.detach().float().cpu().numpy()
    else:
        x2_np = np.asarray(x2, dtype=np.float32)

    all_value = np.concatenate([x1_np.reshape(-1), x2_np.reshape(-1)], axis=0)

    low = np.percentile(all_value, lower_percentile)
    high = np.percentile(all_value, upper_percentile)

    x1_np = np.clip(x1_np, low, high)
    x2_np = np.clip(x2_np, low, high)

    x1_np = (x1_np - low) / (high - low + EPS)
    x2_np = (x2_np - low) / (high - low + EPS)

    return torch.from_numpy(x1_np).float(), torch.from_numpy(x2_np).float()


def _normalize_single_to_01(
    x,
    lower_percentile=1.0,
    upper_percentile=99.0
):
    if torch.is_tensor(x):
        x_np = x.detach().float().cpu().numpy()
    else:
        x_np = np.asarray(x, dtype=np.float32)

    vmin = np.percentile(x_np, lower_percentile)
    vmax = np.percentile(x_np, upper_percentile)

    x_np = np.clip(x_np, vmin, vmax)
    x_np = (x_np - vmin) / (vmax - vmin + EPS)

    return torch.from_numpy(x_np).float()


def _to_spatial_feature_map(x, mode="abs_mean"):
    """
    x: [B, C, H, W]

    return:
        [B, H, W]
    """
    if mode == "abs_mean":
        return x.abs().mean(dim=1)

    elif mode == "mean":
        return x.mean(dim=1)

    elif mode == "max":
        return x.abs().max(dim=1)[0]

    elif mode == "l2":
        return torch.sqrt((x ** 2).mean(dim=1) + EPS)

    else:
        raise ValueError(f"Unsupported mode: {mode}")


def _save_single_map(
    image_map,
    save_path,
    vmin=0.0,
    vmax=1.0,
    cmap="jet"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(4, 4))
    plt.imshow(image_map, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def _fft_power_from_feature(
    feature,
    remove_dc=True
):
    """
    feature: [C, H, W]

    return:
        power: [H, W]
        已 fftshift，低频在中心。
    """
    feat = feature.detach().float().cpu()

    if feat.dim() != 3:
        raise ValueError(f"Expected [C, H, W], but got {feat.shape}")

    if remove_dc:
        feat = feat - feat.mean(dim=(-2, -1), keepdim=True)

    fft = torch.fft.fft2(feat, dim=(-2, -1), norm="ortho")
    power = fft.real ** 2 + fft.imag ** 2
    power = power.mean(dim=0)
    power = torch.fft.fftshift(power)

    return power.numpy()


def _make_radius_grid(h, w):
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cy, cx = h // 2, w // 2

    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius = radius / (radius.max() + EPS)

    return radius


def _radial_energy_profile(
    power,
    num_bins=60
):
    h, w = power.shape
    radius = _make_radius_grid(h, w)

    bin_idx = np.floor(radius * num_bins).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, num_bins - 1)

    energy = np.bincount(
        bin_idx.reshape(-1),
        weights=power.reshape(-1),
        minlength=num_bins
    ).astype(np.float64)

    energy = energy / (energy.sum() + EPS)
    radius_centers = (np.arange(num_bins) + 0.5) / num_bins

    return radius_centers, energy


def _frequency_energy_ratio(
    power,
    low_cut=0.25,
    high_cut=0.50
):
    h, w = power.shape
    radius = _make_radius_grid(h, w)

    total_energy = power.sum() + EPS

    low_energy = power[radius <= low_cut].sum()
    high_energy = power[radius >= high_cut].sum()
    mid_energy = total_energy - low_energy - high_energy

    return {
        "low_energy_ratio": float(low_energy / total_energy),
        "mid_energy_ratio": float(mid_energy / total_energy),
        "high_energy_ratio": float(high_energy / total_energy),
    }


def _save_fft_pair(
    low_power,
    high_power,
    save_path,
    cmap="magma"
):
    low_log = np.log1p(low_power)
    high_log = np.log1p(high_power)

    low_norm, high_norm = _normalize_pair_to_01(
        low_log,
        high_log,
        lower_percentile=1.0,
        upper_percentile=99.0
    )

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(low_norm.numpy(), cmap=cmap, vmin=0, vmax=1)
    plt.title("Low FFT spectrum")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(high_norm.numpy(), cmap=cmap, vmin=0, vmax=1)
    plt.title("High FFT spectrum")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def _save_radial_profile_pair(
    low_power,
    high_power,
    save_path,
    num_bins=60
):
    low_radius, low_profile = _radial_energy_profile(
        low_power,
        num_bins=num_bins
    )

    high_radius, high_profile = _radial_energy_profile(
        high_power,
        num_bins=num_bins
    )

    plt.figure(figsize=(7, 5))
    plt.plot(low_radius, low_profile, label="Low feature")
    plt.plot(high_radius, high_profile, label="High feature")

    plt.xlabel("Normalized frequency radius")
    plt.ylabel("Energy ratio")
    plt.title("Radial frequency energy profile")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def _save_frequency_ratio_bar(
    low_ratio,
    high_ratio,
    save_path
):
    categories = ["low", "mid", "high"]

    low_values = [
        low_ratio["low_energy_ratio"],
        low_ratio["mid_energy_ratio"],
        low_ratio["high_energy_ratio"],
    ]

    high_values = [
        high_ratio["low_energy_ratio"],
        high_ratio["mid_energy_ratio"],
        high_ratio["high_energy_ratio"],
    ]

    x = np.arange(len(categories))
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, low_values, width, label="Low feature")
    plt.bar(x + width / 2, high_values, width, label="High feature")

    plt.xticks(x, categories)
    plt.ylabel("Energy ratio")
    plt.title("Frequency energy ratio")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def _to_diff_feature_map(
    x_low,
    x_high,
    diff_mode="abs_mean"
):
    """
    计算 high - low 的差值特征图。

    输入:
        x_low:  [B, C, H, W]
        x_high: [B, C, H, W]

    diff_mode:
        abs_mean:
            mean(abs(high - low), channel)
            表示差异强度，不区分增强还是减弱。

        signed_mean:
            mean(high - low, channel)
            正值表示 high > low，负值表示 high < low。
    """
    diff = x_high - x_low

    if diff_mode == "abs_mean":
        diff_map = diff.abs().mean(dim=1)

    elif diff_mode == "signed_mean":
        diff_map = diff.mean(dim=1)

    elif diff_mode == "l2":
        diff_map = torch.sqrt((diff ** 2).mean(dim=1) + EPS)

    else:
        raise ValueError(f"Unsupported diff_mode: {diff_mode}")

    return diff_map


def _normalize_signed_to_symmetric_range(
    x,
    percentile=99.0
):
    """
    signed map 使用对称范围归一化。

    输出仍然是原始 signed map，只是返回对称的 vmin / vmax。
    用于 bwr / seismic 这种双极性 colormap。

    蓝色: negative
    白色: near zero
    红色: positive
    """
    if torch.is_tensor(x):
        x_np = x.detach().float().cpu().numpy()
    else:
        x_np = np.asarray(x, dtype=np.float32)

    abs_max = np.percentile(np.abs(x_np), percentile)
    abs_max = max(abs_max, EPS)

    x_np = np.clip(x_np, -abs_max, abs_max)

    return x_np, -abs_max, abs_max


def visualize_high_low_frequency_pair_no_pca(
    x_spatial_low,
    x_spatial_high,
    names,
    save_dir,
    prefix="feature",
    mode="abs_mean",
    cmap="jet",
    fft_cmap="magma",
    save_heatmap=True,
    save_fft=True,
    save_radial=True,
    save_ratio_bar=True,
    save_metrics=True,
    low_cut=0.25,
    high_cut=0.50,
    remove_dc=True,
    num_bins=60,
    fixed_ranges=None,
    high_norm_strategy="auto",
    save_diff=True,
    diff_mode="abs_mean",
    diff_cmap="jet"
):
    """
    不做 PCA，仅可视化 low / high 的空间响应和频谱差异。

    输入:
        x_spatial_low:  [B, C, H, W]
        x_spatial_high: [B, C, H, W]
        names:          长度为 B 的样本名称列表

    保存:
        1. heatmap:
            {sample}_{prefix}_low_feature_heatmap_{mode}.png
            {sample}_{prefix}_high_feature_heatmap_{mode}.png

        2. FFT spectrum:
            {sample}_{prefix}_fft_spectrum_low_vs_high.png

        3. radial profile:
            {sample}_{prefix}_radial_profile_low_vs_high.png

        4. frequency ratio bar:
            {sample}_{prefix}_frequency_ratio_bar.png

        5. CSV:
            {prefix}_frequency_metrics.csv
            {prefix}_frequency_metrics_mean.csv
    """
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(names, str):
        names = [names]
    else:
        names = list(names)

    if x_spatial_low.dim() != 4 or x_spatial_high.dim() != 4:
        raise ValueError(
            "x_spatial_low and x_spatial_high must be [B, C, H, W]"
        )

    if x_spatial_low.shape != x_spatial_high.shape:
        raise ValueError(
            f"x_spatial_low shape {x_spatial_low.shape} "
            f"!= x_spatial_high shape {x_spatial_high.shape}"
        )

    batch_size = x_spatial_low.shape[0]

    if len(names) != batch_size:
        raise ValueError(
            f"样本名称数量为 {len(names)}，但 batch size 为 {batch_size}"
        )

    x_low_cpu = x_spatial_low.detach().float().cpu()
    x_high_cpu = x_spatial_high.detach().float().cpu()

    if save_heatmap:
        low_map = _to_spatial_feature_map(
            x_low_cpu,
            mode=mode
        )

        high_map = _to_spatial_feature_map(
            x_high_cpu,
            mode=mode
        )

    if save_diff:
        diff_map = _to_diff_feature_map(
            x_low=x_low_cpu,
            x_high=x_high_cpu,
            diff_mode=diff_mode
        )

    metric_rows = []

    for i, name in enumerate(names):
        sample_name = _safe_name(name)

        # ============================================================
        # 1. Heatmap 特征图
        # ============================================================
        if save_heatmap:
            if fixed_ranges is not None:
                vmin = fixed_ranges[i]["vmin"]
                vmax = fixed_ranges[i]["vmax"]

                low_norm = _normalize_with_fixed_range(
                    low_map[i],
                    vmin=vmin,
                    vmax=vmax
                )
                '''
                high_norm = _normalize_with_fixed_range(
                    high_map[i],
                    vmin=vmin,
                    vmax=vmax
                )
                '''
                if high_norm_strategy == "fixed":
                    # high 也用第一次范围，绝对强度可比，但可能全红
                    high_norm = _normalize_with_fixed_range(
                        high_map[i],
                        vmin=vmin,
                        vmax=vmax
                    )

                elif high_norm_strategy == "auto":
                    # high 自己归一化，适合看结构，不容易全红
                    high_norm = _normalize_single_to_01(
                        high_map[i],
                        lower_percentile=1.0,
                        upper_percentile=99.0
                    )

                elif high_norm_strategy == "log_auto":
                    # 对 enhanced_high 先做 log 压缩，再单独归一化
                    high_tmp = torch.log1p(high_map[i])
                    high_norm = _normalize_single_to_01(
                        high_tmp,
                        lower_percentile=1.0,
                        upper_percentile=99.0
                    )

                else:
                    raise ValueError(f"Unknown high_norm_strategy: {high_norm_strategy}")
            else:
                low_norm, high_norm = _normalize_pair_to_01(
                    low_map[i],
                    high_map[i],
                    lower_percentile=1.0,
                    upper_percentile=99.0
                )

            low_heatmap_save_path = os.path.join(
                save_dir,
                f"{sample_name}_{prefix}_high_feature_heatmap_{mode}.png"
            )

            high_heatmap_save_path = os.path.join(
                save_dir,
                f"{sample_name}_{prefix}_high_enhanced_feature_heatmap_{mode}.png"
            )

            _save_single_map(
                image_map=low_norm.numpy(),
                save_path=low_heatmap_save_path,
                vmin=0.0,
                vmax=1.0,
                cmap=cmap
            )

            _save_single_map(
                image_map=high_norm.numpy(),
                save_path=high_heatmap_save_path,
                vmin=0.0,
                vmax=1.0,
                cmap=cmap
            )

        # ============================================================
        # 额外保存 enhanced - low 的差值特征图
        # ============================================================
        if save_diff:
            diff_save_path = os.path.join(
                save_dir,
                f"{sample_name}_{prefix}_diff_high_minus_low_heatmap_{diff_mode}.png"
            )

            if diff_mode in ["abs_mean", "l2"]:
                # 差异强度图：越红表示 enhanced 和 low 差异越大
                diff_norm = _normalize_single_to_01_gamma(
                    diff_map[i],
                    lower_percentile=1.0,
                    upper_percentile=97.0,
                    gamma=0.6
                )

                _save_single_map(
                    image_map=diff_norm.numpy(),
                    save_path=diff_save_path,
                    vmin=0.0,
                    vmax=1.0,
                    cmap=diff_cmap
                )

            elif diff_mode == "signed_mean":
                # signed 差值图：
                # 红色表示 high > low
                # 蓝色表示 high < low
                # 白色附近表示差异不大
                diff_np, vmin, vmax = _normalize_signed_to_symmetric_range(
                    diff_map[i],
                    percentile=99.0
                )

                _save_single_map(
                    image_map=diff_np,
                    save_path=diff_save_path,
                    vmin=vmin,
                    vmax=vmax,
                    cmap="bwr"
                )

        # ============================================================
        # 2. FFT 频谱图
        # ============================================================
        low_power = _fft_power_from_feature(
            feature=x_low_cpu[i],
            remove_dc=remove_dc
        )

        high_power = _fft_power_from_feature(
            feature=x_high_cpu[i],
            remove_dc=remove_dc
        )

        if save_fft:
            fft_save_path = os.path.join(
                save_dir,
                f"{sample_name}_{prefix}_fft_spectrum_low_vs_high.png"
            )

            _save_fft_pair(
                low_power=low_power,
                high_power=high_power,
                save_path=fft_save_path,
                cmap=fft_cmap
            )

        # ============================================================
        # 3. 径向频谱能量曲线
        # ============================================================
        if save_radial:
            radial_save_path = os.path.join(
                save_dir,
                f"{sample_name}_{prefix}_radial_profile_low_vs_high.png"
            )

            _save_radial_profile_pair(
                low_power=low_power,
                high_power=high_power,
                save_path=radial_save_path,
                num_bins=num_bins
            )

        # ============================================================
        # 4. 低频 / 高频能量比例统计
        # ============================================================
        low_ratio = _frequency_energy_ratio(
            low_power,
            low_cut=low_cut,
            high_cut=high_cut
        )

        high_ratio = _frequency_energy_ratio(
            high_power,
            low_cut=low_cut,
            high_cut=high_cut
        )

        if save_ratio_bar:
            ratio_bar_save_path = os.path.join(
                save_dir,
                f"{sample_name}_{prefix}_frequency_ratio_bar.png"
            )

            _save_frequency_ratio_bar(
                low_ratio=low_ratio,
                high_ratio=high_ratio,
                save_path=ratio_bar_save_path
            )

        metric_rows.append({
            "sample_name": sample_name,
            "prefix": prefix,
            "feature_type": "low",
            **low_ratio,
            "low_cut": low_cut,
            "high_cut": high_cut,
        })

        metric_rows.append({
            "sample_name": sample_name,
            "prefix": prefix,
            "feature_type": "high",
            **high_ratio,
            "low_cut": low_cut,
            "high_cut": high_cut,
        })

    metrics_df = pd.DataFrame(metric_rows)

    if save_metrics:
        csv_save_path = os.path.join(
            save_dir,
            f"{prefix}_frequency_metrics.csv"
        )
        metrics_df.to_csv(csv_save_path, index=False)

        mean_df = (
            metrics_df
            .groupby(["prefix", "feature_type"])
            [
                [
                    "low_energy_ratio",
                    "mid_energy_ratio",
                    "high_energy_ratio"
                ]
            ]
            .mean()
            .reset_index()
        )

        mean_csv_save_path = os.path.join(
            save_dir,
            f"{prefix}_frequency_metrics_mean.csv"
        )
        mean_df.to_csv(mean_csv_save_path, index=False)

    print(f"[Done] High-low frequency visualization without PCA saved to: {save_dir}")

    return metrics_df