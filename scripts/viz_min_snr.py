"""Visualize Min-SNR, P2, EDM, and Logit-Normal weighting for SD 1.5.

Shows how different loss/sampling strategies redistribute gradient
across timesteps compared to uniform (standard epsilon loss).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm as scipy_norm
from scipy.special import expit as sigmoid


def get_sd15_alphas_cumprod(num_timesteps=1000):
    """Reproduce SD 1.5 linear beta schedule -> alphas_cumprod."""
    beta_start, beta_end = 0.00085, 0.012
    betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2
    alphas = 1.0 - betas
    return np.cumprod(alphas)


def logit_normal_pdf(t_norm, mean=0.0, std=1.0):
    """PDF of logit-normal distribution over t in (0, 1).

    If z ~ N(mean, std^2), then t = sigmoid(z) has this PDF.
    """
    # Avoid log(0) at boundaries
    t_norm = np.clip(t_norm, 1e-6, 1 - 1e-6)
    z = np.log(t_norm / (1 - t_norm))  # logit
    pdf_z = scipy_norm.pdf(z, loc=mean, scale=std)
    # Change of variables: dt/dz = t(1-t)
    return pdf_z / (t_norm * (1 - t_norm))


def main():
    alphas_cumprod = get_sd15_alphas_cumprod()
    t = np.arange(1000)
    t_norm = t / 999.0  # normalize to (0, 1)
    snr = alphas_cumprod / (1 - alphas_cumprod)
    sigma = np.sqrt(1.0 / snr)

    kernel = np.ones(20) / 20  # smoothing kernel

    # === Precompute all gradient distributions ===
    uniform = np.ones(1000)
    uniform_norm = uniform / uniform.sum()
    uniform_smooth = np.convolve(uniform_norm, kernel, mode='same')

    # Min-SNR gamma=5
    weight_minsnr = np.minimum(snr, 5) / snr
    minsnr_norm = weight_minsnr / weight_minsnr.sum()
    minsnr_smooth = np.convolve(minsnr_norm, kernel, mode='same')

    # P2 k=1, p=1
    weight_p2 = 1.0 / (1 + snr)
    p2_norm = weight_p2 / weight_p2.sum()
    p2_smooth = np.convolve(p2_norm, kernel, mode='same')

    # EDM
    P_mean, P_std, sigma_data = -1.2, 1.2, 0.5
    ln_sigma = np.log(sigma)
    edm_sampling = scipy_norm.pdf(ln_sigma, loc=P_mean, scale=P_std)
    edm_loss_weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
    edm_effective = edm_sampling * edm_loss_weight
    edm_norm = edm_effective / edm_effective.sum()
    edm_smooth = np.convolve(edm_norm, kernel, mode='same')

    # Logit-normal distributions (various mean/std)
    logit_configs = [
        (0.0, 1.0, "darkcyan", "-", "mean=0, std=1 (SD3 default)"),
        (0.0, 0.5, "blue", "--", "mean=0, std=0.5 (tighter)"),
        (0.0, 1.5, "teal", "--", "mean=0, std=1.5 (wider)"),
        (-0.5, 1.0, "orange", "--", "mean=-0.5, std=1 (shift left)"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))

    # ===== ROW 1: Logit-Normal sampling distributions =====

    # Panel 1,1: Raw PDF shapes
    ax = axes[0, 0]
    for mean, std, color, ls, label in logit_configs:
        pdf = logit_normal_pdf(t_norm, mean=mean, std=std)
        pdf_norm = pdf / pdf.sum()
        pdf_smooth = np.convolve(pdf_norm, kernel, mode='same')
        ax.plot(t, pdf_smooth, color=color, linewidth=2, linestyle=ls, label=label)
    ax.plot(t, uniform_smooth, color="gray", linewidth=1.5, linestyle=":", label="Uniform")
    ax.set_xlabel("Timestep t")
    ax.set_ylabel("Sampling probability")
    ax.set_title("Logit-Normal Sampling Distributions")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 999)
    ax.grid(True, alpha=0.3)

    # Panel 1,2: SD3 default (mean=0, std=1) — gradient contribution
    ax = axes[0, 1]
    pdf_sd3 = logit_normal_pdf(t_norm, mean=0.0, std=1.0)
    sd3_norm = pdf_sd3 / pdf_sd3.sum()
    sd3_smooth = np.convolve(sd3_norm, kernel, mode='same')
    ax.fill_between(t, uniform_smooth, alpha=0.3, color="gray", label="Uniform")
    ax.fill_between(t, sd3_smooth, alpha=0.4, color="darkcyan", label="Logit-Normal (SD3)")
    ax.plot(t, uniform_smooth, color="gray", linewidth=1.5)
    ax.plot(t, sd3_smooth, color="darkcyan", linewidth=1.5)
    ax.set_xlabel("Timestep t")
    ax.set_ylabel("Relative gradient contribution")
    ax.set_title("SD3 Logit-Normal: Where Learning Happens")
    ax.legend()
    ax.set_xlim(0, 999)
    ax.grid(True, alpha=0.3)

    # Panel 1,3: Logit-Normal vs Min-SNR — direct comparison
    ax = axes[0, 2]
    ax.fill_between(t, uniform_smooth, alpha=0.15, color="gray")
    ax.plot(t, uniform_smooth, color="gray", linewidth=2, label="Uniform")
    ax.plot(t, minsnr_smooth, color="red", linewidth=2, label="Min-SNR gamma=5")
    ax.plot(t, sd3_smooth, color="darkcyan", linewidth=2, label="Logit-Normal (SD3)")
    ax.set_xlabel("Timestep t")
    ax.set_ylabel("Relative gradient contribution")
    ax.set_title("Min-SNR vs Logit-Normal")
    ax.legend()
    ax.set_xlim(0, 999)
    ax.grid(True, alpha=0.3)
    ax.annotate("Logit-Normal is\nnaturally symmetric!", xy=(500, sd3_smooth[500]),
                xytext=(500, max(sd3_smooth) * 0.85),
                fontsize=9, ha='center', color='darkcyan', fontweight='bold')

    # ===== ROW 2: All weighting strategies =====

    # Panel 2,1: SNR curve
    ax = axes[1, 0]
    ax.semilogy(t, snr, color="black", linewidth=2)
    ax.axhline(y=5, color="red", linestyle="--", linewidth=1.5, label="gamma=5")
    ax.set_xlabel("Timestep t")
    ax.set_ylabel("SNR (log scale)")
    ax.set_title("Signal-to-Noise Ratio vs Timestep")
    ax.legend()
    ax.set_xlim(0, 999)
    ax.grid(True, alpha=0.3)

    # Panel 2,2: Min-SNR weights
    ax = axes[1, 1]
    for gamma, color, ls in [(1, "blue", "--"), (5, "red", "-"), (10, "green", "--"), (20, "orange", "--")]:
        weight = np.minimum(snr, gamma) / snr
        ax.plot(t, weight, color=color, linewidth=2, linestyle=ls, label=f"gamma={gamma}")
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="Uniform")
    ax.set_xlabel("Timestep t")
    ax.set_ylabel("Loss weight")
    ax.set_title("Min-SNR Weight: min(SNR, gamma) / SNR")
    ax.legend()
    ax.set_xlim(0, 999)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)

    # Panel 2,3: EDM components
    ax = axes[1, 2]
    edm_samp_norm = edm_sampling / edm_sampling.max()
    edm_lw_norm = edm_loss_weight / edm_loss_weight.max()
    ax.plot(t, edm_samp_norm, color="darkcyan", linewidth=2, label="Sampling density p(sigma)")
    ax.plot(t, edm_lw_norm, color="goldenrod", linewidth=2, linestyle="--", label="Loss weight w(sigma)")
    ax.set_xlabel("Timestep t")
    ax.set_ylabel("Normalized value")
    ax.set_title("EDM: Sampling Density + Loss Weight")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 999)
    ax.grid(True, alpha=0.3)

    # ===== ROW 3: Comparisons =====

    # Panel 3,1: All five strategies head-to-head
    ax = axes[2, 0]
    ax.fill_between(t, uniform_smooth, alpha=0.15, color="gray")
    ax.plot(t, uniform_smooth, color="gray", linewidth=2, label="Uniform")
    ax.plot(t, minsnr_smooth, color="red", linewidth=2, label="Min-SNR gamma=5")
    ax.plot(t, p2_smooth, color="purple", linewidth=2, label="P2 k=1,p=1")
    ax.plot(t, edm_smooth, color="goldenrod", linewidth=2, label="EDM")
    ax.plot(t, sd3_smooth, color="darkcyan", linewidth=2, label="Logit-Normal (SD3)")
    ax.set_xlabel("Timestep t")
    ax.set_ylabel("Relative gradient contribution")
    ax.set_title("All Strategies Compared")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 999)
    ax.grid(True, alpha=0.3)

    # Panel 3,2: Inference-relevant region
    ax = axes[2, 1]
    ax.fill_between(t, uniform_smooth, alpha=0.15, color="gray")
    ax.plot(t, uniform_smooth, color="gray", linewidth=2, label="Uniform")
    ax.plot(t, minsnr_smooth, color="red", linewidth=2, label="Min-SNR gamma=5")
    ax.plot(t, sd3_smooth, color="darkcyan", linewidth=2, label="Logit-Normal (SD3)")
    ax.axvspan(0, 700, alpha=0.08, color='green')
    ax.axvline(x=700, color='green', linewidth=2, linestyle='--', label='noise_strength=0.7')
    ax.axvspan(700, 999, alpha=0.08, color='red')
    ax.set_xlabel("Timestep t")
    ax.set_ylabel("Relative gradient contribution")
    ax.set_title("Inference-Relevant Region (surface mode)")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 999)
    ax.grid(True, alpha=0.3)
    ax.annotate("Used at inference", xy=(350, max(uniform_smooth) * 0.5),
                fontsize=9, ha='center', color='green', fontweight='bold')
    ax.annotate("Never used\n(surface mode)", xy=(850, max(uniform_smooth) * 0.5),
                fontsize=9, ha='center', color='red')

    # Panel 3,3: Cumulative gradient — what fraction of gradient comes from t<X
    ax = axes[2, 2]
    ax.plot(t, np.cumsum(uniform_norm), color="gray", linewidth=2, label="Uniform")
    ax.plot(t, np.cumsum(minsnr_norm), color="red", linewidth=2, label="Min-SNR gamma=5")
    ax.plot(t, np.cumsum(p2_norm), color="purple", linewidth=2, label="P2")
    ax.plot(t, np.cumsum(sd3_norm), color="darkcyan", linewidth=2, label="Logit-Normal (SD3)")
    ax.plot(t, np.cumsum(edm_norm), color="goldenrod", linewidth=2, label="EDM")
    ax.axhline(y=0.5, color="black", linewidth=0.5, linestyle=":")
    ax.axvline(x=700, color='green', linewidth=1.5, linestyle='--', alpha=0.5)
    ax.set_xlabel("Timestep t")
    ax.set_ylabel("Cumulative gradient fraction")
    ax.set_title("CDF: How Much Gradient Below Timestep t")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 999)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = "results/min_snr_viz.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved to {out}")
    plt.close()


if __name__ == "__main__":
    main()
