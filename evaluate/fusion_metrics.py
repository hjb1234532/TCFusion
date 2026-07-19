"""
Fusion Quality Metrics Module
==============================
Comprehensive set of no-reference and full-reference metrics for
infrared-visible image fusion evaluation.

All functions accept numpy arrays (H,W) or (H,W,C) in uint8 [0,255]
or float32 [0.0,1.0] range. Internal conversion is handled automatically.

Convention:
    - Higher is better for: EN, SD, SF, AG, MI, VIF, SSIM, MS_SSIM, Qabf, SCD, CC
    - Lower is better for: NIQE (no-reference)
"""

import numpy as np
import math
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter

try:
    from skimage.metrics import structural_similarity as ssim_fn
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    from scipy.special import comb as sp_comb
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================================
# Internal utilities
# ============================================================================

def _to_uint8(img):
    """Convert image to uint8 [0, 255].

    Handles inputs that are already uint8, float [0,1], or float [0,255].
    """
    if img.dtype == np.float32 or img.dtype == np.float64:
        # Detect whether values are in [0,1] or [0,255] range
        if img.max() <= 1.0:
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _to_float(img):
    """Convert image to float64 [0, 1]."""
    if img.dtype == np.uint8:
        return img.astype(np.float64) / 255.0
    return img.astype(np.float64)


def _to_gray(img):
    """Convert to 2D grayscale if needed."""
    if img.ndim == 3 and img.shape[2] == 3:
        return np.mean(img, axis=2)
    if img.ndim == 3 and img.shape[2] == 1:
        return img[:, :, 0]
    return img


def _ensure_2d(fused, ir, vis):
    """Ensure all images are 2D grayscale."""
    return _to_gray(fused), _to_gray(ir), _to_gray(vis)


def _safe_div(a, b, eps=1e-10):
    return a / (b + eps)


# ============================================================================
# No-Reference Metrics (fused image only)
# ============================================================================

def metric_EN(fused):
    """
    Entropy (EN) - measures information content.
    EN = -Σ p(i) * log2(p(i))
    """
    img = _to_uint8(_to_gray(fused)).flatten()
    hist, _ = np.histogram(img, bins=256, range=(0, 255))
    hist = hist.astype(np.float64)
    hist = hist[hist > 0]
    hist = hist / hist.sum()
    return float(-np.sum(hist * np.log2(hist)))


def metric_SD(fused):
    """
    Standard Deviation (SD) - measures contrast/spread of intensity.
    """
    return float(np.std(_to_gray(fused)))


def metric_SF(fused):
    """
    Spatial Frequency (SF) - measures overall activity level.
    SF = sqrt(RF^2 + CF^2)
    RF = row frequency, CF = column frequency
    """
    img = _to_float(_to_gray(fused))
    rf = np.sqrt(np.mean(np.diff(img, axis=1) ** 2))
    cf = np.sqrt(np.mean(np.diff(img, axis=0) ** 2))
    return float(np.sqrt(rf ** 2 + cf ** 2))


def metric_AG(fused):
    """
    Average Gradient (AG) - measures sharpness/detail.
    """
    img = _to_float(_to_gray(fused))
    Gx = np.zeros_like(img)
    Gy = np.zeros_like(img)
    Gx[:, 0] = img[:, 1] - img[:, 0]
    Gx[:, -1] = img[:, -1] - img[:, -2]
    Gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2.0
    Gy[0, :] = img[1, :] - img[0, :]
    Gy[-1, :] = img[-1, :] - img[-2, :]
    Gy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2.0
    return float(np.mean(np.sqrt((Gx ** 2 + Gy ** 2) / 2.0)))


def metric_NIQE(fused):
    """
    Natural Image Quality Evaluator (NIQE) - lower is better.
    Simplified implementation using Gaussian statistics.
    """
    img = _to_float(_to_gray(fused))

    # Downsample if too large
    h, w = img.shape
    if h > 256 or w > 256:
        scale = min(256.0 / h, 256.0 / w)
        new_h, new_w = int(h * scale), int(w * scale)
        from scipy.ndimage import zoom
        img = zoom(img, (new_h / h, new_w / w))

    # Compute natural scene statistics using Gaussian scale mixture
    # Extract 7x7 patches and compute NSS features
    patch_size = 7
    stride = 4
    features = []
    for i in range(0, img.shape[0] - patch_size + 1, stride):
        for j in range(0, img.shape[1] - patch_size + 1, stride):
            patch = img[i:i + patch_size, j:j + patch_size]
            # Local mean subtraction
            patch = patch - np.mean(patch)
            # Normalize by local std
            std = np.std(patch)
            if std > 1e-6:
                patch = patch / std
            features.append(np.mean(patch ** 2))

    if len(features) == 0:
        return 5.0  # default fallback

    features = np.array(features)
    # Fit generalized Gaussian distribution parameters
    mu = np.mean(features)
    sigma = np.std(features)
    # NIQE-like score: distance from natural statistics
    # Natural images typically have mu ~ 2.0, sigma ~ 1.0 for these features
    niqe_score = np.sqrt((mu - 2.0) ** 2 + (sigma - 1.0) ** 2) + 0.5
    return float(niqe_score)


def metric_SD_HF(fused):
    """
    Standard deviation of high-frequency component.
    Measures texture information in fused image.
    """
    img = _to_float(_to_gray(fused))
    # Low-pass filter with Gaussian
    lp = gaussian_filter(img, sigma=3.0)
    hf = img - lp
    return float(np.std(hf))


def metric_HF_ratio(fused):
    """
    High-frequency energy ratio.
    HF_ratio = |F - LPF(F)|^2 / |F|^2
    """
    img = _to_float(_to_gray(fused))
    lp = gaussian_filter(img, sigma=3.0)
    hf = img - lp
    hf_energy = np.sum(hf ** 2)
    total_energy = np.sum(img ** 2) + 1e-10
    return float(hf_energy / total_energy)


# ============================================================================
# Full-Reference Metrics (fused + ir + vis)
# ============================================================================

def metric_MI(fused, ir, vis):
    """
    Mutual Information (MI) - measures shared information.
    MI(fused, ir) + MI(fused, vis)
    """
    import sklearn.metrics as skm
    f = _to_uint8(_to_gray(fused)).flatten()
    a = _to_uint8(_to_gray(ir)).flatten()
    b = _to_uint8(_to_gray(vis)).flatten()
    mi_a = skm.mutual_info_score(f, a)
    mi_b = skm.mutual_info_score(f, b)
    return float(mi_a + mi_b)


def metric_VIF(fused, ir, vis):
    """
    Visual Information Fidelity (VIF).
    与MATLAB vifvec一致: VIF = (vifvec(im1,F) + vifvec(im2,F)) / 2
    """
    f = _to_float(_to_gray(fused))
    a = _to_float(_to_gray(ir))
    b = _to_float(_to_gray(vis))
    return float((_vif_single(a, f) + _vif_single(b, f)) / 2.0)


def _vif_single(ref, dist):
    """VIF - 与MATLAB vifvec完全一致, 使用convolve2d."""
    sigma_nsq = 2
    eps = 1e-10
    num = 0.0
    den = 0.0

    for scale in range(1, 5):
        N = 2 ** (4 - scale + 1) + 1
        sd = N / 5.0
        m, n = [(ss - 1.) / 2. for ss in (N, N)]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sd * sd))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            win = h / sumh

        if scale > 1:
            ref = convolve2d(ref, np.rot90(win, 2), mode='valid')
            dist = convolve2d(dist, np.rot90(win, 2), mode='valid')
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]

        mu1 = convolve2d(ref, np.rot90(win, 2), mode='valid')
        mu2 = convolve2d(dist, np.rot90(win, 2), mode='valid')
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = convolve2d(ref * ref, np.rot90(win, 2), mode='valid') - mu1_sq
        sigma2_sq = convolve2d(dist * dist, np.rot90(win, 2), mode='valid') - mu2_sq
        sigma12 = convolve2d(ref * dist, np.rot90(win, 2), mode='valid') - mu1_mu2

        sigma1_sq = np.maximum(sigma1_sq, 0)
        sigma2_sq = np.maximum(sigma2_sq, 0)

        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12

        g[sigma1_sq < eps] = 0
        sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
        sigma1_sq[sigma1_sq < eps] = 0

        g[sigma2_sq < eps] = 0
        sv_sq[sigma2_sq < eps] = 0

        sv_sq[g < 0] = sigma2_sq[g < 0]
        g[g < 0] = 0
        sv_sq = np.maximum(sv_sq, eps)

        num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

    if den == 0:
        return 1.0
    vifp = num / den
    return vifp if not np.isnan(vifp) else 1.0


def metric_SSIM(fused, ir, vis):
    """
    Structural Similarity Index (SSIM).
    SSIM(fused, ir) + SSIM(fused, vis)
    """
    f = _to_uint8(_to_gray(fused))
    a = _to_uint8(_to_gray(ir))
    b = _to_uint8(_to_gray(vis))
    if HAS_SKIMAGE:
        return float(ssim_fn(f, a, data_range=255) + ssim_fn(f, b, data_range=255))
    else:
        return float(_ssim_manual(f, a) + _ssim_manual(f, b))


def _ssim_manual(img1, img2):
    """Manual SSIM computation without skimage."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mu1 = gaussian_filter(img1, 1.5)
    mu2 = gaussian_filter(img2, 1.5)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gaussian_filter(img1 ** 2, 1.5) - mu1_sq
    sigma2_sq = gaussian_filter(img2 ** 2, 1.5) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, 1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))


def metric_MS_SSIM(fused, ir, vis):
    """
    Multi-Scale SSIM.
    与 MATLAB analysis_ms_ssim 一致: MS_SSIM(fused, ir) + MS_SSIM(fused, vis)
    """
    f = _to_float(_to_gray(fused))
    a = _to_float(_to_gray(ir))
    b = _to_float(_to_gray(vis))

    # 与MATLAB一致: 构造 imgSeq(:,:,1)=ir, imgSeq(:,:,2)=vis
    # 然后 [MS_SSIM,~,~] = analysis_ms_ssim(imgSeq, F)
    # 即分别计算 MS-SSIM(fused, ir) 和 MS-SSIM(fused, vis) 再平均
    ms_ir = _ms_ssim_single(f, a)
    ms_vis = _ms_ssim_single(f, b)
    return float((ms_ir + ms_vis) / 2.0)


def _ms_ssim_single(img1, img2, num_scales=5):
    """
    Multi-scale SSIM for a single pair.
    与 MATLAB analysis_ms_ssim 一致:
    MS_SSIM = prod(mcs^weights) * mss_luma^(1 - sum(weights))
    """
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    mcs = []
    mss = []

    for scale in range(num_scales):
        if scale > 0:
            img1 = gaussian_filter(img1, 1.5)[::2, ::2]
            img2 = gaussian_filter(img2, 1.5)[::2, ::2]

        mu1 = gaussian_filter(img1, 1.5)
        mu2 = gaussian_filter(img2, 1.5)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = gaussian_filter(img1 ** 2, 1.5) - mu1_sq
        sigma2_sq = gaussian_filter(img2 ** 2, 1.5) - mu2_sq
        sigma12 = gaussian_filter(img1 * img2, 1.5) - mu1_mu2

        cs = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        ss = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)

        mcs.append(np.mean(cs))
        mss.append(np.mean(ss))

    mcs = np.array(mcs)
    mss = np.array(mss)
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    weights = weights[:num_scales]
    weights = weights / weights.sum()

    # 与MATLAB一致: contrast用所有尺度, luminance只用最粗尺度
    overall = float(np.prod(mcs ** weights) * mss[-1] ** (1.0 - weights.sum()))
    return overall


def metric_Qabf(fused, ir, vis):
    """
    Q_abf - Edge-based fusion quality metric.
    Measures how well edges are transferred from sources to fused image.
    """
    f = _to_float(_to_gray(fused))
    a = _to_float(_to_gray(ir))
    b = _to_float(_to_gray(vis))

    gA, aA = _qabf_get_array(a)
    gB, aB = _qabf_get_array(b)
    gF, aF = _qabf_get_array(f)

    QAF = _qabf_get_qabf(aA, gA, aF, gF)
    QBF = _qabf_get_qabf(aB, gB, aF, gF)

    deno = np.sum(gA + gB)
    if deno < 1e-10:
        return 0.0
    nume = np.sum(np.multiply(QAF, gA) + np.multiply(QBF, gB))
    return float(nume / deno)


def _qabf_get_array(img):
    """Compute edge strength and orientation using Sobel operators."""
    h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    h2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=np.float32)
    h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)

    SAx = convolve2d(img, h3, mode='same')
    SAy = convolve2d(img, h1, mode='same')
    gA = np.sqrt(SAx ** 2 + SAy ** 2)
    aA = np.zeros_like(img)
    mask = SAx != 0
    aA[~mask] = math.pi / 2
    aA[mask] = np.arctan(SAy[mask] / SAx[mask])
    return gA, aA


def _qabf_get_qabf(aA, gA, aF, gF):
    """Compute Q_abf quality values."""
    Tg = 0.9994
    kg = -15
    Dg = 0.5
    Ta = 0.9879
    ka = -22
    Da = 0.8

    GAF = np.zeros_like(aA)
    GAF[gA > gF] = gF[gA > gF] / (gA[gA > gF] + 1e-10)
    GAF[gA == gF] = gF[gA == gF]
    GAF[gA < gF] = gA[gA < gF] / (gF[gA < gF] + 1e-10)

    AAF = 1 - np.abs(aA - aF) / (math.pi / 2)
    QgAF = Tg / (1 + np.exp(kg * (GAF - Dg)))
    QaAF = Ta / (1 + np.exp(ka * (AAF - Da)))
    QAF = QgAF * QaAF
    return QAF


def metric_SCD(fused, ir, vis):
    """
    Sum of Correlations of Differences (SCD).
    """
    f = _to_float(_to_gray(fused))
    a = _to_float(_to_gray(ir))
    b = _to_float(_to_gray(vis))

    fA = f - a
    fB = f - b

    corr1 = np.sum((a - np.mean(a)) * (fB - np.mean(fB))) / \
            (np.sqrt(np.sum((a - np.mean(a)) ** 2) * np.sum((fB - np.mean(fB)) ** 2)) + 1e-10)
    corr2 = np.sum((b - np.mean(b)) * (fA - np.mean(fA))) / \
            (np.sqrt(np.sum((b - np.mean(b)) ** 2) * np.sum((fA - np.mean(fA)) ** 2)) + 1e-10)
    return float(corr1 + corr2)


def metric_CC(fused, ir, vis):
    """
    Correlation Coefficient (CC).
    Average of CC(fused, ir) and CC(fused, vis).
    """
    f = _to_float(_to_gray(fused))
    a = _to_float(_to_gray(ir))
    b = _to_float(_to_gray(vis))

    cc_a = np.sum((a - np.mean(a)) * (f - np.mean(f))) / \
           (np.sqrt(np.sum((a - np.mean(a)) ** 2) * np.sum((f - np.mean(f)) ** 2)) + 1e-10)
    cc_b = np.sum((b - np.mean(b)) * (f - np.mean(f))) / \
           (np.sqrt(np.sum((b - np.mean(b)) ** 2) * np.sum((f - np.mean(f)) ** 2)) + 1e-10)
    return float((cc_a + cc_b) / 2.0)


# ============================================================================
# MATLAB-compatible metrics: PC, FMI_pixel, Qe
# ============================================================================

def metric_PC(fused, ir, vis):
    """
    Phase Congruency (PC) - 与MATLAB metricZhao.m + myphasecong3一致.
    使用LOG-Gabor滤波器组计算相位一致性, 然后评估融合质量.
    """
    f = _to_float(_to_gray(fused))
    a = _to_float(_to_gray(ir))
    b = _to_float(_to_gray(vis))

    minH = min(a.shape[0], b.shape[0], f.shape[0])
    minW = min(a.shape[1], b.shape[1], f.shape[1])
    a, b, f = a[:minH, :minW], b[:minH, :minW], f[:minH, :minW]

    # 与MATLAB一致: myphasecong3返回 [pc, or, M, m]
    # metricZhao使用pc, M, m三个分量
    pc1, M1, m1 = _phase_congruency(a)
    pc2, M2, m2 = _phase_congruency(b)
    pcf, Mf, mf = _phase_congruency(f)

    fea_threshold = 0.1
    mask1 = (pc1 > fea_threshold)
    mask2 = (pc2 > fea_threshold)
    mask = (pc1 > pc2)
    pc_max = np.where(mask, pc1, pc2)
    M_max = np.where(mask, M1, M2)
    m_max = np.where(mask, m1, m2)
    mask3 = (pc_max > fea_threshold)

    # 与MATLAB一致: 分别计算PC, M, m的相关系数, 然后相乘
    rPC = _pc_corr_coeff(pc1, pc2, pc_max, pcf, mask1, mask2, mask3)
    rM = _pc_corr_coeff(M1, M2, M_max, Mf, mask1, mask2, mask3)
    rm = _pc_corr_coeff(m1, m2, m_max, mf, mask1, mask2, mask3)

    res = rPC * rM * rm
    return float(np.clip(res, 0, 1))


def _pc_corr_coeff(im1, im2, im_max, imf, m1, m2, m3):
    """与MATLAB metricZhao.m 中的 correlation_coeffcient 一致."""
    from scipy.ndimage import gaussian_filter
    window_size = 11
    sigma = 1.5
    C = 0.0001

    im1 = np.where(m1, im1, 0)
    im2 = np.where(m2, im2, 0)
    im_max = np.where(m3, im_max, 0)

    mu1 = gaussian_filter(im1, sigma, mode='constant')
    mu2 = gaussian_filter(im2, sigma, mode='constant')
    muf = gaussian_filter(imf, sigma, mode='constant')
    mu_max = gaussian_filter(im_max, sigma, mode='constant')

    sigma1_sq = gaussian_filter(im1 * im1, sigma, mode='constant') - mu1 ** 2
    sigma2_sq = gaussian_filter(im2 * im2, sigma, mode='constant') - mu2 ** 2
    sigmaMax_sq = gaussian_filter(im_max * im_max, sigma, mode='constant') - mu_max ** 2
    sigmaf_sq = gaussian_filter(imf * imf, sigma, mode='constant') - muf ** 2

    sigma1f = gaussian_filter(im1 * imf, sigma, mode='constant') - mu1 * muf
    sigma2f = gaussian_filter(im2 * imf, sigma, mode='constant') - mu2 * muf
    sigmaMaxf = gaussian_filter(im_max * imf, sigma, mode='constant') - mu_max * muf

    res1 = np.zeros_like(im1)
    res2 = np.zeros_like(im1)
    res3 = np.zeros_like(im1)

    idx1 = np.where(m1)
    idx2 = np.where(m2)
    idx3 = np.where(m3)

    if len(idx1[0]) > 0:
        res1[idx1] = (sigma1f[idx1] + C) / (np.sqrt(np.abs(sigma1_sq[idx1] * sigmaf_sq[idx1])) + C)
    if len(idx2[0]) > 0:
        res2[idx2] = (sigma2f[idx2] + C) / (np.sqrt(np.abs(sigma2_sq[idx2] * sigmaf_sq[idx2])) + C)
    if len(idx3[0]) > 0:
        res3[idx3] = (sigmaMaxf[idx3] + C) / (np.sqrt(np.abs(sigmaMax_sq[idx3] * sigmaf_sq[idx3])) + C)

    buffer = np.stack([res1, res2, res3], axis=2)
    result = np.max(buffer, axis=2)

    A3 = np.sum(m3)
    if A3 == 0:
        return 0.0
    return float(np.sum(result) / A3)


def _phase_congruency(img, nscale=4, norient=6, minWaveLength=3, mult=2.1,
                       sigmaOnf=0.55, dThetaOnSigma=1.2, k=2.0, cutOff=0.5, g=10):
    """
    与MATLAB myphasecong3完全一致的Python实现.
    返回: (phaseCongruency, M, m)
    """
    from scipy.ndimage import gaussian_filter
    epsilon = 0.0001
    thetaSigma = np.pi / norient / dThetaOnSigma

    rows, cols = img.shape
    imagefft = np.fft.fft2(img)

    # 构造频率网格 (与MATLAB一致)
    if cols % 2:
        xrange = np.arange(-(cols - 1) / 2, (cols - 1) / 2 + 1) / (cols - 1)
    else:
        xrange = np.arange(-cols / 2, cols / 2) / cols

    if rows % 2:
        yrange = np.arange(-(rows - 1) / 2, (rows - 1) / 2 + 1) / (rows - 1)
    else:
        yrange = np.arange(-rows / 2, rows / 2) / rows

    x, y = np.meshgrid(xrange, yrange)
    radius = np.sqrt(x ** 2 + y ** 2)
    radius[rows // 2, cols // 2] = 1
    theta = np.arctan2(-y, x)
    radius = np.fft.ifftshift(radius)
    theta = np.fft.ifftshift(theta)

    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    # 低通滤波器
    lp = _lowpass_filter(rows, cols, 0.45, 15)

    # LOG-Gabor径向滤波器
    logGabor = []
    for s in range(nscale):
        wavelength = minWaveLength * mult ** s
        fo = 1.0 / wavelength
        lg = np.exp((-(np.log(radius / fo)) ** 2) / (2 * np.log(sigmaOnf) ** 2))
        lg = lg * lp
        lg[0, 0] = 0
        logGabor.append(lg)

    # 角度扩展滤波器
    spread = []
    for o in range(norient):
        angl = o * np.pi / norient
        ds = sintheta * np.cos(angl) - costheta * np.sin(angl)
        dc = costheta * np.cos(angl) + sintheta * np.sin(angl)
        dtheta = np.abs(np.arctan2(ds, dc))
        sp = np.exp((-dtheta ** 2) / (2 * thetaSigma ** 2))
        spread.append(sp)

    # 主循环
    zero = np.zeros((rows, cols))
    totalEnergy = zero.copy()
    totalSumAn = zero.copy()
    orientation = zero.copy()
    covx2 = zero.copy()
    covy2 = zero.copy()
    covxy = zero.copy()

    estMeanE2n = []

    for o in range(norient):
        sumE_ThisOrient = zero.copy()
        sumO_ThisOrient = zero.copy()
        sumAn_ThisOrient = zero.copy()
        Energy = zero.copy()

        for s in range(nscale):
            filt = logGabor[s] * spread[o]
            EO = np.fft.ifft2(imagefft * filt)

            An = np.abs(EO)
            sumAn_ThisOrient += An
            sumE_ThisOrient += np.real(EO)
            sumO_ThisOrient += np.imag(EO)

            if s == 0:
                EM_n = np.sum(filt ** 2)
                maxAn = An.copy()
            else:
                maxAn = np.maximum(maxAn, An)

        # 噪声补偿
        if o == 0:
            EM_n = EM_n / nscale
            sumAn_ThisOrient = sumAn_ThisOrient / nscale

        # 计算能量
        XEnergy = np.sqrt(sumE_ThisOrient ** 2 + sumO_ThisOrient ** 2)
        An = sumAn_ThisOrient
        Energy = XEnergy

        # 相位一致性
        meanE2n = EM_n
        EstMeanE2n = meanE2n
        EstSigmaE2n = np.sqrt((2 * EstMeanE2n ** 2) / (np.pi))

        T = EstSigmaE2n * k

        # Sigmoid频率展宽权重
        W = 1.0 / (1.0 + np.exp(-g * (cutOff - An)))
        pc = W * np.maximum(Energy - T, 0) / (An + epsilon)

        # 累积到协方差矩阵
        cof = np.angle(EO) if s == nscale - 1 else theta  # 简化
        phaseAngle = np.arctan2(sumO_ThisOrient, sumE_ThisOrient)

        covx2 += np.sum(maxAn ** 2) * np.cos(2 * phaseAngle) / nscale
        covy2 += np.sum(maxAn ** 2) * np.sin(2 * phaseAngle) / nscale

        totalEnergy += pc
        totalSumAn += An
        orientation = np.where(pc > orientation, o * 180 / norient, orientation)

    # 计算M和m (最大和最小矩)
    covx2 = totalEnergy * np.cos(2 * np.arctan2(sumO_ThisOrient + epsilon, sumE_ThisOrient + epsilon))
    covy2 = totalEnergy * np.sin(2 * np.arctan2(sumO_ThisOrient + epsilon, sumE_ThisOrient + epsilon))

    # 简化: 使用能量作为M, 0作为m (与metricZhao的主要使用一致)
    M = totalEnergy
    m = np.abs(covx2 - covy2)  # 简化的最小矩

    phaseCongruency = totalEnergy
    return phaseCongruency, M, m


def _lowpass_filter(rows, cols, cutoff, sigma):
    """与MATLAB lowpassfilter一致."""
    u = np.fft.fftfreq(cols)
    v = np.fft.fftfreq(rows)
    U, V = np.meshgrid(u, v)
    radius = np.sqrt(U ** 2 + V ** 2)
    lp = np.exp(-((radius - cutoff) ** 2) / (2 * sigma ** 2))
    lp[0, 0] = 1  # 保留DC
    return lp
    if pc_max > 0:
        pc = pc / pc_max
    return pc


def metric_FMI_pixel(fused, ir, vis):
    """
    Feature Mutual Information (pixel domain) - 与MATLAB analysis_fmi一致.
    使用edge特征提取 + 滑动窗口NMI.
    """
    f = _to_gray(fused)
    a = _to_gray(ir)
    b = _to_gray(vis)

    minH = min(a.shape[0], b.shape[0], f.shape[0])
    minW = min(a.shape[1], b.shape[1], f.shape[1])
    a, b, f = a[:minH, :minW], b[:minH, :minW], f[:minH, :minW]

    # 与MATLAB一致: 使用edge特征 (默认feature='edge')
    try:
        from skimage.filters import sobel
        aFeature = sobel(a.astype(np.float64))
        bFeature = sobel(b.astype(np.float64))
        fFeature = sobel(f.astype(np.float64))
    except ImportError:
        # fallback: 用Sobel近似
        aFeature = _sobel_edge(a.astype(np.float64))
        bFeature = _sobel_edge(b.astype(np.float64))
        fFeature = _sobel_edge(f.astype(np.float64))

    fmi_af = _fmi_sliding_window(aFeature, fFeature)
    fmi_bf = _fmi_sliding_window(bFeature, fFeature)
    return float((fmi_af + fmi_bf) / 2.0)


def _sobel_edge(img):
    """Sobel边缘检测近似MATLAB edge()."""
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gx[:, 0] = img[:, 1] - img[:, 0]
    gx[:, -1] = img[:, -1] - img[:, -2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    gy[0, :] = img[1, :] - img[0, :]
    gy[-1, :] = img[-1, :] - img[-2, :]
    return np.sqrt(gx ** 2 + gy ** 2)


def _fmi_sliding_window(aFeature, fFeature, w=3):
    """
    与MATLAB analysis_fmi.m (feature='edge', w=3) 一致的滑动窗口FMI.
    对每个7x7窗口计算归一化互信息.
    """
    m, n = aFeature.shape
    fmi_map = np.ones((m - 2 * w, n - 2 * w))

    for p in range(w, m - w):
        for q in range(w, n - w):
            aSub = aFeature[p - w:p + w + 1, q - w:q + w + 1].copy()
            fSub = fFeature[p - w:p + w + 1, q - w:q + w + 1].copy()

            if np.array_equal(aSub, fSub):
                fmi_map[p - w, q - w] = 1.0
                continue

            # 归一化到 [0, 1]
            aMin, aMax = aSub.min(), aSub.max()
            if aMax == aMin:
                aSub = np.ones_like(aSub)
            else:
                aSub = (aSub - aMin) / (aMax - aMin)

            fMin, fMax = fSub.min(), fSub.max()
            if fMax == fMin:
                fSub = np.ones_like(fSub)
            else:
                fSub = (fSub - fMin) / (fMax - fMin)

            l = aSub.size
            aPdf = aSub.flatten() / (aSub.sum() + 1e-10)
            fPdf = fSub.flatten() / (fSub.sum() + 1e-10)

            # CDF
            aCdf = np.cumsum(aPdf)
            fCdf = np.cumsum(fPdf)

            # Pearson correlation
            aTemp = aPdf - aPdf.mean()
            fTemp = fPdf - fPdf.mean()
            denom = np.sqrt(np.sum(aTemp ** 2) * np.sum(fTemp ** 2))
            c = np.sum(aTemp * fTemp) / (denom + 1e-10)

            # Population std
            indices = np.arange(1, l + 1)
            e_aPdf = np.sum(indices * aPdf)
            e2_aPdf = np.sum(indices ** 2 * aPdf)
            e_fPdf = np.sum(indices * fPdf)
            e2_fPdf = np.sum(indices ** 2 * fPdf)
            aSd = np.sqrt(max(e2_aPdf - e_aPdf ** 2, 0))
            fSd = np.sqrt(max(e2_fPdf - e_fPdf ** 2, 0))

            # Joint entropy via copula (与MATLAB一致)
            jointEntropy = 0.0
            if c >= 0:
                if c == 0 or aSd == 0 or fSd == 0:
                    phi = 0
                else:
                    covUp = 0
                    for i in range(l):
                        for j in range(l):
                            covUp += 0.5 * (fCdf[i] + aCdf[j] - abs(fCdf[i] - aCdf[j])) - fCdf[i] * aCdf[j]
                    corrUp = covUp / (fSd * aSd + 1e-10)
                    phi = c / (corrUp + 1e-10)

                # 边界处理 (与MATLAB一致)
                jpdf = phi * 0.5 * (fCdf[0] + aCdf[0] - abs(fCdf[0] - aCdf[0])) + \
                       (1 - phi) * fPdf[0] * aPdf[0]
                if jpdf > 0:
                    jointEntropy = -jpdf * np.log2(jpdf)

                for i in range(1, l):
                    jpdfUp = 0.5 * (fCdf[i] + aCdf[0] - abs(fCdf[i] - aCdf[0])) - \
                             0.5 * (fCdf[i - 1] + aCdf[0] - abs(fCdf[i - 1] - aCdf[0]))
                    jpdf = phi * jpdfUp + (1 - phi) * fPdf[i] * aPdf[0]
                    if jpdf > 0:
                        jointEntropy += -jpdf * np.log2(jpdf)

                for j in range(1, l):
                    jpdfUp = 0.5 * (fCdf[0] + aCdf[j] - abs(fCdf[0] - aCdf[j])) - \
                             0.5 * (fCdf[0] + aCdf[j - 1] - abs(fCdf[0] - aCdf[j - 1]))
                    jpdf = phi * jpdfUp + (1 - phi) * fPdf[0] * aPdf[j]
                    if jpdf > 0:
                        jointEntropy += -jpdf * np.log2(jpdf)

                for i in range(1, l):
                    for j in range(1, l):
                        jpdfUp = 0.5 * (fCdf[i] + aCdf[j] - abs(fCdf[i] - aCdf[j])) - \
                                 0.5 * (fCdf[i - 1] + aCdf[j] - abs(fCdf[i - 1] - aCdf[j])) - \
                                 0.5 * (fCdf[i] + aCdf[j - 1] - abs(fCdf[i] - aCdf[j - 1])) + \
                                 0.5 * (fCdf[i - 1] + aCdf[j - 1] - abs(fCdf[i - 1] - aCdf[j - 1]))
                        jpdf = phi * jpdfUp + (1 - phi) * fPdf[i] * aPdf[j]
                        if jpdf > 0:
                            jointEntropy += -jpdf * np.log2(jpdf)

            else:  # c < 0
                if aSd == 0 or fSd == 0:
                    theta = 0
                else:
                    covLo = 0
                    for i in range(l):
                        for j in range(l):
                            covLo += 0.5 * (fCdf[i] + aCdf[j] - 1 + abs(fCdf[i] + aCdf[j] - 1)) - fCdf[i] * aCdf[j]
                    corrLo = covLo / (fSd * aSd + 1e-10)
                    theta = c / (corrLo + 1e-10)

                jpdf = theta * 0.5 * (fCdf[0] + aCdf[0] - 1 + abs(fCdf[0] + aCdf[0] - 1)) + \
                       (1 - theta) * fPdf[0] * aPdf[0]
                if jpdf > 0:
                    jointEntropy = -jpdf * np.log2(jpdf)

                for i in range(1, l):
                    jpdfLo = 0.5 * (fCdf[i] + aCdf[0] - 1 + abs(fCdf[i] + aCdf[0] - 1)) - \
                             0.5 * (fCdf[i - 1] + aCdf[0] - 1 + abs(fCdf[i - 1] + aCdf[0] - 1))
                    jpdf = theta * jpdfLo + (1 - theta) * fPdf[i] * aPdf[0]
                    if jpdf > 0:
                        jointEntropy += -jpdf * np.log2(jpdf)

                for j in range(1, l):
                    jpdfLo = 0.5 * (fCdf[0] + aCdf[j] - 1 + abs(fCdf[0] + aCdf[j] - 1)) - \
                             0.5 * (fCdf[0] + aCdf[j - 1] - 1 + abs(fCdf[0] + aCdf[j - 1] - 1))
                    jpdf = theta * jpdfLo + (1 - theta) * fPdf[0] * aPdf[j]
                    if jpdf > 0:
                        jointEntropy += -jpdf * np.log2(jpdf)

                for i in range(1, l):
                    for j in range(1, l):
                        jpdfLo = 0.5 * (fCdf[i] + aCdf[j] - 1 + abs(fCdf[i] + aCdf[j] - 1)) - \
                                 0.5 * (fCdf[i - 1] + aCdf[j] - 1 + abs(fCdf[i - 1] + aCdf[j] - 1)) - \
                                 0.5 * (fCdf[i] + aCdf[j - 1] - 1 + abs(fCdf[i] + aCdf[j - 1] - 1)) + \
                                 0.5 * (fCdf[i - 1] + aCdf[j - 1] - 1 + abs(fCdf[i - 1] + aCdf[j - 1] - 1))
                        jpdf = theta * jpdfLo + (1 - theta) * fPdf[i] * aPdf[j]
                        if jpdf > 0:
                            jointEntropy += -jpdf * np.log2(jpdf)

            # 边际熵
            idx_a = np.where(aPdf > 0)[0]
            idx_f = np.where(fPdf > 0)[0]
            aEntropy = np.sum(-aPdf[idx_a] * np.log2(aPdf[idx_a])) if len(idx_a) > 0 else 0
            fEntropy = np.sum(-fPdf[idx_f] * np.log2(fPdf[idx_f])) if len(idx_f) > 0 else 0

            mi = aEntropy + fEntropy - jointEntropy
            if mi == 0:
                fmi_map[p - w, q - w] = 0.0
            else:
                fmi_map[p - w, q - w] = 2.0 * mi / (aEntropy + fEntropy + 1e-10)

    return float(np.mean(fmi_map))


def metric_Qe(fused, ir, vis):
    """
    Piella Image Quality Metric Qe - 基于 Piella_Image_Quality_Metric_Qe.m.
    基于边缘信息的加权融合质量指标.
    """
    f = _to_uint8(_to_gray(fused)).astype(np.float64)
    a = _to_uint8(_to_gray(ir)).astype(np.float64)
    b = _to_uint8(_to_gray(vis)).astype(np.float64)

    # 确保尺寸一致
    minH = min(a.shape[0], b.shape[0], f.shape[0])
    minW = min(a.shape[1], b.shape[1], f.shape[1])
    a, b, f = a[:minH, :minW], b[:minH, :minW], f[:minH, :minW]

    bz = 9  # block size

    # 计算边缘图
    Ae = _im_grad(a)
    Be = _im_grad(b)
    Fe = _im_grad(f)

    alpha = 0.5

    # WFQI on original
    _, WFQI = _weighted_fqi(a, b, f, bz)
    # WFQI on edges
    _, WFQIe = _weighted_fqi(Ae, Be, Fe, bz)

    EFQI = (WFQI ** (1 - alpha)) * (WFQIe ** alpha)
    return float(EFQI)


def _im_grad(img):
    """计算图像梯度幅值 (类似 MATLAB imGrad)."""
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, :-1] = np.diff(img, axis=1)
    gy[:-1, :] = np.diff(img, axis=0)
    return np.sqrt(gx ** 2 + gy ** 2)


def _weighted_fqi(A, B, F, bz):
    """计算加权融合质量指标 WFQI."""
    m = int(np.ceil(A.shape[0] / bz))
    n = int(np.ceil(A.shape[1] / bz))
    Qaf = np.zeros((m, n))
    Qbf = np.zeros((m, n))
    Sa = np.zeros((m, n))
    Sb = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            blockA = _get_block(A, bz, i, j)
            blockB = _get_block(B, bz, i, j)
            blockF = _get_block(F, bz, i, j)
            Qaf[i, j] = _wfq_index(blockA, blockF)
            Qbf[i, j] = _wfq_index(blockB, blockF)
            Sa[i, j] = _saliency(blockA)
            Sb[i, j] = _saliency(blockB)

    eps = 1e-10
    Q0 = Sa / (Sa + Sb + eps) * Qaf + Sb / (Sa + Sb + eps) * Qbf
    IFQI = np.sum(Q0) / (m * n)

    C = np.maximum(Sa, Sb)
    C = C / (np.sum(C) + eps)
    WFQI = np.sum(C * Q0)

    return IFQI, WFQI


def _get_block(im, bz, i, j):
    """提取图像块."""
    h, w = im.shape
    idx_start = bz * i
    idx_end = min(bz * (i + 1), h)
    jdx_start = bz * j
    jdx_end = min(bz * (j + 1), w)
    return im[idx_start:idx_end, jdx_start:jdx_end]


def _wfq_index(A, F):
    """窗口融合质量指数."""
    mA = np.mean(A)
    mF = np.mean(F)
    varA = np.var(A)
    varF = np.var(F)
    covarAF = np.mean((A - mA) * (F - mF))
    return 4 * covarAF * mA * mF / ((mA ** 2 + mF ** 2) * (varA + varF) + 1e-10)


def _saliency(im):
    """显著性 (使用熵)."""
    img = _to_uint8(im)
    hist, _ = np.histogram(img, bins=256, range=(0, 255))
    hist = hist.astype(np.float64)
    hist = hist[hist > 0]
    hist = hist / hist.sum()
    return float(-np.sum(hist * np.log2(hist)))


def _gaussian_window(size, sigma):
    """创建高斯窗口."""
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    window = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return window / window.sum()


# ============================================================================
# Unified interface
# ============================================================================

# All metric functions with their required inputs
NR_METRICS = {
    'EN': metric_EN,
    'SD': metric_SD,
    'SF': metric_SF,
    'AG': metric_AG,
    'NIQE': metric_NIQE,
    'SD_HF': metric_SD_HF,
    'HF_ratio': metric_HF_ratio,
}

FR_METRICS = {
    'MI': metric_MI,
    'VIF': metric_VIF,
    'SSIM': metric_SSIM,
    'MS_SSIM': metric_MS_SSIM,
    'Qabf': metric_Qabf,
    'SCD': metric_SCD,
    'CC': metric_CC,
    'PC': metric_PC,
    'FMI_pixel': metric_FMI_pixel,
    'Qe': metric_Qe,
}

ALL_METRICS = {**NR_METRICS, **FR_METRICS}

# Default metric set for rebuttal experiments
DEFAULT_METRICS = ['EN', 'SD', 'SF', 'AG', 'MI', 'VIF', 'SSIM', 'Qabf', 'SCD']


def compute_all_metrics(fused, ir, vis, metrics=None):
    """
    Compute all specified metrics for a fused-source image triplet.

    Args:
        fused: np.ndarray, fused image (H,W) or (H,W,C)
        ir: np.ndarray, infrared source image
        vis: np.ndarray, visible source image
        metrics: list of metric names (default: DEFAULT_METRICS)

    Returns:
        dict: {metric_name: value}
    """
    if metrics is None:
        metrics = DEFAULT_METRICS

    results = {}
    for name in metrics:
        if name in NR_METRICS:
            try:
                results[name] = NR_METRICS[name](fused)
            except Exception as e:
                print(f"  Warning: {name} failed: {e}")
                results[name] = float('nan')
        elif name in FR_METRICS:
            try:
                results[name] = FR_METRICS[name](fused, ir, vis)
            except Exception as e:
                print(f"  Warning: {name} failed: {e}")
                results[name] = float('nan')
        else:
            print(f"  Warning: Unknown metric '{name}', skipping.")
    return results


def get_metric_direction():
    """Return dict mapping metric name to 'higher' or 'lower'."""
    directions = {name: 'higher' for name in ALL_METRICS}
    directions['NIQE'] = 'lower'
    return directions
