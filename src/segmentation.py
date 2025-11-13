import numpy as np


def _smooth_histogram(hist: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    # Gaussian smoothing via convolution
    size = int(max(3, int(6 * sigma)))
    if size % 2 == 0:
        size += 1
    x = np.arange(size) - size // 2
    gauss = np.exp(-(x ** 2) / (2 * sigma * sigma))
    gauss = gauss / gauss.sum()
    sm = np.convolve(hist, gauss, mode='same')
    return sm


def valley_threshold_gray(gray: np.ndarray, sigma: float = 2.0) -> int:
    """Find threshold by the 'valley' method on a grayscale image.

    Steps:
    - Compute histogram (256 bins)
    - Smooth histogram with Gaussian kernel
    - Detect peaks (local maxima)
    - If >= 2 peaks, take two highest peaks and find the minimum between them (valley)
    - Otherwise fall back to Otsu
    """
    if gray is None:
        raise ValueError("gray is None")
    if gray.ndim != 2:
        raise ValueError("Input must be a 2D grayscale image")

    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 255))
    hist = hist.astype(np.float32)
    sh = _smooth_histogram(hist, sigma=sigma)

    # find local maxima
    peaks = []
    for i in range(1, len(sh) - 1):
        if sh[i] > sh[i - 1] and sh[i] > sh[i + 1]:
            peaks.append(i)

    if len(peaks) < 2:
        # fallback to Otsu (pure NumPy implementation)
        return otsu_threshold(gray)

    # take two highest peaks by smoothed histogram value
    peak_vals = [(int(p), float(sh[p])) for p in peaks]
    peak_vals.sort(key=lambda x: x[1], reverse=True)
    p1 = peak_vals[0][0]
    p2 = peak_vals[1][0]
    if p1 > p2:
        p1, p2 = p2, p1

    # valley = argmin between p1 and p2
    slice_vals = sh[p1:p2 + 1]
    if slice_vals.size == 0:
        # fallback
        return otsu_threshold(gray)
    valley_rel = int(np.argmin(slice_vals))
    valley_idx = p1 + valley_rel
    return int(valley_idx)


def otsu_threshold(gray: np.ndarray) -> int:
    """Compute Otsu threshold using pure NumPy.

    Returns integer threshold in [0,255].
    """
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 255))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0
    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    # ignore first and last where denom 0
    idx = np.nanargmax(sigma_b2)
    return int(idx)


def segment_color_by_valley(img_color: np.ndarray, sigma: float = 2.0, return_mask: bool = False):
    """Segment a color image using threshold found by valley method on intensity.

    Args:
        img_color: BGR image as uint8 (OpenCV convention)
        sigma: smoothing sigma for histogram
        return_mask: if True, return mask along with segmented image

    Returns:
        segmented_bgr (uint8). If return_mask True returns (segmented_bgr, mask_uint8)
    """
    if img_color is None:
        raise ValueError("img_color is None")
    if img_color.ndim != 3:
        raise ValueError("Input must be a color (3-channel) image")

    # Convert BGR/RGB to grayscale using luminosity method. If the image
    # is in BGR order (OpenCV), this formula still produces a valid intensity
    # map because it's just a linear combination of channels.
    b = img_color[:, :, 0].astype(np.float32)
    g = img_color[:, :, 1].astype(np.float32)
    r = img_color[:, :, 2].astype(np.float32)
    gray = (0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8)

    th = valley_threshold_gray(gray, sigma=sigma)

    # Choose foreground as bright region (pixels > th). If majority is above, invert.
    mask = (gray > th).astype(np.uint8) * 255

    # If mask is almost all 255, invert to prefer smaller foreground
    frac = (mask > 0).mean()
    if frac > 0.85:
        mask = (gray <= th).astype(np.uint8) * 255

    # Apply mask to color image
    segmented = img_color.copy()
    segmented[mask == 0] = 0
    if return_mask:
        return segmented, mask
    return segmented
