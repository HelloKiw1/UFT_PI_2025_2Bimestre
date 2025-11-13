import numpy as np


def _smooth_histogram(hist: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    # Suavização: h_s = h * G_sigma, onde G_sigma(k) = exp(-k^2/(2*sigma^2))/Z
    # Implementado por convolução discreta com um kernel Gaussiano.
    size = int(max(3, int(6 * sigma)))
    if size % 2 == 0:
        size += 1
    x = np.arange(size) - size // 2
    gauss = np.exp(-(x ** 2) / (2 * sigma * sigma))
    gauss = gauss / gauss.sum()
    sm = np.convolve(hist, gauss, mode='same')
    return sm


def valley_threshold_gray(gray: np.ndarray, sigma: float = 2.0) -> int:
    # Método do vale (valley): calcular histograma h(i), suavizar h_s,
    # detectar picos locais p1,p2 e escolher t = argmin_{i in [p1,p2]} h_s(i).
    # Se houver menos de dois picos, usar Otsu como fallback.
    if gray is None:
        raise ValueError("gray é None")
    if gray.ndim != 2:
        raise ValueError("A entrada deve ser uma imagem 2D em tons de cinza")

    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 255))
    hist = hist.astype(np.float32)
    sh = _smooth_histogram(hist, sigma=sigma)

    peaks = []
    for i in range(1, len(sh) - 1):
        if sh[i] > sh[i - 1] and sh[i] > sh[i + 1]:
            peaks.append(i)

    if len(peaks) < 2:
        return otsu_threshold(gray)

    peak_vals = [(int(p), float(sh[p])) for p in peaks]
    peak_vals.sort(key=lambda x: x[1], reverse=True)
    p1 = peak_vals[0][0]
    p2 = peak_vals[1][0]
    if p1 > p2:
        p1, p2 = p2, p1

    slice_vals = sh[p1:p2 + 1]
    if slice_vals.size == 0:
        return otsu_threshold(gray)
    valley_rel = int(np.argmin(slice_vals))
    valley_idx = p1 + valley_rel
    return int(valley_idx)


def otsu_threshold(gray: np.ndarray) -> int:
    # Otsu: p_i = h(i)/N, omega(t)=sum_{i<=t} p_i, mu(t)=sum_{i<=t} i*p_i,
    # mu_T = mu(255). Escolher t que maximiza sigma_B^2(t) = ((mu_T*omega - mu)^2)/(omega*(1-omega)).
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
    idx = np.nanargmax(sigma_b2)
    return int(idx)


def segment_color_by_valley(img_color: np.ndarray, sigma: float = 2.0, return_mask: bool = False):
    # Conversão para intensidade: gray = 0.114*B + 0.587*G + 0.299*R
    # Em seguida encontra-se o limiar via método do vale (ou Otsu) e aplica-se a máscara.
    if img_color is None:
        raise ValueError("img_color é None")
    if img_color.ndim != 3:
        raise ValueError("A entrada deve ser uma imagem colorida (3 canais)")

    b = img_color[:, :, 0].astype(np.float32)
    g = img_color[:, :, 1].astype(np.float32)
    r = img_color[:, :, 2].astype(np.float32)
    gray = (0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8)

    th = valley_threshold_gray(gray, sigma=sigma)

    mask = (gray > th).astype(np.uint8) * 255

    frac = (mask > 0).mean()
    if frac > 0.85:
        mask = (gray <= th).astype(np.uint8) * 255

    segmented = img_color.copy()
    segmented[mask == 0] = 0
    if return_mask:
        return segmented, mask
    return segmented
