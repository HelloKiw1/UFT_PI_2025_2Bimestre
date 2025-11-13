import numpy as np


def erosion(image: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    if image is None:
        raise ValueError("imagem é None")
    if image.ndim != 2:
        raise ValueError("erosion: a entrada deve ser uma imagem 2D em tons de cinza")

    img = image.astype(np.uint8).copy()
    h, w = img.shape
    pad = kernel_size // 2
    for _ in range(iterations):
        padded = np.pad(img, pad, mode='constant', constant_values=255)
        out = np.empty_like(img)
        for i in range(h):
            for j in range(w):
                window = padded[i:i + kernel_size, j:j + kernel_size]
                out[i, j] = window.min()
        img = out
    return img


def dilation(image: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    if image is None:
        raise ValueError("imagem é None")
    if image.ndim != 2:
        raise ValueError("dilation: a entrada deve ser uma imagem 2D em tons de cinza")

    img = image.astype(np.uint8).copy()
    h, w = img.shape
    pad = kernel_size // 2
    for _ in range(iterations):
        padded = np.pad(img, pad, mode='constant', constant_values=0)
        out = np.empty_like(img)
        for i in range(h):
            for j in range(w):
                window = padded[i:i + kernel_size, j:j + kernel_size]
                out[i, j] = window.max()
        img = out
    return img


def top_hat(image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    if image is None:
        raise ValueError("imagem é None")
    if image.ndim != 2:
        raise ValueError("top_hat: a entrada deve ser uma imagem 2D em tons de cinza")

    opened = erosion(image, kernel_size=kernel_size, iterations=1)
    opened = dilation(opened, kernel_size=kernel_size, iterations=1)
    res = image.astype(np.int16) - opened.astype(np.int16)
    res = np.clip(res, 0, 255).astype(np.uint8)
    return res
