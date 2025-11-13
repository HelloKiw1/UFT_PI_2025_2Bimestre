import numpy as np


def erosion(image: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    # Fórmula (erosão): (f ⊖ B)(x) = min_{u in B} f(x + u)
    # - f: imagem de entrada (image)
    # - B: elemento estruturante (kernel quadrado de tamanho kernel_size)
    # Para kernel quadrado k x k, r = kernel_size // 2 e u varia em [-r, r]^2
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
    # Fórmula (dilatação): (f ⊕ B)(x) = max_{u in B} f(x + u)
    # - f: imagem de entrada
    # - B: elemento estruturante (kernel quadrado de lado kernel_size)
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
    # Fórmula (top-hat branco): top_hat(f) = f - (f ∘ B)
    # onde abertura f ∘ B = (f ⊖ B) ⊕ B
    opened = erosion(image, kernel_size=kernel_size, iterations=1)
    opened = dilation(opened, kernel_size=kernel_size, iterations=1)
    res = image.astype(np.int16) - opened.astype(np.int16)
    res = np.clip(res, 0, 255).astype(np.uint8)
    return res
