import numpy as np

# Naive implementations using only NumPy (no OpenCV) for portability and
# educational purposes. These functions implement basic grayscale
# morphological operations using min/max filters.


def erosion(image: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """Perform erosion on a grayscale image.

    Args:
        image: Grayscale image as a 2D NumPy array (dtype uint8).
        kernel_size: Size of the square structuring element (odd integer).
        iterations: Number of erosion iterations.

    Returns:
        Eroded image (uint8).
    """
    if image is None:
        raise ValueError("image is None")
    if image.ndim != 2:
        raise ValueError("erosion: input must be a 2D grayscale image")

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
    """Perform dilation on a grayscale image using a max filter (NumPy).

    This is a naive implementation (nested loops). For large images use
    optimized libraries.
    """
    if image is None:
        raise ValueError("image is None")
    if image.ndim != 2:
        raise ValueError("dilation: input must be a 2D grayscale image")

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
    """Compute white top-hat transform: original - opening.

    Args:
        image: Grayscale image as 2D array.
        kernel_size: size of structuring element used for opening.

    Returns:
        Top-hat transformed image (uint8).
    """
    if image is None:
        raise ValueError("image is None")
    if image.ndim != 2:
        raise ValueError("top_hat: input must be a 2D grayscale image")

    # Opening = erosion followed by dilation
    opened = erosion(image, kernel_size=kernel_size, iterations=1)
    opened = dilation(opened, kernel_size=kernel_size, iterations=1)
    # top-hat = original - opened (clip to unsigned range)
    res = image.astype(np.int16) - opened.astype(np.int16)
    res = np.clip(res, 0, 255).astype(np.uint8)
    return res
