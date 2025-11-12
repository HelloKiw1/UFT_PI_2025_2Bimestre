import cv2
import numpy as np


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
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.erode(image, k, iterations=iterations)


def dilation(image: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """Perform dilation on a grayscale image."""
    if image is None:
        raise ValueError("image is None")
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.dilate(image, k, iterations=iterations)


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
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, k)
