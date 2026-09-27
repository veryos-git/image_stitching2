"""Input resizing and optional Sobel images, applied once before stitching."""
import cv2
import numpy as np


def preprocess_image(image, max_width=0, mode="none"):
    """Preserve aspect ratio, never upscale, and return a three-channel image."""
    if max_width and image.shape[1] > max_width:
        height = max(1, round(image.shape[0] * max_width / image.shape[1]))
        image = cv2.resize(image, (max_width, height), interpolation=cv2.INTER_AREA)
    if mode == "sobel":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(dx, dy)
        peak = float(magnitude.max())
        edges = (np.rint(magnitude * (255.0 / peak)).clip(0, 255).astype(np.uint8)
                 if peak > 0 else np.zeros(gray.shape, np.uint8))
        image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return image
