import cv2
import numpy as np
from scipy.signal import wiener
from scipy.fftpack import fft2, ifft2, fftshift
import os

# Load the distorted image
image_path = r"C:\Users\niros\OneDrive\Desktop\EE 405 FYP clear sight and TSR\Image_pre_processing\distorted_image.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Ensure the image is loaded properly
if image is None:
    print(f"Error: Unable to load image at {image_path}")
    exit()

# Create an output directory
output_dir = r"C:\Users\niros\OneDrive\Desktop\EE 405 FYP clear sight and TSR\Image_pre_processing"
os.makedirs(output_dir, exist_ok=True)

# 1. Smoothing (Gaussian Blur)
smoothed = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imwrite(os.path.join(output_dir, "smoothed.jpg"), smoothed)

# 2. Gradient Operations (Sobel and Laplacian)
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
laplacian = cv2.Laplacian(image, cv2.CV_64F)

cv2.imwrite(os.path.join(output_dir, "sobel_x.jpg"), np.abs(sobel_x).astype(np.uint8))
cv2.imwrite(os.path.join(output_dir, "sobel_y.jpg"), np.abs(sobel_y).astype(np.uint8))
cv2.imwrite(os.path.join(output_dir, "laplacian.jpg"), np.abs(laplacian).astype(np.uint8))

# 3. Wiener Filtration (Noise Reduction)
def wiener_filter(img, kernel_size=5):
    return wiener(img, (kernel_size, kernel_size))

wiener_filtered = wiener_filter(image)
cv2.imwrite(os.path.join(output_dir, "wiener_filtered.jpg"), np.uint8(np.clip(wiener_filtered, 0, 255)))

# 4. Inverse Filtration (Deblurring)
def inverse_filter(img, radius=10):
    h, w = img.shape
    H = np.ones((h, w), dtype=np.float32)
    center = (h // 2, w // 2)

    for i in range(h):
        for j in range(w):
            d = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
            if d < radius:
                H[i, j] = d / radius

    img_fft = fft2(img)
    img_fft_shift = fftshift(img_fft)
    restored_fft = img_fft_shift / (H + 1e-6)
    restored_img = np.abs(ifft2(restored_fft))

    return np.uint8(np.clip(restored_img, 0, 255))

inverse_filtered = inverse_filter(image)
cv2.imwrite(os.path.join(output_dir, "inverse_filtered.jpg"), inverse_filtered)

print(f"Processed images saved in: {output_dir}")
