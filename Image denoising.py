import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tkinter import ttk
import sys


# Define operations
def operation_gaussian_blur(image, ksize=(3, 3), sigma=0):
    return cv2.GaussianBlur(image, ksize, sigma)

def operation_median_blur(image, ksize=3):
    return cv2.medianBlur(image, ksize)

def operation_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def operation_sharpen(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def apply_low_pass_filter(image):
    kernel = np.ones((5, 5), np.float32) / 25
    return cv2.filter2D(image, -1, kernel)

def apply_high_pass_filter(image):
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

def apply_laplacian_filter(image):
    return cv2.Laplacian(image, cv2.CV_64F)

def apply_threshold(image):
    if len(image.shape) == 2:  # image en niveaux de gris
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
    else:  # image en couleur
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    return thresh

def apply_sobel_filter(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

def apply_canny_edge_detection(image):
    if len(image.shape) == 2:  # grayscaled image
        edges = cv2.Canny(image, 100, 200)
    else:  # colored image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return edges

# Calculate quality metrics
def calculate_quality_metrics(original_image, denoised_image):
    psnr_value = cv2.PSNR(original_image, denoised_image)

    # Determine the appropriate win_size for SSIM
    height, width = original_image.shape[:2]
    win_size = min(height, width, 7)
    if win_size % 2 == 0:
        win_size -= 1

    ssim_value = ssim(original_image, denoised_image, multichannel=True, win_size=3)
    mse_value = np.mean((original_image - denoised_image) ** 2)
    return {'psnr': psnr_value, 'ssim': ssim_value, 'mse': mse_value}

# Normalize metrics
def normalize_metric(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

# Calculate score
def calculate_score(metrics, psnr_range, ssim_range, mse_range, weights):
    psnr_normalized = normalize_metric(metrics['psnr'], psnr_range[0], psnr_range[1])
    ssim_normalized = normalize_metric(metrics['ssim'], ssim_range[0], ssim_range[1])
    mse_normalized = normalize_metric(metrics['mse'], mse_range[0], mse_range[1])
    score = (weights['psnr'] * psnr_normalized) + (weights['ssim'] * ssim_normalized) - (weights['mse'] * mse_normalized)
    return score

# Optimize image denoising
def optimize_image_denoising(original_image, operations, max_iterations=5):
    current_image = original_image.copy()
    best_image = current_image
    current_metrics = calculate_quality_metrics(original_image, current_image)

    psnr_range = (0, 100)  # Adjust as necessary for PSNR
    ssim_range = (0, 1)    # SSIM is already between 0 and 1
    mse_range = (0, 10000) # Adjust as necessary for MSE
    weights = {'psnr': 1, 'ssim': 1, 'mse': 1}  # Weights can be adjusted according to importance

    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}")

        best_score = -float('inf')
        for operation in operations:
            new_image = operation(current_image)
            new_metrics = calculate_quality_metrics(original_image, new_image)
            score = calculate_score(new_metrics, psnr_range, ssim_range, mse_range, weights)

            print(f"Operation: {operation.__name__}")
            print(f"Metrics: PSNR={new_metrics['psnr']:.2f}, SSIM={new_metrics['ssim']:.4f}, MSE={new_metrics['mse']:.2f}")
            print(f"Score: {score:.4f}")

            if score > best_score:
                best_score = score
                best_image = new_image
                current_metrics = new_metrics

        if best_score <= calculate_score(current_metrics, psnr_range, ssim_range, mse_range, weights):
            print("No improvement, stopping.")
            break

        current_image = best_image

    return best_image

def open_image():
    global img_tk  
    filepath = filedialog.askopenfilename()
    if filepath:
        original_image = cv2.imread(filepath)
        operations = [operation_gaussian_blur, operation_median_blur, operation_bilateral_filter, operation_sharpen, apply_low_pass_filter, apply_high_pass_filter, apply_sobel_filter, apply_threshold]
        denoised_image = optimize_image_denoising(original_image, operations)

        denoised_image_rgb = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(denoised_image_rgb)
        img_tk = ImageTk.PhotoImage(img)

        label.config(image=img_tk)  
        label.image = img_tk  

def exit_app():
    root.destroy()
    sys.exit()

root = tk.Tk()
root.title("Denoising Tool")

bg_color = "#f0f0f0"  
button_color = "#4caf50"  
button_hover_color = "#45a049"  

root.configure(bg=bg_color)  

style = ttk.Style()
style.configure('TButton', font=('Arial', 12))
style.map('TButton', background=[('!disabled', button_color), ('pressed', button_hover_color)])

notebook = ttk.Notebook(root)
notebook.pack(pady=10, padx=10)

tab1 = tk.Frame(notebook)
tab2 = tk.Frame(notebook)

notebook.add(tab1, text="File")
notebook.add(tab2, text="About")

about_text = """
It's a small application that takes a noisy image and gives you the image with a better quality"""

about_label = tk.Label(tab2, text=about_text, bg="#ffffff")
about_label.pack(padx=10, pady=10)

label = tk.Label(tab1, bg="#ffffff")  
label.pack(padx=10, pady=10)

browse_button = ttk.Button(tab1, text="Open Image", command=open_image)
browse_button.pack(pady=10)

exit_button = ttk.Button(tab1, text="Exit", command=exit_app)
exit_button.pack(pady=5)

root.mainloop()
