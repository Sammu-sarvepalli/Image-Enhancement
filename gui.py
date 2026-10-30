import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# Global variables
img = None
processed_img = None
enhanced_img = None


# ------------------ WCA CORE ------------------ #
def calculate_entropy(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))


def enhance_image_wca(image, population_size=10, max_iter=20):

    def enhance(img, alpha, beta):
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    def fitness(img):
        return calculate_entropy(img)

    def optimize_channel(channel):
        alpha_range = [1.0, 3.0]
        beta_range = [0, 100]

        population = [(random.uniform(*alpha_range), random.uniform(*beta_range))
                      for _ in range(population_size)]

        population.sort(key=lambda x: -fitness(enhance(channel, x[0], x[1])))
        sea = population[0]
        rivers = population[1:3]
        streams = population[3:]

        for _ in range(max_iter):

            # Streams → Rivers
            for i in range(len(streams)):
                target = rivers[i % len(rivers)]
                new_alpha = streams[i][0] + random.uniform(0, 1) * (target[0] - streams[i][0])
                new_beta = streams[i][1] + random.uniform(0, 1) * (target[1] - streams[i][1])

                streams[i] = (
                    np.clip(new_alpha, *alpha_range),
                    np.clip(new_beta, *beta_range)
                )

            # Rivers → Sea
            for i in range(len(rivers)):
                new_alpha = rivers[i][0] + random.uniform(0, 1) * (sea[0] - rivers[i][0])
                new_beta = rivers[i][1] + random.uniform(0, 1) * (sea[1] - rivers[i][1])

                rivers[i] = (
                    np.clip(new_alpha, *alpha_range),
                    np.clip(new_beta, *beta_range)
                )

            # Re-evaluate
            all_individuals = [sea] + rivers + streams
            all_individuals.sort(key=lambda x: -fitness(enhance(channel, x[0], x[1])))

            sea = all_individuals[0]
            rivers = all_individuals[1:3]
            streams = all_individuals[3:]

        best_alpha, best_beta = sea
        return enhance(channel, best_alpha, best_beta)

    # -------- IMAGE TYPE HANDLING -------- #
    if len(image.shape) == 2:
        # True grayscale
        return optimize_channel(image)

    elif len(image.shape) == 3:
        # Check if pseudo grayscale (3 channels but same)
        if np.array_equal(image[:, :, 0], image[:, :, 1]) and np.array_equal(image[:, :, 1], image[:, :, 2]):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return optimize_channel(gray)

        # True color image
        b, g, r = cv2.split(image)

        b_enh = optimize_channel(b)
        g_enh = optimize_channel(g)
        r_enh = optimize_channel(r)

        return cv2.merge((b_enh, g_enh, r_enh))


# ---------- GUI FUNCTIONS ---------- #
def load_image():
    filepath = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.png *.bmp *.jpeg")]
    )
    if not filepath:
        return

    global img, processed_img

    try:
        # FIX: Use UNCHANGED to preserve grayscale properly
        img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

        if img is None:
            raise Exception("Error reading image")

        processed_img = img.copy()
        show_image(processed_img, lbl_original_img)

        lbl_original_text.config(text="Original Image")
        lbl_status.config(text="Image loaded successfully!")

    except Exception as e:
        messagebox.showerror("Error", str(e))


def enhance_and_show():
    global enhanced_img

    if processed_img is None:
        messagebox.showwarning("Warning", "Please load an image first.")
        return

    enhanced_img = enhance_image_wca(processed_img)

    show_image(enhanced_img, lbl_enhanced_img)
    lbl_enhanced_text.config(text="Enhanced Image (WCA)")
    lbl_status.config(text="Enhancement completed!")

    plot_histograms(processed_img, enhanced_img)


def save_image():
    if enhanced_img is None:
        messagebox.showinfo("Info", "No enhanced image to save.")
        return

    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG Files", "*.png")]
    )

    if file_path:
        cv2.imwrite(file_path, enhanced_img)
        messagebox.showinfo("Saved", "Enhanced image saved successfully!")


def show_image(image, label_widget):
    img_resized = cv2.resize(image, (300, 300))

    if len(img_resized.shape) == 2:
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil)

    label_widget.config(image=img_tk)
    label_widget.image = img_tk


def plot_histograms(original, enhanced):
    plt.figure(figsize=(10, 4))

    # Original
    plt.subplot(1, 2, 1)
    if len(original.shape) == 2:
        plt.hist(original.flatten(), bins=256)
        plt.title("Original Grayscale Histogram")
    else:
        colors = ('b', 'g', 'r')
        for i, col in enumerate(colors):
            hist = cv2.calcHist([original], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
        plt.title("Original Color Histogram")

    # Enhanced
    plt.subplot(1, 2, 2)
    if len(enhanced.shape) == 2:
        plt.hist(enhanced.flatten(), bins=256)
        plt.title("Enhanced Grayscale Histogram")
    else:
        colors = ('b', 'g', 'r')
        for i, col in enumerate(colors):
            hist = cv2.calcHist([enhanced], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
        plt.title("Enhanced Color Histogram")

    plt.tight_layout()
    plt.show()


# ---------- GUI SETUP ---------- #
root = tk.Tk()
root.title("Efficient and Effective Blind JPEG Image Improvement with Sequential Feature Processing")
root.geometry("950x670")
root.configure(bg="#f0f2f5")

style = ttk.Style()
style.configure("TButton", font=("Helvetica", 12), padding=6)
style.configure("TLabel", font=("Helvetica", 12))

# Title
frame_title = tk.Frame(root, bg="#283593")
frame_title.pack(fill=tk.X)

title_label = tk.Label(
    frame_title,
    text="Efficient and Effective Blind JPEG Image Improvement with Sequential Feature Processin",
    bg="#283593",
    fg="white",
    font=("Helvetica", 16, "bold"),
    pady=10
)
title_label.pack()

# Description
frame_desc = tk.Frame(root, bg="#f0f2f5")
frame_desc.pack(pady=5)

desc = "Enhances grayscale and color images correctly using WCA"
tk.Label(frame_desc, text=desc, bg="#f0f2f5").pack()

# Buttons
frame_buttons = tk.Frame(root, bg="#f0f2f5")
frame_buttons.pack(pady=10)

ttk.Button(frame_buttons, text="📁 Load Image", command=load_image).grid(row=0, column=0, padx=10)
ttk.Button(frame_buttons, text="✨ Enhance Image", command=enhance_and_show).grid(row=0, column=1, padx=10)
ttk.Button(frame_buttons, text="💾 Save Image", command=save_image).grid(row=0, column=2, padx=10)
ttk.Button(frame_buttons, text="❌ Exit", command=root.quit).grid(row=0, column=3, padx=10)

# Images
frame_images = tk.Frame(root, bg="#f0f2f5")
frame_images.pack(pady=10)

lbl_original_img = tk.Label(frame_images, bg="#e0e0e0", width=300, height=300)
lbl_original_img.grid(row=0, column=0, padx=30)

lbl_enhanced_img = tk.Label(frame_images, bg="#e0e0e0", width=300, height=300)
lbl_enhanced_img.grid(row=0, column=1, padx=30)

# Labels
frame_labels = tk.Frame(root, bg="#f0f2f5")
frame_labels.pack()

lbl_original_text = tk.Label(frame_labels, text="", bg="#f0f2f5")
lbl_original_text.grid(row=0, column=0, padx=140)

lbl_enhanced_text = tk.Label(frame_labels, text="", bg="#f0f2f5")
lbl_enhanced_text.grid(row=0, column=1, padx=140)

# Status
lbl_status = tk.Label(root, text="", fg="green", bg="#f0f2f5")
lbl_status.pack(pady=10)

root.mainloop()