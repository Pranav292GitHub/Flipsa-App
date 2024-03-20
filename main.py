import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageFilter, ImageOps, ImageChops
import cv2 
import io

class FlipsaApp:
    def __init__(self, root=tk.Tk()):
        self.image2 = None
        # Initialize the GUI elements and variables
        self.bg = "dodgerblue"
        self.root = root
        self.root.title("Flipsa - DEMO")
        self.root.configure(bg=self.bg)
        self.root.geometry("900x600")  # Reduced window width

        self.title = tk.Label(root, text="Flipsa", bg=self.bg, foreground="white", font=("Lobster", 64))
        self.title.pack()

        self.version = tk.Label(root, text="DEMO", bg=self.bg, foreground="white", font=("Calibri", 16))
        self.version.pack()
        self.canvas = tk.Canvas(root, bg=self.bg)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.button_frame = tk.Frame(self.canvas, bg=self.bg)
        self.canvas.create_window((0, 0), window=self.button_frame, anchor=tk.NW)

        self.scrollbar = tk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill="y")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.button_frame.bind("<Configure>", self.on_frame_configure)

        self.grayscale_intensity_label = tk.Label(self.button_frame, text="Grayscale Intensity", bg=self.bg, fg="white")
        self.grayscale_intensity_label.pack(pady=2, padx=10)
        self.grayscale_intensity = tk.Scale(self.button_frame, from_=0, to=255, orient=tk.HORIZONTAL, bg=self.bg, fg="white", highlightbackground=self.bg, highlightcolor=self.bg)
        self.grayscale_intensity.set(128)  # Initial intensity value
        self.grayscale_intensity.pack(pady=2, padx=10, fill=tk.X)

        self.grayscale_btn = tk.Button(self.button_frame, text="Grayscale", bg="goldenrod", foreground="white", command=self.apply_grayscale, width=15, height=2, font=("Lobster", 14))
        self.grayscale_btn.pack(pady=5, padx=10, fill=tk.X)

        self.sketch_intensity_label = tk.Label(self.button_frame, text="Sketch Intensity", bg=self.bg, fg="white")
        self.sketch_intensity_label.pack(pady=2, padx=10)
        self.sketch_intensity = tk.Scale(self.button_frame, from_=0, to=255, orient=tk.HORIZONTAL, bg=self.bg, fg="white", highlightbackground=self.bg, highlightcolor=self.bg)
        self.sketch_intensity.set(128)  # Initial intensity value
        self.sketch_intensity.pack(pady=2, padx=10, fill=tk.X)

        self.sketch_btn = tk.Button(self.button_frame, text="Sketch", bg="goldenrod", foreground="white", command=self.apply_sketch, width=15, height=2, font=("Lobster", 14))
        self.sketch_btn.pack(pady=5, padx=10, fill=tk.X)

        self.cartoonize_intensity_label = tk.Label(self.button_frame, text="Cartoonize Intensity", bg=self.bg, fg="white")
        self.cartoonize_intensity_label.pack(pady=2, padx=10)
        self.cartoonize_intensity = tk.Scale(self.button_frame, from_=0, to=100, orient=tk.HORIZONTAL, bg=self.bg, fg="white", highlightbackground=self.bg, highlightcolor=self.bg)
        self.cartoonize_intensity.set(50)  # Initial intensity value
        self.cartoonize_intensity.pack(pady=2, padx=10, fill=tk.X)

        self.cartoonize_btn = tk.Button(self.button_frame, text="Cartoonize", bg="goldenrod", foreground="white", command=self.apply_cartoonize, width=15, height=2, font=("Lobster", 14))
        self.cartoonize_btn.pack(pady=5, padx=10, fill=tk.X)

        self.upload_btn = tk.Button(self.button_frame, text="Upload Image", bg="goldenrod", foreground="white", command=self.upload_image, width=15, height=2, font=("Lobster", 14))
        self.upload_btn.pack(pady=5, padx=10, fill=tk.X)

        self.image_label = tk.Label(root, bg="purple")
        self.image_label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.image = None
        self.photo = None

    def on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def apply_grayscale(self):
        if self.image is not None:
            intensity = self.grayscale_intensity.get()
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            gray_image = cv2.addWeighted(gray_image, intensity / 255.0, np.zeros_like(gray_image), 0, 0)
            self.display_cv2_image(gray_image)

    def apply_sketch(self):
        if self.image is not None:
            intensity = self.sketch_intensity.get()
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            inverted_gray_image = cv2.bitwise_not(gray_image)
            blurred_image = cv2.GaussianBlur(inverted_gray_image, (609, 609), 0)
            inverted_blurred_image = cv2.bitwise_not(blurred_image)
            sketch_image = cv2.divide(gray_image, inverted_blurred_image, scale=256.0)
            sketch_image = cv2.addWeighted(sketch_image, intensity / 255.0, np.zeros_like(sketch_image), 0, 0)
            self.display_cv2_image(sketch_image)

    def apply_cartoonize(self):
        if self.image is not None:
            intensity = self.cartoonize_intensity.get() / 100.0
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            blurred_image = cv2.medianBlur(gray_image, 5)
            edges = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
            color = cv2.bilateralFilter(self.image, 9, 250, 250)
            cartoon_image = cv2.bitwise_and(color, color, mask=edges)
            cartoon_image = cv2.addWeighted(self.image, intensity, cartoon_image, 1 - intensity, 0)
            self.display_cv2_image(cartoon_image)

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            self.display_cv2_image(self.image)

    def display_cv2_image(self, cv2_image):
        if cv2_image is not None:
            rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_image)
            photo = ImageTk.PhotoImage(image=image)
            self.image_label.config(image=photo)
            self.image_label.image = photo
            self.image2 = photo

    #    Still under development :}





if __name__ == "__main__":
    app = FlipsaApp()
    app.root.mainloop()