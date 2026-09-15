import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# method: (slider1 cfg, slider2 cfg hoặc None)
METHOD_CFG = {
    "Negative":            (None, None),
    "Log":                 (None, None),
    "Gamma":               (dict(from_=0.1, to=3.0, resolution=0.1, label="Gamma", default=1.0), None),
    "Contrast Stretching": (dict(from_=0, to=255, resolution=1, label="r1", default=50),
                             dict(from_=0, to=255, resolution=1, label="r2", default=200)),
    "Thresholding":        (dict(from_=0, to=255, resolution=1, label="Ngưỡng T", default=127), None),
    "Bit-plane":           (dict(from_=0, to=7, resolution=1, label="Bit Plane", default=7), None),
    "Histogram Eq":        (None, None),

    # ================== BỔ SUNG: NHÓM XỬ LÝ THEO VÙNG LÂN CẬN ==================
    "Averaging":           (dict(from_=1, to=15, resolution=2, label="Kích thước mask", default=5), None),
    "Median":              (dict(from_=3, to=15, resolution=2, label="Kích thước mask", default=5), None),
    "Laplacian":           (None, None),
    "Unsharp/Highboost":   (dict(from_=0.1, to=3.0, resolution=0.1, label="Hệ số k", default=1.5),
                             dict(from_=3, to=15, resolution=2, label="Kernel làm mờ", default=9)),
    "Roberts":             (None, None),
    "Sobel":               (dict(from_=1, to=7, resolution=2, label="Kích thước mask", default=3), None),

    # ======================= BỔ SUNG: NHÓM XỬ LÝ MÀU =======================
    "RGB - Kênh màu":      (dict(from_=0, to=2, resolution=1, label="0=R 1=G 2=B", default=0), None),
    "CMYK - Kênh màu":     (dict(from_=0, to=3, resolution=1, label="0=C 1=M 2=Y 3=K", default=0), None),
    "HSI - Kênh màu":      (dict(from_=0, to=2, resolution=1, label="0=H 1=S 2=I", default=2), None),
    "Pseudocolor":         (dict(from_=0, to=4, resolution=1, label="Chỉ số Colormap", default=0), None),
    "Color Slicing":       (dict(from_=0, to=179, resolution=1, label="Tâm màu (Hue)", default=0),
                             dict(from_=5, to=90, resolution=1, label="Dung sai", default=15)),
}

# Danh sách colormap dùng cho Pseudocolor (chọn qua slider "Chỉ số Colormap")
PSEUDOCOLOR_MAPS = [cv2.COLORMAP_JET, cv2.COLORMAP_HOT, cv2.COLORMAP_OCEAN,
                    cv2.COLORMAP_RAINBOW, cv2.COLORMAP_BONE]


class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ứng Dụng Xử Lý Ảnh")
        self.img = self.processed_img = None
        self.img_color = None  # BỔ SUNG: ảnh màu gốc (BGR), dùng cho các PP xử lý màu

        ctrl = tk.Frame(root)
        ctrl.pack(side=tk.TOP, fill=tk.X, pady=5)

        tk.Button(ctrl, text="Mở Ảnh", command=self.load_image).pack(side=tk.LEFT, padx=5)

        self.method_var = tk.StringVar(value="Negative")
        cb = ttk.Combobox(ctrl, textvariable=self.method_var, values=list(METHOD_CFG), state="readonly")
        cb.pack(side=tk.LEFT, padx=5)
        cb.bind("<<ComboboxSelected>>", self.update_ui)

        self.slider1 = tk.Scale(ctrl, orient=tk.HORIZONTAL, command=self.apply_filter)
        self.slider1.pack(side=tk.LEFT, padx=5)
        self.slider2 = tk.Scale(ctrl, orient=tk.HORIZONTAL, command=self.apply_filter)

        tk.Button(ctrl, text="Lưu Ảnh", command=self.save_image).pack(side=tk.RIGHT, padx=5)
        tk.Button(ctrl, text="Đặt Lại", command=self.reset_image).pack(side=tk.RIGHT, padx=5)

        disp = tk.Frame(root)
        disp.pack(fill=tk.BOTH, expand=True)
        self.lbl_original = tk.Label(disp, text="Ảnh Gốc")
        self.lbl_original.grid(row=0, column=0, padx=10, pady=5)
        self.lbl_result = tk.Label(disp, text="Ảnh Kết Quả")
        self.lbl_result.grid(row=0, column=1, padx=10, pady=5)

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(8, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=disp)
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=2)

        self.update_ui()

    # --- UI ---
    def update_ui(self, event=None):
        cfg1, cfg2 = METHOD_CFG[self.method_var.get()]
        self.slider2.pack_forget()

        for slider, cfg, show in ((self.slider1, cfg1, True), (self.slider2, cfg2, cfg2 is not None)):
            if cfg:
                slider.config(from_=cfg["from_"], to=cfg["to"], resolution=cfg["resolution"], label=cfg["label"])
                slider.set(cfg["default"])
            else:
                slider.config(label="Không dùng", from_=0, to=0)
            if show:
                slider.pack(side=tk.LEFT, padx=5)

        self.apply_filter()

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self.img_color = cv2.imread(path, cv2.IMREAD_COLOR)  # BỔ SUNG: nạp thêm bản màu để dùng cho PP xử lý màu
            self.apply_filter()

    # --- Xử lý ảnh ---
    def apply_filter(self, val=None):
        if self.img is None:
            return
        method = self.method_var.get()
        img_f = self.img.astype(np.float32)
        p1, p2 = self.slider1.get(), self.slider2.get()

        if method == "Negative":
            self.processed_img = 255 - self.img
        elif method == "Log":
            c = 255 / np.log(1 + 255)
            self.processed_img = np.uint8(c * np.log(1 + img_f))
        elif method == "Gamma":
            self.processed_img = np.uint8(((self.img / 255.0) ** float(p1)) * 255)
        elif method == "Thresholding":
            _, self.processed_img = cv2.threshold(self.img, int(p1), 255, cv2.THRESH_BINARY)
        elif method == "Bit-plane":
            self.processed_img = ((self.img >> int(p1)) & 1) * 255
        elif method == "Contrast Stretching":
            r1, r2 = int(p1), int(p2)
            r1, r2 = min(r1, r2), max(r1, r2)
            self.processed_img = np.piecewise(
                img_f,
                [img_f <= r1, img_f >= r2],
                [lambda r: (0 / max(r1, 1)) * r,
                 lambda r: 255,
                 lambda r: (255 / max(r2 - r1, 1)) * (r - r1)]
            ).astype(np.uint8)
        elif method == "Histogram Eq":
            self.processed_img = cv2.equalizeHist(self.img)

        # ================== BỔ SUNG: NHÓM XỬ LÝ THEO VÙNG LÂN CẬN ==================
        elif method == "Averaging":
            k = max(1, int(p1) | 1)  # đảm bảo số lẻ
            self.processed_img = cv2.blur(self.img, (k, k))
        elif method == "Median":
            k = max(3, int(p1) | 1)  # đảm bảo số lẻ, >=3
            self.processed_img = cv2.medianBlur(self.img, k)
        elif method == "Laplacian":
            lap = cv2.Laplacian(self.img.astype(np.float64), cv2.CV_64F, ksize=3)
            sharp = img_f - lap
            self.processed_img = np.clip(sharp, 0, 255).astype(np.uint8)
        elif method == "Unsharp/Highboost":
            k_blur = max(3, int(p2) | 1)
            blurred = cv2.GaussianBlur(self.img, (k_blur, k_blur), 0).astype(np.float32)
            mask = img_f - blurred
            sharp = img_f + float(p1) * mask
            self.processed_img = np.clip(sharp, 0, 255).astype(np.uint8)
        elif method == "Roberts":
            kx = np.array([[1, 0], [0, -1]], dtype=np.float32)
            ky = np.array([[0, 1], [-1, 0]], dtype=np.float32)
            gx = cv2.filter2D(img_f, -1, kx)
            gy = cv2.filter2D(img_f, -1, ky)
            g = np.sqrt(gx ** 2 + gy ** 2)
            self.processed_img = np.clip(g, 0, 255).astype(np.uint8)
        elif method == "Sobel":
            k = max(1, int(p1) | 1)
            gx = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=k)
            gy = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=k)
            g = np.sqrt(gx ** 2 + gy ** 2)
            self.processed_img = np.clip(g, 0, 255).astype(np.uint8)

        # ======================= BỔ SUNG: NHÓM XỬ LÝ MÀU =======================
        elif method == "RGB - Kênh màu":
            if self.img_color is not None:
                b, g, r = cv2.split(self.img_color)
                self.processed_img = [r, g, b][int(p1)]
        elif method == "CMYK - Kênh màu":
            if self.img_color is not None:
                bgr = self.img_color.astype(np.float64) / 255.0
                b, g, r = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
                k = 1 - np.max(bgr, axis=2)
                denom = np.where(1 - k == 0, 1, 1 - k)
                c = (1 - r - k) / denom
                m = (1 - g - k) / denom
                y = (1 - b - k) / denom
                channel = [c, m, y, k][int(p1)]
                self.processed_img = np.clip(channel * 255, 0, 255).astype(np.uint8)
        elif method == "HSI - Kênh màu":
            if self.img_color is not None:
                bgr = self.img_color.astype(np.float64) / 255.0
                b, g, r = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
                intensity = (r + g + b) / 3.0
                minc = np.minimum(np.minimum(r, g), b)
                saturation = 1 - (3.0 / (r + g + b + 1e-6)) * minc
                num = 0.5 * ((r - g) + (r - b))
                den = np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + 1e-6
                theta = np.arccos(np.clip(num / den, -1, 1))
                hue = np.where(b <= g, theta, 2 * np.pi - theta) / (2 * np.pi)
                channel = [hue, saturation, intensity][int(p1)]
                self.processed_img = np.clip(channel * 255, 0, 255).astype(np.uint8)
        elif method == "Pseudocolor":
            colormap = PSEUDOCOLOR_MAPS[int(p1) % len(PSEUDOCOLOR_MAPS)]
            colored = cv2.applyColorMap(self.img, colormap)
            self.processed_img = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        elif method == "Color Slicing":
            if self.img_color is not None:
                hsv = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2HSV)
                center, tol = int(p1), int(p2)
                lower = max(0, center - tol)
                upper = min(179, center + tol)
                mask = cv2.inRange(hsv, (lower, 60, 60), (upper, 255, 255))
                if center - tol < 0:  # xử lý vòng lặp Hue quanh 0/179
                    mask |= cv2.inRange(hsv, (180 + (center - tol), 60, 60), (179, 255, 255))
                if center + tol > 179:
                    mask |= cv2.inRange(hsv, (0, 60, 60), ((center + tol) - 180, 255, 255))
                out = np.full_like(self.img_color, 30)
                out[mask > 0] = self.img_color[mask > 0]
                self.processed_img = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

        self.display_images()
        self.draw_histograms()

    # --- Hiển thị ---
    def display_images(self):
        for lbl, arr in ((self.lbl_original, self.img), (self.lbl_result, self.processed_img)):
            tk_img = ImageTk.PhotoImage(Image.fromarray(arr).resize((250, 250)))
            lbl.config(image=tk_img)
            lbl.image = tk_img

    def draw_histograms(self):
        for ax, arr, title in ((self.ax1, self.img, "Histogram Gốc"),
                                (self.ax2, self.processed_img, "Histogram Kết Quả")):
            ax.clear()
            ax.hist(arr.ravel(), 256, [0, 256], color='gray')
            ax.set_title(title)
        self.canvas.draw()

    def save_image(self):
        if self.processed_img is not None:
            path = filedialog.asksaveasfilename(defaultextension=".png")
            if path:
                cv2.imwrite(path, self.processed_img)

    def reset_image(self):
        self.update_ui()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()