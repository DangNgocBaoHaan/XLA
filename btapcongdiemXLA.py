import cv2
import numpy as np
import math
from collections import OrderedDict

VIDEO_INPUT_PATH = "C:/Users/PC/Downloads/XLA/conveyor_inspection_student.mp4"
VIDEO_OUTPUT_PATH = "C:/Users/PC/Downloads/XLA/conveyor_output.mp4"

FRAME_DELAY_MS = 20          
END_HOLD_SECONDS = 3         
END_WAIT_MS = 0              

INSPECTION_LINE_RATIO = 0.20

MIN_CONTOUR_AREA = 1200       # Loại bỏ nhiễu nhỏ
PASS_MIN_AREA = 2600          # Nhỏ hơn 2600 -> FAIL (Size nhỏ)
PASS_MAX_AREA = 5500          # Lớn hơn 5500 -> FAIL (Size quá khổ)

# Ngưỡng hình học giữ rộng như Bản 2 -> Kiện hàng nằm xéo/nghiêng vẫn PASS bình thường
PASS_MIN_SOLIDITY = 0.82      # Độ đặc (Area / Convex Hull Area)
PASS_MIN_ASPECT_RATIO = 0.55  
PASS_MAX_ASPECT_RATIO = 2.00  

# Ngưỡng màu sắc: Sản phẩm chuẩn màu Cam có Hue từ 5 đến 35
PASS_HUE_MIN = 5
PASS_HUE_MAX = 35
PASS_MIN_SATURATION = 50
PASS_MIN_VALUE = 60

MAX_TRACKING_DISTANCE = 85
MAX_DISAPPEARED_FRAMES = 8

# LỚP THEO DÕI SẢN PHẨM TRÊN BĂNG TẢI 
class ProductTracker:
    def __init__(self, max_disappeared=MAX_DISAPPEARED_FRAMES, max_distance=MAX_TRACKING_DISTANCE):
        self.next_object_id = 1
        self.objects = OrderedDict()        
        self.disappeared = OrderedDict()    
        self.counted = OrderedDict()       
        self.status_history = OrderedDict()
        self.final_status = OrderedDict()   
        self.prev_x = OrderedDict()         
        self.decided = OrderedDict()        
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

        self.total_count = 0
        self.pass_count = 0
        self.fail_count = 0

    def register(self, centroid, is_pass):
        obj_id = self.next_object_id
        self.objects[obj_id] = centroid
        self.disappeared[obj_id] = 0
        self.counted[obj_id] = False
        self.prev_x[obj_id] = centroid[0]
        self.status_history[obj_id] = [is_pass]
        self.final_status[obj_id] = "PASS" if is_pass else "FAIL"
        self.next_object_id += 1

    def deregister(self, obj_id):
        del self.objects[obj_id]
        del self.disappeared[obj_id]
        del self.counted[obj_id]
        del self.prev_x[obj_id]
        del self.status_history[obj_id]
        del self.final_status[obj_id]
        self.decided.pop(obj_id, None)

    def decide(self, obj_id):
        """Chốt PASS/FAIL (biểu quyết đa số) và cộng thống kê. Chỉ gọi một lần khi vật cắt qua vạch."""
        self.counted[obj_id] = True
        self.decided[obj_id] = self.final_status[obj_id]
        self.total_count += 1
        if self.decided[obj_id] == "PASS":
            self.pass_count += 1
        else:
            self.fail_count += 1

    def update(self, detections):
        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        input_centroids = np.array([d["centroid"] for d in detections])
        input_passes = [d["is_pass"] for d in detections]

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_passes[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array(list(self.objects.values()))

            dist_matrix = np.linalg.norm(object_centroids[:, np.newaxis] - input_centroids, axis=2)
            rows = dist_matrix.min(axis=1).argsort()
            cols = dist_matrix.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if dist_matrix[row, col] > self.max_distance:
                    continue

                obj_id = object_ids[row]
                self.prev_x[obj_id] = self.objects[obj_id][0]
                self.objects[obj_id] = input_centroids[col]
                self.disappeared[obj_id] = 0
                self.status_history[obj_id].append(input_passes[col])

        
                pass_votes = sum(self.status_history[obj_id])
                majority_pass = pass_votes >= (len(self.status_history[obj_id]) / 2.0)
                self.final_status[obj_id] = "PASS" if majority_pass else "FAIL"

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, dist_matrix.shape[0])).difference(used_rows)
            unused_cols = set(range(0, dist_matrix.shape[1])).difference(used_cols)

            for row in unused_rows:
                obj_id = object_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)

            for col in unused_cols:
                self.register(input_centroids[col], input_passes[col])

        return self.objects


# CÁC HÀM XỬ LÝ ẢNH TRUYỀN THỐNG 
def preprocess_frame(frame):
    """
    1. Lọc trung vị (Median Blur) khử nhiễu hạt.
    2. Cân bằng sáng thích nghi CLAHE trên không gian màu LAB (xử lý thay đổi độ sáng).
    3. Làm sắc nét biên Unsharp Masking với Gaussian Blur (xử lý mờ chuyển động).
    """
    denoised = cv2.medianBlur(frame, 5)

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_equalized = clahe.apply(l_channel)
    lab_merged = cv2.merge((l_equalized, a_channel, b_channel))
    illuminated_fixed = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)

    gaussian_blur = cv2.GaussianBlur(illuminated_fixed, (5, 5), 1.5)
    sharpened = cv2.addWeighted(illuminated_fixed, 1.5, gaussian_blur, -0.5, 0)

    return sharpened


def segment_products(preprocessed_frame):
    """
    Kết hợp cách tách biên sắc nét của Bản 2 + Giới hạn đúng lòng băng tải (28% - 80%)
    để không bị nhiễu răng cưa ở mép trên và mép dưới.
    """
    h_frame, w_frame = preprocessed_frame.shape[:2]
    gray = cv2.cvtColor(preprocessed_frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(preprocessed_frame, cv2.COLOR_BGR2HSV)
    _, s_channel, _ = cv2.split(hsv)

    # Cố định vùng lòng băng tải để cắt bỏ hoàn toàn lỗi nhận nhầm viền dưới (ID:10)
    belt_top = int(h_frame * 0.28)
    belt_bottom = int(h_frame * 0.80)

    roi_mask = np.zeros((h_frame, w_frame), dtype=np.uint8)
    roi_mask[belt_top:belt_bottom, :] = 255

    # Phân ngưỡng kênh Saturation + Otsu trong lòng băng tải (như Bản 2)
    _, sat_mask = cv2.threshold(s_channel, 45, 255, cv2.THRESH_BINARY)

    belt_gray_roi = gray[belt_top:belt_bottom, :]
    _, otsu_roi = cv2.threshold(belt_gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_full = np.zeros((h_frame, w_frame), dtype=np.uint8)
    otsu_full[belt_top:belt_bottom, :] = otsu_roi

    combined_mask = cv2.bitwise_or(sat_mask, otsu_full)
    combined_mask = cv2.bitwise_and(combined_mask, roi_mask)

    # Phép toán Hình thái học như Bản 2
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

    morph_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    morph_mask = cv2.morphologyEx(morph_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    return morph_mask, (belt_top, belt_bottom)


def extract_features_and_classify(contour, original_frame):
    """
    Trích xuất đặc trưng như Bản 2 (Area, WxH, Centroid, Solidity, Màu HSV)
    và bổ sung kiểm tra Hue để bắt cục màu Xanh dương thành FAIL (Color).
    """
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)

    # Tính Centroid bằng Moments
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = x + w // 2, y + h // 2

    # Tính Độ đặc (Solidity) và Tỷ lệ khung hình (Aspect Ratio)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    aspect_ratio = float(w) / h if h > 0 else 0

    # Trích xuất màu sắc trung bình bên trong Contour
    mask_roi = np.zeros(original_frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask_roi, [contour], -1, 255, -1)
    hsv_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2HSV)
    mean_h, mean_s, mean_v, _ = cv2.mean(hsv_frame, mask=mask_roi)

    # Phân loại PASS / FAIL
    fail_reasons = []
    if not (PASS_MIN_AREA <= area <= PASS_MAX_AREA):
        fail_reasons.append("Size")
    if solidity < PASS_MIN_SOLIDITY or not (PASS_MIN_ASPECT_RATIO <= aspect_ratio <= PASS_MAX_ASPECT_RATIO):
        fail_reasons.append("Shape")
    # Kiểm tra màu Cam chuẩn (Hue từ 5 đến 35) và độ bão hòa/độ sáng
    if not (PASS_HUE_MIN <= mean_h <= PASS_HUE_MAX) or mean_s < PASS_MIN_SATURATION or mean_v < PASS_MIN_VALUE:
        fail_reasons.append("Color")

    is_pass = (len(fail_reasons) == 0)

    return {
        "contour": contour,
        "bbox": (x, y, w, h),
        "centroid": (cx, cy),
        "area": area,
        "solidity": solidity,
        "aspect_ratio": aspect_ratio,
        "mean_hsv": (int(mean_h), int(mean_s), int(mean_v)),
        "is_pass": is_pass,
        "fail_reasons": fail_reasons
    }



def draw_label(img, text, org, color, scale=0.5, thick=1):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    x = max(2, min(x, img.shape[1] - tw - 2))   # không tràn ra ngoài khung
    y = max(th + 4, y)
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + base), (20, 20, 20), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def main():
    cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
    if not cap.isOpened():
        print(f"[LỖI] Không thể mở file video: {VIDEO_INPUT_PATH}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

    tracker = ProductTracker()

    counting_line_x = int(frame_width * INSPECTION_LINE_RATIO)

    # Lưu khung cuối để giữ lại trên màn hình khi video kết thúc
    last_display = None
    last_mask = None

    cv2.namedWindow("Conveyor Inspection - Result", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Binary Mask (Segmentation)", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Conveyor Inspection - Result", 30, 50)
    cv2.moveWindow("Binary Mask (Segmentation)", 30 + frame_width + 20, 50)

    print("[INFO] Đang xử lý video băng tải... Nhấn 'q' để thoát, phím cách để tạm dừng.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()

        # Bước 1 & 2: Tiền xử lý
        preprocessed = preprocess_frame(frame)

        # Bước 3 & 4: Phân đoạn sản phẩm trên vùng băng tải & Hình thái học
        mask, (belt_top, belt_bottom) = segment_products(preprocessed)

        # Bước 5: Tìm Contours & Trích xuất đặc trưng
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        current_detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if x <= 8 or (x + w) >= (frame_width - 8):
                continue

            features = extract_features_and_classify(cnt, preprocessed)
            current_detections.append(features)

        # Bước 6: Cập nhật Tracker, chỉ chốt PASS/FAIL và đếm khi tâm sản phẩm cắt qua vạch
        tracked_objects = tracker.update(current_detections)

        crossed_now = False
        for obj_id, (ocx, ocy) in tracked_objects.items():
            if tracker.counted[obj_id] or tracker.disappeared[obj_id] > 0:
                continue
            px = tracker.prev_x[obj_id]
            # Cắt vạch = tâm đổi phía so với vạch giữa 2 frame (không bỏ sót dù vật nhảy xa)
            if (px < counting_line_x <= ocx) or (ocx <= counting_line_x < px):
                tracker.decide(obj_id)
                crossed_now = True

        total_count = tracker.total_count
        pass_count = tracker.pass_count
        fail_count = tracker.fail_count

        # Vẽ giới hạn vùng băng tải (ROI) và Vạch kiểm tra dọc (Inspection Line)
        cv2.line(display_frame, (0, belt_top), (frame_width, belt_top), (180, 180, 180), 1)
        cv2.line(display_frame, (0, belt_bottom), (frame_width, belt_bottom), (180, 180, 180), 1)
        cv2.line(display_frame, (counting_line_x, belt_top), (counting_line_x, belt_bottom), (0, 255, 255), 2)
        cv2.putText(display_frame, "INSPECTION LINE", (counting_line_x - 75, belt_bottom + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        if crossed_now:
            cv2.line(display_frame, (counting_line_x, belt_top), (counting_line_x, belt_bottom), (255, 255, 255), 4)

        # Bước 7: Vẽ kết quả lên từng sản phẩm
        for det in current_detections:
            x, y, w, h = det["bbox"]
            cx, cy = det["centroid"]
            cnt = det["contour"]

            matched_id = None
            for obj_id, obj_centroid in tracked_objects.items():
                if math.hypot(cx - obj_centroid[0], cy - obj_centroid[1]) < 15:
                    matched_id = obj_id
                    break

            # Chưa cắt qua vạch: chỉ quan sát (xám). Cắt qua vạch: dùng kết quả đã chốt.
            status_str = tracker.decided.get(matched_id)
            if status_str is None:
                color = (170, 170, 170)
            else:
                color = (0, 210, 0) if status_str == "PASS" else (0, 0, 255)

            # Vẽ Contour, Bounding Box và Centroid
            cv2.drawContours(display_frame, [cnt], -1, (255, 200, 0), 2)
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            cv2.circle(display_frame, (cx, cy), 5, (0, 255, 255), -1)

            id_text = f"#{matched_id}" if matched_id is not None else "#?"
            if status_str is None:
                label_top = f"{id_text} CHECKING"
            else:
                reason_suffix = f"({','.join(det['fail_reasons'])})" if status_str == "FAIL" and det["fail_reasons"] else ""
                label_top = f"{id_text} {status_str}{reason_suffix}"
            label_bottom = f"A:{int(det['area'])} S:{det['solidity']:.2f}"

            draw_label(display_frame, label_top, (x, y - 8), color)
            draw_label(display_frame, label_bottom, (x, y + h + 18), (255, 255, 255), scale=0.4)

        # Bước 8: Vẽ bảng Dashboard thống kê
        dash_w, dash_h = 190, 92
        x0, y0 = 10, frame_height - dash_h - 10
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + dash_w, y0 + dash_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, display_frame, 0.25, 0, display_frame)

        cv2.putText(display_frame, "CONVEYOR INSPECTION", (x0 + 8, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(display_frame, f"TOTAL : {total_count}", (x0 + 8, y0 + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, f"PASS  : {pass_count}", (x0 + 8, y0 + 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 0), 1)
        cv2.putText(display_frame, f"FAIL  : {fail_count}", (x0 + 8, y0 + 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Lưu lại khung hiện tại để giữ khi video kết thúc
        last_display = display_frame.copy()
        last_mask = mask.copy()

        out_writer.write(display_frame)
        cv2.imshow("Conveyor Inspection - Result", display_frame)
        cv2.imshow("Binary Mask (Segmentation)", mask)

        key = cv2.waitKey(FRAME_DELAY_MS) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)  # Tạm dừng cho đến khi nhấn phím bất kỳ

    # Giữ khung cuối trên màn hình và trong video xuất ra
    if last_display is not None:
        for _ in range(int(fps * END_HOLD_SECONDS)):
            out_writer.write(last_display)

        cv2.imshow("Conveyor Inspection - Result", last_display)
        cv2.imshow("Binary Mask (Segmentation)", last_mask)
        print("[INFO] Video đã hết. Nhấn phím bất kỳ để đóng cửa sổ.")
        cv2.waitKey(END_WAIT_MS)

    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()
    print(f"[HOÀN TẤT] Đã lưu video kết quả tại: {VIDEO_OUTPUT_PATH}")
    pending = [i for i in tracker.objects if not tracker.counted[i]]
    if pending:
        print(f"[LƯU Ý] Video hết khi còn {len(pending)} vật chưa qua vạch (ID: {pending}), chưa được tính. "
              f"Có thể giảm INSPECTION_LINE_RATIO để dời vạch sang trái.")
    t, ps, fl = tracker.total_count, tracker.pass_count, tracker.fail_count
    print(f"Thống kê cuối cùng -> Tổng: {t} | PASS: {ps} | FAIL: {fl}")


if __name__ == "__main__":
    main()