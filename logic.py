import os
import cv2 as cv
import shutil
import numpy as np
import pytesseract
from keras import models
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap
from ultralytics import YOLO

# Đường dẫn tới Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load YOLO model và MNIST model
model = YOLO('best.pt')
mnist_model = models.load_model('mnist_keras_model.h5')

def split_digits(binary_img, debug_dir="./debug/", filename_prefix="digit_split"):
    os.makedirs(debug_dir, exist_ok=True)

    # Tìm contour để tách các chữ số
    contours, _ = cv.findContours(binary_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    digit_images = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        padding = 2
        if w > 5 and h > 10:  # Loại bỏ các vùng nhiễu quá nhỏ
            padding = 2
            x = max(0, x - padding)
            y = max(0, y - padding)
            w += 2 * padding
            h += 2 * padding
            digit_img = binary_img[y:y + h, x:x + w]
            digit_images.append((x, digit_img))
    digit_images = sorted(digit_images, key=lambda x: x[0])  # Sắp xếp từ trái sang phải

    # Lưu từng chữ số để kiểm tra
    for i, (_, digit_img) in enumerate(digit_images):
        digit_path = os.path.join(debug_dir, f"{filename_prefix}_{i + 1}.jpg")
        cv.imwrite(digit_path, digit_img)

    return [digit_img for _, digit_img in digit_images]

def process_and_resize_image(cropped_img, debug_dir="./debug/", filename_prefix="digit_processed"):
    os.makedirs(debug_dir, exist_ok=True)

    # Chuyển sang grayscale và binary
    gray_img = cv.cvtColor(cropped_img, cv.COLOR_BGR2GRAY)
    _, binary_img = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # Tách chữ số
    digits = split_digits(binary_img, debug_dir, filename_prefix)

    # Resize từng chữ số về 28x28
    resized_digits = []
    for i, digit_img in enumerate(digits):
        resized_digit = cv.resize(digit_img, (28, 28), interpolation=cv.INTER_AREA)
        resized_digits.append(resized_digit)

        # Lưu từng chữ số đã resize
        debug_path = os.path.join(debug_dir, f"{filename_prefix}_resized_{i + 1}.jpg")
        cv.imwrite(debug_path, resized_digit)

    return resized_digits

def recognize_digits_from_images(digit_images):
    recognized_digits = []
    for digit_img in digit_images:
        # Chuẩn hóa ảnh đầu vào
        normalized_img = digit_img.astype("float32") / 255.0
        normalized_img = np.expand_dims(normalized_img, axis=-1)  # Thêm kênh màu
        normalized_img = np.expand_dims(normalized_img, axis=0)   # Thêm batch size

        # Dự đoán chữ số
        prediction = mnist_model.predict(normalized_img)
        predicted_digit = np.argmax(prediction, axis=1)[0]
        recognized_digits.append(str(predicted_digit))

    return "".join(recognized_digits)

def recognize_digit_from_box(img, box, debug_dir="./debug/", filename_prefix="digit"):
    os.makedirs(debug_dir, exist_ok=True)
    x1, y1, x2, y2 = box
    cropped_img = img[y1:y2, x1:x2]

    # Xử lý ảnh và tách chữ số
    digit_images = process_and_resize_image(cropped_img, debug_dir, filename_prefix)

    # Nhận diện từng chữ số
    return recognize_digits_from_images(digit_images)

def recognize_text_from_box(img, box):
    x1, y1, x2, y2 = box
    cropped_img = img[y1:y2, x1:x2]
    gray_img = cv.cvtColor(cropped_img, cv.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray_img, config='--psm 8 digits')
    return text.strip()

def select_image(label_update_callback, date_update_callback):
    file_path, _ = QFileDialog.getOpenFileName(None, "Chọn ảnh", "", "Image Files (*.jpg *.jpeg *.png)")
    if file_path:
        val_folder = "./images/val/"
        result_folder = "./result/"
        debug_dir = "./debug/"  # Thư mục debug

        os.makedirs(val_folder, exist_ok=True)
        os.makedirs(result_folder, exist_ok=True)
        os.makedirs(debug_dir, exist_ok=True)

        new_image_name = f"FileTest_{len(os.listdir(val_folder))}.jpg"
        new_image_path = os.path.join(val_folder, new_image_name)
        shutil.copy(file_path, new_image_path)

        # Đọc ảnh gốc và tạo bản sao
        img = cv.imread(new_image_path)  # Ảnh gốc để xử lý
        show_img = img.copy()  # Ảnh để vẽ bounding box

        # Nhận diện bounding box với YOLO
        result = model(new_image_path, conf=0.6)
        box_ngay, box_thang, box_nam = None, None, None

        for r in result:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = box.conf[0].item()

                if conf > 0.6:
                    if cls == 0:
                        box_ngay = (x1, y1, x2, y2)
                    elif cls == 1:
                        box_thang = (x1, y1, x2, y2)
                    elif cls == 2:
                        box_nam = (x1, y1, x2, y2)

        if box_ngay and box_nam:
            day_value, month_value, year_value = "", "", ""

            if box_ngay:
                day_value_box = (box_ngay[2], box_ngay[1], box_thang[0] if box_thang else box_nam[0], box_ngay[3])
                cv.rectangle(show_img, (day_value_box[0], day_value_box[1]),
                            (day_value_box[2], day_value_box[3]), (128, 0, 128), 2)
                day_value = recognize_digit_from_box(img, day_value_box, debug_dir, 'ngay')

                # Kiểm tra nếu day_value rỗng
                if not day_value or not day_value.isdigit() or int(day_value) > 31:
                    label_update_callback(new_image_path)
                    date_update_callback("Không hợp lệ", "", "")
                    return

            if box_thang:
                month_value_box = (box_thang[2], box_thang[1], box_nam[0] if box_nam else img.shape[1], box_thang[3])
                cv.rectangle(show_img, (month_value_box[0], month_value_box[1]),
                            (month_value_box[2], month_value_box[3]), (0, 255, 0), 2)
                month_value = recognize_digit_from_box(img, month_value_box, debug_dir, 'thang')

                # Kiểm tra nếu month_value rỗng
                if not month_value or not month_value.isdigit() or int(month_value) > 12:
                    label_update_callback(new_image_path)
                    date_update_callback("", "Không hợp lệ", "")
                    return

            if box_nam:
                year_value_box = (box_nam[2], box_nam[1], img.shape[1], box_nam[3])
                cv.rectangle(show_img, (year_value_box[0], year_value_box[1]),
                            (year_value_box[2], year_value_box[3]), (0, 165, 255), 2)
                year_value = recognize_text_from_box(img, year_value_box)


            # Cập nhật kết quả ngày/tháng/năm lên giao diện
            date_update_callback(day_value, month_value, year_value)

        result_image_path = os.path.join(result_folder, new_image_name)
        cv.imwrite(result_image_path, show_img)

        # Cập nhật ảnh kết quả lên giao diện
        label_update_callback(result_image_path)
