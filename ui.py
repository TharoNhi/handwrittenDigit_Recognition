from PyQt5.QtWidgets import QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt


class AutoLabelingApp(QWidget):
    def __init__(self, select_image_callback):
        super().__init__()
        self.setWindowTitle("Auto Labeling with YOLOv8, MNIST, and OCR")
        self.setGeometry(100, 100, 800, 600)

        # Layout chính
        main_layout = QVBoxLayout()

        # Nút ESC để thoát
        self.esc_button = QPushButton("ESC")
        self.esc_button.setFixedHeight(30)
        self.esc_button.clicked.connect(self.close)  # Gắn chức năng thoát ứng dụng
        main_layout.addWidget(self.esc_button, alignment=Qt.AlignLeft)

        # Label hiển thị ảnh
        self.image_label = QLabel("Vui lòng chọn ảnh")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid black;")
        main_layout.addWidget(self.image_label, stretch=1)

        # Layout cho ngày/tháng/năm
        date_layout = QHBoxLayout()
        self.day_label = QLabel("Ngày:")
        self.month_label = QLabel("Tháng:")
        self.year_label = QLabel("Năm:")

        for label in [self.day_label, self.month_label, self.year_label]:
            label.setStyleSheet("font-size: 16px;")
            date_layout.addWidget(label)

        main_layout.addLayout(date_layout)

        # Nút chọn ảnh
        self.select_button = QPushButton("Chọn Ảnh")
        self.select_button.setFixedHeight(50)
        self.select_button.clicked.connect(select_image_callback)  # Gắn callback xử lý
        main_layout.addWidget(self.select_button)

        self.setLayout(main_layout)

    def update_image(self, image_path):
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(pixmap.scaled(500, 500, Qt.KeepAspectRatio))

    def update_date_labels(self, day, month, year):
        self.day_label.setText(f"Ngày: {day}")
        self.month_label.setText(f"Tháng: {month}")
        self.year_label.setText(f"Năm: {year}")
