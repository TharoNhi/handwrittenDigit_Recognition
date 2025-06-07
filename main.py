import sys
from PyQt5.QtWidgets import QApplication
from ui import AutoLabelingApp
from logic import select_image


def main():
    app = QApplication(sys.argv)

    # Khởi tạo giao diện và gắn callback
    window = AutoLabelingApp(select_image_callback=lambda: select_image(window.update_image, window.update_date_labels))
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
