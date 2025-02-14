import sys
from PyQt5.QtWidgets import QApplication

from views.mainwindow import MainWindow


if __name__ == '__main__':
    app = QApplication(sys.argv)

    center_x = app.primaryScreen().availableGeometry().center().x()
    center_y = app.primaryScreen().availableGeometry().center().y()

    window = MainWindow()
    window.move(center_x - window.width() // 2, center_y - window.height() // 2)
    window.show()

    sys.exit(app.exec())
