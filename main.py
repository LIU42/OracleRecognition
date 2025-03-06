from PyQt5.QtWidgets import QApplication

from views.mainwindow import AppMainWindow


def main():
    app = QApplication([])

    center_x = app.primaryScreen().availableGeometry().center().x()
    center_y = app.primaryScreen().availableGeometry().center().y()

    window = AppMainWindow()
    window.move(center_x - window.width() // 2, center_y - window.height() // 2)
    window.show()

    return app.exec()


if __name__ == '__main__':
    main()
