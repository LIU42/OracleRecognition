import cv2
import yaml

from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap

from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QAction
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QGraphicsScene
from PyQt5.QtWidgets import QGraphicsView

import inferences.inference as inference


with open('configs/languages.yaml', 'r', encoding='utf-8') as languages:
    languages = yaml.load(languages, Loader=yaml.FullLoader)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.source_image = None
        self.output_image = None
        self.output_scene = QGraphicsScene()

        layout = QHBoxLayout()
        layout.addWidget(QGraphicsView(self.output_scene))

        central_widget = QWidget()
        central_widget.setLayout(layout)

        self.setWindowTitle(languages['main-title'])
        self.setMinimumSize(680, 706)
        self.setCentralWidget(central_widget)

        self.open_action = QAction(languages['open-action'])
        self.save_action = QAction(languages['save-action'])
        self.exit_action = QAction(languages['exit-action'])

        self.open_action.setShortcut('Ctrl+O')
        self.save_action.setShortcut('Ctrl+S')

        self.open_action.setEnabled(True)
        self.save_action.setEnabled(False)

        self.open_action.triggered.connect(self.inference)
        self.save_action.triggered.connect(self.save)
        self.exit_action.triggered.connect(self.close)

        menuber = self.menuBar()

        file_menu = menuber.addMenu(languages['file-menu'])
        help_menu = menuber.addMenu(languages['help-menu'])

        file_menu.addAction(self.open_action)
        file_menu.addAction(self.save_action)
        help_menu.addAction(self.exit_action)

    @property
    def converted_output_pixmap(self):
        data = self.output_image.data

        w = self.output_image.shape[1]
        h = self.output_image.shape[0]

        return QPixmap.fromImage(QImage(data, w, h, QImage.Format_RGB888))

    def inference(self):
        selected_path, _ = QFileDialog.getOpenFileName(caption=languages['open-title'], filter=languages['types-description'])

        if selected_path.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            self.open_action.setEnabled(False)
            self.save_action.setEnabled(False)

            self.source_image = cv2.imread(selected_path)
            self.output_image = inference.inference(self.source_image)

            self.output_scene.clear()
            self.output_scene.addPixmap(self.converted_output_pixmap)

            self.open_action.setEnabled(True)
            self.save_action.setEnabled(True)

    def save(self):
        selected_path, _ = QFileDialog.getSaveFileName(caption=languages['save-title'], filter=languages['types-description'])

        if selected_path.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            self.open_action.setEnabled(False)
            self.save_action.setEnabled(False)

            cv2.imwrite(selected_path, self.output_image)

            self.open_action.setEnabled(True)
            self.save_action.setEnabled(True)
