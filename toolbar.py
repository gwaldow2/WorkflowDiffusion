
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QColorDialog, QSizePolicy
from PyQt5.QtCore import Qt

class Toolbar(QWidget):
    def __init__(self, canvas, parent=None):
        super().__init__(parent)

        self.canvas = canvas
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignTop)
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self.canvas.clear_canvas)
        self.layout.addWidget(clear_button)
        color_button = QPushButton("Select Color")
        color_button.clicked.connect(self.open_color_dialog)
        self.layout.addWidget(color_button)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layout.addWidget(spacer)

        self.setLayout(self.layout) 

    def open_color_dialog(self):
        color = QColorDialog.getColor(self.canvas.pen_color, self, "Select Color")
        if color.isValid():
            self.canvas.pen_color = color
