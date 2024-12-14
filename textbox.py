# textbox.py
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QTextEdit, QLabel
)
from PyQt5.QtCore import Qt

class ChatBoxWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Last entered prompt
        self.last_prompt = None

        # Main layout
        self.layout = QVBoxLayout()

        # Chat area
        self.chat_area = QTextEdit()
        self.chat_area.setReadOnly(True)
        self.chat_area.setStyleSheet("background-color: #f4f4f4;")
        self.layout.addWidget(QLabel("Prompt Box:"))
        self.layout.addWidget(self.chat_area)

        # Input area
        self.input_layout = QHBoxLayout()
        self.text_box = QLineEdit()
        self.text_box.setPlaceholderText("Enter prompt...")
        self.draw_button = QPushButton("Draw")

        self.input_layout.addWidget(self.text_box)
        self.input_layout.addWidget(self.draw_button)

        self.layout.addLayout(self.input_layout)
        self.setLayout(self.layout)

    def get_prompt(self):
        return self.text_box.text().strip()

    def append_to_chat(self, message):
        self.chat_area.append(message)
