# canvas.py
import numpy as np
from PyQt5.QtWidgets import QWidget, QShortcut
from PyQt5.QtGui import QPainter, QPen, QPixmap, QMouseEvent, QImage, QKeySequence
from PyQt5.QtCore import Qt, QPoint
import cv2

class Canvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: white;")  
        self.setFixedSize(512, 512)  
        self.generated_layer = QPixmap(self.size())
        self.drawing_layer = QPixmap(self.size())
        self.generated_layer.fill(Qt.white)  
        self.drawing_layer.fill(Qt.transparent)  

        self.last_point = QPoint()
        self.drawing = False
        self.pen_color = Qt.black  
        self.pen_width = 3  

        self.strokes = [] 
        self.current_stroke = []
        # doesnt work :/
        undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        undo_shortcut.activated.connect(self.undo)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()
            self.current_stroke = [self.last_point]

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() & Qt.LeftButton and self.drawing:
            painter = QPainter(self.drawing_layer)
            pen = QPen(self.pen_color, self.pen_width, Qt.SolidLine)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            painter.end()
            self.current_stroke.append(self.last_point) 
            self.update()  # trigger a repaint

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            if self.current_stroke:
                self.strokes.append(self.current_stroke) 
                self.current_stroke = []

    def paintEvent(self, event):
        """
        Handles painting the widget by combining the generated image and the drawing layer.
        """
        canvas_painter = QPainter(self)

        # half opacity
        if not self.generated_layer.isNull():
            canvas_painter.setOpacity(0.5)
            canvas_painter.drawPixmap(0, 0, self.generated_layer)

        # full opacity
        canvas_painter.setOpacity(1.0)
        canvas_painter.drawPixmap(0, 0, self.drawing_layer)

    def clear_canvas(self):
        """
        Clears the drawing layer (user-drawn content) and resets the history.
        """
        self.drawing_layer.fill(Qt.transparent)
        self.strokes.clear()
        self.update()

    def convert_pixmap_to_numpy(self):
        """
        Converts the current state of the drawing layer to a NumPy array.
        """
        # Convert QPixmap to QImage with RGBA format
        image = self.drawing_layer.toImage().convertToFormat(QImage.Format_RGBA8888)
        width = image.width()
        height = image.height()
        ptr = image.bits()

        ptr.setsize(image.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)  # RGBA

        # Replace transparent pixels with white
        alpha = arr[:, :, 3].astype(float) / 255.0
        arr[:, :, :3] = arr[:, :, :3] * alpha[:, :, np.newaxis] + 255 * (1 - alpha[:, :, np.newaxis])
        arr = arr[:, :, :3].astype(np.uint8)  # Remove alpha channel

        return arr

    def update_image(self, image):
        # Assume image is a numpy array
        image = np.uint8(image)
        image = cv2.resize(image, (self.width(), self.height()))
        height, width, channel = image.shape
        qimage = QImage(image.data, width, height, channel * width, QImage.Format_RGB888)
        self.generated_layer = QPixmap.fromImage(qimage)
        self.update()

    def undo(self):
        """
        Removes the last stroke and redraws the drawing layer.
        """
        if self.strokes:
            self.strokes.pop()  
            self.redraw_strokes()
            self.update()

    def redraw_strokes(self):
        """
        Clears the drawing layer and redraws all strokes from the history.
        """
        self.drawing_layer.fill(Qt.transparent)
        painter = QPainter(self.drawing_layer)
        pen = QPen(self.pen_color, self.pen_width, Qt.SolidLine)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        for stroke in self.strokes:
            if len(stroke) < 2:
                continue
            for i in range(1, len(stroke )):
                painter.drawLine(stroke[i - 1], stroke[i])

        painter.end()
