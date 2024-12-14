# main.py
import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QWidget, QPushButton, QVBoxLayout
from PyQt5.QtCore import QThread, pyqtSignal
from canvas import Canvas
from toolbar import Toolbar
from textbox import ChatBoxWidget
from model_binding import prompt_model
import numpy as np
from PIL import Image


class ImageGenerationThread(QThread):
    """
    QThread subclass to handle image generation without freezing the UI.
    """
    generated = pyqtSignal(tuple)  
    error = pyqtSignal(str)         

    def __init__(self, text_prompt, sketch_path, output_path):
        super().__init__()
        self.text_prompt = text_prompt
        self.sketch_path = sketch_path
        self.output_path = output_path

    def run(self):
        try:
            # Perform image generation
            generated_image = prompt_model(self.text_prompt, self.sketch_path)
            # Emit the generated image along with the desired output path
            self.generated.emit((generated_image, self.output_path))
        except Exception as e:
            # Emit the error message
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ControlNet Diffusion Model Integration")
        self.setGeometry(100, 100, 1200, 800)
        main_layout = QHBoxLayout()
        self.canvas = Canvas()
        self.toolbar = Toolbar(self.canvas)
        self.chatbox_widget = ChatBoxWidget()
        self.chatbox_widget.setFixedWidth(300) 
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)
        main_layout.addWidget(self.chatbox_widget)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        self.thread = None

        # connect ChatBox draw button to the im
        self.chatbox_widget.draw_button.clicked.connect(self.generate_image)

        # test output button
        self.add_test_button()

    def add_test_button(self):
        test_button = QPushButton("Generate from Test Image")
        test_button.clicked.connect(self.generate_test_image)
        self.toolbar.layout.insertWidget(2, test_button) 

    def generate_image(self):
        text_prompt = self.chatbox_widget.get_prompt()

        # reuse last prompt if needed
        if not text_prompt and not self.chatbox_widget.last_prompt:
            self.chatbox_widget.append_to_chat("Error: Please enter a prompt first!")
        if text_prompt:
            self.chatbox_widget.last_prompt = text_prompt
            self.chatbox_widget.append_to_chat(f"Prompt: {text_prompt}")
            self.chatbox_widget.text_box.clear()

        self.chatbox_widget.append_to_chat("Generating image...")

        
        sketch_path = "temp_sketch.png" 
        sketch_array = self.canvas.convert_pixmap_to_numpy()
        sketch_image = Image.fromarray(sketch_array)
        sketch_image = sketch_image.resize((512, 512)) 
        sketch_image.save(sketch_path)

        
        sketch_image.save("sketch_output.png")

        
        output_path = "generated_image.png"

        # gen
        self.thread = ImageGenerationThread(self.chatbox_widget.last_prompt, sketch_path, output_path)
        self.thread.generated.connect(self.display_generated_image)
        self.thread.error.connect(self.handle_generation_error)
        self.thread.start()

    def generate_test_image(self):
        """
        Generates an image from 'test.png' with the prompt and saves the result as 'testoutput.png'.
        """
        prompt = self.chatbox_widget.get_prompt()
        test_image_path = "test.png"  
        output_path = "testoutput.png"

        if not os.path.exists(test_image_path):
            self.chatbox_widget.append_to_chat(f"Error: '{test_image_path}' not found!")
            return

        self.chatbox_widget.append_to_chat("Generating image from 'test.png'...")

        
        sketch_image = Image.open(test_image_path).convert("RGB")
        sketch_image = sketch_image.resize((512, 512))  # Resize to match the model's input size
        sketch_image.save("sketch_output.png")

        
        self.thread = ImageGenerationThread(prompt, "sketch_output.png", output_path)
        self.thread.generated.connect(self.display_test_generated_image)
        self.thread.error.connect(self.handle_generation_error)
        self.thread.start()

    def display_generated_image(self, data):
        generated_image, output_path = data
        self.canvas.update_image(generated_image)
        self.chatbox_widget.append_to_chat("Image generation completed!")

        #Cleanup temporary sketch file
        if os.path.exists("temp_sketch.png"):
            os.remove("temp_sketch.png")

    def display_test_generated_image(self, data):
        generated_image, output_path = data
        # Save generated image as 'testoutput.png'
        Image.fromarray(generated_image).save(output_path)
        self.chatbox_widget.append_to_chat(f"Test image generated and saved as '{output_path}'.")

    def handle_generation_error(self, error_message):
        self.chatbox_widget.append_to_chat(f"Error during image generation: {error_message}")
        if os.path.exists("temp_sketch.png"):
            os.remove("temp_sketch.png")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
