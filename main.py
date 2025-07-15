import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import pytesseract
import os
from utils import process_image, process_video

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class PlateRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Number Plate Recognition")
        self.root.geometry("800x600")

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        btn_frame = tk.Frame(root)
        btn_frame.pack()

        img_btn = tk.Button(btn_frame, text="Upload Image", command=self.load_image)
        img_btn.grid(row=0, column=0, padx=10)

        vid_btn = tk.Button(btn_frame, text="Upload Video", command=self.load_video)
        vid_btn.grid(row=0, column=1, padx=10)

        self.result_text = tk.Text(root, height=10, width=100)
        self.result_text.pack(pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            image = cv2.imread(file_path)
            plate_texts, output_image = process_image(image)

            img = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)

            self.image_label.config(image=img)
            self.image_label.image = img

            self.result_text.delete("1.0", tk.END)
            for text in plate_texts:
                self.result_text.insert(tk.END, text + "\n")

    def load_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
        if file_path:
            detected_plates = process_video(file_path)
            self.result_text.delete("1.0", tk.END)
            for plate in detected_plates:
                self.result_text.insert(tk.END, plate + "\n")

if __name__ == '__main__':
    root = tk.Tk()
    app = PlateRecognitionApp(root)
    root.mainloop()
