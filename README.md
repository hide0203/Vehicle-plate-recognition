Certainly! Here is your final, professional `README.md` tailored for your completed Vehicle License Plate Recognition System as demonstrated by your success screenshot:

# Vehicle License Plate Recognition System

**Last updated:** July 20, 2025

**Author:** [Your Name]  
**Project Type:** BCA Major Project  
**Function:** Real-time detection and recognition of vehicle number plates using computer vision and OCR

## 📸 Live Demo

![Recent Detections Screenshot](https://pplx-res.cloudinary.com/image/private/user_uploads/81445994/76afcfd6-6d72-4952-99cb 🚗 Features

- **Live Camera Feed:** Real-time plate detection using your webcam or connected cameras
- **Automatic License Plate Recognition:** Optical Character Recognition (OCR) for extracting plate numbers
- **Recent Detections Panel:** Shows most recent successful readings with confidence scores
- **Database Logging:** Stores number plates, confidence, and timestamps (SQLite)
- **Responsive Web UI:** Control the system and view results from any device in your LAN
- **Configurable & Extendable:** Easily adapt settings or add new features

## ⚙️ Requirements

- **Python 3.8+** (Tested on Python 3.10)
- **Pip packages** (see requirements.txt):
    - `opencv-python`
    - `flask`
    - `pytesseract`
    - `numpy`
    - `pillow`
    - `imutils`
    - `requests`
- **Tesseract OCR** [Download](https://github.com/tesseract-ocr/tesseract)
- **Haar Cascade File:**  
  Download `haarcascade_russian_plate_number.xml` from [OpenCV GitHub](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_russian_plate_number.xml) and place it in `data/haarcascades/`

## 🚀 Quick Start

### 1. Clone or Download the Repository

```bash
git clone https://github.com/yourusername/vehicle-plate-recognition.git
cd vehicle-plate-recognition
```

### 2. Set Up Python Environment

```bash
python -m venv venv
venv\Scripts\activate      # (Windows)
# or
source venv/bin/activate   # (Linux/Mac)
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR

- **Windows:** Download and install from [here](https://github.com/tesseract-ocr/tesseract).
- Set the correct path in `app/config.py`, e.g.:
  ```python
  TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
  ```
- **Linux:** `sudo apt-get install tesseract-ocr`

### 5. (First Time Only) Download Haar Cascade

```bash
curl -o data/haarcascades/haarcascade_russian_plate_number.xml https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_russian_plate_number.xml
```
Or [download manually](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_russian_plate_number.xml) and save to `data/haarcascades/`

### 6. Run the Application

```bash
python run.py
```
Access via your browser at [http://localhost:5000](http://localhost:5000)  
Or on your LAN at `http://:5000`

## 🖥️ Usage

- **Start Camera:** Use the interface to begin live recognition. Detected plates appear in "Recent Detections" along with confidence.
- **Stop Camera:** Ends live processing.
- **Recent Detections:** Shows last recognized plates, timestamps, and confidence scores.
- **Admin/Debug:** Watch terminal logs for detailed OCR, detection, and system debug info.

## 📂 File Structure

```plaintext
vehicle-plate-recognition/
├── app/
│   ├── config.py
│   ├── core/
│   ├── models/
│   ├── web/
│   ├── static/
│   └── ...
├── data/
│   ├── haarcascades/
│   ├── captured_plates/
├── logs/
├── run.py
├── requirements.txt
└── README.md
```

## 🧑‍💻 How It Works

1. **Video frames** are captured from your camera.
2. **Haarcascade detector** locates possible license plate regions.
3. **ROI preprocessesed:** Cropped, upscaled, contrast-adjusted, and thresholded.
4. **Tesseract OCR** attempts to extract plate number text.
5. **Post-processing:** Only plausible numbers (with confidence) are recorded and shown.

## 💡 Tips for Best Results

- **Good Lighting:** Use daylight or strong white light to reduce glare/shadow.
- **High-Resolution Plates:** Present plates directly facing camera, large in frame, and in focus.
- **Test with real/classic printed plates:** Even a black-on-white printout in Arial Bold works great for demo.
- **Tune Parameters:** Adjust ROI size, minimum/maximum area, whitelist characters, or preprocessing for your use case.

## 🚧 Troubleshooting

- **Blank or junk OCR results:** Check ROI clarity, light, focus, and preprocessing. Inspect `captured_plates/` for debug images.
- **No detections:** Check that the cascade file path is correct, frame is well-lit, and the camera is functioning.
- **Tesseract issues:** Ensure `TESSERACT_CMD` is set and Tesseract runs from the command line.
- **Multiple/False Plates:** Haarcascade is simple; for best performance, consider using a deep learning detector.

## 📜 License

This project is for educational purposes (BCA Major Project).  
For commercial or production use, adapt preprocessing and detection to your application and region.

**Built with ❤️ using Python, OpenCV, Flask, and Tesseract-OCR.**

*For questions, improvements, or feedback, [contact the project maintainer](mailto:youremail@domain.com) or [open an issue](https://github.com/yourusername/vehicle-plate-recognition/issues).*