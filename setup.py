from setuptools import setup, find_packages

setup(
    name="vehicle-plate-recognition",
    version="1.0.0",
    description="Real-time vehicle license plate recognition system",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "opencv-python==4.8.1.78",
        "pytesseract==0.3.10",
        "numpy==1.24.3",
        "pillow==10.0.1",
        "flask==2.3.3",
        "ultralytics==8.0.196",
        "imutils==0.5.4",
        "flask-socketio==5.3.6",
        "requests==2.31.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "plate-recognition=run:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
