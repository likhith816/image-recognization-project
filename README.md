# image-recognization-project
# Face Recognition System using LBPH (OpenCV)

This project implements a **Face Recognition System** using the **Local Binary Patterns Histograms (LBPH)** algorithm provided by OpenCV's `face` module. It allows you to train a model with labeled facial images and later recognize individuals based on that model.

---

## 🚀 Features

- Grayscale image preprocessing using Pillow
- Face ID extraction from filenames
- Training the LBPH model using OpenCV's `cv2.face` module
- Saving the trained model as `classifier.xml`
- Modular code for easy extension (e.g., real-time recognition)

---

## 🧠 Requirements

Install the following Python packages:

```bash
pip install numpy pillow opencv-contrib-python
```

> ⚠️ Make sure you **install `opencv-contrib-python`**, not just `opencv-python`, to access the `cv2.face` module.

If you face an `AttributeError` like:
```
AttributeError: module 'cv2' has no attribute 'face'
```
Uninstall and reinstall with:
```bash
pip uninstall opencv-python opencv-contrib-python
pip install opencv-contrib-python==4.5.1.48
```

---

## 🗂️ Folder Structure

```
your-project/
│
├── data/                    # Folder containing training images
│   ├── User.1.jpg
│   ├── User.1.1.jpg
│   ├── User.2.jpg
│   └── ...
│
├── train.py                 # Training script
├── classifier.xml           # Output: trained model
└── README.md                # This file
```

---

## 🖼️ Image Format

Your training images must follow this naming pattern:
```
User.<ID>.<any>.jpg
```
For example:
- `User.1.1.jpg` (ID = 1)
- `User.2.3.jpg` (ID = 2)

---

## 🧪 How to Train

```bash
python train.py
```

This will read images from the `data/` folder, extract the IDs, train the LBPH face recognizer, and save the model as `classifier.xml`.

---

## 🛠️ Training Code (train.py)

```python
import os
import cv2
from PIL import Image
import numpy as np

def train_classifier(data_dir):
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    
    faces = []
    ids = []
    
    for image in path:
        try:
            img = Image.open(image).convert('L')
            imageNp = np.array(img, 'uint8')
            id = int(os.path.split(image)[1].split(".")[1])
            
            faces.append(imageNp)
            ids.append(id)
        except Exception as e:
            print(f"Skipping {image}: {e}")
    
    ids = np.array(ids)
    
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")
    print("Training complete.")

train_classifier("data")
```

---

## 📌 Notes

- This project focuses on training. Real-time recognition (via webcam) can be added using OpenCV’s `VideoCapture` and `clf.predict()` on detected faces.
- This script does not perform face detection — it assumes cropped face images are already available in the `data/` folder.

---


