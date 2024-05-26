import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLOv10
import supervision as sv
import numpy as np

# Inisialisasi model dengan bobot pre-trained YOLOv10
model = YOLOv10('weights/yolov10n.pt')

# Melakukan deteksi objek pada gambar dengan confidence threshold 0.25
results = model(source='images/lomba.jpg', conf=0.25)

# Membaca gambar asli
image = cv2.imread('images/lomba.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Mengubah dari BGR (OpenCV default) ke RGB

# Mengambil hasil deteksi bounding box, label, dan confidence
boxes = results[0].boxes.xyxy.cpu().numpy()  # Memastikan dalam bentuk numpy array
labels = results[0].boxes.cls.cpu().numpy()
confidences = results[0].boxes.conf.cpu().numpy()

# Mendefinisikan daftar nama kelas
category_dict = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

# Membuat pemetaan warna unik untuk setiap kelas
num_classes = len(category_dict)
colors = plt.cm.get_cmap('hsv', num_classes)

# Inisialisasi BoxAnnotator dari supervision
box_annotator = sv.BoxAnnotator()

# Siapkan list untuk bounding boxes, labels, dan confidences
annotations = []

# Iterasi melalui hasil deteksi dan menambahkan ke list
for box, label, confidence in zip(boxes, labels, confidences):
    x1, y1, x2, y2 = box.astype(int)
    label_name = category_dict[int(label)]
    confidence_text = f"{label_name} {confidence:.2f}"
    
    # Memperoleh warna unik untuk kelas
    class_color = colors(int(label) / num_classes)[:3]  # Ambil komponen RGB, buang alpha
    class_color = [int(c * 255) for c in class_color]  # Konversi ke rentang 0-255
    
    annotations.append({
        "box": (x1, y1, x2, y2),
        "label": confidence_text,
        "color": class_color
    })

# Menggambar bounding box dan label pada gambar menggunakan BoxAnnotator
for annotation in annotations:
    x1, y1, x2, y2 = annotation["box"]
    confidence_text = annotation["label"]
    color = annotation["color"]
    
    # Menggambar bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    # Menghitung ukuran teks
    font_scale = 1.0  # Ukuran huruf lebih besar
    thickness = 2  # Ketebalan huruf lebih besar
    (text_width, text_height), baseline = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    
    # Menggambar latar belakang kotak teks
    cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
    
    # Menambahkan teks label dengan warna putih
    cv2.putText(image, confidence_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

# Menampilkan gambar dengan bounding box dan label
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('off')  # Menghilangkan axis
plt.show()
