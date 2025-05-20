import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Dictionary untuk mapping indeks kelas ke nama kelas
dic = {0: 'dyed-lifted-polyps', 1: 'dyed-resection-margins', 2: 'esophagitis', 3: 'normal-cecum', 
       4: 'normal-pylorus', 5: 'normal-z-line', 6: 'polyps', 7: 'ulcerative-colitis'}

# Muat model
model = load_model('hasil_ensemble.keras')

# Path ke gambar yang ingin diuji
img_path = 'static/polyps.jpg'  # Ganti dengan path gambar kamu

# Muat dan preprocess gambar
img = image.load_img(img_path, target_size=(416, 416))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
# Hapus baris ini untuk menonaktifkan normalisasi:
# img = img / 255.0  # Normalisasi

# Prediksi
predictions = model.predict(img)
predicted_class = np.argmax(predictions, axis=1)[0]

# Tampilkan hasil prediksi
print("Probabilitas prediksi:", predictions)
print("Kelas prediksi:", dic[predicted_class])