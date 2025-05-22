from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import csv
import io
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'gastrointestinal_analysis_key'  # Secret key for flashing messages

# Dictionary mapping indeks kelas ke nama kelas
dic = {
    0: 'dyed-lifted-polyps', 1: 'dyed-resection-margins', 2: 'esophagitis',
    3: 'normal-cecum', 4: 'normal-pylorus', 5: 'normal-z-line',
    6: 'polyps', 7: 'ulcerative-colitis'
}

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Muat semua model yang ingin disediakan
try:
    model_resnet = load_model('modelResnet50Epoch.keras')
    model_inception = load_model('modelResnet50Epoch.keras')
    model_averageEnsemble = load_model('hasil_ensembleStacking.keras')
    model_stackingEnsemble = load_model('hasil_ensembleStacking.keras')

    # Jika diperlukan, aktifkan predict_function (tergantung versi TF/Keras)
    model_resnet.make_predict_function()
    model_inception.make_predict_function()
    model_averageEnsemble.make_predict_function()
    model_stackingEnsemble.make_predict_function()
except Exception as e:
    print(f"Error loading models: {e}")
    # We'll handle this gracefully when prediction is requested

# Fungsi untuk memprediksi label gambar
def predict_label(img_path, model, target_size):
    try:
        # Sesuaikan target_size dengan kebutuhan masing-masing model
        img = image.load_img(img_path, target_size=target_size)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        # (Jika perlu, lakukan normalisasi sesuai preprocessing yang digunakan saat training)
        p = model.predict(img)
        
        # Tentukan threshold confidence score
        threshold = 0.8     # Ambil kelas dengan confidence score tertinggi

        confidence = np.max(p)
        if confidence < threshold:
            return "Kelas tidak ditemukan", confidence

        predicted_class = np.argmax(p, axis=1)[0]
        return dic[predicted_class], confidence
    except Exception as e:
        print(f"Error predicting image: {e}")
        return "Error", 0

# Route untuk halaman utama
@app.route("/", methods=['GET'])
def main():
    return render_template("classification.html")

# Route untuk memproses gambar yang diupload dan pilihan model
@app.route("/submit", methods=['POST'])
def get_output():
    # Initialize with error
    error_msg = None
    
    # Check if the post request has the file part
    if 'my_image' not in request.files:
        return render_template("classification.html", error="Tidak ada file yang dipilih")
    
    img = request.files['my_image']
    
    # If user does not select a file, browser submits an empty file
    if img.filename == '':
        return render_template("classification.html", error="Tidak ada file yang dipilih")
    
    # Check if file extension is allowed
    if not allowed_file(img.filename):
        return render_template("classification.html", 
                              error="Format file tidak didukung. Hanya file gambar (PNG, JPG, JPEG, GIF, BMP, TIFF) yang diizinkan.")
    
    try:
        # Ambil pilihan model dari form
        model_choice = request.form.get('model_choice')
        export_csv = request.form.get('export_csv') == 'on'  # Check if CSV export is requested

        # Tentukan path dan simpan gambar
        img_path = "static/" + img.filename 
        img.save(img_path)

        # Pilih model dan target size yang sesuai
        if model_choice == "ResNet50":
            selected_model = model_resnet
            target_size = (416, 416)  # Misalnya, ukuran input ensemble
        elif model_choice == "InceptionV3":
            selected_model = model_inception
            target_size = (416, 416)  # Ukuran input ResNet50
        elif model_choice == "Average Ensemble":
            selected_model = model_averageEnsemble
            target_size = (416, 416)  # Ukuran input Inception
        elif model_choice == "Stacking Ensemble":
            selected_model = model_stackingEnsemble
            target_size = (416, 416)  # Ukuran input VGG
        else:
            # Jika tidak ada pilihan yang valid
            os.remove(img_path)  # Clean up the uploaded file
            return render_template("classification.html", error="Model tidak ditemukan")

        # Lakukan prediksi dengan model yang dipilih
        prediction, confidence = predict_label(img_path, selected_model, target_size)
        
        if prediction == "Error":
            os.remove(img_path)  # Clean up the uploaded file
            return render_template("classification.html", error="Gagal memproses gambar. Silakan coba lagi.")
        
        # If CSV export is requested, create the CSV file
        csv_filename = None
        if export_csv:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"classification_result_{timestamp}.csv"
            csv_path = os.path.join("static", csv_filename)
            
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Image', 'Model', 'Prediction', 'Confidence'])
                writer.writerow([img.filename, model_choice, prediction, f"{confidence:.4f}"])
        
        return render_template("classification.html", 
                              prediction=prediction, 
                              confidence=confidence, 
                              img_path=img_path,
                              csv_filename=csv_filename)
    
    except Exception as e:
        # If any error occurs during processing, provide clear feedback
        print(f"Error processing image: {e}")
        # Try to clean up the uploaded file if it exists
        try:
            if 'img_path' in locals() and os.path.exists(img_path):
                os.remove(img_path)
        except:
            pass
            
        return render_template("classification.html", error="Terjadi kesalahan saat memproses gambar. Silakan coba lagi.")

# Add a new route for CSV file download
@app.route('/download_csv/<filename>')
def download_csv(filename):
    try:
        return send_file(f"static/{filename}", 
                        mimetype='text/csv',
                        as_attachment=True,
                        download_name=filename)
    except Exception as e:
        print(f"Error downloading CSV: {e}")
        flash("Gagal mengunduh file CSV", "error")
        return redirect(url_for('main'))

if __name__ == '__main__':
    app.run(debug=True)
