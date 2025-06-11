import os
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Cấu hình thư mục uploads
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load mô hình
model = load_model('model/weather_cnn.keras')

# Định nghĩa các nhãn lớp
class_labels = ['dew', 'fogsmog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rainbow', 'rime', 'sandstorm', 'snow']

# Hàm tiền xử lý ảnh
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Không thể đọc ảnh.")
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Hàm dự đoán
def predict_weather(image_path):
    try:
        processed_img = preprocess_image(image_path)
        prediction = model.predict(processed_img)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][predicted_class_index] * 100
        predicted_class = class_labels[predicted_class_index]
        return predicted_class, confidence
    except Exception as e:
        return None, str(e)

# Route cho trang chính
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Kiểm tra nếu có file được tải lên
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Lưu file với tên an toàn
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Dự đoán
            predicted_class, confidence = predict_weather(file_path)
            
            if predicted_class is not None:
                return render_template('index.html', 
                                      prediction=f"{predicted_class} (Confidence: {confidence:.2f}%)",
                                      image_url=url_for('static', filename='uploads/' + filename))
            else:
                return render_template('index.html', prediction=f"Lỗi: {confidence}")
    
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)