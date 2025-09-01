import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

def extract_angle(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        data_pts = np.array(largest, dtype=np.float64).reshape(-1, 2)
        mean, eigenvectors, _ = cv2.PCACompute2(data_pts, mean=np.array([]))
        angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]) * 180 / np.pi
        return angle % 180
    else:
        return None

def classify_asl_letter(angle):
    if angle is None:
        return "unknown"
    if angle < 25 or angle > 155:
        return "G"
    elif 65 < angle < 115:
        return "U"
    else:
        return "unknown"

UPLOAD_FORM = """
<!doctype html>
<title>ASL G vs U Classifier</title>
<h2>Upload an ASL hand image (G or U)</h2>
<form method=post enctype=multipart/form-data action="/upload">
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(UPLOAD_FORM)

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"})
    filepath = "uploaded_image.png"
    file.save(filepath)
    angle = extract_angle(filepath)
    prediction = classify_asl_letter(angle)
    if angle is None:
        return jsonify({"error": "No contour detected"})
    return jsonify({
        "detected_angle_degrees": round(angle, 2),
        "predicted_letter": prediction
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
