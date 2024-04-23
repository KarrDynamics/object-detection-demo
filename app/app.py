from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import cv2
import torch
from PIL import Image
import base64
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)

# Initialize Flask-Limiter
limiter = Limiter(
    app,
    default_limits=["5 per minute", "1 per second"]
)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).eval()

@app.route('/', methods=['GET', 'POST'])
@limiter.limit("1 per minute")
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        image = Image.open(file)
        image_np = np.array(image)

        # Perform object detection with YOLOv5
        results = model(image_np)

        objects_detected = []

        # Process the results to draw bounding boxes
        for result in results.pred:
            boxes = result[:, :4].detach().numpy()
            confidences = result[:, 4].detach().numpy()
            class_ids = result[:, 5].detach().numpy().astype(int)

            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)
                class_name = model.names[class_id]
                objects_detected.append((class_name, confidence))
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_np, f"{class_name}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert the image back to RGB (OpenCV uses BGR)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        # Encode image to base64
        _, buffer = cv2.imencode('.jpg', image_np)
        img_str = base64.b64encode(buffer).decode('utf-8')

        return render_template('index.html', image=img_str, objects_detected=objects_detected)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
