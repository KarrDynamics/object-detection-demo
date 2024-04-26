import os
from flask import Flask, render_template, request, redirect, url_for, send_file
import cv2
import torch
import base64
import numpy as np

app = Flask(__name__)

# Get the absolute path of the app directory
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Configure upload folder
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static/uploads/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLOv5 model
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).eval()
except Exception as e:
    print(f"Error loading YOLOv5 model: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    image_base64 = None
    objects_detected = []

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file:
            # Save the file to the uploads directory
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Check if it's an image or video
            if file.filename.endswith(('.jpg', '.jpeg', '.png')):
                image_base64, objects_detected = detect_objects_in_image(file_path)
            elif file.filename.endswith(('.mp4', '.avi', '.mov')):
                detect_objects_in_video(file_path)
                # For video, we won't return anything as we are not displaying video
            else:
                return "Unsupported file format", 400

    return render_template('index.html', image=image_base64, objects_detected=objects_detected)

def detect_objects_in_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    original_image = image.copy()

    # Perform object detection with YOLOv5
    results = model(original_image)

    detected_objects = []

    # Process the results to draw bounding boxes
    for result in results.pred:
        boxes = result[:, :4].detach().numpy()
        confidences = result[:, 4].detach().numpy()
        class_ids = result[:, 5].detach().numpy().astype(int)

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[class_id]}: {confidence:.2f}"
            detected_objects.append({
                'label': label,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'confidence': confidence
            })

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert the modified image to base64
    retval, buffer = cv2.imencode('.jpg', image)
    base64_image = base64.b64encode(buffer).decode('utf-8')

    return base64_image, detected_objects


def detect_objects_in_video(video_path):
    # Open video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return None

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_video.mp4')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    detected_objects = []

    # Process the video frame by frame
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Finished processing video.")
            break

        # Perform object detection with YOLOv5
        results = model(frame)

        # Process the results to draw bounding boxes
        for result in results.pred:
            boxes = result[:, :4].detach().numpy()
            confidences = result[:, 4].detach().numpy()
            class_ids = result[:, 5].detach().numpy().astype(int)

            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)
                label = f"{model.names[class_id]}: {confidence:.2f}"
                detected_objects.append({
                    'label': label,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'confidence': confidence
                })

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the frame to the output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()

    print(f"Processed video saved at: {output_path}")

    return output_path, detected_objects

if __name__ == '__main__':
    app.run(debug=True)
