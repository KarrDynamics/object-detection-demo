# Object Detection Demo with YOLO and Flask

This project is a simple web-based object detection demo powered by YOLO and Flask. It allows users to upload an image and performs object detection on the uploaded image, displaying the detected objects with bounding boxes and confidence scores.

## Features

- **Object Detection:** Detects objects in uploaded images using YOLO.
- **Real-time Processing:** Provides real-time object detection results.
- **User-friendly Interface:** Sleek and modern dashboard-style user interface.

## Getting Started

### Prerequisites

- Python (3.7 or higher)
- pip

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/KarrDynamics/object-detection-demo.git
    ```

2. Navigate to the project directory:

    ```bash
    cd object-detection-demo
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

Run the following command to start the Flask application:

```bash
python app/app.py
```

The application will be live at `http://127.0.0.1:5000/`.

## Usage

1. Open the browser and go to `http://127.0.0.1:5000/`.
2. Click on the "Choose file" button and upload an image.
3. Click the "Upload" button to perform object detection.
4. The detected objects with bounding boxes and confidence scores will be displayed below the uploaded image.

## Built With

- [Flask](https://flask.palletsprojects.com/) - Web framework
- [YOLO](https://github.com/ultralytics/ultralytics) - Object detection model
- [Bootstrap](https://getbootstrap.com/) - Front-end framework

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for providing YOLOv5.

---