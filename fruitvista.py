from flask import Flask, render_template, Response
import cv2
import math
from ultralytics import YOLO

app = Flask(__name__)

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("best.pt")

# Object classes
classNames = ["apple", "banana", "carambola", "cherry", "dragon fruit", "grape", "guava", "kiwi", "lemon", "mango",
              "mulberries", "orange", "papaya", "passion_fruit", "peach", "pear", "pineapple", "strawberry", "tomato", "watermelon"]

def generate_frames():

    while True:
        success, img = cap.read()
        results = model(img, stream=True)

        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100

                # class name
                cls = int(box.cls[0])
                class_name = classNames[cls]

                # object details
                org = (x1, y1 - 20)  # Adjust the y-coordinate for displaying text above the box
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, f"{class_name} ({confidence})", org, font, fontScale, color, thickness)
                
        _, buffer = cv2.imencode('.jpg', img)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
@app.route('/')
def index():
    return render_template('index.html')

@app.route("/inspect")
def inspect():
    return render_template('portfolio-details.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, port=5050, use_reloader=False)