from flask import Flask, render_template, request, redirect, url_for
import os
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)

# Configure paths
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load your trained model
model = YOLO("weights/best.pt")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle uploaded file
        file = request.files["image"]
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Run detection
            results = model.predict(filepath, conf=0.92, iou=0.45)
            result_img_path = os.path.join(RESULT_FOLDER, file.filename)
            
            if len(results[0].boxes) > 0:  # If sheep detected
                # Draw boxes/labels on the image
                img = Image.open(filepath)
                draw = ImageDraw.Draw(img)
                font = ImageFont.load_default()

                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                    draw.text((x1, y1 - 10), "Sheep", fill="red", font=font)

                img.save(result_img_path)
                message = "Sheep detected!"
            else:
                # Create a blank result with "not detected" text
                img = Image.open(filepath)
                draw = ImageDraw.Draw(img)
                draw.text((10, 10), "Cannot detect sheep", fill="red", font=ImageFont.load_default())
                img.save(result_img_path)
                message = "No sheep found."

            return render_template("index.html", 
                                uploaded_image=filepath,
                                result_image=result_img_path,
                                message=message)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)