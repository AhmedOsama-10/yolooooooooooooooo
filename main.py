#!/usr/bin/env python
# coding: utf-8


from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from ultralytics import YOLO  # type: ignore
from PIL import Image, ImageDraw, ImageFont
import io

# Initialize FastAPI app
app = FastAPI()

# Load the pre-trained YOLO model
#model = YOLO("C:/Users/ahmed/Downloads/brain_tumor_app/app/models/best.pt")
model = YOLO("models/best.pt")


# Define class names (you should replace these with your model's specific class names)
class_names = ["Tumor", "No Tumor"]  # Example, adjust as needed

@app.get("/")
def read_root():
    return {"message": "Brain Tumor Classification API is up and running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Run the YOLO model for inference
        results = model.predict(image, imgsz=416, conf=0.5)
        
        # Extract results
        detections = results[0].boxes.data.cpu().numpy()  # Extract detection boxes

        # Annotate the image with bounding boxes
        annotated_image = annotate_image(image, detections)
        
        # Save the annotated image
        annotated_image_path = "annotated_image.jpg"
        annotated_image.save(annotated_image_path)
        
        # Format the results for JSON response
        response = []
        for box in detections:
            x1, y1, x2, y2, score, class_id = box
            response.append({
                "class_name": class_names[int(class_id)],  # Display class name
                "confidence": float(score),
                "box": [int(x1), int(y1), int(x2), int(y2)]
            })
        
        # Return the annotated image along with the predictions
        return FileResponse(annotated_image_path, headers={"Prediction": str(response)})

    except Exception as e:
        return {"error": str(e)}

# Define class names (you should replace these with your model's specific class names)
class_names = ["pituitary", "meningioma", "glioma", "notumor"]

def annotate_image(image: Image, detections: list):
    """Annotates the image with bounding boxes and labels, with enhanced text styling."""
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()  # You can load a custom font if needed, or increase font size

    for box in detections:
        x1, y1, x2, y2, score, class_id = box
        # Draw the bounding box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # Prepare the label text using the updated class names
        label = f"{class_names[int(class_id)]}: {score:.2f}"  # Use class name and confidence score
        
        # Get text size using textbbox
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]  # Width of the bounding box
        text_height = text_bbox[3] - text_bbox[1]  # Height of the bounding box
        
        # Draw a filled rectangle behind the text for better contrast
        padding = 5
        draw.rectangle([x1, y1 - text_height - padding, x1 + text_width + padding, y1], fill="red")
        
        # Add the prediction label (class name and score) with improved readability
        draw.text((x1 + padding, y1 - text_height - padding), label, fill="white", font=font)
    
    return image




