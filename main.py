from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tempfile import NamedTemporaryFile
import os
import torch
from PIL import Image
from collections import Counter
import random
from torchvision import models, transforms
import torch.nn as nn

app = FastAPI()

# Load fine-tuned custom model
model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'best.pt',
                        force_reload=True, trust_repo=True)

model.conf = 0.75  # NMS confidence threshold
model.iou = 0.20  # NMS IoU threshold
model.classes = [0]

confidence_level = 0.70

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def calculate_average_rgb(img, box_size=100):
    # Open the image
    width, height = img.size

    # Calculate the coordinates for the bounding box
    left = max(0, width // 2 - box_size // 2)
    top = max(0, height // 2 - box_size // 2)
    right = min(width, width // 2 + box_size // 2)
    bottom = min(height, height // 2 + box_size // 2)

    # Crop the image
    cropped_img = img.crop((left, top, right, bottom))

    # Calculate the average RGB value
    rgb_values = list(cropped_img.getdata())

    # Calculate the average RGB values
    avg_color = (
        sum([pixel[0] for pixel in rgb_values]) // len(rgb_values),
        sum([pixel[1] for pixel in rgb_values]) // len(rgb_values),
        sum([pixel[2] for pixel in rgb_values]) // len(rgb_values)
    )

    # Return the average RGB value as a color
    return avg_color


apple_colors = {
    "colorless": [(0, 150, 0), (150, 255, 255)],  # Adjust ranges as needed
    "average": [(150, 150, 0), (255, 255, 255)],  # Adjust ranges as needed
    "lime apple": [(150, 80, 0), (225, 150, 255)],
    "Red delicious": [(0, 0, 0), (255, 80, 255)]
}

def find_color_range(rgb_value, color_ranges):
    for color, ranges in color_ranges.items():
        min_range, max_range = ranges
        if all(min_val <= val <= max_val for val, (min_val, max_val) in zip(rgb_value, zip(min_range, max_range))):
            return color
    return "Unknown"


async def process_image(file: UploadFile, layers: str):
    try:
        layers = int(layers)
        # Save the uploaded file to a temporary location
        with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(file.file.read())
            temp_file_path = temp_file.name

        # Perform YOLO predictions
        results = model(temp_file_path)

# Get the bounding boxes, scores, and class IDs
        boxes = results.xyxy[0].tolist()
        img = Image.open(temp_file_path).convert('RGB')

        list =[]
        spot=[]

        if len(boxes) >= 10:
            for i in range(10):
                random_integer = random.randint(0, len(boxes)-1)
                first_box = results.xyxy[0][random_integer].cpu().numpy()
                x_min, y_min, x_max, y_max = map(int, first_box[:4])
                cropped_img = img.crop((x_min, y_min, x_max, y_max))

                rgb_value_to_check = calculate_average_rgb(cropped_img)

                result = find_color_range(rgb_value_to_check, apple_colors)
                list.append(result)

        elif len(boxes) < 10:
            for i in range(len(boxes)):
                first_box = results.xyxy[0][i].cpu().numpy()
                x_min, y_min, x_max, y_max = map(int, first_box[:4])
                cropped_img = img.crop((x_min, y_min, x_max, y_max))

                rgb_value_to_check = calculate_average_rgb(cropped_img)

                result = find_color_range(rgb_value_to_check, apple_colors)
                list.append(result)
        
        else:
            return JSONResponse(content={"error": "No apples detected"}, status_code=500)
       
        element_counts = Counter(list)

        if len(list) == 0:
            most_common_element = "No apples detected"
        elif element_counts.most_common(1)[0][0] == 'Unknown':
            del element_counts['Unknown']
            most_common_element = element_counts.most_common(1)[0][0]
        else:
            most_common_element = element_counts.most_common(1)[0][0]

        list = []
        
        count_of_spot = spot.count(0)
        spot = []

        grade = 'A'

        if most_common_element == 'colorless':
            grade = 'B'
        
        if count_of_spot > 0 and len(boxes) < 10:
            grade = 'C'
        
        if count_of_spot > 3 and len(boxes) > 10:
            grade = 'C'

        total = len(boxes)* layers
        size = 'large'
        if total < 100:
            size = 'large'
        elif total < 125 and total > 100:
            size = 'medium'
        elif total < 175 and total > 125:
            size = 'small'
        elif total < 210 and total > 175 and layers == 6:
            size = 'extra small 6 layers'
        elif total < 200 and total > 175 and layers == 5:
            size = 'extra small 5 layers'
        elif total < 210 and total > 175 and layers != 5 and layers != 6:
            size = 'extra small'
        else:
            size = 'extra small more than 240'
        
        print(len(boxes))
        print(count_of_spot)
        
        if len(boxes) == 0:
            return JSONResponse(content={"error": "No apples detected"}, status_code=500)
        
        return {"size of Apples": size, "Type": most_common_element, "Grade": grade}

    except Exception as e:
        if str(e) == "list index out of range":
            return JSONResponse(content={"error": "No apples detected"}, status_code=500)
        else:
            return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        # Delete the temporary file after processing
        if temp_file_path:
            os.remove(temp_file_path)

@app.post("/predict")
async def predict_image(file: UploadFile = File(...), layers: str = Form(...)):
    return await process_image(file, layers)
