from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tempfile import NamedTemporaryFile
import os
import torch

app = FastAPI()

model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'best.pt',
                        force_reload=True, trust_repo=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def process_image(file: UploadFile):
    try:
        # Save the uploaded file to a temporary location
        with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(file.file.read())
            temp_file_path = temp_file.name

        # Perform YOLO predictions
        results = model(temp_file_path)

# Get the bounding boxes, scores, and class IDs
        boxes = results.xyxy[0].tolist()

        # Process the results and format them as needed
        # For example, you might want to return the detected objects and their bounding boxes
        return {"Number of Apples": len(boxes)}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        # Delete the temporary file after processing
        if temp_file_path:
            os.remove(temp_file_path)

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    return await process_image(file)