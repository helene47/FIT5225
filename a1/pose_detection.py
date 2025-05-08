from fastapi import FastAPI, Request
from pydantic import BaseModel
import cv2
import logging
import tempfile
import base64
import os
import numpy as np
from ultralytics import YOLO
import uvicorn

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImgData(BaseModel):
    id: str
    img_base64: str

def resize_image(img, file_path, max_size=128):
    """
    Resize image while maintaining aspect ratio
    :param img: Input image in OpenCV format
    :param max_size: Maximum size for the longer edge in pixels
    :return: Resized image
    """
    height, width = img.shape[:2]
    
    # Calculate scaling ratio based on the longer edge
    if width > height:
        scale = max_size / width
    else:
        scale = max_size / height
    
    # Return original image if it's already smaller than max_size
    if scale >= 1:
        return img
    
    # Save the resized image with JPEG compression
    cv2.imwrite(file_path, img, [cv2.IMWRITE_JPEG_QUALITY, 80])

def pose_estimation(data: ImgData, request: Request):
    """
    Perform pose estimation on the input image
    :param data: ImgData object containing image and ID
    :param request: FastAPI request object containing the model
    :return: Dictionary containing detection results
    """
    # Create a temporary file to store the decoded image
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as file:
        file_path = file.name
        file.write(base64.b64decode(data.img_base64))

    try:
        # Read the image from the temporary file
        img = cv2.imread(file_path)
        if img is None:
            logger.error(f"Error: Could not read image at {file_path}")
            return {"error": "Could not read image"}
        
        # Resize the image to optimize processing
        resize_image(img, file_path)

        # Run model inference using the model stored in the application state
        results = request.app.state.model(
            file_path,
            imgsz=128,
            device="cpu"
        )

        # Extract keypoints and bounding boxes from the results
        keypoints_list, boxes_list = process_keypoints_and_boxes(results)

        # Prepare the result dictionary with detection information
        result_dict = {
                "id": data.id,
                "count": len(results[0].boxes),
                "boxes": boxes_list,
                "keypoints": keypoints_list,
            }
            
        # Add speed information if available in the results
        if hasattr(results[0], 'speed'):
            result_dict.update({
                "speed-preprocess": results[0].speed["preprocess"],
                "speed-inference": results[0].speed["inference"],
                "speed-postprocess": results[0].speed["postprocess"],
            })

        return result_dict
    except Exception as e:
        logger.error(f"Error in pose_estimation: {str(e)}")
        return {"error": str(e)}
    finally:
        # Clean up the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

def process_keypoints_and_boxes(results):
    """
    Process model results to extract keypoints and bounding boxes
    :param results: Model detection results
    :return: Tuple of (keypoints_list, boxes_list)
    """
    keypoints_list = []
    boxes_list = []
    try:
        for i, result in enumerate(results):
            # Extract keypoints if available
            keypoints = result.keypoints
            if keypoints is not None:
                keypoints_list.extend(extract_keypoints(keypoints))
                # Clear reference to reduce memory usage
                keypoints = None
            
            # Extract bounding boxes if available
            boxes = result.boxes
            if boxes is not None:
                boxes_list.extend(extract_boxes(boxes))
                # Clear reference to reduce memory usage
                boxes = None
    except Exception as e:
        logger.error(f"Error processing results: {str(e)}")
    
    return keypoints_list, boxes_list

def extract_keypoints(keypoints):
    """
    Extract keypoint coordinates and confidence scores
    :param keypoints: Keypoints object from model results
    :return: List of (x, y, confidence) tuples
    """
    keypoints_xy_conf = []
    try:
        # Validate keypoints has the required attributes
        if hasattr(keypoints, 'xy') and keypoints.xy is not None and hasattr(keypoints, 'conf') and keypoints.conf is not None:
            keypoints_xy = keypoints.xy[0]
            keypoints_conf = keypoints.conf[0]
            
            # Process each keypoint and extract coordinates and confidence
            keypoints_xy_conf = []
            for k, (x, y) in enumerate(keypoints_xy):
                if k < len(keypoints_conf):
                    # Convert tensor values to Python native types
                    x_val = int(x.item()) if hasattr(x, 'item') else int(x)
                    y_val = int(y.item()) if hasattr(y, 'item') else int(y)
                    conf_val = float(keypoints_conf[k].item()) if hasattr(keypoints_conf[k], 'item') else float(keypoints_conf[k])
                    keypoints_xy_conf.append((x_val, y_val, conf_val))
            
            # Clear references to reduce memory usage
            keypoints_xy = None
            keypoints_conf = None
    except Exception as e:
        logger.error(f"Error extracting keypoints: {str(e)}")
    
    return keypoints_xy_conf

def extract_boxes(boxes):
    """
    Extract bounding box coordinates and confidence scores
    :param boxes: Boxes object from model results
    :return: List of dictionaries containing box information
    """
    boxes_data = []
    try:
        # Process each bounding box
        for box in boxes:
            # Extract and convert coordinates and dimensions to native types
            x = int(box.xywh[0][0].item()) if hasattr(box.xywh[0][0], 'item') else int(box.xywh[0][0])
            y = int(box.xywh[0][1].item()) if hasattr(box.xywh[0][1], 'item') else int(box.xywh[0][1])
            width = int(box.xywh[0][2].item()) if hasattr(box.xywh[0][2], 'item') else int(box.xywh[0][2])
            height = int(box.xywh[0][3].item()) if hasattr(box.xywh[0][3], 'item') else int(box.xywh[0][3])
            probability = float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf)
            
            # Store bounding box data in a dictionary
            boxes_data.append({
                "x": x, "y": y, "width": width, "height": height, "probability": probability
            })
    except Exception as e:
        logger.error(f"Error extracting box data: {str(e)}")
    
    return boxes_data

def pose_estimation_annotation(data: ImgData, request: Request):
    """
    Generate annotated image with pose keypoints and connections
    :param data: ImgData object containing image and ID
    :param request: FastAPI request object containing the model
    :return: Dictionary containing annotated image in base64 format
    """
    # Create a temporary file to store the decoded image
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as file:
        file_path = file.name
        file.write(base64.b64decode(data.img_base64))
    try:
        # Load image at reduced resolution
        img = cv2.imread(file_path)
        if img is None:
            logger.error(f"Error: Could not read image at {file_path}")
            return {"error": "Could not read image"}
        
        # Resize the image to optimize processing
        resize_image(img, file_path)

        # Run model inference
        results = request.app.state.model(
            file_path,
            imgsz=128,
            device="cpu"
        )
        
        # Check if results are valid
        if results is None or len(results) == 0:
            logger.warning("No results from model")
            return {"error": "No detection results"}
        
        # Process only the first detection result to reduce memory usage
        result = results[0]
        keypoints = result.keypoints
        
        # Validate keypoints data structure
        if keypoints is None or not hasattr(keypoints, 'xy') or keypoints.xy is None or \
           not hasattr(keypoints, 'conf') or keypoints.conf is None or \
           len(keypoints.xy) == 0 or len(keypoints.conf) == 0:
            return {"error": "No valid keypoints detected"}

        try:
            # Get keypoint coordinates and confidence scores
            keypoints_xy = keypoints.xy[0]
            keypoints_conf = keypoints.conf[0]

            # Draw keypoints and labels
            for k, (x, y) in enumerate(keypoints_xy):
                if k < len(keypoints_conf) and keypoints_conf[k] > 0.5:
                    # Draw keypoint as green circle
                    cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
                    # Draw keypoint index as red text
                    cv2.putText(img, str(k), (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Define connections between keypoints (shoulders and hips)
            connections = [[5, 6], [5, 11], [6, 12], [11, 12]]

            # Draw connections between keypoints
            for connection in connections:
                if all(idx < len(keypoints_xy) for idx in connection):
                    p1 = (int(keypoints_xy[connection[0]][0]), int(keypoints_xy[connection[0]][1]))
                    p2 = (int(keypoints_xy[connection[1]][0]), int(keypoints_xy[connection[1]][1]))

                    if all(idx < len(keypoints_conf) for idx in connection) and \
                       keypoints_conf[connection[0]] > 0.5 and keypoints_conf[connection[1]] > 0.5:
                        # Draw connection as red line
                        cv2.line(img, p1, p2, (0, 0, 255), 2)

            # Encode with reduced quality to save memory
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
            _, buffer = cv2.imencode('.jpg', img, encode_params)
            
            # Convert to base64
            annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                "id": data.id,
                "annotated_image": annotated_image_base64
            }
        except Exception as e:
            logger.error(f"Error processing keypoints: {str(e)}")
            return {"error": f"Error processing keypoints: {str(e)}"}
    except Exception as e:
        logger.error(f"Error in pose_estimation_annotation: {str(e)}")
        return {"error": str(e)}
    finally:
        # Clean up the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

def load_model():
    """
    Load the YOLO pose estimation model
    :return: Loaded YOLO model instance
    """
    model_path = "yolo11l-pose.pt"
    model = YOLO(model_path)
    return model

# ===== FastAPI Application =====
app = FastAPI()

# Load the model globally to optimize the workflow for subsequent calls
@app.on_event("startup")
async def startup_event():
    app.state.model = load_model()

# Endpoint to verify if the service is deployed correctly
@app.get("/health_check")
def health_check():
    return {"status": "ok"}

# Returns JSON response with pose estimation results
@app.post("/api/pose_estimation")
async def pose_estimation_endpoint(data: ImgData, request: Request):   
    # Using request to access the globally stored model
    return pose_estimation(data, request)
    
# Returns annotated image with pose estimation visualization
@app.post("/api/pose_estimation_annotation")
async def pose_estimation_annotation_endpoint(data: ImgData, request: Request):
    return pose_estimation_annotation(data, request)

# Server configuration parameters
if __name__ == "__main__":
    uvicorn.run(
        "pose_detection:app",  # Updated to use the new file name
        host="0.0.0.0",
        port=60000,
        workers=1,  
        limit_concurrency=10,
        timeout_keep_alive=30
    ) 