import logging
from fastapi import Request
from ImgData import ImgData
import cv2
import logging
import tempfile
import base64
import os
import gc
import numpy as np
import psutil
from ultralytics import YOLO

logger = logging.getLogger(__name__)

def resize_image(img, max_size=128):
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
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image using area interpolation (best for downscaling)
    # Resize in-place to avoid creating new array
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_img

def write_temp_file(data: ImgData):
    """
    Write base64 encoded image data to a temporary file
    :param data: ImgData object containing base64 encoded image
    :return: Path to the temporary file
    """
    try:
        # Create a more controlled temp file that we explicitly delete
        temp_file_path = os.path.join(tempfile.gettempdir(), f"pose_temp_{data.id}.jpg")
        # Directly decode and write to file in one step to reduce memory copies
        with open(temp_file_path, "wb") as f:
            f.write(base64.b64decode(data.img_base64))
        # Clear the base64 data from memory
        data.img_base64 = None
        gc.collect()
        return temp_file_path
    except Exception as e:
        logger.error(f"Error writing temp file: {str(e)}")
        return None

def pose_estimation(data: ImgData, request: Request):
    """
    Perform pose estimation on the input image
    :param data: ImgData object containing image and ID
    :param request: FastAPI request object containing the model
    :return: Dictionary containing detection results
    """
    temp_file_path = None
    try:
        temp_file_path = write_temp_file(data)
        if not temp_file_path:
            return {"error": "Failed to write temporary file"}

        # Use IMREAD_REDUCED to load smaller image directly
        img = cv2.imread(temp_file_path, cv2.IMREAD_REDUCED_COLOR_4)
        if img is None:
            logger.error(f"Error: Could not read image at {temp_file_path}")
            return {"error": "Could not read image"}
        
        # Run model inference - no need to resize since we already loaded a reduced image
        # results = request.app.state.model(temp_file_path)
        results = request.app.state.model(
            temp_file_path,
            conf=0.25,
            imgsz=640,
            device='cpu',
            half=False,
            max_det=10
        )
        
        # Clean up the temp file as soon as model is done with it
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            temp_file_path = None

        if results is None:
            return {"error": "Failed to predict pose"}
        
        # Process detection results
        keypoints_list, boxes_list = process_keypoints_and_boxes(results)
        
        # Clear references to large objects
        img = None
        
        # Prepare result dictionary
        result_dict = {
            "id": data.id,
            "count": len(results[0].boxes),
            "boxes": boxes_list,
            "keypoints": keypoints_list,
        }
        
        # Add speed info if available
        if hasattr(results[0], 'speed'):
            result_dict.update({
                "speed-preprocess": results[0].speed["preprocess"],
                "speed-inference": results[0].speed["inference"],
                "speed-postprocess": results[0].speed["postprocess"],
            })
        
        # Clear references to results
        results = None
        gc.collect()
        
        return result_dict
    except Exception as e:
        logger.error(f"Error in pose_estimation: {str(e)}")
        return {"error": str(e)}
    finally:
        # Clean up temp file if it still exists
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

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
        if hasattr(keypoints, 'xy') and keypoints.xy is not None and hasattr(keypoints, 'conf') and keypoints.conf is not None:
            keypoints_xy = keypoints.xy[0]
            keypoints_conf = keypoints.conf[0]
            
            # Pre-allocate list with estimated size to avoid resizing
            keypoints_xy_conf = []
            for k, (x, y) in enumerate(keypoints_xy):
                if k < len(keypoints_conf):
                    # Convert tensor values to Python native types
                    x_val = int(x.item()) if hasattr(x, 'item') else int(x)
                    y_val = int(y.item()) if hasattr(y, 'item') else int(y)
                    conf_val = float(keypoints_conf[k].item()) if hasattr(keypoints_conf[k], 'item') else float(keypoints_conf[k])
                    keypoints_xy_conf.append((x_val, y_val, conf_val))
            
            # Clear references
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
        for box in boxes:
            x = int(box.xywh[0][0].item()) if hasattr(box.xywh[0][0], 'item') else int(box.xywh[0][0])
            y = int(box.xywh[0][1].item()) if hasattr(box.xywh[0][1], 'item') else int(box.xywh[0][1])
            width = int(box.xywh[0][2].item()) if hasattr(box.xywh[0][2], 'item') else int(box.xywh[0][2])
            height = int(box.xywh[0][3].item()) if hasattr(box.xywh[0][3], 'item') else int(box.xywh[0][3])
            probability = float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf)
            
            # Use a tuple instead of dict to save memory
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
    temp_file_path = None
    try:
        temp_file_path = write_temp_file(data)
        if not temp_file_path:
            return {"error": "Failed to write temporary file"}

        # Load image at reduced resolution
        img = cv2.imread(temp_file_path, cv2.IMREAD_REDUCED_COLOR_2)
        if img is None:
            logger.error(f"Error: Could not read image at {temp_file_path}")
            return {"error": "Could not read image"}
        
        # Run model inference
        results = request.app.state.model(temp_file_path)
        
        # Clean up temp file immediately after model use
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            temp_file_path = None
        
        if results is None or len(results) == 0:
            logger.warning("No results from model")
            return {"error": "No detection results"}
        
        # Process only the first detection result to reduce memory usage
        result = results[0]
        keypoints = result.keypoints
        
        # Validate keypoints data
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
            
            # Clear references to large objects
            keypoints_xy = None
            keypoints_conf = None
            keypoints = None
            results = None

            # Encode with reduced quality to save memory
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
            _, buffer = cv2.imencode('.jpg', img, encode_params)
            
            # Convert to base64
            annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Clear references
            buffer = None
            img = None
            
            # Force garbage collection
            gc.collect()
            
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
        # Ensure temp file is deleted
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        
        # Force garbage collection
        gc.collect()

def load_model():
    """
    加载YOLO模型并记录内存使用情况
    """
    # process = psutil.Process()

    # 记录加载模型前的内存使用
    # before_mem = process.memory_info().rss / 1024 / 1024  # 转换为MB
    # logger.info(f"内存使用（加载前）: {before_mem:.2f} MB")

    # 获取模型文件大小
    model_path = "yolo11l-pose.pt"
    # model_size = os.path.getsize(model_path) / 1024 / 1024  # 转换为MB
    # logger.info(f"模型文件大小: {model_size:.2f} MB")

    # 加载模型
    # logger.info(">> Loading model...")
    model = YOLO(model_path)
    model.conf = 0.25  # 降低置信度阈值
    model.max_det = 10  # 限制检测数量
    # logger.info(">> Model loaded.")

    # 记录加载模型后的内存使用
    # after_mem = process.memory_info().rss / 1024 / 1024
    # logger.info(f"内存使用（加载后）: {after_mem:.2f} MB")
    # logger.info(f"模型实际占用内存: {(after_mem - before_mem):.2f} MB")

    return model


