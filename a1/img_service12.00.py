import gc
import logging
import os
from fastapi import Request
from ImgData import ImgData
import cv2
import logging
import tempfile
import base64

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
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_img

def write_temp_file(data: ImgData):
    """
    Write base64 encoded image data to a temporary file
    :param data: ImgData object containing base64 encoded image
    :return: Path to the temporary file
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(base64.b64decode(data.img_base64))
    return temp_file_path

def pose_estimation(data: ImgData, request: Request):
    """
    Perform pose estimation on the input image
    :param data: ImgData object containing image and ID
    :param request: FastAPI request object containing the model
    :return: Dictionary containing detection results
    """
    temp_file_path = write_temp_file(data)

    # Read and validate image
    # img = cv2.imread(temp_file_path)
    # if img is None:
    #     logger.error(f"Error: Could not read image at {temp_file_path}")
    #     return {}
    
    # # Resize image to reduce memory usage
    # img = resize_image(img)
    
    # Run model inference
    # results = request.app.state.model(temp_file_path)
    results = request.app.state.model(temp_file_path, conf=0.5,  imgsz=640,
    device='cpu',
    half=False,   
    max_det=10 )

    if results is None:
        return {"error": "Failed to predict pose"}
    
    # Process detection results
    keypoints_list, boxes_list = process_keypoints_and_boxes(results)
 
    # 处理完成后删除临时文件
    if os.path.exists(temp_file_path):
        os.unlink(temp_file_path)
    # del img
    gc.collect()

    return {
        "id": data.id,
        "count": len(results[0].boxes),
        "boxes": boxes_list,
        "keypoints": keypoints_list,
        "speed-preprocess": results[0].speed["preprocess"],
        "speed-inference": results[0].speed["inference"],
        "speed-postprocess": results[0].speed["postprocess"],
    }

def process_keypoints_and_boxes(results):
    """
    Process model results to extract keypoints and bounding boxes
    :param results: Model detection results
    :return: Tuple of (keypoints_list, boxes_list)
    """
    keypoints_list = []
    boxes_list = []
    for i, result in enumerate(results):
        try:
            # Extract keypoints if available
            keypoints = result.keypoints
            if keypoints is not None:
                keypoints_list.extend(extract_keypoints(keypoints))
            
            # Extract bounding boxes if available
            boxes = result.boxes
            if boxes is not None:
                boxes_list.extend(extract_boxes(boxes))
        except Exception as e:
            logger.error(f"Error processing result #{i}: {str(e)}")
    return keypoints_list, boxes_list

def extract_keypoints(keypoints):
    """
    Extract keypoint coordinates and confidence scores
    :param keypoints: Keypoints object from model results
    :return: List of (x, y, confidence) tuples
    """
    keypoints_xy_conf = []
    if hasattr(keypoints, 'xy') and keypoints.xy is not None and hasattr(keypoints, 'conf') and keypoints.conf is not None:
        keypoints_xy = keypoints.xy[0]
        keypoints_conf = keypoints.conf[0]
        for k, (x, y) in enumerate(keypoints_xy):
            if k < len(keypoints_conf):
                # Convert tensor values to Python native types
                x_val = int(x.item()) if hasattr(x, 'item') else int(x)
                y_val = int(y.item()) if hasattr(y, 'item') else int(y)
                conf_val = float(keypoints_conf[k].item()) if hasattr(keypoints_conf[k], 'item') else float(keypoints_conf[k])
                keypoints_xy_conf.append((x_val, y_val, conf_val))
    return keypoints_xy_conf

def extract_boxes(boxes):
    """
    Extract bounding box coordinates and confidence scores
    :param boxes: Boxes object from model results
    :return: List of dictionaries containing box information
    """
    boxes_data = []
    for box in boxes:
        try:
            # Extract box coordinates and confidence
            x = int(box.xywh[0][0].item())
            y = int(box.xywh[0][1].item())
            width = int(box.xywh[0][2].item())
            height = int(box.xywh[0][3].item())
            probability = float(box.conf.item())
            boxes_data.append({
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "probability": probability
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
    temp_file_path = write_temp_file(data)

    # Read and validate image
    img = cv2.imread(temp_file_path)
    if img is None:
        logger.error(f"Error: Could not read image at {temp_file_path}")
        return {"error": "Could not read image"}
    
    # Resize image to reduce memory usage
    img = resize_image(img)
    
    # Run model inference
    results = request.app.state.model(temp_file_path)

    if results is None or len(results) == 0:
        logger.warning("No results from model")
        return {"error": "No detection results"}

    for result in results:
        keypoints = result.keypoints
        
        # Validate keypoints data
        if keypoints is None:
            logger.warning("No keypoints detected")
            continue

        if not hasattr(keypoints, 'xy') or keypoints.xy is None or \
           not hasattr(keypoints, 'conf') or keypoints.conf is None or \
           len(keypoints.xy) == 0 or len(keypoints.conf) == 0:
            logger.warning("Invalid keypoints data")
            continue

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

            # Convert annotated image to base64
            _, buffer = cv2.imencode('.jpg', img)
            annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    
            return {
                "id": data.id,
                "annotated_image": annotated_image_base64
            }
        except Exception as e:
            logger.error(f"Error processing keypoints: {str(e)}")
            continue
        finally:
            # 处理完成后删除临时文件
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            del img
            del buffer
            del annotated_image_base64
            gc.collect()

    return {"error": "No valid keypoints detected in the image"}


