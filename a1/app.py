from fastapi import FastAPI, Request
from ImgData import ImgData
from img_service import pose_estimation, pose_estimation_annotation, load_model  
import logging
from ultralytics import YOLO
import uvicorn



app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model globally to optimize the workflow for subsequent calls
@app.on_event("startup")
async def startup_event():
    app.state.model = load_model()

# def load_model():
#     app.state.model = YOLO('./yolo11l-pose.pt')
#     logger.info("Model loaded")

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

# 配置服务器参数
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=60000,
        workers=1,  # 使用单worker
        limit_concurrency=10,  # 限制并发连接数
        timeout_keep_alive=30,  # 设置keep-alive超时
    )

