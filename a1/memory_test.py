import psutil
import time
import logging
import requests
import base64
from PIL import Image
import io
import gc

# 配置日志
logging.basicConfig(
    filename='memory_test.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_memory_usage():
    """获取当前进程的内存使用情况"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        'rss': memory_info.rss / 1024 / 1024,  # MB
        'vms': memory_info.vms / 1024 / 1024,  # MB
        'percent': process.memory_percent()
    }

def log_memory_usage(step_name):
    """记录内存使用情况"""
    memory = get_memory_usage()
    logger.info(f"{step_name} - Memory usage: {memory}")
    return memory

def test_pose_estimation(image_path, server_url, num_requests=10):
    """测试姿态估计服务的内存使用"""
    # 读取测试图片
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    # 记录初始内存使用
    initial_memory = log_memory_usage("Initial")
    
    # 发送多个请求并监控内存
    for i in range(num_requests):
        start_memory = log_memory_usage(f"Request {i+1} Start")
        start_time = time.time()
        
        try:
            # 发送请求
            response = requests.post(
                f"{server_url}/api/pose_estimation",
                json={"id":"123", "img_base64": image_data}
            )
            
            # 记录请求完成后的内存使用
            end_time = time.time()
            end_memory = log_memory_usage(f"Request {i+1} End")
            
            # 计算内存变化和响应时间
            memory_diff = {
                'rss_diff': end_memory['rss'] - start_memory['rss'],
                'vms_diff': end_memory['vms'] - start_memory['vms']
            }
            response_time = end_time - start_time
            
            logger.info(
                f"Request {i+1} completed - "
                f"Memory diff: {memory_diff}, "
                f"Response time: {response_time:.2f}s, "
                f"Status: {response.status_code}"
            )
            
            # 强制垃圾回收
            gc.collect()
            time.sleep(1)  # 等待1秒
            
        except Exception as e:
            logger.error(f"Request {i+1} failed - Error: {str(e)}")
    
    # 记录最终内存使用
    final_memory = log_memory_usage("Final")
    total_memory_increase = {
        'rss_increase': final_memory['rss'] - initial_memory['rss'],
        'vms_increase': final_memory['vms'] - initial_memory['vms']
    }
    logger.info(f"Total memory increase: {total_memory_increase}")

if __name__ == "__main__":
    # 测试参数
    image_path = "test.jpg"  # 测试图片路径
    server_url = "http://158.179.28.3:60000"
    num_requests = 10  # 请求次数
    
    # 运行测试
    test_pose_estimation(image_path, server_url, num_requests)