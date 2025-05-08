from locust import HttpUser, task, between
import os
import uuid
import base64
import glob

# Cache for pre-encoded images
encoded_images = []
# Index to track the current image
current_image_index = 0

def load_images():
    """
    Load and encode all JPEG images from the input folder
    """
    image_path = "/home/ubuntu/inputfolder"
    image_files = [str(f) for f in glob.glob(os.path.join(image_path, "*.jpg"))]
    for image_file in image_files:
        with open(image_file, 'rb') as image_file:
            base64_str = base64.b64encode(image_file.read()).decode('utf-8')
            encoded_images.append(base64_str)


class CloudPoseUser(HttpUser):
    # Users wait 1-3 seconds between requests
    wait_time = between(1, 3)
    
    def on_start(self):
        """
        Executed when a user starts - ensures images are loaded
        """
        if not encoded_images:
            load_images()

    def get_next_image():
        """
        Get the next image in sequence, looping back to the first image after the last one
        """
        global current_image_index
        
        if not encoded_images:
            return None
            
        # Get current image
        image = encoded_images[current_image_index]
        
        # Move to next image, loop back to first if needed
        current_image_index = (current_image_index + 1) % len(encoded_images)
        
        return image

    @task(1)
    def send_pose_estimation_request(self):
        """
        Send request to the pose estimation endpoint that returns JSON data
        Weight: 1
        """
        if not encoded_images:
            print("No encoded images available!")
            return
        
        image = CloudPoseUser.get_next_image()
        if image is None:
            print("Failed to get next image!")
            return
            
        data = {
            "id": str(uuid.uuid4()),  # Generate unique ID for each request
            "img_base64": image  # Use sequential image
        }
        self.client.post("/api/pose_estimation", json=data)

    @task(1)
    def send_pose_estimation_annotation_request(self):
        """
        Send request to the pose estimation annotation endpoint that returns annotated images
        Weight: 1
        """
        if not encoded_images:
            print("No encoded images available!")
            return
        
        image = CloudPoseUser.get_next_image()
        if image is None:
            print("Failed to get next image!")
            return
            
        data = {
            "id": str(uuid.uuid4()),  # Generate unique ID for each request
            "img_base64": image  # Use sequential image
        }
        self.client.post("/api/pose_estimation_annotation", json=data)
