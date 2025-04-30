from pydantic import BaseModel

class ImgData(BaseModel):
    id: str
    img_base64: str
