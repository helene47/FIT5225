FROM python:3.12-slim

WORKDIR /code

COPY img_service.py /code/
COPY requirements.txt /code/
COPY ImgData.py /code/
COPY app.py /code/
COPY yolo11l-pose.pt /code/

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    procps \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

EXPOSE 60000

CMD ["fastapi", "run", "app.py", "--port", "60000"]