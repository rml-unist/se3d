# docker build -f docker/Dockerfile -t se3d .
# docker run --gpus all --ipc=host -v /data/SE3D:/data/SE3D -it se3d
FROM python:3.9-slim
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
WORKDIR /workspace/se3d
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
