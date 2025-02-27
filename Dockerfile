# # Use an official Python runtime as a parent image
# FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

# # Import NVIDIA CUDA repository GPG key
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
FROM python:3.9-slim
# Update apt repository
RUN apt-get update

RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1-mesa-glx  # Install libGL.so.1 dependency
RUN apt-get install -y python3-pip

# Set working directory
WORKDIR /app

# Copy application files
COPY . .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run main.py when the container launches  
CMD ["python", "main.py"]
