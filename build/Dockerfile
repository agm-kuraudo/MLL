FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variable to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and pip
RUN apt-get update -y && apt-get install -y python3 python3-pip && \
    python3 -m pip install --upgrade pip

# Install necessary packages for adding repositories
RUN apt-get install -y gnupg2 curl

# Install TensorRT
RUN apt-get update -y && \
    apt-get install -y tensorrt

# Install TensorFlow and other Python packages with NumPy <2
RUN python3 -m pip install "numpy<2" tensorflow==2.8.0 pillow scipy tensorflow-datasets beautifulsoup4 matplotlib PyQt5 tk-tools pandas scikit-learn yfinance==0.2.58 keras-tuner

# Set python3 as the default python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Downgrade protobuf to 3.20.x
RUN python3 -m pip install protobuf==3.20.0

# Install SQL Alchemy
RUN python3 -m pip install sqlalchemy psycopg2-binary

# Set environment variables for CUDA and PTXAS
ENV PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
ENV XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-11.8

# Copy application files
COPY .. /app
