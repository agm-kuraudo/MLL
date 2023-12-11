FROM tensorflow/tensorflow:latest-gpu

RUN python -m pip install pillow

RUN python -m pip install scipy

COPY . /app