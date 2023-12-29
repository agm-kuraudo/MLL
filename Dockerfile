FROM tensorflow/tensorflow:latest-gpu

RUN python -m pip install pillow

RUN python -m pip install scipy

RUN python -m pip install tensorflow-datasets

COPY . /app