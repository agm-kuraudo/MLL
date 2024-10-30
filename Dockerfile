FROM tensorflow/tensorflow:latest-gpu

RUN python -m pip install pillow
RUN python -m pip install scipy
RUN python -m pip install tensorflow-datasets
RUN python -m pip install beautifulsoup4
RUN python -m pip install matplotlib
RUN python -m pip install PyQt5
RUN python -m pip install tk-tools

COPY . /app