FROM tensorflow/tensorflow:latest-gpu

RUN python -m pip install pillow

ENTRYPOINT ["top", "-b"]