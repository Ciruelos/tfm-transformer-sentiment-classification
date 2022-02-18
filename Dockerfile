FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

ENV DEBIAN_FRONTEND="noninteractive"

RUN apt-get update --fix-missing
RUN apt-get install -y libgl1-mesa-glx libglib2.0-0 git

RUN pip install -r requirements.txt

ENV PYTHONPATH '${PYTHONPATH}:/workspace'

WORKDIR /workspace
