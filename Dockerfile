FROM python:3.8
WORKDIR /app
COPY . /app
RUN pip install pip -U -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install numpy pillow -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install torch torchvision torchaudio -i https://download.pytorch.org/whl/cpu
CMD [ "python","kitti_predict.py"]

