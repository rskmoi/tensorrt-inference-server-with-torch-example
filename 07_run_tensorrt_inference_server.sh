#!/bin/bash
sudo docker run --gpus=1 --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
-v $PWD/models:/models nvcr.io/nvidia/tensorrtserver:19.10-py3 trtserver --model-repository=/models