#!/bin/bash
cp ./outputs/resnet_18.onnx ./models/onnx_example/1/model.onnx
cp ./outputs/resnet_18.plan ./models/trt_example/1/model.plan
