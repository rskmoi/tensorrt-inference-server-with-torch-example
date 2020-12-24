# tensorrt-inference-server-with-torch-example

Sample of serving pytorch model with TensorRT Inference Server.

In this sample, TensorRT and ONNX are used as the model format.

**NOTICE**

TensorRT Inference Server was renamed Triton Inference Server in March 2020.

## Version

### pip

```
torch==1.7.1
onnx==1.6.0
onnxruntime==1.4.0
tensorrt==6.0.1.5
tensorrtserver==1.11.0
```

### docker 

```
nvcr.io/nvidia/tensorrtserver:19.10-py3
```

### other

```
cuda:10.1
cudnn:7.5.0
onnx-tensorrt:6.0
```

cudnn 7.5.0 is important. I failed build onnx-tensorrt 6.0 with cudnn 7.6.4.


## Usage
```
# Training Pytorch Model
python 01_train_model_with_torch.py
# Convert Pytorch model to ONNX model
python 02_pth_to_onnx.py
# Inference example using ONNX model on local
python 03_onnxruntime_local.py
# Convert ONNX model to TensorRT model 
./04_onnx_to_tensorrt.sh
# Inference example using TensorRT model on local
python 05_tensorrt_local.py
# Copy and rename models for using TensorRT Inference Server
./06_prepare_model.sh
# Run TensorRT Inference Server
./07_run_tensorrt_inference_server.sh
#  Inference example using ONNX and TensorRT model with TensorRT Inference Server.
python 08_tensorrt_inferense_server_client.py
```


## onnx_model_cli
onnx_model_cli helps checking onnx model structure.

Inspired by tensorflow [saved_model_cli](https://www.tensorflow.org/guide/saved_model#details_of_the_savedmodel_command_line_interface).

### Usage
```
python onnx_model_cli.py show --path foo.onnx
```

## Great Links
- [TensorRT Inference Server official](https://github.com/triton-inference-server/server/tree/r19.10)
    - [client sample](https://github.com/triton-inference-server/server/tree/r19.10/docs/examples/model_repository)
- [onnx-tensorrt official](https://github.com/onnx/onnx-tensorrt)
- [NVIDIA Triton Inference Server で推論してみた](https://qiita.com/dcm_yamaya/items/985a57598d516e77894f)
