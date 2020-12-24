import numpy as np
from tools import defaults
from tools import data_util
from tensorrtserver.api import ServerHealthContext, ServerStatusContext, InferContext, ProtocolType


class TISClient:
    def __init__(self, protocol_type, fqdn):
        assert protocol_type in [ProtocolType.HTTP, ProtocolType.GRPC]
        self.protocol_type = protocol_type
        self._fqdn = fqdn
        if protocol_type == ProtocolType.HTTP:
            self._port = "8000"
        elif protocol_type == ProtocolType.GRPC:
            self._port = "8001"

    @property
    def url(self):
        return f"{self._fqdn}:{self._port}"

    def health(self):
        # Create a health context, get the ready and live state of server.
        health_ctx = ServerHealthContext(self.url, self.protocol_type)
        return health_ctx

    def status(self):
        # Create a status context and get server status
        status_ctx = ServerStatusContext(self.url, self.protocol_type)
        return status_ctx

    def infer(self, input_images, model_name, model_version=-1):
        # Create the inference context for the model.
        infer_ctx = InferContext(self.url, self.protocol_type, model_name, model_version)
        batch_size = len(input_images)
        # Send inference request to the inference server. Get results for
        # both output tensors.
        result = infer_ctx.run({'input_0': input_images},
                               {'output_0': InferContext.ResultFormat.RAW},
                               batch_size)

        outputs = result["output_0"]
        return outputs


def inference(protocol_type, model_name):
    fqdn = "localhost"
    client = TISClient(protocol_type, fqdn)
    # health check
    health_ctx = client.health()
    print("Live: {}".format(health_ctx.is_live()))
    print("Ready: {}".format(health_ctx.is_ready()))
    # status check
    status_ctx = client.status()
    print(status_ctx.get_server_status())
    # inference
    input_images = [data_util.get_transformed_array(_color).copy().squeeze() for _color in defaults.COLORS]
    outputs = client.infer(input_images, model_name)

    for _output, _color in zip(outputs, defaults.COLORS):
        pred = np.argmax(_output)
        assert defaults.COLORS[pred] == _color
        print("Great! Label = {}, Pred = {}.".format(defaults.COLORS[pred], _color))


if __name__ == '__main__':
    for model_name in ["onnx_example", "trt_example"]:
        inference(ProtocolType.HTTP, model_name)
        inference(ProtocolType.GRPC, model_name)