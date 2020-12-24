import numpy as np
import onnxruntime
import tools.defaults as defaults
import tools.data_util as data_util


def inference(onnx_model_path):
    sess = onnxruntime.InferenceSession(onnx_model_path)
    colors = defaults.COLORS
    for color in colors:
        input_array = data_util.get_transformed_array(color)
        output = sess.run(['output_0'], {'input_0': input_array})
        pred = np.argmax(np.squeeze(output))

        assert defaults.COLORS[pred] == color
        print("Great! Label = {}, Pred = {}.".format(defaults.COLORS[pred], color))


if __name__ == '__main__':
    inference(defaults.ONNX_MODEL_PATH)