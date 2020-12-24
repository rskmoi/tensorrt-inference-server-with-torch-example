import click
import onnx
import os


@click.group()
def cli():
    pass


@cli.command()
@click.option("--path")
@click.option("--input_index_list")
@click.option("--input_pattern")
def show(path, input_index_list, input_pattern):
    dtype_dict = {}
    for k, v in onnx.TensorProto.DataType.items():
        dtype_dict[v] = k

    model = onnx.load(path)
    model_file_name = os.path.basename(path)
    model_input_info_list = _get_input_info(model, dtype_dict, index_list=input_index_list, pattern=input_pattern)
    model_output_info_list = _get_output_info(model, dtype_dict)

    print("File: {}".format(model_file_name))
    for _input_info in model_input_info_list:
        print("    inputs['{}'] info:".format(_input_info["name"]))
        print("        dtype: {}".format(_input_info["dtype"]))
        print("        shape: {}".format(_input_info["shape"]))

    for _output_info in model_output_info_list:
        print("    outputs['{}'] info:".format(_output_info["name"]))
        print("        dtype: {}".format(_output_info["dtype"]))
        print("        shape: {}".format(_output_info["shape"]))


def _get_input_info(model, dtype_dict, index_list=None, pattern=None):
    if pattern:
        info_list = []
        for _input in model.graph.input:
            if pattern not in _input.name:
                continue
            _info = _get_info_from_value_info(_input, dtype_dict)
            info_list.append(_info)
        return info_list

    if index_list is None:
        index_list = [0]

    info_list = []
    for index in index_list:
        _info = _get_info_from_value_info(model.graph.input[index], dtype_dict)
        info_list.append(_info)
    return info_list


def _get_output_info(model, dtype_dict):
    info_list = []
    for _output in model.graph.output:
        _info = _get_info_from_value_info(_output, dtype_dict)
        info_list.append(_info)
    return info_list


def _get_info_from_value_info(value_info, dtype_dict):
    name = value_info.name
    dtype_str = dtype_dict[value_info.type.tensor_type.elem_type]
    shape_str = _get_shape_str(value_info)
    return {"name": name, "dtype": dtype_str, "shape": shape_str}


def _get_shape_str(value_info):
    shapes = []
    for _dim in value_info.type.tensor_type.shape.dim:
        if _dim.dim_value != 0:
            shapes.append(str(_dim.dim_value))
        elif _dim.dim_param != "":
            shapes.append(_dim.dim_param)
    return "[{}]".format(", ".join(shapes))


@cli.command()
def run():
    print("will be implemented...")


if __name__ == '__main__':
    cli()