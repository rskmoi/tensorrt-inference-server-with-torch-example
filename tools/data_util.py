from PIL import Image
import numpy as np

def get_image(color, image_size=64):
    return Image.new("RGB", (image_size, image_size), color=color)

def transform(image_array: np.ndarray):
    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.
    image_array = image_array.astype(np.float32)
    return image_array

def get_transformed_array(color, image_size=64):
    img = get_image(color, image_size)
    return transform(np.asarray(img))