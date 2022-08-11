import numpy as np
from PIL import Image
import cv2
from io import BytesIO
import random

def transform_JPEGcompression(image, compress_range = (30, 100)):
    '''
        Perform random JPEG Compression
    '''
    if random.random() < 0.15:
        assert compress_range[0] < compress_range[1], "Lower and higher value not accepted: {} vs {}".format(compress_range[0], compress_range[1])
        jpegcompress_value = random.randint(compress_range[0], compress_range[1])
        out = BytesIO()
        image.save(out, 'JPEG', quality=jpegcompress_value)
        out.seek(0)
        rgb_image = Image.open(out)
        return rgb_image
    else:
        return image

def transform_gaussian_noise(img_pil, mean = 0.0, var = 10.0):
    '''
        Perform random gaussian noise
    '''
    if random.random() < 0.15:
        img = np.array(img_pil)
        height, width, channels = img.shape
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma,(height, width, channels))
        noisy = img + gauss
        cv2.normalize(noisy, noisy, 0, 255, cv2.NORM_MINMAX, dtype=-1)
        noisy = noisy.astype(np.uint8)
        return Image.fromarray(noisy)
    else:
        return img_pil

def transform_resize(image, resize_range = (32, 112), target_size = 112):
    if random.random() < 0.15:
        assert resize_range[0] < resize_range[1], "Lower and higher value not accepted: {} vs {}".format(resize_range[0], resize_range[1])
        resize_value = random.randint(resize_range[0], resize_range[1])
        resize_image = image.resize((resize_value, resize_value))
        return resize_image.resize((target_size, target_size))
    else:
        return image

def transform_eraser(image):
    if random.random() < 0.15:
        mask_range = random.randint(0, 3)
        image_array = np.array(image, dtype=np.uint8)
        image_array[(7-mask_range)*16:, :, :] = 0
        return Image.fromarray(image_array)
    else:
        return image