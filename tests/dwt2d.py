
import numpy as np
import pywt
import cv2
from pathlib import Path
import json
import argparse
import os
import contextlib
import shutil




def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     dest='image_filename',
    #     help='Image',
    # )
    # parser.add_argument(
    #     'wavelet', 'w',
    #     dest='wavelet',
    #     choices=pywt.wavelist(),
    #     required=True,
    #     help='Wavelet',
    # )
    # parser.add_argument(
    #     '--mode', '-m',
    #     action='append',
    #     default=['symmetric'],
    #     type=list,
    #     dest='modes',
    #     help='Boundary mode used to generate test cases',
    # )
    # parser.add_argument(
    #     '--dry-run', '-d',
    #     action='store_true',
    #     help='Generate test cases, but do not save files',
    # )
    # parser.add_argument(
    #     '--verbose', '-v',
    #     action='store_true',
    #     help='Print manifest entries generation',
    # )
    # parser.add_argument(
    #     '--manifest',
    #     default='cases_manifest.json',
    #     help='The JSON manifest path',
    # )
    # parser.add_argument(
    #     '--root',
    #     default='data',
    #     help='The root path for generated test case data',
    # )
    # args = parser.parse_args()

    input_filename = 'inputs/lena.png'
    wavelet = 'db1'

    original = cv2.imread(input_filename)
    print(original.shape, original.dtype)
    print(original.max(), original.min())
    print('original:')
    print('    type =', original.dtype)
    print('    min =', original.min(), 'max =', original.max())



    axes = (0, 1)

    image = original.astype(np.float32)
    image = image / 255
    # image = (image - np.mean(image)) / np.std(image)

    print('image:')
    print('    type =', image.dtype)
    print('    min =', image.min(), 'max =', image.max())
    print()

    coeffs = pywt.wavedec2(image, wavelet, level=2, axes=axes)
    dwt_image, slices = pywt.coeffs_to_array(coeffs, axes=axes)

    h = 2 * np.max(np.abs(dwt_image))

    print('dwt_image:')
    print('    dwt_image =', image.dtype, image.shape)
    print('    min =', dwt_image.min(), 'max =', dwt_image.max())
    print('    h =', h)
    print()

    final_image = (dwt_image / h) + 0.5
    final_image = 255 * final_image
    final_image = final_image.astype(original.dtype)

    print('final_image:')
    print('    type =', final_image.dtype, final_image.shape)
    print('    min =', final_image.min(), 'max =', final_image.max())
    print()

    cv2.imshow('Image', original)
    cv2.imshow('DWT', final_image)
    cv2.waitKey(0)



if __name__ == '__main__':
    # inputs = [
    #     np.ones([32, 32], dtype=np.float32),
    #     ('inputs/lena_gray.png', cv2.IMREAD_GRAYSCALE),
    # ]
    main()



