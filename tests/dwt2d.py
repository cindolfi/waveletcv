
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




def make_checkerboard(shape, dtype):
    return np.tile(
        np.array([[0, 1], [1, 0]], dtype=dtype),
        (shape[0] // 2, shape[1] // 2),
    )

def make_horizontal_lines(shape, dtype):
    return np.tile(
        np.array([[1, 1], [0, 0]], dtype=dtype),
        (shape[0] // 2, shape[1] // 2),
    )


def make_vertical_lines(shape, dtype):
    return np.tile(
        np.array([[1, 0], [1, 0]], dtype=dtype),
        (shape[0] // 2, shape[1] // 2),
    )


def test_pattern_ones(wavelet, shape, dtype, border_mode):
    pattern = np.ones(shape, dtype=dtype)
    run_test('ones', pattern, wavelet, border_mode)


def test_pattern_checkerboard(wavelet, shape, dtype, border_mode):
    pattern = make_checkerboard(shape, dtype)
    run_test('checkerboard', pattern, wavelet, border_mode)


def test_pattern_checkerboard2(wavelet, shape, dtype, border_mode):
    pattern = make_checkerboard(shape, dtype)
    run_test('checkerboard2', 1 - pattern, wavelet, border_mode)


def test_pattern_vertical_lines(wavelet, shape, dtype, border_mode):
    pattern = make_vertical_lines(shape, dtype)
    run_test('vertical_lines', pattern, wavelet, border_mode)


def test_pattern_vertical_lines2(wavelet, shape, dtype, border_mode):
    pattern = make_vertical_lines(shape, dtype)
    run_test('vertical_lines2', 1 - pattern, wavelet, border_mode)


def test_pattern_horizontal_lines(wavelet, shape, dtype, border_mode):
    pattern = make_horizontal_lines(shape, dtype)
    run_test('horizontal_lines', pattern, wavelet, border_mode)


def test_pattern_horizontal_lines2(wavelet, shape, dtype, border_mode):
    pattern = make_horizontal_lines(shape, dtype)
    run_test('horizontal_lines2', 1 - pattern, wavelet, border_mode)




def run_test(title, pattern, wavelet, border_mode):
    np.set_printoptions(linewidth=240, precision=2)

    axes = (0, 1)
    coeffs = pywt.wavedec2(pattern, wavelet, axes=axes, mode=border_mode)
    coeffs, _ = pywt.coeffs_to_array(coeffs, axes=axes)

    print(title)
    print('wavelet =', wavelet)
    print('pattern =')
    print(pattern)
    print(pattern.shape)
    print()
    print('coeffs =')
    print(coeffs)
    print(coeffs.shape)
    print()
    print('-' * 80)
    print()

if __name__ == '__main__':
    # inputs = [
    #     np.ones([32, 32], dtype=np.float32),
    #     ('inputs/lena_gray.png', cv2.IMREAD_GRAYSCALE),
    # ]
    wavelet = 'db1'
    dtype = np.float32
    shape = [16, 16]
    border_mode = 'reflect'

    wavelet = pywt.Wavelet('db1')

    print(wavelet.filter_bank)

    # test_pattern_ones(wavelet, shape, dtype, border_mode)
    # test_pattern_vertical_lines(wavelet, shape, dtype, border_mode)
    # test_pattern_vertical_lines2(wavelet, shape, dtype, border_mode)
    # test_pattern_horizontal_lines(wavelet, shape, dtype, border_mode)
    # test_pattern_horizontal_lines2(wavelet, shape, dtype, border_mode)
    # test_pattern_checkerboard(wavelet, shape, dtype, border_mode)
    # test_pattern_checkerboard2(wavelet, shape, dtype, border_mode)

    # main()



