
import numpy as np
import pywt
import cv2
from pathlib import Path
import json
import argparse
import os
import contextlib
import shutil
import matplotlib.pyplot as plt




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


def test_pattern_zeros(wavelet, shape, dtype, border_mode):
    pattern = np.zeros(shape, dtype=dtype)
    run_test('zeros', pattern, wavelet, border_mode)


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




def run_test(title, pattern, wavelet, border_mode, level=None):
    np.set_printoptions(linewidth=240, precision=2)

    axes = (0, 1)
    coeffs = pywt.wavedec2(pattern, wavelet, axes=axes, mode=border_mode, level=level)
    coeffs, _ = pywt.coeffs_to_array(coeffs, axes=axes)

    print(title)
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


def sure(x, t, stdev):
    y = abs(x / stdev)
    thresh = t / stdev
    return  len(y) + np.sum(np.minimum(y, thresh)**2) - 2 * np.sum(y <= thresh)



def plot_sure():
    n = 512 * 512
    x = np.linspace(0, 2 * np.pi, n)
    noise = 0.9 * np.random.randn(len(x))
    # noise = 0
    # y = np.sin(x) + noise
    # y = np.sin(x) + np.sin(2 * x) + noise
    w = np.linspace(0.5, 10, len(x))
    y = np.sin(w * x) + noise

    stdev = np.std(y)

    y = y / stdev
    risk = np.zeros_like(y)
    a = np.zeros_like(y)
    b = np.zeros_like(y)
    # thresholds = sorted(abs(y)
    print(max(abs(y)))

    thresholds = np.linspace(0, max(abs(y)), len(y))
    # thresholds = thresholds / stdev
    for i, t in enumerate(thresholds):
        risk[i] = sure(y, t, 1)
        # a[i] = np.sum(np.minimum(y, t)**2)
        # b[i] = -2 * np.sum(y <= t)


    # plt.plot(x, y)
    plt.plot(thresholds, risk)
    # plt.plot(thresholds, a)
    # plt.plot(thresholds, b)



if __name__ == '__main__':
    # inputs = [
    #     np.ones([32, 32], dtype=np.float32),
    #     ('inputs/lena_gray.png', cv2.IMREAD_GRAYSCALE),
    # ]
    plot_sure()
    plt.show()
    exit()
    wavelet = 'db1'
    dtype = np.float32
    shape = [16, 16]
    border_mode = 'reflect'
    level = None

    wavelet = pywt.Wavelet('db1')

    # print(wavelet.filter_bank)

    # test_pattern_zeros(wavelet, shape, dtype, border_mode, level=level)
    # test_pattern_ones(wavelet, shape, dtype, border_mode, level=level)
    # test_pattern_horizontal_lines(wavelet, shape, dtype, border_mode, level=level)
    # test_pattern_horizontal_lines2(wavelet, shape, dtype, border_mode, level=level)
    # test_pattern_vertical_lines(wavelet, shape, dtype, border_mode, level=level)
    # test_pattern_vertical_lines2(wavelet, shape, dtype, border_mode, level=level)
    # test_pattern_checkerboard(wavelet, shape, dtype, border_mode, level=level)
    # test_pattern_checkerboard2(wavelet, shape, dtype, border_mode, level=level)


    # x = -7 + np.arange(16)
    # stdev = np.std(x)

    x = 1 + np.arange(16)
    stdev = np.std(x)


    # g = [0, 7, 15]
    # h = [-0.5, 0, 0.5]
    # offsets = set()
    # for w in g:
    #     for i in h:
    #         offsets.add(w + i)

    # offsets = sorted(offsets)

    # x = np.ones(16)
    # offsets = np.linspace(0, 1, 5)
    # stdev = 1

    # x = np.zeros(16)
    # offsets = np.linspace(0, 1, 5) - 1
    # stdev = 1



    # print('x =', x)
    # print('stdev =', stdev)
    # print('t = 0.0: ', sure(x, 0, stdev))
    # print()
    # for i in offsets:
    #     t = i + x[0]
    #     t = i - 1
    #     t = i
    #     print(f't = {t:0.1f}: ', sure(x, t, stdev))


    print('x =', x)
    print('stdev =', stdev)
    print()
    for t in x:
        print(f't = {t}: ', sure(x, t, stdev))


    # print('x =', x)
    # print('stdev =', stdev)
    # print()
    # for i in offsets:
    #     t = i
    #     print(f't = {t}: ', sure(x, t, stdev))

    # print('t = 0.5: ', sure(x, 0.5, stdev))
    # print('t = 1.0: ', sure(x, 1.0, stdev))
    # print('t = 1.5: ', sure(x, 1.5, stdev))
    # print('t = 7.5: ', sure(x, 7.5, stdev))
    # print('t = 8.0: ', sure(x, 8.0, stdev))
    # print('t = 8.5: ', sure(x, 8.5, stdev))
    # print('t = 15.5:', sure(x, 15.5, stdev))
    # print('t = 16.0:', sure(x, 16.0, stdev))
    # print('t = 16.5:', sure(x, 16.5, stdev))

    # main()



