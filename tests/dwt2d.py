
import numpy as np
import pywt
import cv2
from pathlib import Path
import json
import argparse
import os
from collections import abc
import contextlib
import shutil
import textwrap
import mpmath
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



def ss(k, n, m):
    k += 1
    q = 2**(-k)
    return q * (n - m) + m

def s(k, w, L, m):
    k += 1
    q = 2**(L - k - 1) / (2**(L - 1) - 1)
    return q * (w - L * m) + m

def t(k, w, L, m):
    k += 1
    q = 2**(L - k) * ((2**(k - 1) - 1) / (2**(L - 1) - 1))
    return q * (w - L * m) + k * m



if __name__ == '__main__':
    # inputs = [
    #     np.ones([32, 32], dtype=np.float32),
    #     ('inputs/lena_gray.png', cv2.IMREAD_GRAYSCALE),
    # ]
    # plot_sure()
    # plt.show()


    x = np.zeros([128, 128])
    coeffs = pywt.wavedec2(x, 'db4', level=3)
    coeffs, slices = pywt.coeffs_to_array(coeffs)

    print(coeffs.shape)

    exit()

    def mad(x):
        m = np.median(x)
        # print(m)
        return np.median(np.abs(x - m))

    def mad_std(x):
        return mad(x) / 0.675


    def get_subband(x, slices):
        return x[..., slices[0], slices[1]]

    def bayes_threshold(coeffs, noise_std):
        noise_variance = noise_std**2
        obs_variance = np.sum(coeffs**2) / coeffs.size

        sig_variance = max(obs_variance - noise_variance, 0)
        if sig_variance == 0:
            return np.max(abs(coeffs))

        return noise_variance / np.sqrt(sig_variance)

    def subband_bayes_threshold(x, subband_slices, noise_std):
        return {
            name: np.array([
                bayes_threshold(get_subband(x[i], rect_slices), noise_std[i])
                for i in range(4)
            ])
            for name, rect_slices in subband_slices.items()
        }


    def levelwise_bayes_threshold(x, subband_slices, noise_std):
        def get_level_coeffs(i, level):
            return np.concatenate([
                get_subband(x[i], subband_slices[f'h{level}']).ravel(),
                get_subband(x[i], subband_slices[f'v{level}']).ravel(),
                get_subband(x[i], subband_slices[f'd{level}']).ravel(),
            ])
        return {
            f'level{level}': np.array([
                bayes_threshold(get_level_coeffs(i, level), noise_std[i])
                for i in range(4)
            ])
            for level in range(levels)
        }


    def global_bayes_threshold(x, subband_slices, noise_std, levels):
        def get_global_coeffs(i):
            return np.concatenate([
                get_subband(x[i], s).ravel()
                for name, s in subband_slices.items()
                if int(name[-1]) in levels
            ])
        threshold = np.array([
            bayes_threshold(get_global_coeffs(i), noise_std[i])
            for i in range(4)
        ])
        return {'global': threshold}


    levels = 3
    n = 16
    decimals = 4
    fmt = f'0.{decimals}f'
    fmt2 = '0.16f'

    x = np.random.randn(4, n, n)
    # x = np.linspace(0, 1, 4 * n * n).reshape(4, n, n)
    # x = np.arange(4 * n * n).reshape(4, n, n)
    x = np.round(x, decimals)

    subband_slices = dict()
    for i in range(levels):
        k = n // 2**i
        j = k // 2
        subband_slices[f'h{i}'] = (slice(j, k), slice(0, j))
        subband_slices[f'v{i}'] = (slice(0, j), slice(j, k))
        subband_slices[f'd{i}'] = (slice(j, k), slice(j, k))

    for i in range(x.shape[-2]):
        print(f'//  row {i}')
        for j in range(x.shape[-1]):
            print(f'cv::Scalar({x[0, i, j]:{fmt}}, {x[1, i, j]:{fmt}}, {x[2, i, j]:{fmt}}, {x[3, i, j]:{fmt}}), ')
    print()

    noise_std = np.array([
        mad_std(get_subband(x[0], subband_slices['d0'])),
        mad_std(get_subband(x[1], subband_slices['d0'])),
        mad_std(get_subband(x[2], subband_slices['d0'])),
        mad_std(get_subband(x[3], subband_slices['d0'])),
    ])
    print(f'cv::Scalar({noise_std[0]:{fmt2}}, {noise_std[1]:{fmt2}}, {noise_std[2]:{fmt2}}, {noise_std[3]:{fmt2}})')
    print()

    def doit(compute_thresholds, *args):
        thresholds = compute_thresholds(x, subband_slices, noise_std, *args)
        for name, t in thresholds.items():
            print(f'cv::Scalar({t[0]:{fmt2}}, {t[1]:{fmt2}}, {t[2]:{fmt2}}, {t[3]:{fmt2}}),')
        print()

    print('//  Subband Partition')
    doit(subband_bayes_threshold)
    print('//  Level Partition')
    doit(levelwise_bayes_threshold)
    print('//  All Levels')
    doit(global_bayes_threshold, [0, 1, 2])
    print('//  First Level')
    doit(global_bayes_threshold, [0])
    print('//  First Two Levels')
    doit(global_bayes_threshold, [0, 1])
    print('//  Last Level')
    doit(global_bayes_threshold, [2])
    print('//  Last Two Levels')
    doit(global_bayes_threshold, [1, 2])

    exit()

    # image = cv2.imread('inputs/lena.png', cv2.IMREAD_COLOR)
    # image = image / 255
    # wavelet = pywt.Wavelet('db1')
    # coeffs = pywt.wavedec2(image, wavelet, axes=(0, 1), level=1)
    # coeffs, _ = pywt.coeffs_to_array(coeffs, axes=(0, 1))

    # # print(coeffs)
    # print(coeffs.shape)
    # print(image.shape)

    # cv2.imshow('image', image)
    # cv2.imshow('coeffs', coeffs)

    # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
    # exit()

    def transform(image, wavelet, level=None):
        axes = (0, 1)
        # coeffs = pywt.wavedec2(image, wavelet, level=2, axes=axes)
        coeffs = pywt.wavedec2(image, wavelet, level=level, axes=axes)
        dwt_image, slices = pywt.coeffs_to_array(coeffs, axes=axes)

        return dwt_image


    def get_sizes_and_offsets(n, L, wavelet):
        sizes = list()
        for i in range(L):
            sizes.append(
                pywt.dwt_coeff_len(n, wavelet, 'symmetric')
            )
            n = sizes[-1]

        sizes = np.array(sizes)
        offsets = np.cumsum(sizes)

        return sizes, offsets

    np.set_printoptions(linewidth=2000)
    wavelet = pywt.Wavelet('db6')

    n = 1024
    # filter_length = wavelet.dec_len
    filter_length = 10
    m = filter_length - 1
    # filter_length = m + 1
    L = pywt.dwt_max_level(n, filter_length)
    z = n - m

    t = n
    W = 0
    for k in range(L):
        t = pywt.dwt_coeff_len(t, filter_length, mode='symmetric')
        W += t
    W += t

    # W2 = transform(np.zeros([n, n]), wavelet, L).shape[0]

    sizes, offsets = get_sizes_and_offsets(n, L, wavelet)
    w = offsets[-1]
    print('sizes   =', sizes)
    print('offsets =', offsets)
    print('n =', n)
    # print('L =', L, 'm =', m, 'W =', W, 'W2 =', W2)
    print('L =', L, 'm =', m, 'W =', W)
    print('w =', w)
    print()

    def print_floor_series(y):
        y_bits = np.array([int(x) for x in bin(y)[2:]])
        terms = np.array([np.floor(y / 2**k) for k in range(1, len(y_bits))])
        series = np.sum(terms)
        hammond_dist = np.sum(y_bits)
        print('y =', y, ' y_hat =', hammond_dist + series)
        print(y_bits, 'len =', len(y_bits))
        print('series =', series)
        print('hammond_dist =', hammond_dist)


    # print(np.floor(1 + np.log2(z)), L)
    # print_floor_series(z)
    # print()

    # y = 100
    # print_floor_series(y)
    # print(2**L * m, n)
    # print(W - m // 2)
    # print(W - n)

    print(W - W / 2)
    print(W - W / 4)
    print(W - W / 8)
    print(W - W // 2**(L-1))













    exit()
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



