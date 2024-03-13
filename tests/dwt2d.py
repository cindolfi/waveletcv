
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







INDENT = 4 * ' '
OUTPUT_FILENAME = 'dwt_test_cases.hpp'
COEFFICIENT_PRECISION = 20

class PatternsTest:
    def __init__(
            self,
            wavelets: abc.Sequence[str | pywt.Wavelet] | str | pywt.Wavelet,
            patterns: dict[str, np.ndarray],
            border_mode: str = 'reflect',
            levels: abc.Sequence[int] | int | None = None,
            dtype: np.dtype = np.float64,
            precision: int = 8,
            zero_tolerance: float = 1e-7,
            base_variable_name: str = 'dwt2d_test_cases',
            header_guard: str = 'WAVELET_DWT2D_TEST_HPP',
        ):
        if isinstance(wavelets, str):
            wavelets = [pywt.Wavelet(wavelets)]
        elif isinstance(wavelets, pywt.Wavelet):
            wavelets = [wavelets]
        else:
            wavelets = [
                pywt.Wavelet(wavelet) if isinstance(wavelet, str) else wavelet
                for wavelet in wavelets
            ]

        if not isinstance(levels, (list, tuple)):
            levels = [levels]

        self.wavelets = wavelets
        self.patterns = patterns
        self.dtype = dtype
        self.border_mode = border_mode
        self.levels = levels
        self.precision = precision
        self.zero_tolerance = zero_tolerance
        self.base_variable_name = base_variable_name
        self.header_guard = header_guard
        self._indent = 0


    @staticmethod
    def make_horizontal_lines(shape, inverted=False, dtype=np.float64):
        pattern = np.tile(
            np.array([[1, 1], [0, 0]], dtype=dtype),
            (shape[0] // 2, shape[1] // 2),
        )
        return pattern if not inverted else 1 - pattern


    @staticmethod
    def make_vertical_lines(shape, inverted=False, dtype=np.float64):
        pattern = np.tile(
            np.array([[1, 0], [1, 0]], dtype=dtype),
            (shape[0] // 2, shape[1] // 2),
        )
        return pattern if not inverted else 1 - pattern


    @staticmethod
    def make_diagonal_lines(shape, inverted=False, dtype=np.float64):
        pattern = np.tile(
            np.array([[1, 0], [0, 1]], dtype=dtype),
            (shape[0] // 2, shape[1] // 2),
        )
        return pattern if not inverted else 1 - pattern


    def transform_pattern(self, wavelet, pattern, level):
        axes = (0, 1)
        coeffs = pywt.wavedec2(
            pattern,
            wavelet,
            axes=axes,
            mode=self.border_mode,
            level=level,
        )
        coeffs = pywt.coeffs_to_array(coeffs, axes=axes)[0]
        self._clamp_zero(coeffs, self.zero_tolerance)
        return self._trim_coeffs_to_match_pattern(wavelet, pattern, coeffs)


    def _clamp_zero(self, coeffs, zero_tolerance):
        if zero_tolerance > 0:
            coeffs[abs(coeffs) <= zero_tolerance] = 0


    def _trim_coeffs_to_match_pattern(self, wavelet, pattern, coeffs):
        if coeffs.shape != pattern.shape:
            padding = np.asarray(coeffs.shape) - np.asarray(pattern.shape)
            front_padding = padding // 2
            back_padding = front_padding + (padding % 2)

            trimmed_coeffs = coeffs[
                front_padding[0]: -back_padding[0],
                front_padding[1]: -back_padding[1],
            ]
            assert trimmed_coeffs.shape == pattern.shape, \
                f'\ncoeffs.shape = {coeffs.shape}\npattern.shape = {pattern.shape}\ntrimmed_coeffs.shape = {trimmed_coeffs.shape}\nwavelet =\n{wavelet}'
        else:
            trimmed_coeffs = coeffs

        return trimmed_coeffs


    def run(self):
        def print_run(wavelet, pattern_name, pattern, coeffs):
            print(wavelet.name, '-', pattern_name)
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

        for wavelet in self.wavelets:
            for pattern_name, pattern in self.patterns.items():
                coeffs = self.transform_pattern(wavelet, pattern)
                print_run(wavelet, pattern_name, pattern, coeffs)


    @contextlib.contextmanager
    def indent(self):
        self._indent += 4
        yield
        self._indent -= 4


    def write(self, *lines):
        print(textwrap.indent('\n'.join(lines), self._indent * ' '))


    def generate_test_cases_hpp(self, filename):
        with open(filename, 'w') as file:
            with contextlib.redirect_stdout(file):
                self.write(
                    f'#ifndef {self.header_guard}',
                    f'#define {self.header_guard}',
                    '',
                    '#include <vector>',
                    '#include <map>',
                    '#include <string>',
                    '',
                )
                self._write_input_patterns()
                self._write_all_test_cases()
                self.write(f'#endif  // {self.header_guard}')
                self.write()


    def _write_input_patterns(self):
        variable_name = f'{self.base_variable_name}_inputs'
        self.write(f'std::map<std::string, std::vector<double>> {variable_name} = {{')
        for pattern_name, pattern in self.patterns.items():
            with self.indent():
                self._write_name_array_pair(
                    name=pattern_name,
                    array=pattern,
                    comment=f'{variable_name}["{pattern_name}"]',
                )
        self.write(f'}}; // {variable_name}')
        self.write()


    def _write_all_test_cases(self):
        for level in self.levels:
            self.write(f'//  {"=" * 76}')
            self.write()
            self._write_test_cases_for_level(level)
            self.write()


    def _write_test_cases_for_level(self, level):
        if level is None:
            variable_name = f'{self.base_variable_name}_all_levels'
        else:
            variable_name = f'{self.base_variable_name}_{level}_levels'
        self.write(
            f'std::map<std::string, std::map<std::string, std::vector<double>>> {variable_name} = {{'
        )
        for wavelet in self.wavelets:
            with self.indent():
                self.write(f'//  {"-" * (76 - self._indent)}')
                self.write(f'{{  //  {variable_name}["{wavelet.name}"]')
                with self.indent():
                    self.write(f'"{wavelet.name}",',)
                    self.write('{')
                    with self.indent():
                        for pattern_name, pattern in self.patterns.items():
                            self._write_single_test_case(
                                wavelet=wavelet,
                                pattern_name=pattern_name,
                                pattern=pattern,
                                level=level,
                                variable_name=variable_name,
                            )
                    self.write('},')
                self.write(f'}}, //  {variable_name}["{wavelet.name}"]')
        self.write(f'}}; // {variable_name}')


    def _write_single_test_case(
            self,
            wavelet,
            pattern_name,
            pattern,
            level,
            variable_name,
        ):
        self._write_name_array_pair(
            name=pattern_name,
            array=self.transform_pattern(wavelet, pattern.astype(self.dtype), level),
            comment=f'{variable_name}["{wavelet.name}"]["{pattern_name}"]',
        )


    def _write_name_array_pair(self, name, array, comment):
        self.write(f'{{  // {comment}')
        with self.indent():
            self.write(f'"{name}",')
            self._write_array(array)

        self.write('},')


    def _write_array(self, array):
        array = np.array2string(
            array,
            max_line_width=2000,
            precision=self.precision,
            suppress_small=True,
            separator=', ',
        )
        self.write('{')
        with self.indent():
            self.write(array.replace('[[', '').replace(' [', '').replace(']', ''))
        self.write('},')



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
    # plot_sure()
    # plt.show()

    shape = [16, 16]
    patterns_test = PatternsTest(
        wavelets=[
            'db2',
            'db3',
            # 'db4',
        ],
        patterns = dict(
            zeros=np.zeros(shape),
            ones=np.ones(shape),
            horizontal_lines=PatternsTest.make_horizontal_lines(shape),
            inverted_horizontal_lines=PatternsTest.make_horizontal_lines(shape, inverted=True),
            vertical_lines=PatternsTest.make_vertical_lines(shape),
            inverted_vertical_lines=PatternsTest.make_vertical_lines(shape, inverted=True),
            diagonal_lines=PatternsTest.make_diagonal_lines(shape),
            inverted_diagonal_lines=PatternsTest.make_diagonal_lines(shape, inverted=True),
        ),
        dtype=np.float64,
        border_mode='reflect',
        precision=8,
        levels=[1, None],
    )

    # np.set_printoptions(linewidth=240, precision=2)
    # patterns_test.run()
    patterns_test.generate_test_cases_hpp(
        filename=OUTPUT_FILENAME,
    )

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



