"""
Generates test case for wavelet unit tests
"""
from collections import abc
from dataclasses import dataclass
import json

import numpy as np
import pywt


class CoefficientTestCases:
    def __init__(self, families: abc.Sequence[str] | str):
        if isinstance(families, str):
            families = [families]

        self.families = families


    def generate(self, filename):
        wavelets = [
            pywt.Wavelet(name)
            for family in self.families
            for name in pywt.wavelist(family)
        ]

        with open(filename, 'w') as file:
            json.dump(
                wavelets,
                file,
                cls=TestCaseJsonEncoder,
            )




@dataclass
class TestCase:
    wavelet: str
    patterns: str
    levels: abc.Sequence[int]

@dataclass
class DWT2DParam:
    wavelet_name: str
    input_name: str
    levels: int
    coeffs: np.ndarray

class DWT2DTestCases:
    def __init__(
            self,
            test_cases: abc.Sequence[TestCase],
            patterns: dict[str, dict[str, np.ndarray]],
            border_mode: str = 'reflect',
            dtype: np.dtype = np.float64,
        ):
        self.test_cases = test_cases
        self.patterns = patterns
        self.dtype = dtype
        self.border_mode = border_mode


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


    def generate(self, filename):
        def make_input_name(set_name, name):
            return f'{set_name}_{name}'

        inputs = {
            make_input_name(pattern_set_name, pattern_name): pattern
            for pattern_set_name, patterns in self.patterns.items()
            for pattern_name, pattern in patterns.items()
        }

        test_cases = list()
        for test_case in self.test_cases:
            wavelet = pywt.Wavelet(test_case.wavelet)
            for levels in test_case.levels:
                for pattern_name in self.patterns[test_case.patterns]:
                    input_name = make_input_name(test_case.patterns, pattern_name)
                    test_cases.append(
                        DWT2DParam(
                            wavelet_name=wavelet.name,
                            input_name=input_name,
                            levels=levels,
                            coeffs=self.transform_pattern(wavelet, inputs[input_name], levels),
                        )
                    )

        with open(filename, 'w') as file:
            json.dump(
                dict(inputs=inputs, test_cases=test_cases),
                file,
                cls=TestCaseJsonEncoder,
            )


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

        return coeffs.astype(self.dtype)




class TestCaseJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pywt.Wavelet):
            result = dict(
                vanishing_moments_psi=obj.vanishing_moments_psi,
                vanishing_moments_phi=obj.vanishing_moments_phi,
                orthogonal=obj.orthogonal,
                biorthogonal=obj.biorthogonal,
                symmetry=obj.symmetry,
                family=obj.family_name,
                name=obj.name,
                decompose_lowpass=obj.dec_lo,
                decompose_highpass=obj.dec_hi,
                reconstruct_lowpass=obj.rec_lo,
                reconstruct_highpass=obj.rec_hi,
            )
        elif isinstance(obj, DWT2DParam):
            result = dict(
                wavelet_name=obj.wavelet_name,
                input_name=obj.input_name,
                coeffs=obj.coeffs,
                levels=obj.levels,
            )
        elif isinstance(obj, np.ndarray):
            result = dict(
                shape=obj.shape,
                dtype=str(obj.dtype),
                data=obj.ravel().tolist(),
            )
        else:
            result = super().default(obj)

        return result




def main():
    #   ------------------------------------------------------------------------
    coeffs_test_cases = CoefficientTestCases(
        families=[
            'haar',
            'db',
            'sym',
            'coif',
            'bior',
            # 'rbior',
        ],
    )
    coeffs_test_cases.generate('wavelet_test_data.json')

    #   ------------------------------------------------------------------------
    def create_patterns(shape):
        return dict(
            zeros=np.zeros(shape, dtype=np.float64),
            ones=np.ones(shape, dtype=np.float64),
            horizontal_lines=DWT2DTestCases.make_horizontal_lines(shape),
            inverted_horizontal_lines=DWT2DTestCases.make_horizontal_lines(shape, inverted=True),
            vertical_lines=DWT2DTestCases.make_vertical_lines(shape),
            inverted_vertical_lines=DWT2DTestCases.make_vertical_lines(shape, inverted=True),
            diagonal_lines=DWT2DTestCases.make_diagonal_lines(shape),
            inverted_diagonal_lines=DWT2DTestCases.make_diagonal_lines(shape, inverted=True),
            # random=np.random.randn(*shape),
        )

    def make_square_tall_and_wide_shapes(size):
        shapes = [
            (size, size),
            (size, size // 2),
            (size // 2, size)
        ]
        return {f'{shape[0]}x{shape[1]}': shape for shape in shapes}

    SMALL_SIZE = 16
    MEDIUM_SIZE = 64
    LARGE_SIZE = 128

    small_shapes = make_square_tall_and_wide_shapes(SMALL_SIZE)
    medium_shapes = make_square_tall_and_wide_shapes(MEDIUM_SIZE)
    large_shapes = make_square_tall_and_wide_shapes(LARGE_SIZE)

    small_square_name, small_tall_name, small_wide_name = tuple(small_shapes.keys())
    medium_square_name, medium_tall_name, medium_wide_name = tuple(medium_shapes.keys())
    large_square_name, large_tall_name, large_wide_name = tuple(large_shapes.keys())

    dwt2d_test_cases = DWT2DTestCases(
        border_mode='reflect',
        patterns={
            name: create_patterns(shape)
            # for name, shape in {**small_shapes, **medium_shapes, **large_shapes}.items()
            for name, shape in (small_shapes | medium_shapes | large_shapes).items()
        },
        test_cases=[
            TestCase(
                wavelet='haar',
                patterns=small_square_name,
                levels=[1, 2, 3, 4],
            ),
            TestCase(
                wavelet='haar',
                patterns=small_tall_name,
                levels=[1, 2, 3],
            ),
            TestCase(
                wavelet='haar',
                patterns=small_wide_name,
                levels=[1, 2, 3],
            ),
            #   ----------------------------------------------------------------
            TestCase(
                wavelet='db1',
                patterns=small_square_name,
                levels=[1, 2, 3, 4],
            ),
            TestCase(
                wavelet='db1',
                patterns=small_tall_name,
                levels=[1, 2, 3],
            ),
            TestCase(
                wavelet='db1',
                patterns=small_wide_name,
                levels=[1, 2, 3],
            ),
            TestCase(
                wavelet='db2',
                patterns=medium_square_name,
                levels=[1, 2, 3],
            ),
            TestCase(
                wavelet='db2',
                patterns=medium_tall_name,
                levels=[1, 2, 3],
            ),
            TestCase(
                wavelet='db2',
                patterns=medium_wide_name,
                levels=[1, 2, 3],
            ),
            TestCase(
                wavelet='db4',
                patterns=large_square_name,
                levels=[1, 2, 3, 4],
            ),
            TestCase(
                wavelet='db4',
                patterns=large_tall_name,
                levels=[1, 2, 3],
            ),
            TestCase(
                wavelet='db4',
                patterns=large_wide_name,
                levels=[1, 2, 3],
            ),
            #   ----------------------------------------------------------------
            TestCase(
                wavelet='sym2',
                patterns=medium_square_name,
                levels=[1, 2, 3],
            ),
            TestCase(
                wavelet='sym2',
                patterns=medium_tall_name,
                levels=[1, 2, 3],
            ),
            TestCase(
                wavelet='sym2',
                patterns=medium_wide_name,
                levels=[1, 2, 3],
            ),
            TestCase(
                wavelet='sym3',
                patterns=large_square_name,
                levels=[1, 2, 3, 4],
            ),
            TestCase(
                wavelet='sym3',
                patterns=large_tall_name,
                levels=[1, 2, 3],
            ),
            TestCase(
                wavelet='sym3',
                patterns=large_wide_name,
                levels=[1, 2, 3],
            ),
            #   ----------------------------------------------------------------
            TestCase(
                wavelet='coif1',
                patterns=medium_square_name,
                levels=[1, 2, 3],
            ),
            TestCase(
                wavelet='coif1',
                patterns=medium_tall_name,
                levels=[1, 2],
            ),
            TestCase(
                wavelet='coif1',
                patterns=medium_wide_name,
                levels=[1, 2],
            ),
            TestCase(
                wavelet='coif2',
                patterns=large_square_name,
                levels=[1, 2, 3],
            ),
            TestCase(
                wavelet='coif2',
                patterns=large_tall_name,
                levels=[1, 2],
            ),
            TestCase(
                wavelet='coif2',
                patterns=large_wide_name,
                levels=[1, 2],
            ),
            #   ----------------------------------------------------------------
            TestCase(
                wavelet='bior1.1',
                patterns=small_square_name,
                levels=[1, 2, 3, 4],
            ),
            TestCase(
                wavelet='bior1.1',
                patterns=small_tall_name,
                levels=[1, 2, 3],
            ),
            TestCase(
                wavelet='bior1.1',
                patterns=small_wide_name,
                levels=[1, 2, 3],
            ),
            TestCase(
                wavelet='bior2.2',
                patterns=medium_square_name,
                levels=[1, 2, 3],
            ),
            TestCase(
                wavelet='bior2.2',
                patterns=medium_tall_name,
                levels=[1, 2],
            ),
            TestCase(
                wavelet='bior2.2',
                patterns=medium_wide_name,
                levels=[1, 2],
            ),
            TestCase(
                wavelet='bior4.4',
                patterns=large_square_name,
                levels=[1, 2, 3],
            ),
            TestCase(
                wavelet='bior4.4',
                patterns=large_tall_name,
                levels=[1, 2],
            ),
            TestCase(
                wavelet='bior4.4',
                patterns=large_wide_name,
                levels=[1, 2],
            ),
        ],
        dtype=np.float64,
    )

    dwt2d_test_cases.generate(filename='dwt2d_test_data.json')




if __name__ == '__main__':
    main()

