"""
Generates test case for wavelet unit tests
"""
import collections
from collections import abc
from collections.abc import Callable
from dataclasses import dataclass
import json
import os
import warnings
from typing import Any, Optional, SupportsIndex

import numpy as np
from numpy.typing import ArrayLike
import pywt
import cv2

WAVELET_TEST_DATA_FILENAME = 'wavelet_test_data.json'
DWT2D_TEST_DATA_FILENAME = 'dwt2d_test_data.json'
SHRINK_THRESHOLDS_TEST_DATA_FILENAME = 'shrink_thresholds_test_data.json'
LENA_FILEPATH = 'images/lena.png'
SEED = 42

random = np.random.default_rng(SEED)


#   ============================================================================
#   Wavelet
#   ============================================================================
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



def generate_coefficients_test_cases():
    coeffs_test_cases = CoefficientTestCases(
        families=[
            'haar',
            'db',
            'sym',
            'coif',
            'bior',
            'rbio',
        ],
    )
    coeffs_test_cases.generate(WAVELET_TEST_DATA_FILENAME)


#   ============================================================================
#   DWT2D Transformation
#   ============================================================================
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
        return DWT2DTestCases._make_pattern(
            shape=shape,
            tile=[[1, 1], [0, 0]],
            inverted=inverted,
            dtype=dtype,
        )


    @staticmethod
    def make_vertical_lines(shape, inverted=False, dtype=np.float64):
        return DWT2DTestCases._make_pattern(
            shape=shape,
            tile=[[1, 0], [1, 0]],
            inverted=inverted,
            dtype=dtype,
        )


    @staticmethod
    def make_diagonal_lines(shape, inverted=False, dtype=np.float64):
        return DWT2DTestCases._make_pattern(
            shape=shape,
            tile=[[1, 0], [0, 1]],
            inverted=inverted,
            dtype=dtype,
        )


    @staticmethod
    def _make_pattern(shape, tile, inverted=False, dtype=np.float64):
        rows, cols = (shape[0] + shape[0] % 2, shape[1] + shape[1] % 2)
        tile = np.array(tile, dtype=dtype)
        pattern = np.tile(tile, (rows // 2, cols // 2))
        pattern = pattern[:shape[0], :shape[1]]

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
        try:
            coeffs = pywt.wavedec2(
                pattern,
                wavelet,
                axes=axes,
                mode=self.border_mode,
                level=level,
            )
            coeffs = pywt.coeffs_to_array(coeffs, axes=axes)[0]
        except ValueError:
            coeffs = np.zeros([0, 0], dtype=self.dtype)

        return coeffs.astype(self.dtype)




def generate_dwt2d_test_cases():
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
            random=random.normal(0, 1, shape),
            random_rgb=random.normal(0, 1, (*shape, 3)),
            random_rgba=random.normal(0, 1, (*shape, 4)),
        )

    def make_square_tall_and_wide_shapes(size):
        shapes = [
            (size, size),
            (size, (size // 2) - 1),
            ((size // 2) + 1, size)
        ]
        return {f'{shape[1]}x{shape[0]}': shape for shape in shapes}

    SMALL_SIZE = 16
    MEDIUM_SIZE = 64
    LARGE_SIZE = 128

    small_shapes = make_square_tall_and_wide_shapes(SMALL_SIZE)
    medium_shapes = make_square_tall_and_wide_shapes(MEDIUM_SIZE)
    large_shapes = make_square_tall_and_wide_shapes(LARGE_SIZE)

    small_square_name, small_tall_name, small_wide_name = tuple(small_shapes.keys())
    medium_square_name, medium_tall_name, medium_wide_name = tuple(medium_shapes.keys())
    large_square_name, large_tall_name, large_wide_name = tuple(large_shapes.keys())

    levels = [1, 2, 3, 4]

    dwt2d_test_cases = DWT2DTestCases(
        border_mode='reflect',
        dtype=np.float64,
        patterns={
            name: create_patterns(shape)
            for name, shape in (small_shapes | medium_shapes | large_shapes).items()
        },
        test_cases=[
            TestCase(
                wavelet='haar',
                patterns=small_square_name,
                levels=levels,
            ),
            TestCase(
                wavelet='haar',
                patterns=small_tall_name,
                levels=levels,
            ),
            TestCase(
                wavelet='haar',
                patterns=small_wide_name,
                levels=levels,
            ),
            #   ----------------------------------------------------------------
            TestCase(
                wavelet='db1',
                patterns=small_square_name,
                levels=levels,
            ),
            TestCase(
                wavelet='db1',
                patterns=small_tall_name,
                levels=levels,
            ),
            TestCase(
                wavelet='db1',
                patterns=small_wide_name,
                levels=levels,
            ),
            TestCase(
                wavelet='db2',
                patterns=medium_square_name,
                levels=levels,
            ),
            TestCase(
                wavelet='db2',
                patterns=medium_tall_name,
                levels=levels,
            ),
            TestCase(
                wavelet='db2',
                patterns=medium_wide_name,
                levels=levels,
            ),
            TestCase(
                wavelet='db4',
                patterns=large_square_name,
                levels=levels,
            ),
            TestCase(
                wavelet='db4',
                patterns=large_tall_name,
                levels=levels,
            ),
            TestCase(
                wavelet='db4',
                patterns=large_wide_name,
                levels=levels,
            ),
            #   ----------------------------------------------------------------
            TestCase(
                wavelet='sym2',
                patterns=medium_square_name,
                levels=levels,
            ),
            TestCase(
                wavelet='sym2',
                patterns=medium_tall_name,
                levels=levels,
            ),
            TestCase(
                wavelet='sym2',
                patterns=medium_wide_name,
                levels=levels,
            ),
            TestCase(
                wavelet='sym3',
                patterns=large_square_name,
                levels=levels,
            ),
            TestCase(
                wavelet='sym3',
                patterns=large_tall_name,
                levels=levels,
            ),
            TestCase(
                wavelet='sym3',
                patterns=large_wide_name,
                levels=levels,
            ),
            #   ----------------------------------------------------------------
            TestCase(
                wavelet='coif1',
                patterns=medium_square_name,
                levels=levels,
            ),
            TestCase(
                wavelet='coif1',
                patterns=medium_tall_name,
                levels=levels,
            ),
            TestCase(
                wavelet='coif1',
                patterns=medium_wide_name,
                levels=levels,
            ),
            TestCase(
                wavelet='coif2',
                patterns=large_square_name,
                levels=levels,
            ),
            TestCase(
                wavelet='coif2',
                patterns=large_tall_name,
                levels=levels,
            ),
            TestCase(
                wavelet='coif2',
                patterns=large_wide_name,
                levels=levels,
            ),
            #   ----------------------------------------------------------------
            TestCase(
                wavelet='bior1.1',
                patterns=small_square_name,
                levels=levels,
            ),
            TestCase(
                wavelet='bior1.1',
                patterns=small_tall_name,
                levels=levels,
            ),
            TestCase(
                wavelet='bior1.1',
                patterns=small_wide_name,
                levels=levels,
            ),
            TestCase(
                wavelet='bior2.2',
                patterns=medium_square_name,
                levels=levels,
            ),
            TestCase(
                wavelet='bior2.2',
                patterns=medium_tall_name,
                levels=levels,
            ),
            TestCase(
                wavelet='bior2.2',
                patterns=medium_wide_name,
                levels=levels,
            ),
            TestCase(
                wavelet='bior4.4',
                patterns=large_square_name,
                levels=levels,
            ),
            TestCase(
                wavelet='bior4.4',
                patterns=large_tall_name,
                levels=levels,
            ),
            TestCase(
                wavelet='bior4.4',
                patterns=large_wide_name,
                levels=levels,
            ),
            #   ----------------------------------------------------------------
            TestCase(
                wavelet='rbio1.1',
                patterns=small_square_name,
                levels=levels,
            ),
            TestCase(
                wavelet='rbio1.1',
                patterns=small_tall_name,
                levels=levels,
            ),
            TestCase(
                wavelet='rbio1.1',
                patterns=small_wide_name,
                levels=levels,
            ),
            TestCase(
                wavelet='rbio2.2',
                patterns=medium_square_name,
                levels=levels,
            ),
            TestCase(
                wavelet='rbio2.2',
                patterns=medium_tall_name,
                levels=levels,
            ),
            TestCase(
                wavelet='rbio2.2',
                patterns=medium_wide_name,
                levels=levels,
            ),
            TestCase(
                wavelet='rbio4.4',
                patterns=large_square_name,
                levels=levels,
            ),
            TestCase(
                wavelet='rbio4.4',
                patterns=large_tall_name,
                levels=levels,
            ),
            TestCase(
                wavelet='rbio4.4',
                patterns=large_wide_name,
                levels=levels,
            ),
        ],
    )

    dwt2d_test_cases.generate(filename=DWT2D_TEST_DATA_FILENAME)


#   ============================================================================
#   Shrink Threshold
#   ============================================================================
@dataclass
class Coeffs:
    matrix: np.ndarray
    wavelet: str
    image_size: tuple[int, int]
    levels: int
    subband_slices: dict[str, Any]

    @property
    def shape(self):
        return self.matrix.shape

    @property
    def ndim(self):
        return self.matrix.ndim

    def __array__(self):
        return self.matrix


@dataclass
class ShrinkThresholdParam:
    coeffs: str
    shrinker: dict[str, Any]
    expected_stdev: np.ndarray
    expected_thresholds: np.ndarray


ThresholdFunction = Callable[[ArrayLike, ArrayLike, int | tuple[int], bool], np.ndarray]
NoiseStdFunction = Callable[[Coeffs, int | tuple[int]], np.ndarray]
Axis = SupportsIndex | None


def flatten(array: ArrayLike, *, axis: Axis = None) -> np.ndarray:
    axis = resolve_axis(axis, array.ndim)
    not_flat_axis = tuple(
        flat_dim for flat_dim in range(array.ndim)
        if flat_dim not in axis
    )
    array = np.transpose(array, axis + not_flat_axis)

    return np.reshape(array, (-1, *array.shape[-len(not_flat_axis):]))


def resolve_axis(axis: Axis, ndim: int) -> Axis:
    if axis is None:
        axis = tuple(range(ndim))
    elif isinstance(axis, int):
        axis = (axis,)

    return axis


def subsize(array: ArrayLike, *, axis: Axis = None):
    array = np.asarray(array)
    return np.prod([array.shape[dim] for dim in resolve_axis(axis, array.ndim)])


def mad(array: ArrayLike, *, axis: Axis = None):
    array = np.asarray(array)
    median = np.median(array, axis=axis)
    return np.median(np.abs(array - median), axis=axis)


def mad_std(array: ArrayLike, *, axis: Axis = None):
    return mad(array, axis=axis) / 0.675


def universal_threshold(
        coeffs: ArrayLike,
        stdev: ArrayLike,
        *,
        axis: Axis = None,
        keepdims: bool = False,
    ) -> np.ndarray:
    coeffs = np.asarray(coeffs)
    stdev = np.asarray(stdev)
    axis = resolve_axis(axis, coeffs.ndim - 1)

    threshold = stdev * np.sqrt(2 * np.log(subsize(coeffs, axis=axis)))
    if keepdims:
        threshold = np.expand_dims(threshold, axis=axis)

    return threshold


def bayes_threshold(
        coeffs: ArrayLike,
        noise_std: ArrayLike,
        *,
        axis: Axis = None,
        keepdims: bool = False,
    ) -> np.ndarray:
    coeffs = np.asarray(coeffs)
    noise_std = np.asarray(noise_std)
    axis = resolve_axis(axis, coeffs.ndim - 1)
    noise_variance = noise_std**2
    obs_variance = np.mean(coeffs**2, axis=axis, keepdims=keepdims)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'invalid value encountered in sqrt')
        threshold1 = noise_variance / np.sqrt(obs_variance - noise_variance)
        threshold2 = np.max(np.abs(coeffs), axis=axis, keepdims=keepdims)

    return np.where(np.isnan(threshold1), threshold2, threshold1)


def sure_risk(
        coeffs: ArrayLike,
        threshold: ArrayLike,
        stdev: Optional[ArrayLike] = None,
    ) -> np.ndarray:
    coeffs = np.asarray(coeffs)
    threshold = np.asarray(threshold)
    if coeffs.ndim == 1:
        coeffs = coeffs[:, np.newaxis]
    elif coeffs.ndim == 2:
        if threshold.ndim != 1:
            raise ValueError(f'threshold must be 1-dim, got threshold.ndim = {threshold.ndim}')
    else:
        raise ValueError(f'coeffs must be 1-dim or 2-dim, got coeffs.ndim = {coeffs.ndim}')

    if threshold.shape[0] != coeffs.shape[1]:
        raise ValueError(
            f'Shape mismatch between coeffs and threshold. '
            f'Must have thresholds.shape[0] == coeffs.shape[1], '
            f'got threshold.shape = {threshold.shape}, coeffs.shape = {coeffs.shape}'
        )

    if stdev is not None:
        coeffs = coeffs / stdev
        threshold = threshold / stdev

    coeffs = np.abs(coeffs)
    return (
        coeffs.shape[0]
        + np.sum(np.minimum(coeffs, threshold)**2, axis=0)
        - 2 * np.sum(coeffs <= threshold, axis=0)
    )


def sure_threshold(
        coeffs: ArrayLike,
        stdev: ArrayLike,
        *,
        axis: Axis = None,
        keepdims: bool = False,
    ) -> np.ndarray:
    coeffs = np.asarray(coeffs)
    stdev = np.asarray(stdev)
    axis = resolve_axis(axis, coeffs.ndim - 1)
    if len(axis) != coeffs.ndim - 1:
        raise ValueError(
            f'The number axis dimensions must be one less than coeffs.ndim. '
            f'Got coeffs.ndim = {coeffs.ndim} and axis = {axis}'
        )

    coeffs = coeffs / stdev
    coeffs = flatten(coeffs, axis=axis)
    thresholds = np.abs(coeffs)
    risks = np.array([
        sure_risk(coeffs, threshold)
        for threshold in thresholds
    ])

    min_risk_indices = np.argmin(risks, axis=0)
    threshold = thresholds[min_risk_indices, np.arange(thresholds.shape[-1])]
    if keepdims:
        threshold = np.expand_dims(threshold, tuple(range(len(axis))))

    return stdev * threshold


def hybrid_sure_threshold(
        coeffs: ArrayLike,
        stdev: ArrayLike,
        *,
        axis: Axis = None,
        keepdims: bool = False,
    ) -> np.ndarray:
    axis = resolve_axis(axis, coeffs.ndim - 1)
    coeffs = np.asarray(coeffs)
    n = subsize(coeffs, axis=axis)
    lhs = np.mean((coeffs / stdev)**2 - 1, axis=axis, keepdims=keepdims)
    rhs = np.log2(n)**1.5 / np.sqrt(n)
    return np.where(
        lhs <= rhs,
        universal_threshold(coeffs, stdev, axis=axis, keepdims=keepdims),
        sure_threshold(coeffs, stdev, axis=axis, keepdims=keepdims),
    )


def get_subband_coeffs(coeffs: Coeffs, level: int, subband_name: str) -> np.ndarray:
    slices = coeffs.subband_slices[level][subband_name]
    return coeffs.matrix[slices]


def mad_std_from_subband_d0(coeffs: Coeffs, *, axis: Axis = None) -> np.ndarray:
    return mad_std(
        get_subband_coeffs(coeffs, 0, 'd'),
        axis=resolve_axis(axis, coeffs.ndim - 1),
    )


class ShrinkThresholdTestCaseGenerator:
    def __init__(self, coeffs: abc.Mapping[str, Any], shrinkers: abc.Mapping[str, Any]):
        self.coeffs = dict(coeffs)
        self.shrinkers = shrinkers


    def generate(self, filename: os.PathLike):
        params = list()
        for shrinker in self.shrinkers:
            shrinker_type: str = shrinker['shrinker_type']
            compute_threshold: ThresholdFunction = shrinker['threshold']
            compute_noise_std: NoiseStdFunction = shrinker['noise_std']
            parameter_sets = shrinker['parameter_sets']
            for coeffs_name, coeffs in self.coeffs.items():
                noise_std = compute_noise_std(coeffs)
                for shrinker_args in parameter_sets:
                    match shrinker_args['partition']:
                        case 'subbands':
                            expected_thresholds = self.compute_subband_partition_thresholds(
                                compute_threshold,
                                coeffs,
                                noise_std,
                            )
                        case 'levels':
                            expected_thresholds = self.compute_level_partition_thresholds(
                                compute_threshold,
                                coeffs,
                                noise_std,
                            )
                        case 'global':
                            expected_thresholds = self.compute_global_thresholds(
                                compute_threshold,
                                coeffs,
                                noise_std,
                                self.create_global_level_sets(coeffs.levels),
                            )
                        case _:
                            raise ValueError('invalid partition')

                    params.append(
                        ShrinkThresholdParam(
                            coeffs=coeffs_name,
                            shrinker=dict(
                                label=self.build_shrinker_label(
                                    shrinker_type,
                                    shrinker_args,
                                ),
                                type=shrinker_type,
                                args=shrinker_args,
                            ),
                            expected_stdev=noise_std,
                            expected_thresholds=expected_thresholds,
                        )
                    )

        with open(filename, 'w') as file:
            json.dump(
                dict(
                    coeffs=self.coeffs,
                    params=params,
                ),
                file,
                cls=TestCaseJsonEncoder,
            )

        return params


    def build_shrinker_label(self, shrinker_type: str, shrinker_args: dict[str, Any]):
        return '_'.join([shrinker_type, *shrinker_args.values()])


    def create_global_level_sets(self, levels: int):
        global_level_sets = {
            'All Levels': list(range(levels)),
            'First Level': [0],
        }
        if levels >= 1:
            global_level_sets['Last Level'] = [levels - 1]

        if levels >= 2:
            global_level_sets['First Two Levels'] = [0, 1]

        if levels > 2:
            global_level_sets['Last Two Levels'] = [levels - 2, levels - 1]

        return global_level_sets


    def get_level_set_coeffs(self, coeffs: Coeffs, level_set: list[int]):
        return np.concatenate([
            get_subband_coeffs(
                coeffs,
                level,
                subband_key,
            ).reshape([-1, coeffs.matrix.shape[-1]])
            for level in level_set
            for subband_key in ['h', 'v', 'd']
        ])


    def compute_subband_partition_thresholds(
            self,
            compute_threshold: ThresholdFunction,
            coeffs: Coeffs,
            noise_std: np.ndarray,
        ):
        return np.array([
            [
                compute_threshold(
                    get_subband_coeffs(coeffs, level, subband_key),
                    noise_std
                )
                for subband_key in ['h', 'v', 'd']
            ]
            for level in range(coeffs.levels)
        ])


    def compute_level_partition_thresholds(
            self,
            compute_threshold: ThresholdFunction,
            coeffs: Coeffs,
            noise_std: np.ndarray
        ):
        return np.array([
            compute_threshold(
                self.get_level_set_coeffs(coeffs, [level]),
                noise_std,
                keepdims=True,
            )
            for level in range(coeffs.levels)
        ])


    def compute_global_thresholds(
            self,
            compute_threshold: ThresholdFunction,
            coeffs: Coeffs,
            noise_std: np.ndarray,
            global_level_sets: list[int],
        ):
        return np.array([
            compute_threshold(
                self.get_level_set_coeffs(coeffs, level_set),
                noise_std,
                keepdims=True,
            )
            for level_set in global_level_sets.values()
        ])




def generate_shrink_test_cases():
    linear_shape = (16, 16, 3)
    linear_levels = 4
    lena_shape = (64, 64)
    lena_wavelet = 'db2'
    noise_std = 0.1

    def create_linear_coeffs(shape, start, end):
        return (
            np.linspace(start, end, np.prod(shape))
            .reshape(shape[2], shape[0], shape[1])
            .transpose([1, 2, 0])
        )

    def create_lena_image(shape, *, grayscale=False):
        lena_image = cv2.imread(
            LENA_FILEPATH,
            cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        )
        lena_image = lena_image / 255.0
        lena_image = cv2.resize(lena_image, shape)
        if grayscale:
            lena_image = lena_image[:, :, np.newaxis]

        return lena_image

    def create_noisy(coeffs, noise_std):
        return coeffs + random.normal(0, noise_std, coeffs.shape)

    def build_coeffs(*, name, image=None, coeffs=None, wavelet=None, levels=None):
        def translate_slices(subband_slices):
            return tuple(
                dict(
                    h=level_subband_slices['da'],
                    v=level_subband_slices['ad'],
                    d=level_subband_slices['dd'],
                )
                for level_subband_slices in list(reversed(subband_slices))[:-1]
            )

        if coeffs is not None:
            image_shape = coeffs.shape
            wavelet = 'haar'
            dummy_coeffs = pywt.wavedec2(coeffs, wavelet, level=levels, axes=(0, 1))
            levels = len(dummy_coeffs) - 1
            _, subband_slices = pywt.coeffs_to_array(dummy_coeffs, axes=(0, 1))
        elif image is not None:
            image_shape = image.shape
            coeffs = pywt.wavedec2(image, wavelet, level=levels, axes=(0, 1))
            levels = len(coeffs) - 1
            coeffs, subband_slices = pywt.coeffs_to_array(coeffs, axes=(0, 1))

        return {
            '_'.join([name, wavelet, str(levels)]): Coeffs(
                matrix=coeffs,
                subband_slices=translate_slices(subband_slices),
                wavelet=wavelet,
                image_size=image_shape[:2],
                levels=levels,
            )
        }

    def build_sure_shrink(variant, optimizer):
        return dict(
            shrinker_type='SureShrink',
            threshold=sure_threshold,
            noise_std=mad_std_from_subband_d0,
            parameter_sets=[
                dict(
                    partition='subbands',
                    variant=variant,
                    optimizer=optimizer,
                ),
                dict(
                    partition='levels',
                    variant=variant,
                    optimizer=optimizer,
                ),
                dict(
                    partition='global',
                    variant=variant,
                    optimizer=optimizer,
                ),
            ],
        )

    #   Using a perfectly symmetric linear range (i.e. -1 to 1) causes the
    #   SureShrink optimizer to fail at a single channel in a single pixel.
    #   Slightly pertubing the end points rectifies this.
    linear = create_linear_coeffs(linear_shape, start=-0.95, end=1.01)
    positive_linear = create_linear_coeffs(linear_shape, start=0.05, end=1.01)
    negative_linear = create_linear_coeffs(linear_shape, start=-1.01, end=-0.05)

    shrink_test_case_generator = ShrinkThresholdTestCaseGenerator(
        #   --------------------------------------------------------------------
        #   Coefficients
        #   --------------------------------------------------------------------
        coeffs=collections.ChainMap(
            # build_coeffs(
            #     name='linear',
            #     coeffs=linear,
            #     levels=linear_levels,
            # ),
            # build_coeffs(
            #     name='negative_linear',
            #     coeffs=negative_linear,
            #     levels=linear_levels,
            # ),
            # build_coeffs(
            #     name='positive_linear',
            #     coeffs=positive_linear,
            #     levels=linear_levels,
            # ),
            # build_coeffs(
            #     name='noisy_linear',
            #     coeffs=create_noisy(linear, noise_std),
            #     levels=linear_levels,
            # ),
            # build_coeffs(
            #     name='noisy_negative_linear',
            #     coeffs=create_noisy(negative_linear, noise_std),
            #     levels=linear_levels,
            # ),
            # build_coeffs(
            #     name='noisy_positive_linear',
            #     coeffs=create_noisy(positive_linear, noise_std),
            #     levels=linear_levels,
            # ),
            build_coeffs(
                name='lena',
                image=create_lena_image(lena_shape, grayscale=False),
                wavelet=lena_wavelet,
                levels=1,
            ),
            build_coeffs(
                name='lena',
                image=create_lena_image(lena_shape, grayscale=False),
                wavelet=lena_wavelet,
                levels=4,
            ),
            build_coeffs(
                name='lena_gray',
                image=create_lena_image(lena_shape, grayscale=True),
                wavelet=lena_wavelet,
                levels=3,
            ),
        ),
        #   --------------------------------------------------------------------
        #   Shrinkers
        #   --------------------------------------------------------------------
        shrinkers=[
            dict(
                shrinker_type='VisuShrink',
                threshold=universal_threshold,
                noise_std=mad_std_from_subband_d0,
                parameter_sets=[
                    dict(
                        partition='subbands',
                    ),
                    dict(
                        partition='levels',
                    ),
                    dict(
                        partition='global',
                    ),
                ],
            ),
            dict(
                shrinker_type='BayesShrink',
                threshold=bayes_threshold,
                noise_std=mad_std_from_subband_d0,
                parameter_sets=[
                    dict(
                        partition='subbands',
                    ),
                    dict(
                        partition='levels',
                    ),
                    dict(
                        partition='global',
                    ),
                ],
            ),
            build_sure_shrink(variant='strict', optimizer='brute_force'),
            build_sure_shrink(variant='strict', optimizer='sbplx'),
            build_sure_shrink(variant='strict', optimizer='nelder_mead'),
            build_sure_shrink(variant='strict', optimizer='direct'),
            build_sure_shrink(variant='strict', optimizer='direct_l'),
            build_sure_shrink(variant='strict', optimizer='cobyla'),
            build_sure_shrink(variant='strict', optimizer='bobyqa'),
            build_sure_shrink(variant='hybrid', optimizer='brute_force'),
        ]
    )

    shrink_test_case_generator.generate(SHRINK_THRESHOLDS_TEST_DATA_FILENAME)


#   ============================================================================
#   Main
#   ============================================================================
class TestCaseJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pywt.Wavelet):
            result = dict(
                vanishing_moments_psi=obj.vanishing_moments_psi,
                vanishing_moments_phi=obj.vanishing_moments_phi,
                orthogonal=obj.orthogonal,
                biorthogonal=obj.biorthogonal,
                symmetry=obj.symmetry,
                family=obj.family_name.title(),
                name=self.fix_reverse_biorthogonal_name(obj.name),
                decompose_lowpass=obj.dec_lo,
                decompose_highpass=obj.dec_hi,
                reconstruct_lowpass=obj.rec_lo,
                reconstruct_highpass=obj.rec_hi,
            )
        elif isinstance(obj, DWT2DParam):
            result = dict(
                wavelet_name=self.fix_reverse_biorthogonal_name(obj.wavelet_name),
                input_name=obj.input_name,
                coeffs=obj.coeffs,
                levels=obj.levels,
            )
        elif isinstance(obj, ShrinkThresholdParam):
            result = dict(
                coeffs=obj.coeffs,
                shrinker=obj.shrinker,
                expected_stdev=obj.expected_stdev,
                expected_thresholds=obj.expected_thresholds,
            )
        elif isinstance(obj, Coeffs):
            result = dict(
                matrix=obj.matrix,
                wavelet=obj.wavelet,
                image_size=obj.image_size,
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


    def fix_reverse_biorthogonal_name(self, wavelet_name):
        #   Fix naming mismatch between pywt and opencv wavelet lib
        #   for reverse biorthogonal wavelets.
        if wavelet_name.startswith('rbio'):
            wavelet_name = wavelet_name.replace('o', 'or')

        return wavelet_name


def main():
    generate_coefficients_test_cases()
    generate_dwt2d_test_cases()
    generate_shrink_test_cases()


if __name__ == '__main__':
    exit(main())

