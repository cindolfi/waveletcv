
import argparse

import numpy as np
import scipy as sp


def mad(x):
    m = np.median(x)
    return np.median(np.abs(x - m))


def mad_std(x):
    return mad(x) / 0.675


def universal_threshold(coeffs, stdev):
    return stdev * np.sqrt(2 * np.log(coeffs.size))


def bayes_threshold(coeffs, noise_std):
    noise_variance = noise_std**2
    obs_variance = np.sum(coeffs**2) / coeffs.size

    sig_variance = max(obs_variance - noise_variance, 0)
    if sig_variance == 0:
        return np.max(abs(coeffs))

    return noise_variance / np.sqrt(sig_variance)


def sure_risk(coeffs, threshold, stdev):
    coeffs = abs(coeffs / stdev)
    threshold = threshold / stdev
    return  len(coeffs) + np.sum(np.minimum(coeffs, threshold)**2) - 2 * np.sum(coeffs <= threshold)


def sure_threshold(coeffs, stdev):
    assert(1 <= coeffs.ndim <= 2)
    coeffs = coeffs.ravel()
    thresholds = np.sort(coeffs)
    risks = np.array([
        sure_risk(coeffs, threshold, stdev)
        for threshold in thresholds
    ])
    i = np.argmin(risks)
    return abs(thresholds[i])


    # return argmin


def hybrid_sure_threshold(coeffs, stdev):
    lhs = np.mean((coeffs / stdev)**2 - 1)
    rhs = np.log2(coeffs.size)**1.5 / np.sqrt(coeffs.size)
    return universal_threshold(coeffs, stdev) if lhs <= rhs else sure_threshold(coeffs, stdev)





class TestCaseGenerator:
    SUBBAND_LABELS = dict(h='Horizontal', v='Vertical', d='Diagonal')

    def __init__(self, compute_threshold, global_level_sets=None, precision=4):
        self.compute_threshold = compute_threshold
        self.precision = precision
        self.coeff_format = f'0.{self.precision}f'
        self.threshold_format = f'0.16f'
        self.global_level_sets = global_level_sets


    def generate(self, coeffs, levels):
        assert(coeffs.ndim == 3)
        assert(coeffs.shape[0] == coeffs.shape[1])
        assert(coeffs.shape[2] == 4)

        if self.global_level_sets is None:
            self.global_level_sets = {'All Levels': list(range(levels))}

        size = coeffs.shape[0]
        self._create_subband_slices_and_masks(size, levels)

        # self.print_header('Coeffs')
        # coeffs = np.round(coeffs, self.precision)
        # self.print_coeffs(coeffs)
        # print()

        self.print_header('Noise Std Dev')
        noise_std = np.array([
            mad_std(self.get_subband(coeffs, 'd0', channel))
            for channel in range(4)
        ])
        print_scalar(noise_std, self.threshold_format, trailing_comma=False)
        print()

        self.print_header('Subband Partition')
        thresholds = self.compute_subband_partition_thresholds(coeffs, noise_std)
        self.print_thresholds(thresholds)
        print()

        self.print_header('Level Partition')
        thresholds = self.compute_level_partition_thresholds(coeffs, noise_std, levels)
        self.print_thresholds(thresholds)
        print()

        self.print_header('Global Partition')
        thresholds = self.compute_global_thresholds(coeffs, noise_std)
        self.print_thresholds(thresholds)
        print()


    def _create_subband_slices_and_masks(self, size, levels):
        self.subband_slices = dict()
        for level in range(levels):
            k = size // 2**level
            j = k // 2
            self.subband_slices[f'h{level}'] = (slice(j, k), slice(0, j))
            self.subband_slices[f'v{level}'] = (slice(0, j), slice(j, k))
            self.subband_slices[f'd{level}'] = (slice(j, k), slice(j, k))


    def get_subband(self, coeffs, subband_name, channel=None):
        slices = self.subband_slices[subband_name]
        if channel is None:
            subband = coeffs[slices[0], slices[1]]
        else:
            subband = coeffs[slices[0], slices[1], channel]

        return subband


    def compute_subband_partition_thresholds(self, coeffs, noise_std):
        def create_comment(subband_key):
            return f'Level {subband_key[-1]}, {self.SUBBAND_LABELS[subband_key[0]]}'

        return {
            create_comment(subband_key): np.array([
                self.compute_threshold(
                    self.get_subband(coeffs, subband_key, channel),
                    noise_std[channel]
                )
                for channel in range(4)
            ])
            for subband_key in self.subband_slices
        }


    def compute_level_partition_thresholds(self, coeffs, noise_std, levels):
        def get_level_coeffs(channel, level):
            return np.concatenate([
                self.get_subband(coeffs, f'h{level}', channel).ravel(),
                self.get_subband(coeffs, f'v{level}', channel).ravel(),
                self.get_subband(coeffs, f'd{level}', channel).ravel(),
            ])

        return {
            f'Level {level}': np.array([
                self.compute_threshold(
                    get_level_coeffs(channel, level),
                    noise_std[channel]
                )
                for channel in range(4)
            ])
            for level in range(levels)
        }


    def compute_global_thresholds(self, coeffs, noise_std):
        def get_global_coeffs(channel, level_set):
            return np.concatenate([
                self.get_subband(coeffs, subband_key, channel).ravel()
                for subband_key in self.subband_slices
                if int(subband_key[-1]) in level_set
            ])

        return {
            comment: np.array([
                self.compute_threshold(
                    get_global_coeffs(channel, level_set),
                    noise_std[channel]
                )
                for channel in range(4)
            ])
            for comment, level_set in self.global_level_sets.items()
        }


    def print_header(self, header):
        print(banner(header, '-'))
        # print('{:-^80}'.format(' ' + header + ' '))


    # def print_scalar(self, scalar, format, trailing_comma=True):
    #     print(
    #         f'cv::Scalar('
    #         f'{scalar[0]:{format}}, '
    #         f'{scalar[1]:{format}}, '
    #         f'{scalar[2]:{format}}, '
    #         f'{scalar[3]:{format}})',
    #         end=',\n' if trailing_comma else '\n'
    #     )


    # def print_coeffs(self, coeffs):
    #     for row in range(coeffs.shape[0]):
    #         print(f'//  Row {row}')
    #         for column in range(coeffs.shape[1]):
    #             self.print_scalar(
    #                 coeffs[row, column],
    #                 self.coeff_format,
    #                 trailing_comma=(row != coeffs.shape[0] - 1 or column != coeffs.shape[1] - 1),
    #             )


    def print_thresholds(self, thresholds):
        for i, (comment, threshold) in enumerate(thresholds.items()):
            if comment:
                print(f'//  {comment}')
            print_scalar(
                threshold,
                self.threshold_format,
                trailing_comma=i < len(thresholds) - 1,
            )




def banner(title, fill):
    title = f' {title} '
    return f'{title:{fill}^80}'


def print_scalar(scalar, format, trailing_comma=True):
    print(
        f'cv::Scalar('
        f'{scalar[0]:{format}}, '
        f'{scalar[1]:{format}}, '
        f'{scalar[2]:{format}}, '
        f'{scalar[3]:{format}})',
        end=',\n' if trailing_comma else '\n'
    )


def print_coeffs(coeffs, coeffs_format):
    for row in range(coeffs.shape[0]):
        print(f'//  Row {row}')
        for column in range(coeffs.shape[1]):
            print_scalar(
                coeffs[row, column],
                coeffs_format,
                trailing_comma=(row != coeffs.shape[0] - 1 or column != coeffs.shape[1] - 1),
            )


def create_global_level_sets(levels):
    global_level_sets = {
        'All Levels': list(range(levels)),
        'First Level': [0],
        # 'First Two Levels': [0, 1],
        # 'Last Level': [args.levels - 1],
        # 'Last Two Levels': [args.levels - 2, args.levels - 1],
    }
    if levels >= 1:
        global_level_sets['Last Level'] = [levels - 1]

    if levels >= 2:
        global_level_sets['First Two Levels'] = [0, 1]

    if levels > 2:
        global_level_sets['Last Two Levels'] = [levels - 2, levels - 1]

    return global_level_sets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--linear',
        action='store_true',
        help='Generate linearly spaced coefficients',
    )
    parser.add_argument(
        '--random',
        action='store_true',
        help='Generate coefficients drawn from standard normal distribution',
    )
    parser.add_argument(
        '--bayes',
        action='store_true',
        help='Generate BayesShrink test cases',
    )
    parser.add_argument(
        '--strict-sure',
        action='store_true',
        help='Generate SureShrink (strict variant) test cases',
    )
    parser.add_argument(
        '--hybrid-sure',
        action='store_true',
        help='Generate SureShrink (hybrid variant) test cases',
    )
    parser.add_argument(
        '--seed',
        action='store',
        type=int,
        default=0,
        help='Set the random number generator seed (default: 0)',
    )
    parser.add_argument(
        '--start',
        action='store',
        type=float,
        default=0,
        help='Set the initial value for linearly spaced coefficients (default: 0)',
    )
    parser.add_argument(
        '--end',
        action='store',
        type=float,
        default=1,
        help='Set the final value for linearly spaced coefficients (default: 1)',
    )
    parser.add_argument(
        '--noise',
        action='store',
        type=float,
        default=0.1,
        help='Set the standard deviation of the noise added to the linearly spaced coefficients (default: 0.1)',
    )
    parser.add_argument(
        '--size',
        action='store',
        type=int,
        default=16,
        help='Set the size of the square coefficients matrix (default: 16)',
    )
    parser.add_argument(
        '--levels',
        action='store',
        type=int,
        default=3,
        help='Set the number of decomposition levels represented by the coefficients matrix (default: 3)',
    )
    parser.add_argument(
        '--precision',
        action='store',
        type=int,
        default=4,
        help='Set the precision of the coefficients (default: 4)',
    )

    args = parser.parse_args()
    if args.size < 1:
        parser.error('size must be greater than or equal to 1')

    if args.levels < 1:
        parser.error('levels must be greater than or equal to 1')

    if args.seed < 0:
        parser.error('seed must be a positive integer')

    return args


def main():
    def create_coeffs(title, coeffs):
        coeffs = np.round(coeffs, args.precision)
        print(banner(title, '%'))
        print_coeffs(coeffs, f'0.{args.precision}f')
        print()
        return coeffs

    def generate_test_cases(title, threshold):
        generator = TestCaseGenerator(threshold, global_level_sets)
        if args.linear:
            print(banner(f'{title} - Linearly Spaced' , '#'))
            generator.generate(linear_space_coeffs, args.levels)
            print()

        if args.random:
            print(banner(f'{title} - Random' , '#'))
            generator.generate(random_coeffs, args.levels)
            print()

    args = parse_args()
    global_level_sets = create_global_level_sets(args.levels)

    if args.linear:
        linear = (
            np.linspace(args.start, args.end, args.size * args.size * 4)
            .reshape(4, args.size, args.size)
            .transpose([1, 2, 0])
        )
        if args.noise > 0:
            linear += np.random.default_rng(args.seed).normal(0, args.noise, [args.size, args.size, 4])

        linear_space_coeffs = create_coeffs('Linearly Spaced Coeffs', linear)

    if args.random:
        # random_coeffs = sp.sparse.random_array(
        #     (4 * args.size, args.size),
        #     random_state=np.random.default_rng(args.seed),
        #     data_sampler=sp.stats.norm(loc=0, scale=0.1).rvs
        # )

        # random_coeffs = 10 * random_coeffs.toarray().reshape([args.size, args.size, 4])
        # # random_coeffs = 10 * np.random.default_rng(args.seed).laplace(0, 0.1, [args.size, args.size, 4])
        # random_coeffs += np.random.default_rng(args.seed).normal(0, 1, [args.size, args.size, 4])
        # random_coeffs = create_coeffs(
        #     'Random Coeffs',
        #     random_coeffs
        # )
        random_coeffs = create_coeffs(
            'Random Coeffs',
            np.random.default_rng(args.seed).normal(0, 1, [args.size, args.size, 4])
        )

    if args.bayes:
        generate_test_cases('Bayes', bayes_threshold)

    if args.strict_sure:
        generate_test_cases('SURE (Strict)', sure_threshold)

    if args.hybrid_sure:
        generate_test_cases('SURE (Hybrid)', hybrid_sure_threshold)



if __name__ == '__main__':
    main()
