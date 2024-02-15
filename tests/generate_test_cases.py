
import numpy as np
import pywt
import cv2
from pathlib import Path
import json
import argparse
import os
import contextlib
import shutil

class TestCaseGenerator:
    def __init__(
            self, 
            manifest_path: Path | str, 
            root_path: Path | str, 
            input_filename: str = 'input.tiff', 
            coeffs_filename: str = 'coeffs.tiff',
            dry_run=False,
            verbose=False,
        ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root_path = Path(root_path)
        self.input_filename = input_filename
        self.coeffs_filename = coeffs_filename
        self.dry_run = dry_run
        self.verbose = verbose

        self.manifest = list()
        if not self.dry_run:
            self.manifest_path.unlink(missing_ok=True)
            shutil.rmtree(self.root_path)
            self.root_path.mkdir(parents=True)

        # try:
        #     with open(self.manifest_path) as file:
        #         self.manifest = json.load(file)
        # except FileNotFoundError:
        #     self.manifest = list()

        

        
    def discard_existing_cases(self):
        for item in self.manifest:
            Path(item['path']).unlink()

        self.manifest = list()


    def create_manifest_entry(self, name, wavelet, mode, flags):
        path = self.create_path(name, wavelet, mode)
        return dict(
            wavelet=wavelet,
            mode=mode,
            name=name,
            flags=flags,
            path=str(path),
            input_filename=self.input_filename,
            coeffs_filename=self.coeffs_filename,
        )
    

    def format_manifest_entry(self, entry):
        return '\n'.join(f'{key}: {value}' for key, value in entry.items())
        
            
    def create_path(self, name, wavelet, mode) -> Path:
        return self.root_path / wavelet / mode / str(name)
    

    def generate_case(self, data, name, wavelet, mode, flags=-1):
        if isinstance(data, (str, os.PathLike)):
            data_path = data
            data = cv2.imread(str(data), flags)
        else:
            data_path = None

        #   Update the manifest
        manifest_entry = self.create_manifest_entry(
            name=name,
            wavelet=wavelet,
            mode=mode,
            flags=flags,
        )
        self.manifest.append(manifest_entry)

        #   Perform the DWT
        coeffs = pywt.wavedec2(data, wavelet, mode)
        coeffs, _ = pywt.coeffs_to_array(coeffs)
        
        if self.verbose:
            print(self.format_manifest_entry(manifest_entry))
            print('-' * 80)

        #   Write files
        if not self.dry_run:
            path = Path(manifest_entry['path'])
            input_path = path / self.input_filename
            coeffs_path = path / self.coeffs_filename
            path.mkdir(parents=True, exist_ok=True)
            
            cv2.imwrite(str(coeffs_path), coeffs)
            if data_path:
                input_path.unlink(missing_ok=True)
                # input_path.symlink_to(data_path)
                shutil.copy(data_path, input_path)
            else:
                cv2.imwrite(str(input_path), data)

            with open(self.manifest_path, 'w') as file:
                json.dump(self.manifest, file, indent=4)



def doit():
    # image = cv2.imread('inputs/lena.png')
    # gray_image = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)
    # gray_image = gray_image.astype(float) / 255
    # print(gray_image.shape, gray_image.dtype)
    # cv2.imwrite('inputs/lena_gray.png', gray_image, )
    pass


def main(inputs):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        action='append',
        dest='wavelets',
        help='Wavelet used to generate test cases',
    )
    parser.add_argument(
        '--mode', '-m',
        action='append',
        default=['symmetric'],
        type=list,
        dest='modes',
        help='Boundary mode used to generate test cases',
    )
    parser.add_argument(
        '--dry-run', '-d',
        action='store_true', 
        help='Generate test cases, but do not save files',
    )
    parser.add_argument(
        '--verbose', '-v', 
        action='store_true',
        help='Print manifest entries generation',
    )
    parser.add_argument(
        '--manifest',
        default='cases_manifest.json',
        help='The JSON manifest path',
    )
    parser.add_argument(
        '--root',
        default='data',
        help='The root path for generated test case data',
    )
    args = parser.parse_args()
    
    generator = TestCaseGenerator(
        manifest_path=args.manifest,
        root_path=args.root,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    for wavelet in args.wavelets:
        for mode in args.modes:
            for name, data in enumerate(inputs):
                data, flags = data if isinstance(data, (tuple, list)) else (data, cv2.IMREAD_UNCHANGED)
                generator.generate_case(
                    data=data,
                    name=name,
                    wavelet=wavelet,
                    mode=mode,
                    flags=flags,
                )




if __name__ == '__main__':
    inputs = [
        np.ones([32, 32], dtype=np.float32),
        ('inputs/lena_gray.png', cv2.IMREAD_GRAYSCALE),
    ]
    main(inputs)



