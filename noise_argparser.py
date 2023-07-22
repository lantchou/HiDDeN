import argparse
import re
import torch
from noise_layers.cropout import Cropout
from noise_layers.crop import Crop
from noise_layers.identity import Identity
from noise_layers.dropout import Dropout
from noise_layers.resize import Resize
from noise_layers.quantization import Quantization
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.rotate import Rotate
from noise_layers.gaussian_blur import GaussianBlur
from noise_layers.jpeg_diff import JpegDiff
from noise_layers.translate import Translate


def parse_pair(match_groups):
    heights = match_groups[0].split(',')
    hmin = float(heights[0])
    hmax = float(heights[1])
    widths = match_groups[1].split(',')
    wmin = float(widths[0])
    wmax = float(widths[1])
    return (hmin, hmax), (wmin, wmax)


def parse_crop(crop_command):
    matches = re.match(r'crop\(\((\d+\.*\d*,\d+\.*\d*)\),\((\d+\.*\d*,\d+\.*\d*)\)\)', crop_command)
    (hmin, hmax), (wmin, wmax) = parse_pair(matches.groups())
    return Crop((hmin, hmax), (wmin, wmax))


def parse_cropout(cropout_command):
    matches = re.match(r'cropout\(\((\d+\.*\d*,\d+\.*\d*)\),\((\d+\.*\d*,\d+\.*\d*)\)\)', cropout_command)
    (hmin, hmax), (wmin, wmax) = parse_pair(matches.groups())
    return Cropout((hmin, hmax), (wmin, wmax))


def parse_dropout(dropout_command):
    matches = re.match(r'dropout\((\d+\.*\d*,\d+\.*\d*)\)', dropout_command)
    ratios = matches.groups()[0].split(',')
    keep_min = float(ratios[0])
    keep_max = float(ratios[1])
    return Dropout((keep_min, keep_max))


def parse_resize(resize_command):
    matches = re.match(r'resize\((\d+,\d+)\)', resize_command)
    ratios = matches.groups()[0].split(',')
    min_ratio = int(ratios[0])
    max_ratio = int(ratios[1])
    return Resize((min_ratio, max_ratio))


def parse_rotate(rotate_command):
    matches = re.match(r'rotate\((\d+,\d+)\)', rotate_command)
    angles = matches.groups()[0].split(',')
    min_angle = int(angles[0])
    max_angle = int(angles[1])
    return Rotate(min_angle, max_angle)


def parse_blur(blur_command):
    # parse single int between parentheses
    matches = re.match(r'blur\((\d+)\)', blur_command)
    sigma = int(matches.groups()[0])
    return GaussianBlur(sigma)


def parse_jpeg_diff(jpeg_command, device):
    matches = re.match(r'diffjpeg\((\d+)\)', jpeg_command)
    quality = int(matches.groups()[0])
    return JpegDiff(device, quality=quality)


def parse_translate(translate_command):
    matches = re.match(r'translate\((\d+\.*\d*,\d+\.*\d*)\)', translate_command)
    ratios = matches.groups()[0].split(',')
    min_ratio = float(ratios[0])
    max_ratio = float(ratios[1])
    return Translate((min_ratio, max_ratio))

class NoiseArgParser(argparse.Action):
    def __init__(self,
                 option_strings,
                 dest,
                 nargs=None,
                 const=None,
                 default=None,
                 type=None,
                 choices=None,
                 required=False,
                 help=None,
                 metavar=None,
                 device=torch.device('cpu'),):
        argparse.Action.__init__(self,
                                 option_strings=option_strings,
                                 dest=dest,
                                 nargs=nargs,
                                 const=const,
                                 default=default,
                                 type=type,
                                 choices=choices,
                                 required=required,
                                 help=help,
                                 metavar=metavar,
                                 )
        self.device = device

    @staticmethod
    def parse_cropout_args(cropout_args):
        pass

    @staticmethod
    def parse_dropout_args(dropout_args):
        pass

    def __call__(self, parser, namespace, values,
                 option_string=None):

        layers = []
        split_commands = values[0].split('+')

        for command in split_commands:
            # remove all whitespace
            command = command.replace(' ', '')
            if command[:len('cropout')] == 'cropout':
                layers.append(parse_cropout(command))
            elif command[:len('crop')] == 'crop':
                layers.append(parse_crop(command))
            elif command[:len('dropout')] == 'dropout':
                layers.append(parse_dropout(command))
            elif command[:len('resize')] == 'resize':
                layers.append(parse_resize(command))
            elif command[:len('jpeg')] == 'jpeg':
                layers.append('JpegPlaceholder')
            elif command[:len('diffjpeg')] == 'diffjpeg':
                layers.append(parse_jpeg_diff(command, self.device))
            elif command[:len('quant')] == 'quant':
                layers.append('QuantizationPlaceholder')
            elif command[:len('rotate')] == 'rotate':
                layers.append(parse_rotate(command))
            elif command[:len('blur')] == 'blur':
                layers.append(parse_blur(command))
            elif command[:len('mirror')] == 'mirror':
                layers.append('Mirror')
            elif command[:len('translate')] == 'translate':
                layers.append(parse_translate(command))
            elif command[:len('shear')] == 'shear':
                layers.append('Shear')
            elif command[:len('identity')] == 'identity':
                # We are adding one Identity() layer in Noiser anyway
                pass
            else:
                raise ValueError('Command not recognized: \n{}'.format(command))
        setattr(namespace, self.dest, layers)
