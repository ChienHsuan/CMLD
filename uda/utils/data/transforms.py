import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
import random
import math
import numpy as np


class RandomPadding(object):
    """Random padding
    """
    def __init__(self, p=0.5, padding=(0, 10), **kwargs):
        self.p = p
        self.padding_limits = padding

    def __call__(self, img):
        if random.uniform(0., 1.) > self.p:
            return img

        rnd_padding = [random.randint(self.padding_limits[0], self.padding_limits[1]) for _ in range(4)]
        rnd_fill = random.randint(0, 255)

        img = TF.pad(img, rnd_padding, fill=rnd_fill, padding_mode='constant')

        return img


class Random2DTranslation(object):
    """Randomly translates the input image with a probability.

    Specifically, given a predefined shape (height, width), the input is first
    resized with a factor of 1.125, leading to (height*1.125, width*1.125), then
    a random crop is performed. Such operation is done with a probability.

    Args:
        height (int): target image height.
        width (int): target image width.
        p (float, optional): probability that this operation takes place.
            Default is 0.5.
        interpolation (int, optional): desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """
    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img.resize((self.width, self.height), self.interpolation)

        new_width, new_height = int(round(self.width * 1.125)
                                    ), int(round(self.height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop(
            (x1, y1, x1 + self.width, y1 + self.height)
        )
        return croped_img


class RandomColorJitter(object):
    def __init__(self, p=0.5, brightness=0.2, contrast=0.15, saturation=0, hue=0, **kwargs):
        self.p = p
        self.transform = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img

        img = self.transform(img)

        return img


class GaussianBlur(object):
    def __init__(self, p=0.5, sigma=(0.1, 2.0)):
        self.p = p
        self.sigma = sigma

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img

        sigma = random.uniform(self.sigma[0], self.sigma[1])
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))

        return img


class RandomRotate(object):
    def __init__(self, p=0.5, degrees=(-10, 10), **kwargs):
        self.p = p
        self.transform = T.RandomRotation(degrees)

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img

        img = self.transform(img)

        return img


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


def build_transforms(
    height,
    width,
    transforms=['random_flip'],
    norm_mean=[0.485, 0.456, 0.406],
    norm_std=[0.229, 0.224, 0.225]
):
    if len(transforms) > 0:
        transforms = [t.lower() for t in transforms]

    normalize = T.Normalize(mean=norm_mean, std=norm_std)

    print('Building train transforms ...')
    transform_tr = []

    if 'random_padding' in transforms:
        print('+ random padding')
        transform_tr += [RandomPadding(p=0.5)]

    if 'random_crop' in transforms:
        print(
            '+ random crop (enlarge to {}x{} and '
            'crop {}x{})'.format(
                int(round(height * 1.125)), int(round(width * 1.125)), height,
                width
            )
        )
        transform_tr += [Random2DTranslation(height, width, p=0.5)]

    print('+ resize to {}x{}'.format(height, width))
    transform_tr += [T.Resize((height, width))]

    if 'random_flip' in transforms:
        print('+ random flip')
        transform_tr += [T.RandomHorizontalFlip(p=0.5)]

    if 'color_jitter' in transforms:
        print('+ color jitter')
        transform_tr += [
            RandomColorJitter(p=0.5, brightness=0.15, contrast=0.15, saturation=0.1, hue=0.1)
        ]

    if 'gaussian_blur' in transforms:
        print('+ gaussian blur')
        transform_tr += [GaussianBlur(p=0.5)]

    if 'random_rotate' in transforms:
        print('+ random rotate')
        transform_tr += [RandomRotate(p=0.5)]

    print('+ to torch tensor of range [0, 1]')
    transform_tr += [T.ToTensor()]

    print('+ normalization (mean={}, std={})'.format(norm_mean, norm_std))
    transform_tr += [normalize]

    if 'random_erase' in transforms:
        print('+ random erase')
        transform_tr += [RandomErasing(probability=0.5, mean=norm_mean)]

    transform_tr = T.Compose(transform_tr)

    print('Building test transforms ...')
    print('+ resize to {}x{}'.format(height, width))
    print('+ to torch tensor of range [0, 1]')
    print('+ normalization (mean={}, std={})'.format(norm_mean, norm_std))

    transform_te = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize,
    ])

    return transform_tr, transform_te
