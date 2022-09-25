import os
import numbers
import random
import cv2
import numpy as np

import data.processing.functional as F

__all__ = ["Compose", "ToTensor"]

class Compose:
    """Composes several transforms together.
    
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string

class ToTensor:
    """Convert a ``numpy.ndarray`` to tensor.

    Converts a numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class RandomBase:
    r"""Base class for randomly applying transform
    Args:
        p (float): probability of applying the transform.
            Default: 0.5
    """
    def __init__(self, p:float=0.5):
        assert isinstance(p, numbers.Number) and p >= 0, 'p should be a positive value'
        self.p = p
        self.params = {}

    def apply(self, image, **params):
        """Dummy, use the appropriate function in each class"""
        return image

    def __call__(self, image, **params):
        """
        Args:
            image (np.ndarray): Image to be transformed.

        Returns:
            np.ndarray: Randomly transformed image.
        """
        if random.random() < self.p:
            return self.apply(image, **self.params)
        return image

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomGaussianNoise(RandomBase):
    """Apply gaussian noise on the given image randomly with a
    given probability.
    Args:
        p (float): probability of the image being noised.
            Default value is 0.5
        mean (float): Mean (“center”) of the Gaussian distribution.
            Default=0.0
        var_limit ((float, float) or float): variance range for noise.
            If var_limit is a single float, the range will be
            (0, var_limit). Should be in range [0, 255] if using
            `sigma_calc='sig'` or squared values of that range if
            `sigma_calc='var'`. Default: (10.0, 50.0).
        prob_color: Probably for selecting the type of Gaussian noise
            to add, either colored or grayscale (``color`` or ``gray``),
            in range [0.0, 1.0], higher means more chance of `color`.
            (Note: Color type can introduce color noise during training)
        multi: select to randomly generate multichannel-AWGN (MC-AWGN)
            in addition to regular AWGN.
        mode: select between Gaussian or Speckle noise modes
        sigma_calc: select if var_limit is to be considered as the
            variance (final sigma will be ``sigma = var ** 0.5``) or
            sigma range (var_limit will be used directly for sigma).
            In: `var`, `sig`.
    """

    def __init__(self, p:float=0.5, mean:float=0.0,
        var_limit=(10.0, 50.0), prob_color:float=0.5,
        multi:bool=True, mode:str='gauss', sigma_calc:str='sig'):
        super(RandomGaussianNoise, self).__init__(p=p)

        if not isinstance(mean, numbers.Number) or mean < 0:
            raise ValueError('Mean should be a positive value')
        if isinstance(var_limit, (tuple, list)):
            if var_limit[0] < 0 or var_limit[1] < 0:
                raise ValueError(
                    f"var_limit values: {var_limit} should be non negative.")
            self.var_limit = var_limit
        elif isinstance(var_limit, (int, float)):
            if var_limit < 0:
                raise ValueError("var_limit should be non negative.")
            self.var_limit = (0, var_limit)
        else:
            raise TypeError(
                "Expected var_limit type to be one of (int, float, "
                f"tuple, list), got {type(var_limit)}"
            )

        if not isinstance(prob_color, (int, float)):
            raise ValueError('prob_color must be a number in [0, 1]')
        self.prob_color = prob_color
        self.mean = mean
        self.mode = mode
        self.multi = multi
        self.sigma_calc = sigma_calc
        self.params = self.get_params()

    def apply(self, img, **params):
        return F.noise_gaussian(img, **params)

    def get_params(self):
        """Get parameters for gaussian noise
        Returns:
            dict: params to be passed to the affine transformation
        """
        # mean = random.uniform(-self.mean, self.mean) #= 0

        gtype = 'color' if random.random() < self.prob_color else 'gray'

        multi = False
        if self.multi and random.random() > 0.66 and gtype == 'color':
            # will only apply MC-AWGN 33% of the time
            multi = True
        if multi:
            lim = self.var_limit
            sigma = [random.uniform(lim[0], lim[1]) for _ in range(3)]
            if self.mode == "gauss":
                sigma = [(v ** 0.5) for v in sigma]
        else:
            # ref wide range: (4, 200)
            var = random.uniform(self.var_limit[0], self.var_limit[1])

            if self.mode == "gauss":
                if self.sigma_calc == 'var':
                    sigma = (var ** 0.5)
                elif self.sigma_calc == 'sig':
                    # no need to var/255 if image range in [0,255]
                    sigma = var
            elif self.mode == "speckle":
                sigma = var

        return {"mean": self.mean,
                "std": sigma,
                "mode": self.mode,
                "gtype": gtype,
                "rounds": False,
                "clip": True,
            }