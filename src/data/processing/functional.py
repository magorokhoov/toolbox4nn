import cv2
import numpy as np
import torch


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def to_tensor(img):
    r"""Convert a ``numpy.ndarray`` to tensor. (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    See ``ToTensor`` for more details.
    Args:
        img (np.ndarray, torch.Tensor): Image to be converted to tensor, (H x W x C[RGB]).
    Returns:
        Tensor: Converted image.
    """
    # if not(_is_numpy_image(img)):
        # raise TypeError('img should be ndarray. Got {}'.format(type(img)))

    # handle numpy array
    # img = torch.from_numpy(img.transpose((2, 0, 1)))
    if _is_numpy_image(img):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = torch.from_numpy(img.transpose((2, 0, 1))).contiguous()
        # backward compatibility
        if isinstance(img, torch.ByteTensor) or img.max() > 127 or img.dtype==torch.uint8:
            return img.float().div(255)
        return img

    elif _is_tensor_image(img):
        return img
    else:
        raise TypeError(f'img should be ndarray or tensor. Got {type(img)}')

def noise_gaussian(img:np.ndarray, mean:float=0.0,
    std=0.5, mode:str='gauss', gtype:str='color',
    rounds:bool=False, clip:bool=True) -> np.ndarray:
    r"""Add Gaussian noise (Additive) to the image.
    Alternatively, can also add Speckle noise to the image
    when using mode='speckle'
    Args:
        img: Image to be augmented.
        mean: Mean (“center”) of the Gaussian distribution,
            in range [0.0, 1.0].
        std: Standard deviation sigma (spread or “width”) of the
            Gaussian distribution that defines the noise level,
            either a single value for AWGN or a list of one per
            channel for multichannel (MC-AWGN). Values in range [0,
            255] for gaussian mode and [0.0, 1.0] for speckle mode.
            (Note: sigma = var ** 0.5)
        mode: select between standard purely additive gaussian
            noise (default) or speckle noise, in: `gauss`
            `speckle`.
        gtype: Type of Gaussian noise to add, either colored or
            grayscale (``color`` or ``bw``/``gray``).
            Default='color' (Note: can introduce color noise during
            training)
        rounds
    Returns:
        numpy ndarray: version of the image with the noise added.
    """
    h, w, c = img.shape
    img = img.astype(np.float32)

    mc = False
    if isinstance(std, list):
        mc = True
        if len(std) != c:
            std = [std[0]] * c

    if gtype in ('bw', 'gray'):
        if mc:
            std = std[0]
        noise = np.random.normal(
            loc=mean, scale=std, size=(h,w)).astype(np.float32)
        noise = np.expand_dims(noise, axis=2).repeat(c, axis=2)
    else:
        if mc:
            noise = np.zeros_like(img, dtype=np.float32)
            for ch, sig in enumerate(std):
                noise[..., ch] = np.random.normal(
                    loc=mean, scale=sig, size=(h,w))
        else:
            noise = np.random.normal(
                loc=mean, scale=std, size=(h,w,c)).astype(np.float32)

    if mode == 'speckle':
        noisy = (1 + noise) * img
    else:
        noisy = img + noise

    if rounds:
        noisy = round_up(noisy)

    if clip:
        noisy = np.clip(noisy, 0, 255)

    return noisy