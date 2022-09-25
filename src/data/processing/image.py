import numpy as np
import cv2


def read_image(img_path, mode, loader='cv', warning=True) -> np.array:
    '''
    Arguments:
        img_path: path to image
        mode:
            1. if loader==cv =>
                1.1 gray
                1.2 rgb
                1.3 rgba
                1.4 bgr
                1.5 bgra
                1.6 unchanged
            2. if loader==numpy (not implemented)
        loader: cv, numpy
        warning: True, False (for example, mode is rgba but it is grayscale)
    '''

    img = None
    if loader == 'cv':
        if mode == 'gray':
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        elif mode == 'gray3dim':
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = np.expand_dims(img, axis=2)

        elif mode in ('rgb', 'bgr'):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if mode == 'rgb':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        elif mode in ('rgba', 'bgra'):
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if len(img.shape) == 2: # Grayscale images
                if warning:
                    print(f'{img_path} is grayscale, but mode is {mode}')

                if mode == 'rgba':
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
                elif mode == 'bgra':
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)

            elif img.shape[2] == 3:
                if warning:
                    print(f'{img_path} has 3 color channels, but mode is {mode}')
                
                if mode == 'rgba':
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
                elif mode == 'bgra':
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        elif mode == 'unchanged':
            raise NotImplementedError(f'mode [{mode}] is not implemented')
        else:
            raise NotImplementedError(f'image read mode {mode} is not implemented')
    else:
        raise NotImplementedError(f'imafe read loader {loader} is not implemented')

    return img