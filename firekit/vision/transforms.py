"""
Classes and functions for transforming images.
"""

# Imports ---------------------------------------------------------------------

import numpy as np

from torchvision.transforms.functional import pad

# Transform -------------------------------------------------------------------

class SquarePad:
    
    """
    Pad an image with zeros to the length of the longest dimension.
    """
    
    def __call__(self, image):

        height = image.shape[1]
        width = image.shape[2]
        
        if height != width:
            l = 0
            t = 0
            r = 0
            b = 0
            if height > width:
                height_pad = height - width
                if height_pad % 2 == 0:
                    l = int(height_pad / 2)
                    r = l
                else:
                    l = int(np.floor(height_pad / 2))
                    r = int(np.ceil(height_pad / 2))
            if height < width:
                width_pad = width - height
                if width_pad % 2 == 0:
                    t = int(width_pad / 2)
                    b = t
                else:
                    t = int(np.floor(width_pad / 2))
                    b = int(np.ceil(width_pad / 2))
            image = pad(image, (l, t, r, b))
        
        return image