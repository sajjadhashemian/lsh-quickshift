from cpp import qs
import numpy as np
from math import log

from skimage._shared.filters import gaussian
from skimage._shared.utils import _supported_float_type
from skimage.color import rgb2lab
from skimage.util import img_as_float


def fast_quickshift(
    image,
    k=None,
    _k=None,
    c = 2,
    ratio=1.0,
    sigma=0,
    convert2lab=True,
    *,
    channel_axis=-1,
):

    image = img_as_float(np.atleast_3d(image))
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)

    if image.ndim > 3:
        raise ValueError("Only 2D color images are supported")

    # move channels to last position as expected by the Cython code
    image = np.moveaxis(image, source=channel_axis, destination=-1)

    if convert2lab:
        if image.shape[-1] != 3:
            raise ValueError("Only RGB images can be converted to Lab space.")
        image = rgb2lab(image)

    image = gaussian(image, sigma=[sigma, sigma, 0], mode='reflect', channel_axis=-1)
    image = np.ascontiguousarray(image * ratio)


    __shape = (image.shape[0], image.shape[1])
    image = np.reshape(image, (-1,3))
    
    if(k == None):
        k = int(log(image.shape[0])**c)

    if(_k == None):
        _k = int(log(image.shape[0]**0.5)**c)
    
    segment_mask = qs.fast_lsh_quickshift(image, k, _k)
    segment_mask = np.array(segment_mask)
    # counter = 0
    # cluster_index = dict({})
    # for i, x in enumerate(segment_mask):
    #     if(x not in cluster_index.keys()):
    #         cluster_index[x] = counter
    #         counter += 1
    #     segment_mask[i] =  cluster_index[x]
    u, segment_mask = np.unique(segment_mask,  return_inverse=True)
    segment_mask = np.reshape(segment_mask, newshape=__shape)

    return segment_mask