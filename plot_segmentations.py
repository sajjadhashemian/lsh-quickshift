import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from math import log
from copy import copy
from time import time

from sklearn.preprocessing import normalize
from skimage.data import astronaut, coffee
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io, color

from fastqs import fast_quickshift

t=[]
t.append(time())

def color_segmentes(image, labels):
    labels = labels.astype(int)
    num_clusters = len(np.unique(labels))
    cmap = plt.cm.get_cmap('tab20', num_clusters)
    return cmap(labels)[:, :, :3]

def read_image(name):
    # img = img_as_float(coffee()[::2, ::2])
    img = mpimg.imread('./images/'+str(name)+'.jpg')
    img = img_as_float(img[::4, ::4])
    return img

def __plot(img, segments_fz, segments_slic, segments_quick, segments_watershed, segments_qs):
    fig, ax = plt.subplots(3, 2, figsize=(10, 10), sharex=True, sharey=True)

    ax[0, 0].imshow(color_segmentes(img, segments_fz))
    ax[0, 0].set_title("Felzenszwalbs's method")
    ax[0, 1].imshow(color_segmentes(img, segments_slic))
    ax[0, 1].set_title('SLIC')
    ax[1, 0].imshow(color_segmentes(img, segments_quick))
    ax[1, 0].set_title('Quickshift')
    ax[1, 1].imshow(color_segmentes(img, segments_watershed))
    ax[1, 1].set_title('Compact watershed')
    ax[2, 0].imshow(color_segmentes(img, segments_qs))
    ax[2, 0].set_title('LSH Quickshift')
    ax[2, 1].imshow(img)
    ax[2, 1].set_title('Original Image')

    for a in ax.ravel():
        a.set_axis_off()

    plt.tight_layout()
    plt.show()


def segment(img, _c):
    segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    segments_slic = slic(img, n_segments=250, compactness=10, sigma=1, start_label=1)
    segments_quick = quickshift(img, kernel_size=6, max_dist=6, ratio=0.5)
    gradient = sobel(rgb2gray(img))
    segments_watershed = watershed(gradient, markers=250, compactness=0.001)
    segments_qs = fast_quickshift(img, c = _c, ratio=0.5, sigma=1.5)

    print(f'Felzenszwalb number of segments: {len(np.unique(segments_fz))}')
    print(f'SLIC number of segments: {len(np.unique(segments_slic))}')
    print(f'Quickshift number of segments: {len(np.unique(segments_quick))}')
    print(f'Watershed number of segments: {len(np.unique(segments_watershed))}')
    print(f'LSH Quickshift number of segments: {len(np.unique(segments_qs))}')

    __plot(img, segments_fz, segments_slic, segments_quick, segments_watershed, segments_qs)

if __name__=="__main__":
    img = read_image(4)
    segment(img, 1.2)