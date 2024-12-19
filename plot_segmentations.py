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
from sklearn.cluster import estimate_bandwidth, MeanShift

class RESULT:
    def __init__(self, algo_name, lable, t, img):
        self.name = algo_name
        self.lable = lable
        self.t = t
        self.img = img

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

def __plot(res):
    fig, ax = plt.subplots(len(res), 7, figsize=(12, len(res)+1), sharex=True, sharey=True)
    
    for row, img in enumerate(res):
        ax[row, 0].imshow(img[0].img, aspect='equal')
        ax[row, 0].set_title('Original Image', fontsize=7)
        for col, alg in enumerate(res[row]):
            ax[row, col+1].imshow(color_segmentes(img, alg.lable), aspect='equal')
            ax[row, col+1].set_title(f'{alg.name}, #: {len(np.unique(alg.lable))}, (s): {alg.t:.2f}', fontsize=7)

    for a in ax.ravel():
        a.set_axis_off()
    plt.tight_layout()
    plt.savefig('segmentation.svg')
    # plt.show()


def segment(img, _c):
    t, res = [], []

    t.append(time())    
    segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    t.append(time()-t[-1])
    res.append(RESULT('Felzenszwalb', segments_fz, t[-1], img))

    t.append(time())
    segments_watershed = watershed(sobel(rgb2gray(img)), markers=25, compactness=0.001)
    t.append(time()-t[-1])
    res.append(RESULT('Watershed', segments_watershed, t[-1], img))

    t.append(time())    
    segments_slic = slic(img, n_segments=30, compactness=10, sigma=1, start_label=1)
    t.append(time()-t[-1])
    res.append(RESULT('SLIC', segments_slic, t[-1], img))

    t.append(time())    
    segments_quick = quickshift(img, kernel_size=6, max_dist=8, ratio=0.5)
    t.append(time()-t[-1])
    res.append(RESULT('Quickshift', segments_quick, t[-1], img))

    t.append(time())
    segments_qs = fast_quickshift(img, c = _c, ratio=0.5, sigma=1.5)
    t.append(time()-t[-1])
    res.append(RESULT('LSH-Quickshift', segments_qs, t[-1], img))

    t.append(time())
    image = np.reshape(img, (-1,3))
    segments_ms = MeanShift(bandwidth = estimate_bandwidth(image, n_samples=100)/2, bin_seeding=True, max_iter=400).fit(image)
    segments_ms = np.reshape(segments_ms.labels_, (img.shape[0], img.shape[1]))
    t.append(time()-t[-1])
    res.append(RESULT('Meanshift', segments_ms, t[-1], img))

    return res

if __name__=="__main__":
    images = [img_as_float(coffee()[::4, ::4])]
    for i in range(9):
        images.append(read_image(i+1))
    res = []
    for i, img in enumerate(images):
        print(i)
        res.append(segment(img, 1.3))
    
    __plot(res)