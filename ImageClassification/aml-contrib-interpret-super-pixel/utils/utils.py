import matplotlib.pylab as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from skimage.segmentation import slic

colors = []
for l in np.linspace(1, 0, 100):
    colors.append((245/255, 39/255, 87/255, l))
for l in np.linspace(0, 1, 100):
    colors.append((24/255, 196/255, 93/255, l))
cmap = LinearSegmentedColormap.from_list("shap", colors)


def fill_segmentation(values, segmentation):
    out = np.zeros(segmentation.shape)
    for i in range(len(values)):
        out[segmentation == i] = values[i]
    return out


def show_explanation(img,n_segments, shap_values, lables, preds):
    segments_slic = slic(img, n_segments=n_segments, compactness=30, sigma=3)
    inds = np.argsort(-preds)[0]

    fig, axes = plt.subplots(nrows=1, ncols=len(lables)+1, figsize=(12,4))
    axes[0].imshow(img)
    axes[0].axis('off')
    max_val = np.max([np.max(np.abs(shap_values[i][:,:-1])) for i in range(len(shap_values))])
    for i in range(len(lables)):
        axes[i+1].set_title(lables[i][1]+':  '+str(lables[i][2]))
        axes[i+1].imshow(img.convert('LA'), alpha=0.15)
        im = axes[i+1].imshow(fill_segmentation(shap_values[inds[i]][0], segments_slic), cmap=cmap, vmin=-max_val, vmax=max_val)
        axes[i+1].axis('off')
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), label="SHAP value", orientation="horizontal", aspect=60)
    cb.outline.set_visible(False)
    return fig


def prep_shaps(y, n, n_segments):
    shap_values = []
    for c in range(n):
        _shap = []
        for _val in y.local_importance_values:
            _shap.append(np.array(_val[c]).reshape(1, n_segments))
        shap_values.append(_shap)

    return shap_values
