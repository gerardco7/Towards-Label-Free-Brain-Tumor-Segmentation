import numpy as np
import matplotlib.pyplot as plt

def ensamble_method(diffs_post_all, ensamble):

    diffs_post_all = np.array(diffs_post_all)

    if ensamble == 'mean':
        diffs_post = np.mean(diffs_post_all, axis=0)
    elif ensamble == 'sum':
        diffs_post = np.sum(diffs_post_all, axis=0)
    elif ensamble == 'voting':
        diffs_post = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=diffs_post_all.astype(int))
    elif ensamble == 'intersection':
        diffs_post = np.sum(diffs_post_all, axis=0) > 1

    return (diffs_post > 0).astype(int)