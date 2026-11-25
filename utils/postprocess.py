import numpy as np


def spixel_to_pixel_labels(sp_level_label, association_mat):
    sp_level_label = np.reshape(sp_level_label, (-1, 1))
    pixel_level_label = (association_mat @ sp_level_label).reshape(-1)
    return pixel_level_label.astype('int')
