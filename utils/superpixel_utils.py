import numpy as np
from scipy.sparse import csr_matrix
from skimage.measure import regionprops
from skimage.segmentation import slic, mark_boundaries, find_boundaries
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale


from sklearn.neighbors import kneighbors_graph

def HSI_to_superpixels(img, n_superpixels, compactness):
    return slic(
        img, n_segments=n_superpixels, compactness=compactness,
        start_label=0, convert2lab=False,
    )


def show_superpixel(label, x=None, save_path='superpixels.pdf'):
    color = (162 / 255, 169 / 255, 175 / 255)
    n_row, n_col, n_band = x.shape
    if x is not None:
        x = minmax_scale(x.reshape(n_row * n_col, n_band)).reshape(n_row, n_col, n_band)
        mask = mark_boundaries(x, label, color=(1, 1, 0), mode='subpixel')
        # mask = x
    else:
        mask_boundary = find_boundaries(label, mode='subpixel')
        mask = np.ones((n_row, n_col, 3))
        mask[mask_boundary] = color
    fig = plt.figure()
    plt.imshow(mask)
    plt.axis('off')
    plt.tight_layout()
    fig.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0)
    plt.close()


def create_association_mat(superpixel_labels):
    labels, columns = np.unique(superpixel_labels.ravel(), return_inverse=True)
    rows = np.arange(columns.size)
    return csr_matrix((np.ones(rows.size), (rows, columns)), shape=(rows.size, labels.size))



def create_spixel_graph(source_img, superpixel_labels, n_neighbors=50):
    s = source_img.reshape((-1, source_img.shape[-1]))
    a = create_association_mat(superpixel_labels)
    # t = superpixel_labels.reshape(-1)
    mean_fea = a.T @ s
    regions = regionprops(superpixel_labels + 1)
    n_labels = np.unique(superpixel_labels).shape[0]
    center_indx = np.zeros((n_labels, 2))
    for i, props in enumerate(regions):
        center_indx[i, :] = props.centroid  # centroid coordinates
    ss_fea = np.concatenate((mean_fea, center_indx), axis=1)
    ss_fea = minmax_scale(ss_fea)
    try:
        adj = kneighbors_graph(ss_fea, n_neighbors=n_neighbors, mode='distance', include_self=False).toarray()
    except:
        adj = kneighbors_graph(ss_fea, n_neighbors=np.unique(superpixel_labels).shape[0] // 2, mode='distance', include_self=False).toarray()

    # # # show initial graph
    # import matplotlib.pyplot as plt
    # adj_ = np.copy(adj)
    # adj_[np.where(adj != 0)] = 1
    # plt.imshow(adj_, cmap='hot')
    # plt.show()

    # # auto calculate gamma in Gaussian kernel
    X_var = ss_fea.var()
    gamma = 1.0 / (ss_fea.shape[1] * X_var) if X_var != 0 else 1.0
    adj[np.where(adj != 0)] = np.exp(-np.power(adj[np.where(adj != 0)], 2) * gamma)

    # adj = euclidean_dist(ss_fea, ss_fea).numpy()
    # adj = np.exp(-np.power(adj, 2) * gamma)
    np.fill_diagonal(adj, 0)

    return adj


def extract_superpixel_features(img, labels, patch_size):
    num_labels = np.unique(labels).size
    assert patch_size % 2 == 1, "Patch size must be odd"
    features = np.zeros((num_labels, patch_size, patch_size, img.shape[2]), dtype='float32')  # 初始化特征向量矩阵
    regions = regionprops(labels + 1)
    pad_size = patch_size // 2
    pad_img = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant')
    for i, props in enumerate(regions):
        center = props.centroid  # centroid coordinates
        x, y = center
        x = int(x + pad_size)
        y = int(y + pad_size)
        feat = pad_img[x - pad_size:x + pad_size + 1, y - pad_size:y + pad_size + 1, :]
        features[i, :] = feat
    features = np.transpose(features, (0, 3, 1, 2))  # 转置为[n, c, h, w]
    shape = features.shape
    features = minmax_scale(features.reshape(features.shape[0], -1), axis=1)  # 归一化
    features = features.reshape(shape)
    return features
