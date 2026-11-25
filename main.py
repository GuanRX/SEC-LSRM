import os
from argparse import ArgumentParser
from pathlib import Path
import random
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import Adam
from time import time
from copy import deepcopy

import warnings
warnings.filterwarnings('ignore')

from dataset import MMDataset, get_config
from utils.evaluation import cluster_accuracy, get_parameter_number
from utils.postprocess import spixel_to_pixel_labels
from torch_clustering import PyTorchKMeans
from model.multiview3 import MultiViewModel
from model.BDR_lowrank import get_sampleMatrix
from model.loss import correlation_reduction_loss, sampling_contrastive_loss


PROJECT_DIR = Path(__file__).resolve().parent
config = get_config(PROJECT_DIR / 'config.json')
if config['dataset_name'] != 'Trento':
    color_tabel = np.array([[0, 0, 0],
                            [110, 240, 70], [0, 0, 243], [230, 50, 30], [70, 157, 71], [230, 50, 240],
                            [255, 255, 84], [83, 183, 223], [227, 203, 244], [130, 80, 180],
                            [10, 35, 78], [170, 50, 100], [113, 180, 50], [200, 100, 60],
                            [60, 128, 161], [210, 146, 60], [84, 185, 54], [122, 75, 50], ])
else:
    color_tabel = np.array([[0, 0, 0],
                            [56, 108, 53], [228, 55, 38], [230, 240, 85], [170, 228, 49],
                            [147, 75, 48], [0, 204, 0], ])



# color_tabel = minmax_scale(color_tabel, feature_range=(0, 1))
color_tabel = color_tabel / 255.0

def visualization(points, shape, true, pred, save_path=None):
    fig_true = np.zeros((shape[0], shape[1], 3))
    fig_pred = np.zeros((shape[0], shape[1], 3))
    colors_true = [color_tabel[true[i]] for i in range(true.shape[0])]
    colors_pred = [color_tabel[pred[i]] for i in range(pred.shape[0])]
    fig_true[points] = colors_true
    fig_pred[points] = colors_pred
    # for i in range(points.shape[0]):
    #     p = points[i]
    #     fig_true[p[0], p[1], :] = colors_true[i]
    #     fig_pred[p[0], p[1], :] = colors_pred[i]

    plt.figure(figsize=(10, 7))
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(fig_true)
    ax1.set_title('true')
    ax1.axis('off')
    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(fig_pred)
    ax2.set_title('predict')
    ax2.axis('off')
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print('### Save figure in {}. ###'.format(save_path))
    plt.close()


def evaluate(y_pre_sp, association_mat, gt):
    if isinstance(y_pre_sp, torch.Tensor):
        y_pre_sp = y_pre_sp.cpu().numpy()
    if isinstance(association_mat, torch.Tensor):
        association_mat = association_mat.cpu().numpy()
    y_pre_pixel = spixel_to_pixel_labels(y_pre_sp, association_mat)
    class_map = y_pre_pixel.reshape(gt.shape)

    indx = np.where(gt != 0)
    y_target = gt[indx]
    y_predict = class_map[indx]
    y_predict, acc, kappa, nmi, ari, pur, ca = cluster_accuracy(y_target, y_predict, return_aligned=True)
    return (acc, kappa, nmi, ari, pur), ca, (indx, y_target.astype(int), y_predict)



def print_table(data, headers, col_width=12):
    num_cols = len(headers)
    col_spacing = 3
    
    # create row format
    row_format = "".join(["{:>"+str(col_width)+"}"]*(num_cols))
    row_separator = "".join(["{:->"+str(col_width)+"}"]*(num_cols))
    
    # print header
    print(row_format.format(*headers))
    print(row_separator.format(*["" for _ in range(num_cols)]))
    
    # print data rows
    for i, row in enumerate(data):
        row_values = [f"{val:.4f}" if isinstance(val, float) else val for val in row]
        print(row_format.format(*row_values))



def train(config, pretrain=True):
    dataset_name = config['dataset_name']
    dataset = MMDataset(
        dataset_name, 
        config[dataset_name]['mm_data_path'], 
        config[dataset_name]['gt_path'],
        config[dataset_name]['n_pc'],
        config[dataset_name]['patch_size'],
        config[dataset_name]['rgb'],
        config[dataset_name]['n_neighbors'],
        config[dataset_name]['n_superpixels'],
        slic_compactness=config[dataset_name].get('slic_compactness', 0.3),
        superpixel_path=Path(config['result_path']) / f'{dataset_name}_sp.pdf',
    )
    print(dataset)
    if config['remove_bkg']:
        dataset.remove_background()
    n_clusters = dataset.n_classes
    n_samples = len(dataset)
    lr = config['lr']
    device = torch.device(config['device'])
    EPOCH = config['n_epoches']
    
    gae_n_enc_1, gae_n_enc_3 = config['gae_n_enc_1'], config['gae_n_enc_3']
    gae_n_dec_1, gae_n_dec_3 = config['gae_n_dec_1'], config['gae_n_dec_3']
    max_iter, lambda_val, gamma, zs, ratio, tau = config[dataset_name]['max_iter'], config[dataset_name]['lambda_val'], \
        config[dataset_name]['gamma'], config[dataset_name]['zs'], config[dataset_name]['ratio'], config[dataset_name]['tau']
    alpha_recon, alpha_contrast, alpha_align = config[dataset_name]['alpha_recon'], config[dataset_name]['alpha_contrast'], config[dataset_name]['alpha_align']
    fuse_type = config['fuse_type']

    save_path = config['save_path'] + '/' + dataset_name + '.pkl'
    n_views = len(dataset.x)

    x = [torch.from_numpy(it).float().to(device) for it in dataset.x]
    # x.append(torch.cat(x, dim=1))
    adj = []
    for matrix in dataset.adj:
        symmetric = np.maximum(matrix, matrix.T)
        rows, cols = symmetric.nonzero()
        edge_index = torch.from_numpy(np.stack((rows, cols))).to(device)
        edge_weight = torch.from_numpy(symmetric[rows, cols]).float().to(device)
        adj.append((edge_index, edge_weight))

    x_sp = [torch.from_numpy(it).reshape(it.shape[0], -1).float().to(device) for it in dataset.x]
    # img = [torch.from_numpy(it).float().unsqueeze(0).to(device) for it in dataset.mm_data]

    model = MultiViewModel(
        in_channels=[it.shape[1] for it in x_sp],
        gae_n_enc_1=gae_n_enc_1,
        gae_n_enc_3=gae_n_enc_3,
        gae_n_dec_1=gae_n_dec_1,
        gae_n_dec_3=gae_n_dec_3,
        n_samples=n_samples,
        patch_size=config[dataset_name]['patch_size'],
        img_channels=[it.shape[1] for it in x]).to(device)
    
    get_parameter_number(model)
    optimizer = Adam(model.parameters(), lr=lr)
    kmeans = PyTorchKMeans(n_clusters=n_clusters, init='k-means++', distributed=True, verbose=False)

    # 直接用原数据测试一下
    z = torch.cat(x_sp, dim=1)         # 不同模态cat起来的
    y_pred = kmeans.fit_predict(z).cpu().numpy()
    if config['remove_bkg']:
        y_pred = dataset.recover_background(y_pred)
    mts, ca, eta = evaluate(y_pred, dataset.association_mat, dataset.gt)
    print('[initial accuracy]')
    print_table([mts], ['ACC', 'Kappa', 'NMI', 'ARI', 'Purity'])
    print()

    t0 = time()
    if pretrain:
        print('start training.')
        best_acc = (-1.0, 0.0, 0.0, 0.0, 0.0)
        for epoch in range(1, EPOCH + 1):
            model.train()
            output = model(x, adj, x_sp, fuse_type)    # output: feat, feat_img, feat_sp, recon_img, recon_sp, latent_recon

            recon_loss = 0
            all_z = []
            for v in range(n_views):
                recon_loss += F.mse_loss(output[v][3], x[v])
                recon_loss += F.mse_loss(output[v][4], x_sp[v])
                recon_loss += F.mse_loss(output[v][0], output[v][-1])
                all_z.append(output[v][0])

            matrices = [
                get_sampleMatrix(z, n_clusters, max_iter, lambda_val, gamma, zs, ratio)
                for z in all_z
            ]
            contrast_loss = sampling_contrastive_loss(all_z, matrices, tau)

            align_loss = correlation_reduction_loss(all_z)

            loss = 0
            loss += alpha_recon * recon_loss + alpha_contrast * contrast_loss + alpha_align * align_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch}: loss = {round(loss.item(), 4)}, \
                  recon_loss = {round(recon_loss.item(), 4)}, \
                  contrast_loss = {round(contrast_loss.item(), 4)},\
                  align_loss = {round(align_loss.item(), 4)}')
            
            model.eval()
            with torch.no_grad():
                output = model(x, adj, x_sp, fuse_type)    # output: feat, recon, latent_recon, adj_hat
            z = torch.cat([output[v][0] for v in range(n_views)], dim=1) # 不同模态cat起来的
            y_pred = kmeans.fit_predict(z).cpu().numpy()
            if config['remove_bkg']:
                y_pred = dataset.recover_background(y_pred)
            matrics, ca, eta = evaluate(y_pred, dataset.association_mat, dataset.gt)
            if matrics[0] > best_acc[0]:
                best_acc = matrics
                weights = deepcopy(model.state_dict())
                indx, true, pred = eta
                visualization(indx, dataset.gt.shape, true, pred.astype(int), 
                              config['result_path'] + '/' + dataset_name + f'/{dataset_name}_{round(best_acc[0], 4)}.png')
                
            print_table([best_acc, matrics], ['ACC', 'Kappa', 'NMI', 'ARI', 'Purity'])
        torch.save(weights, save_path)
        print(f'best weight saved in {save_path}')
        print('finish.')

    
    return best_acc, time() - t0


def rand_seed(seed):
    if seed == -1:
        return
    # 设置随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    os.chdir(PROJECT_DIR)
    parser = ArgumentParser()
    parser.add_argument('--run-name', default='SEC-LSRM')
    args = parser.parse_args()

    metrics = ['ACC', 'Kappa', 'NMI', 'ARI', 'Purity']
    results = []
    dataset_name = config['dataset_name']
    print(f"{dataset_name} | {torch.cuda.get_device_name(config['device']) if config['device'].startswith('cuda') else config['device']} | {config['n_epoches']} epochs")
    Path('result').mkdir(exist_ok=True)
    for seed in config['seeds']:
        rand_seed(seed)
        run = dict(config)
        for key in ('save_path', 'result_path'):
            run[key] = str(Path(config[key]) / args.run_name / f'seed_{seed}')
        Path(run['save_path']).mkdir(parents=True, exist_ok=True)
        (Path(run['result_path']) / dataset_name).mkdir(parents=True, exist_ok=True)
        print(f'Seed {seed}')
        best, elapsed = train(run)
        results.append(best)
        print(f'Seed {seed} completed in {elapsed:.2f}s')
        print_table([best], metrics)

    print_table(
        [[str(seed), *row] for seed, row in zip(config['seeds'], results)]
        + [['Mean', *np.mean(results, axis=0)], ['Sample SD', *np.std(results, axis=0, ddof=1)]],
        ['Seed', *metrics],
    )
