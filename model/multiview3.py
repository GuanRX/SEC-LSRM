import torch
import torch.nn as nn
import torch.nn.functional as F

from .gcn import GraphAutoencoderBranch
from .attention import Attention


class SelfExpressiveLayer(nn.Module):
    def __init__(self, num_samples, init_affinity=None):
        super(SelfExpressiveLayer, self).__init__()
        self.init_affinity = init_affinity
        self.affinity_mat = nn.Parameter(torch.randn(num_samples, num_samples), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_affinity is not None:
            if not isinstance(self.init_affinity, torch.Tensor):
                self.init_affinity = torch.from_numpy(self.init_affinity).float()
            self.affinity_mat.data = self.init_affinity
        # else:
        #     nn.init.constant_(self.affinity_mat, 1.)

    def forward(self, x):
        latent_recon = torch.matmul(self.affinity_mat, x)
        return latent_recon


class SingleViewModel(nn.Module):
    def __init__(self, in_channel, gae_n_enc_1, gae_n_enc_3, gae_n_dec_1, gae_n_dec_3, n_samples, patch_size, img_channel):
        super(SingleViewModel, self).__init__()
        self.encoder = GraphAutoencoderBranch(in_channel, gae_n_enc_1, gae_n_enc_3)
        self.decoder = GraphAutoencoderBranch(gae_n_dec_1, gae_n_dec_3, in_channel)

        self.self_expr_layer = SelfExpressiveLayer(n_samples, init_affinity=None)
        self.patch_size = patch_size

        # self.img_encoder = Enc_VAE(img_channel, 512, None)
        self.img_encoder = nn.Sequential(
            nn.Conv2d(img_channel, 64, (1, 1), stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3, 3), stride=1, padding=0),  # # N*(W-2)*(W-2)*64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        self.img_decoder = nn.Sequential(
            nn.Conv2d(512, 64, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, img_channel, (1, 1), stride=1, padding=0),
        )

        self.mlp = nn.Sequential(
            nn.Linear(gae_n_enc_3, gae_n_enc_3),
            nn.BatchNorm1d(gae_n_enc_3),
            nn.Sigmoid(),
            nn.Dropout(0.3),
        )

        self.sp_lin1 = nn.Sequential(nn.Linear(256, 512),
                                     nn.BatchNorm1d(512),
                                     nn.ReLU(inplace=True),
                                     # nn.Dropout(0.5)
                                     )
        
        self.sp_lin2 = nn.Sequential(nn.Linear(1024, 512),
                                     nn.BatchNorm1d(512),
                                     nn.ReLU(inplace=True),
                                     # nn.Dropout(0.5)
                                     )
        self.attention = Attention(512)
        self.alpha = nn.Parameter(torch.Tensor(1,))
        self.alpha.data = torch.tensor(0.1)
        # self.alpha.data=torch.tensor(0.99999).to(args.device)


    def forward(self, x, adj, x_sp, fuse_type):
        feat_img = self.img_encoder(x)                                    # 整张图像进行卷积
        feat_img = self.sp_lin1(feat_img)

        edge_index, edge_weight = adj
        feat_sp = self.encoder(x_sp, edge_index, edge_weight)

        if type(fuse_type) == str:
            if fuse_type == 'attention':
                feat, _ = self.attention(torch.stack([feat_img, feat_sp], dim=1))
            elif fuse_type == 'concat':
                feat = torch.cat([feat_sp, feat_img], dim=1)
                feat = self.sp_lin2(feat)
        else:
            # feat = (1 - fuse_type) * feat_sp + fuse_type * feat_img
            feat = (1 - self.alpha) * feat_sp + self.alpha * feat_img

        # latent_recon = self.self_expr_layer(feat)
        # latent_recon = self.mlp(latent_recon)
        feat_img_to_patch = F.interpolate(feat_img.unsqueeze(-1).unsqueeze(-1), size=(self.patch_size, self.patch_size),
                                        mode='bilinear',
                                        align_corners=True)
        recon_img = self.img_decoder(feat_img_to_patch)
        recon_sp = self.decoder(feat_sp, edge_index, edge_weight)
        latent_recon = self.self_expr_layer(feat)
        latent_recon = self.mlp(latent_recon)
        # recon, z_hat_adj  = self.decoder(latent_recon, adj)

        return feat, feat_img, feat_sp, recon_img, recon_sp, latent_recon
    

class MultiViewModel(nn.Module):    

    def __init__(self, in_channels, gae_n_enc_1, gae_n_enc_3, gae_n_dec_1, gae_n_dec_3, 
                 n_samples, patch_size, img_channels):
        super(MultiViewModel, self).__init__()
        self.aes = []
        for i in range(len(in_channels)):
            self.aes.append(SingleViewModel(in_channels[i], 
                                            gae_n_enc_1, gae_n_enc_3, 
                                            gae_n_dec_1, gae_n_dec_3, 
                                            n_samples, patch_size, img_channels[i]))
        self.aes = nn.ModuleList(self.aes)


    def forward(self, x, adj, x_sp, fuse_type):
        # x: patch格式的超像素特征
        # x_sp: n * d 的超像素特征
        output = []
        for i in range(len(x)):
            out = self.aes[i](x[i], adj[i], x_sp[i], fuse_type)
            output.append(out)
        return output
    

