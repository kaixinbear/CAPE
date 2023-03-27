# ------------------------------------------------------------------------
# Copyright (c) 2022 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------

import math
from einops import rearrange, repeat
import torch
from typing import List
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.registry import ATTENTION
import copy
from mmdet.models.utils.transformer import inverse_sigmoid
import numpy as np
import torch.utils.checkpoint as cp

class QcR_Modulation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale_emb = nn.Sequential(nn.Linear(9, dim), nn.LayerNorm(dim), nn.Sigmoid())
    
    def forward(self, x, R):
        bs, num_cam = R.shape[:2]
        R = R.flatten(2)
        scale_emb = self.scale_emb(R)
        x = x[:, None].repeat(1, num_cam, 1, 1)
        x = x * scale_emb[:, :, None]
        return x
    
class V_R_Modulation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale_emb = nn.Sequential(nn.Linear(9, dim), nn.LayerNorm(dim), nn.Sigmoid())
    
    def forward(self, feature, R):
        bs, num_cam = R.shape[:2]
        R = R.flatten(2)
        scale_emb = self.scale_emb(R)
        feature = feature * scale_emb[:, :, None]
        return feature

class Ego_emb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ego_emb = nn.Sequential(nn.Linear(9, dim), nn.LayerNorm(dim), nn.Sigmoid())
        
    def forward(self, img_metas, x):
        ego_matrix = self.get_curlidar2prevlidar(img_metas, x)[:3, :3][None, None] # 1, 1, 3, 3
        ego_emb = self.ego_emb(ego_matrix.flatten(2)) # 1, 1, d
        return ego_emb
    
    def get_curlidar2prevlidar(self, img_metas, x):
        '''
            get ego motion matrix in lidar axis.
            cur_lidar----->prev cam------>prev_lidar. 
            curlidar2prevcam @ prevcam2prevlidar =  curlidar2prevcam @ curcam2curlidar = curlidar2prevcam @ inverse(curlidar2curcam)
            
        '''
        curlidar2prevcam = x.new_tensor(img_metas[0]['extrinsics'][6].T) # (4, 4)
        curlidar2curcam = x.new_tensor(img_metas[0]['extrinsics'][0].T)  # (4, 4)
        prevcam2prevlidar = torch.inverse(curlidar2curcam)
        return (prevcam2prevlidar @ curlidar2prevcam)
    
class MLP_Fusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj_k_a = nn.Linear(dim, dim)
        self.proj_k_b = nn.Linear(dim, dim)
        self.proj_v_a = nn.Linear(dim, dim)
        self.proj_v_b = nn.Linear(dim, dim)
        self.fc = nn.Sequential(nn.Linear(dim * 2, dim), nn.LayerNorm(dim), nn.Sigmoid())
        self.ego_emb = Ego_emb(dim)
        
    def forward(self, a, b, img_metas):
        '''
            a: (b Q d)
            b: (b Q d)
        '''
        k_a = self.proj_k_a(a)
        k_b = self.proj_k_b(b)   
        ego_emb = self.ego_emb(img_metas, k_b)    
        ego_k_b = k_b * ego_emb
        w = self.fc(torch.cat([k_a, ego_k_b], -1))
        v_a = self.proj_v_a(a)
        v_b = self.proj_v_b(b)
        a = w * v_a 
        b = (1-w) * v_b
        return a, b
    
class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Linear(channels, channels)
        self.act1 = act_layer()
        self.conv_expand = nn.Linear(channels, channels)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm):
        super().__init__()

        self.heads = heads
        self.dim_head = dim_head

        self.proj_dict = nn.ModuleDict({
            'q_g':nn.Linear(dim, heads * dim_head, bias=qkv_bias),
            'k_g': nn.Linear(dim, heads * dim_head, bias=qkv_bias),
            'q_a': nn.Linear(dim, heads * dim_head, bias=qkv_bias),
            'k_a': nn.Linear(dim, heads * dim_head, bias=qkv_bias),
            'v':nn.Linear(dim, heads * dim_head, bias=qkv_bias)
        })
        
        self.proj = nn.Linear(heads * dim_head, dim)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)
        
        self.scale =  nn.ParameterDict({
            'g': nn.Parameter(torch.tensor([dim_head ** (-0.5) for _ in range(self.heads)])),
            'a': nn.Parameter(torch.tensor([dim_head ** (-0.5) for _ in range(self.heads)]))
        })

    def forward(self, k_g, q_g, k_a, q_a, v, mask):
        """
        k_g: (b n K d)
        q_g: (b n Q d)
        k_a: (b n K d)
        q_a: (b Q d)
        v:   (b n K d)
        mask: (b n K)
        """
        
        b, n, Q, d = q_g.shape
        
        skip = q_a
        # Project with multiple heads
        k_g = self.proj_dict['k_g'](k_g)                                # b n K (heads dim_head)
        q_g = self.proj_dict['q_g'](q_g)                                # b n Q (heads dim_head)
        k_a = self.proj_dict['k_a'](k_a)                                # b n K (heads dim_head)
        q_a = self.proj_dict['q_a'](q_a)                                # b Q (heads dim_head)
        v = self.proj_dict['v'](v)                                      # b n K (heads dim_head)
        q_a = q_a[:, None].expand(b, n, Q, d)
        
        # Group the head dim with batch dim
        k_g = rearrange(k_g, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        q_g = rearrange(q_g, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        k_a = rearrange(k_a, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        q_a = rearrange(q_a, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b n k (m d) -> (b m) (n k) d', m=self.heads, d=self.dim_head)
        
        # Dot product attention along cameras   
        dot_g =  torch.einsum('b n Q d, b n K d -> b n Q K', q_g, k_g)
        dot_a =  torch.einsum('b n Q d, b n K d -> b n Q K', q_a, k_a)
        dot_g = rearrange(dot_g, '(b m) ... -> b m ...', m=self.heads) 
        dot_a = rearrange(dot_a, '(b m) ... -> b m ...', m=self.heads)
        dot_a = dot_a * self.scale['a'].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        dot_g = dot_g * self.scale['g'].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        dot = dot_a + dot_g # b, m, n, Q, K
        dot.masked_fill_(mask[:, None, :, None, :], float('-inf')) # padding region
        dot = rearrange(dot, 'b m n Q K -> (b m) Q (n K)')
        att = dot.softmax(dim=-1)

        # Combine values (image level features).
        a = torch.einsum('b Q K, b K d -> b Q d', att, v)
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)

        # Combine multiple heads
        z = self.proj(a)

        # Optional skip connection
        if skip is not None:
            z = z + skip

        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)

        return z


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@TRANSFORMER.register_module()
class CAPETransformer(nn.Module):
    """
    Args:
        encoder (`mmcv.ConfigDict` | Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Defaults to None.
    """

    def __init__(self, 
        num_cameras: int,
        num_layers: int,
        feat_dim: int,
        feat_stride: int,
        image_height: int,
        image_width: int,
        bound: List[float],
        with_fpe: bool,
        depth_start: int,
        depth_num: int,
        att_layer=dict(type='CrossViewAttention',
            hidden_dim=512,
            num_queries=900,
            qkv_bias=True,
            heads=4,
            dim_head= 32,
            conditional=True),
        tf_layer=dict(
            groups=30,
            heads=8,
            hidden_dim=512),
        scalar = 10,
        noise_scale = 1.0,
        noise_trans = 0.0,
        num_classes=10,
        with_time=True
        ):
        
        super().__init__()
        self.num_cameras = num_cameras
        self.num_queries = att_layer['num_queries'] 
        self.hidden_dim = att_layer['hidden_dim']
        
        ys, xs = torch.meshgrid(
            torch.arange(feat_stride / 2, image_height, feat_stride),
            torch.arange(feat_stride / 2, image_width, feat_stride))
        image_plane = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1).flatten(0, 1) # hw * 3
        self.register_buffer('image_plane', image_plane, persistent=False)
        
        self.register_buffer('bound', torch.tensor(bound).view(2, 3), persistent=False)
        self.with_time = with_time
        self.reference_points = nn.Embedding(self.num_queries, 3)
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)

        if self.with_time:
            self.mf = nn.ModuleList([MLP_Fusion(tf_layer['hidden_dim']) for _ in range(num_layers)])

        self.cva_layers = nn.ModuleList([ATTENTION.build(att_layer) for _ in range(num_layers)])
        self.cva_layers[0].conditional = None
        
        self.content_prior = nn.Embedding(self.num_queries, self.hidden_dim)
        
        #TODO: use camera PE
        self.camera_embedding = nn.Embedding(num_cameras, self.hidden_dim)
        # self.obj_pe = nn.Embedding(self.num_queries, self.hidden_dim)
        
        self.bev_embed = MLP(self.hidden_dim * 3 // 2, self.hidden_dim, self.hidden_dim, 2)
        # self.img_embed = MLP(self.hidden_dim * 3 // 2, self.hidden_dim, self.hidden_dim, 2)
        
        self.feature_linear = nn.Linear(feat_dim, self.hidden_dim)

        self.with_fpe = with_fpe
        if self.with_fpe:
            self.fpe = SELayer(self.hidden_dim)
        
        self.depth_start = depth_start
        self.depth_num = depth_num
        # point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.cam_position_range = [bound[1], -bound[5], self.depth_start, bound[4], -bound[2], bound[3]]
        self.register_buffer('cam_bound', torch.tensor(self.cam_position_range).view(2, 3), persistent=False)
        
        self.position_dim = 3 * self.depth_num
        self.position_encoder = nn.Sequential(
            nn.Conv2d(self.position_dim, self.hidden_dim*4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim*4, self.hidden_dim, kernel_size=1, stride=1, padding=0),
        )
        
        self.query_embedding = nn.Sequential(
            nn.Linear(self.hidden_dim*3//2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.dn_query_embedding = nn.Sequential(
            nn.Linear(self.hidden_dim*3//2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.QcR = QcR_Modulation(self.hidden_dim)
        self.V_R = V_R_Modulation(self.hidden_dim)

        self.num_classes =num_classes
        self.scalar = scalar
        self.bbox_noise_scale = noise_scale
        self.bbox_noise_trans = noise_trans
        self.split = 0.75
        self.pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

    def prepare_for_dn(self, batch_size, reference_points, img_metas):
        if self.training:
            targets = [torch.cat((img_meta['gt_bboxes_3d']._data.gravity_center, img_meta['gt_bboxes_3d']._data.tensor[:, 3:]),dim=1) for img_meta in img_metas ]
            labels = [img_meta['gt_labels_3d']._data for img_meta in img_metas ]
            known = [(torch.ones_like(t)).cuda() for t in labels]
            know_idx = known
            unmask_bbox = unmask_label = torch.cat(known)
            known_num = [t.size(0) for t in targets]
            labels = torch.cat([t for t in labels])
            boxes = torch.cat([t for t in targets])
            batch_idx = torch.cat([torch.full((t.size(0), ), i) for i, t in enumerate(targets)])

            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)
            # add noise
            groups = 600 // max(known_num)
            known_indice = known_indice.repeat(groups, 1).view(-1)
            known_labels = labels.repeat(groups, 1).view(-1).long().to(reference_points.device)
            known_bid = batch_idx.repeat(groups, 1).view(-1)
            known_bboxs = boxes.repeat(groups, 1).to(reference_points.device)
            known_bbox_center = known_bboxs[:, :3].clone()
            known_bbox_scale = known_bboxs[:, 3:6].clone()

            if self.bbox_noise_scale > 0:
                diff = known_bbox_scale / 2 + self.bbox_noise_trans
                rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0
                known_bbox_center += torch.mul(rand_prob,
                                            diff) * self.bbox_noise_scale
                known_bbox_center[..., 0:1] = (known_bbox_center[..., 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
                known_bbox_center[..., 1:2] = (known_bbox_center[..., 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
                known_bbox_center[..., 2:3] = (known_bbox_center[..., 2:3] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
                known_bbox_center = known_bbox_center.clamp(min=0.0, max=1.0)
                mask = torch.norm(rand_prob, 2, 1) > self.split
                known_labels[mask] = self.num_classes
            
            single_pad = int(max(known_num))
            pad_size = int(single_pad * groups)
            padding_bbox = torch.zeros(pad_size, 3).to(reference_points.device)
            padded_reference_points = torch.cat([padding_bbox, reference_points], dim=0).unsqueeze(0).repeat(batch_size, 1, 1)

            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(groups)]).long()
            if len(known_bid):
                padded_reference_points[(known_bid.long(), map_known_indice)] = known_bbox_center.to(reference_points.device)

            tgt_size = pad_size + self.num_queries
            attn_mask = torch.ones(tgt_size, tgt_size).to(reference_points.device) < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(groups):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == groups - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True

            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'know_idx': know_idx,
                'pad_size': pad_size
            }
            
        else:
            padded_reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
            attn_mask = None
            mask_dict = None

        return padded_reference_points, attn_mask, mask_dict

    def prepare_emb(self, feature, mask, I_inv, R_inv, t, ref_points, img_metas, seq_length):
        img_embed, _ = self.position_embeding(feature, I_inv, img_metas, mask)
        img_embed = rearrange(img_embed, 'b n d h w -> b n (h w) d')
        
        bs, nc = feature.shape[:2]
        ref_points_unormalized = ref_points * (self.bound[1] - self.bound[0]) + self.bound[0]
        
        feature = rearrange(feature, 'b n d h w -> (b n) (h w) d')
        feature = self.feature_linear(feature)
        feature = rearrange(feature, '(b n) k d -> b n k d', b=bs, n=nc)
        
        if self.with_fpe:
            img_embed = self.fpe(img_embed, feature)
        
        mask = rearrange(mask, 'b n h w -> b n (h w)')
        # ref_points_unormalized.shape: bs N 3 
        # print("111 ref_points_unormalized", ref_points_unormalized.shape)
        ref_points_unormalized = ref_points_unormalized[:, None].repeat(seq_length, 1, 1, 1) # 2 * bs 1 Q 3
        # print(ref_points_unormalized.shape)
        R = torch.inverse(R_inv)
        world = ref_points_unormalized + t[:, :, None] # b n Q 3
        world = (R[:, :, None] @ world[..., None]).squeeze(-1)
        bev_embed = self.bev_embed(pos2posemb3d(world, self.hidden_dim // 2))                    # b n Q d
        return feature, mask, img_embed, bev_embed, R
        
    def forward(self, feature, mask, I_inv, R_inv, t, img_metas, return_prev_query=False):
        return_list = []
        return_prev_list = []
        bs, nc = feature.shape[:2]
        f=2 if self.with_time else 1
        feature = rearrange(feature, 'b (f n) d h w -> (b f) n d h w', f=f, n=self.num_cameras)
        mask = rearrange(mask, 'b (f n) h w -> (b f) n h w', f=f)
        I_inv = rearrange(I_inv, 'b (f n) h w -> (b f) n h w', f=f)
        R_inv = rearrange(R_inv, 'b (f n) h w -> (b f) n h w', f=f)
        t = rearrange(t, 'b (f n) h w -> (b f) n (h w)', f=f)
        x = self.content_prior.weight.unsqueeze(0).repeat(bs, 1, 1)
        
        ref_points = self.reference_points.weight   # N, 3
        ref_points, attn_mask, mask_dict = self.prepare_for_dn(bs, ref_points, img_metas)   # bs N 3
        if self.training:
            pad_size = mask_dict['pad_size']
            dn_query = self.dn_query_embedding(pos2posemb3d(ref_points[:, :pad_size, :], self.hidden_dim // 2))#.repeat(bs, 1, 1)
            x = torch.cat([dn_query, x], 1) # bs N 256
            
        lidar_obj_pe = self.query_embedding(pos2posemb3d(ref_points, self.hidden_dim // 2))#.repeat(bs, 1, 1) # bs, N, 256
        cam_pe = self.camera_embedding.weight.repeat(bs, 1, 1)

        if self.with_time:
            cur_x = x
            prev_x = x
            lidar_obj_pe = torch.cat([lidar_obj_pe, lidar_obj_pe], 0)   # 2 * bs, N, 256
            cam_pe = torch.cat([cam_pe, cam_pe], 0)
            feature, mask, img_embed, bev_embed, R = self.prepare_emb(feature, mask, I_inv, R_inv, t, ref_points, img_metas, f)
            for mf, cva in zip(self.mf, self.cva_layers):
                x = torch.cat([cur_x, prev_x], 0)   # 2 * bs N 256
                modulated_x = self.QcR(x, R) # b num_cam Q d
                modulated_v = self.V_R(feature, R)
                x = cva(x, modulated_x, lidar_obj_pe, modulated_v, cam_pe, mask, img_embed, bev_embed, attn_mask)
                cur_x, prev_x = torch.split(x, [bs, bs])    # bs N 256
                # cur_x, prev_x = x[0:1, ...], x[1:2, ...]
                cur_x, prev_x = mf(cur_x, prev_x, img_metas)
                return_list.append(cur_x)  
                return_prev_list.append(prev_x)
            if not return_prev_query:
                return torch.stack(return_list), ref_points
            else:
                return torch.stack(return_list), torch.stack(return_prev_list), ref_points, mask_dict
        else:
            feature, mask, img_embed, bev_embed, R = self.prepare_emb(feature, mask, I_inv, R_inv, t, ref_points, img_metas, f)
            for cva in self.cva_layers:
                modulated_x = self.QcR(x, R) # b num_cam Q d
                modulated_v = self.V_R(feature, R)
                x = cva(x, modulated_x, lidar_obj_pe, modulated_v, cam_pe, mask, img_embed, bev_embed, attn_mask)
                return_list.append(x) 
            return torch.stack(return_list), ref_points, mask_dict
 
    def position_embeding(self, img_feats, I_inv, img_metas, masks=None):
        eps = 1e-5
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        B, N, C, H, W = img_feats.shape
        coords_h = torch.arange(H, device=img_feats.device).float() * pad_h / H
        coords_w = torch.arange(W, device=img_feats.device).float() * pad_w / W

        index  = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats.device).float()
        index_1 = index + 1
        bin_size = (self.cam_position_range[5] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
        coords_d = self.depth_start + bin_size * index * index_1

        D = coords_d.shape[0]
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0) # W, H, D, 3
        # coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)

        coords = coords.view(1, 1, W, H, D, 3, 1).repeat(B, N, 1, 1, 1, 1, 1)
        I_inv = I_inv.view(B, N, 1, 1, 1, 3, 3).repeat(1, 1, W, H, D, 1, 1)
        # coords3d is on cam coords, pay attention the axis direction is different from lidar coord.
        coords3d = torch.matmul(I_inv, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.cam_position_range[0]) / (self.cam_position_range[3] - self.cam_position_range[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.cam_position_range[1]) / (self.cam_position_range[4] - self.cam_position_range[1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.cam_position_range[2]) / (self.cam_position_range[5] - self.cam_position_range[2])

        coords_mask = (coords3d > 1.0) | (coords3d < 0.0) 
        coords_mask = coords_mask.flatten(-2).sum(-1) > (D * 0.5)
        coords_mask = masks | coords_mask.permute(0, 1, 3, 2)
        coords3d = coords3d.permute(0, 1, 4, 5, 3, 2).contiguous().view(B*N, -1, H, W)
        coords3d = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(coords3d)
        
        return coords_position_embeding.view(B, N, self.hidden_dim, H, W), coords_mask

@ATTENTION.register_module()
class CrossViewAttention(BaseModule):
    """A wrapper for ``torch.nn.MultiheadAttention``.
    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.
    Args:
     
    """

    def __init__(self,
                num_queries: int,
                hidden_dim: int,
                qkv_bias: bool,
                heads: int = 4,
                dim_head: int = 32,
                conditional: bool = True):
        super(CrossViewAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.cross_attend = CrossAttention(hidden_dim, heads, dim_head, qkv_bias)
        self.conditional = MLP(hidden_dim, hidden_dim, hidden_dim, 2) if conditional else None
        self.sl_layer = nn.MultiheadAttention(hidden_dim, heads, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.1)

    def forward(self,
                x: torch.FloatTensor,
                modulated_x: torch.FloatTensor,
                lidar_obj_pe: torch.FloatTensor,
                feature: torch.FloatTensor,
                camera_pe: torch.FloatTensor,
                mask: torch.BoolTensor, 
                img_embed: torch.FloatTensor, 
                bev_embed: torch.FloatTensor,
                attn_mask: torch.FloatTensor):
        """
        x: (b, Q, d)
        obj_pe: (b, Q, d)
        feature: (b, n, K, d)
        camera_pe: (b, n, d)
        mask: (b, n, K)
        img_embed: (b, n, K, d)
        bev_embed: (b, n, Q, d)

        Returns: (b, d, H, W)
        """
        b, n, _, _ = feature.shape

        if self.conditional is not None:
            bev_embed = self.conditional(modulated_x) * bev_embed

        val = feature
        k_a, q_a = feature + camera_pe[:, :, None], x + lidar_obj_pe
        if self.training:
            updated_x = cp.checkpoint(self.cross_attend, img_embed, bev_embed, k_a, q_a, val, mask)
        else:
            updated_x = self.cross_attend(img_embed, bev_embed, k_a, q_a, val, mask)
        
        q = k = updated_x + lidar_obj_pe
        tgt = self.sl_layer(q, k, value=updated_x, attn_mask=attn_mask)[0]
        
        return self.norm1(updated_x + self.dropout1(tgt))
        
        
if __name__ == '__main__':
    num_cameras = 6
    num_layers = 2
    feat_dim=256
    feat_stride = 32
    image_height = 768
    image_width = 768
    bound = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    att_layer=dict(type='CrossViewAttention',
            hidden_dim=512,
            num_queries=900,
            qkv_bias=True,
            heads=8,
            dim_head= 64,
            conditional=True)

    model = CAPETransformer(num_cameras, num_layers, feat_dim, feat_stride, image_height, image_width, bound, att_layer).cuda()
    feature = torch.rand(2, 6, 256, 24, 24).cuda()
    mask = torch.rand(2, 6, 24, 24).cuda()
    I_inv = torch.rand(2, 6, 3, 3).cuda()
    R_inv = torch.rand(2, 6, 3, 3).cuda()
    t = torch.rand(2, 6, 3).cuda()
    y = model(feature, mask, I_inv, R_inv, t)
    print(y.shape)