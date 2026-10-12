# """
# D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
# Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
# ---------------------------------------------------------------------------------
# Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
# Copyright (c) 2023 lyuwenyu. All Rights Reserved.
# """
#
# import math
# import copy
# import functools
# from collections import OrderedDict
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.nn.init as init
# from typing import List
#
# from .dfine_utils import weighting_function, distance2bbox
# from .denoising import get_contrastive_denoising_training_group
# from .utils import deformable_attention_core_func_v2, get_activation, inverse_sigmoid
# from .utils import bias_init_with_prob
# from ...core import register
# from scipy.optimize import linear_sum_assignment
# from torch.nn.utils.rnn import pad_sequence
# __all__ = ['DFINETransformer']
#
#
# class MLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
#         super().__init__()
#         self.num_layers = num_layers
#         h = [hidden_dim] * (num_layers - 1)
#         self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
#         self.act = get_activation(act)
#
#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
#         return x
#
#
# class MSDeformableAttention(nn.Module):
#     def __init__(
#         self,
#         embed_dim=256,
#         num_heads=8,
#         num_levels=4,
#         num_points=4,
#         method='default',
#         offset_scale=0.5,
#     ):
#         """Multi-Scale Deformable Attention
#         """
#         super(MSDeformableAttention, self).__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.num_levels = num_levels
#         self.offset_scale = offset_scale
#
#         if isinstance(num_points, list):
#             assert len(num_points) == num_levels, ''
#             num_points_list = num_points
#         else:
#             num_points_list = [num_points for _ in range(num_levels)]
#
#         self.num_points_list = num_points_list
#
#         num_points_scale = [1/n for n in num_points_list for _ in range(n)]
#         self.register_buffer('num_points_scale', torch.tensor(num_points_scale, dtype=torch.float32))
#
#         self.total_points = num_heads * sum(num_points_list)
#         self.method = method
#
#         self.head_dim = embed_dim // num_heads
#         assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
#
#         self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
#         self.attention_weights = nn.Linear(embed_dim, self.total_points)
#
#         self.ms_deformable_attn_core = functools.partial(deformable_attention_core_func_v2, method=self.method)
#
#         self._reset_parameters()
#
#         if method == 'discrete':
#             for p in self.sampling_offsets.parameters():
#                 p.requires_grad = False
#
#     def _reset_parameters(self):
#         # sampling_offsets
#         init.constant_(self.sampling_offsets.weight, 0)
#         thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
#         grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
#         grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
#         grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
#         scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
#         grid_init *= scaling
#         self.sampling_offsets.bias.data[...] = grid_init.flatten()
#
#         # attention_weights
#         init.constant_(self.attention_weights.weight, 0)
#         init.constant_(self.attention_weights.bias, 0)
#
#
#     def forward(self,
#                 query: torch.Tensor,
#                 reference_points: torch.Tensor,
#                 value: torch.Tensor,
#                 value_spatial_shapes: List[int]):
#         """
#         Args:
#             query (Tensor): [bs, query_length, C]
#             reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
#                 bottom-right (1, 1), including padding area
#             value (Tensor): [bs, value_length, C]
#             value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
#
#         Returns:
#             output (Tensor): [bs, Length_{query}, C]
#         """
#         bs, Len_q = query.shape[:2]
#
#         sampling_offsets: torch.Tensor = self.sampling_offsets(query)
#         sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.num_heads, sum(self.num_points_list), 2)
#
#         attention_weights = self.attention_weights(query).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))
#         attention_weights = F.softmax(attention_weights, dim=-1)
#
#         if reference_points.shape[-1] == 2:
#             offset_normalizer = torch.tensor(value_spatial_shapes)
#             offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
#             sampling_locations = reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2) + sampling_offsets / offset_normalizer
#         elif reference_points.shape[-1] == 4:
#             # reference_points [8, 480, None, 1,  4]
#             # sampling_offsets [8, 480, 8,    12, 2]
#             num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
#             offset = sampling_offsets * num_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
#             sampling_locations = reference_points[:, :, None, :, :2] + offset
#         else:
#             raise ValueError(
#                 "Last dim of reference_points must be 2 or 4, but get {} instead.".
#                 format(reference_points.shape[-1]))
#
#         output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights, self.num_points_list)
#
#         return output
#
#
# class TransformerDecoderLayer(nn.Module):
#     def __init__(self,
#                  d_model=256,
#                  n_head=8,
#                  dim_feedforward=1024,
#                  dropout=0.,
#                  activation='relu',
#                  n_levels=4,
#                  n_points=4,
#                  cross_attn_method='default',
#                  layer_scale=None):
#         super(TransformerDecoderLayer, self).__init__()
#         if layer_scale is not None:
#             dim_feedforward = round(layer_scale * dim_feedforward)
#             d_model = round(layer_scale * d_model)
#
#         # self attention
#         self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
#         self.dropout1 = nn.Dropout(dropout)
#         self.norm1 = nn.LayerNorm(d_model)
#
#         # cross attention
#         self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points, \
#                                                 method=cross_attn_method)
#         self.dropout2 = nn.Dropout(dropout)
#
#         # gate
#         self.gateway = Gate(d_model)
#
#         # ffn
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.activation = get_activation(activation)
#         self.dropout3 = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)
#         self.dropout4 = nn.Dropout(dropout)
#         self.norm3 = nn.LayerNorm(d_model)
#
#         self._reset_parameters()
#
#     def _reset_parameters(self):
#         init.xavier_uniform_(self.linear1.weight)
#         init.xavier_uniform_(self.linear2.weight)
#
#     def with_pos_embed(self, tensor, pos):
#         return tensor if pos is None else tensor + pos
#
#     def forward_ffn(self, tgt):
#         return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
#
#     def forward(self,
#                 target,
#                 reference_points,
#                 value,
#                 spatial_shapes,
#                 attn_mask=None,
#                 query_pos_embed=None):
#
#         # self attention
#         q = k = self.with_pos_embed(target, query_pos_embed)
#
#         target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
#         target = target + self.dropout1(target2)
#         target = self.norm1(target)
#
#         # cross attention
#         target2 = self.cross_attn(\
#             self.with_pos_embed(target, query_pos_embed),
#             reference_points,
#             value,
#             spatial_shapes)
#
#         target = self.gateway(target, self.dropout2(target2))
#
#         # ffn
#         target2 = self.forward_ffn(target)
#         target = target + self.dropout4(target2)
#         target = self.norm3(target.clamp(min=-65504, max=65504))
#
#         return target
#
#
# class Gate(nn.Module):
#     def __init__(self, d_model):
#         super(Gate, self).__init__()
#         self.gate = nn.Linear(2 * d_model, 2 * d_model)
#         bias = bias_init_with_prob(0.5)
#         init.constant_(self.gate.bias, bias)
#         init.constant_(self.gate.weight, 0)
#         self.norm = nn.LayerNorm(d_model)
#
#     def forward(self, x1, x2):
#         gate_input = torch.cat([x1, x2], dim=-1)
#         gates = torch.sigmoid(self.gate(gate_input))
#         gate1, gate2 = gates.chunk(2, dim=-1)
#         return self.norm(gate1 * x1 + gate2 * x2)
#
#
# class Integral(nn.Module):
#     """
#     A static layer that calculates integral results from a distribution.
#
#     This layer computes the target location using the formula: `sum{Pr(n) * W(n)}`,
#     where Pr(n) is the softmax probability vector representing the discrete
#     distribution, and W(n) is the non-uniform Weighting Function.
#
#     Args:
#         reg_max (int): Max number of the discrete bins. Default is 32.
#                        It can be adjusted based on the dataset or task requirements.
#     """
#
#     def __init__(self, reg_max=32):
#         super(Integral, self).__init__()
#         self.reg_max = reg_max
#
#     def forward(self, x, project):
#         shape = x.shape
#         x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
#         x = F.linear(x, project.to(x.device)).reshape(-1, 4)
#         return x.reshape(list(shape[:-1]) + [-1])
#
#
# class LQE(nn.Module):
#     def __init__(self, k, hidden_dim, num_layers, reg_max):
#         super(LQE, self).__init__()
#         self.k = k
#         self.reg_max = reg_max
#         self.reg_conf = MLP(4 * (k + 1), hidden_dim, 1, num_layers)
#         init.constant_(self.reg_conf.layers[-1].bias, 0)
#         init.constant_(self.reg_conf.layers[-1].weight, 0)
#
#     def forward(self, scores, pred_corners):
#         B, L, _ = pred_corners.size()
#         prob = F.softmax(pred_corners.reshape(B, L, 4, self.reg_max+1), dim=-1)
#         prob_topk, _ = prob.topk(self.k, dim=-1)
#         stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
#         quality_score = self.reg_conf(stat.reshape(B, L, -1))
#         return scores + quality_score
#
#
# class TransformerDecoder(nn.Module):
#     """
#     Transformer Decoder implementing Fine-grained Distribution Refinement (FDR).
#
#     This decoder refines object detection predictions through iterative updates across multiple layers,
#     utilizing attention mechanisms, location quality estimators, and distribution refinement techniques
#     to improve bounding box accuracy and robustness.
#     """
#
#     def __init__(self, hidden_dim, decoder_layer, decoder_layer_wide, num_layers, num_head, reg_max, reg_scale, up,
#                  eval_idx=-1, layer_scale=2):
#         super(TransformerDecoder, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.layer_scale = layer_scale
#         self.num_head = num_head
#         self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
#         self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max
#         self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)] \
#                     + [copy.deepcopy(decoder_layer_wide) for _ in range(num_layers - self.eval_idx - 1)])
#         self.lqe_layers = nn.ModuleList([copy.deepcopy(LQE(4, 64, 2, reg_max)) for _ in range(num_layers)])
#
#     def value_op(self, memory, value_proj, value_scale, memory_mask, memory_spatial_shapes):
#         """
#         Preprocess values for MSDeformableAttention.
#         """
#         value = value_proj(memory) if value_proj is not None else memory
#         value = F.interpolate(memory, size=value_scale) if value_scale is not None else value
#         if memory_mask is not None:
#             value = value * memory_mask.to(value.dtype).unsqueeze(-1)
#         value = value.reshape(value.shape[0], value.shape[1], self.num_head, -1)
#         split_shape = [h * w for h, w in memory_spatial_shapes]
#         return value.permute(0, 2, 3, 1).split(split_shape, dim=-1)
#
#     def convert_to_deploy(self):
#         self.project = weighting_function(self.reg_max, self.up, self.reg_scale, deploy=True)
#         self.layers = self.layers[:self.eval_idx + 1]
#         self.lqe_layers = nn.ModuleList([nn.Identity()] * (self.eval_idx) + [self.lqe_layers[self.eval_idx]])
#
#     def forward(self,
#                 target,
#                 ref_points_unact,
#                 memory,
#                 spatial_shapes,
#                 bbox_head,
#                 score_head,
#                 query_pos_head,
#                 pre_bbox_head,
#                 integral,
#                 up,
#                 reg_scale,
#                 attn_mask=None,
#                 memory_mask=None,
#                 dn_meta=None):
#         output = target
#         output_detach = pred_corners_undetach = 0
#         value = self.value_op(memory, None, None, memory_mask, spatial_shapes)
#
#         dec_out_bboxes = []
#         dec_out_logits = []
#         dec_out_pred_corners = []
#         dec_out_refs = []
#         if not hasattr(self, 'project'):
#             project = weighting_function(self.reg_max, up, reg_scale)
#         else:
#             project = self.project
#
#         ref_points_detach = F.sigmoid(ref_points_unact)
#
#         for i, layer in enumerate(self.layers):
#             ref_points_input = ref_points_detach.unsqueeze(2)
#             query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)
#
#             # TODO Adjust scale if needed for detachable wider layers
#             if i >= self.eval_idx + 1 and self.layer_scale > 1:
#                 query_pos_embed = F.interpolate(query_pos_embed, scale_factor=self.layer_scale)
#                 value = self.value_op(memory, None, query_pos_embed.shape[-1], memory_mask, spatial_shapes)
#                 output = F.interpolate(output, size=query_pos_embed.shape[-1])
#                 output_detach = output.detach()
#
#             output = layer(output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed)
#
#             if i == 0 :
#                 # Initial bounding box predictions with inverse sigmoid refinement
#                 pre_bboxes = F.sigmoid(pre_bbox_head(output) + inverse_sigmoid(ref_points_detach))
#                 pre_scores = score_head[0](output)
#                 ref_points_initial = pre_bboxes.detach()
#
#             # Refine bounding box corners using FDR, integrating previous layer's corrections
#             pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
#             inter_ref_bbox = distance2bbox(ref_points_initial, integral(pred_corners, project), reg_scale)
#
#             if self.training or i == self.eval_idx:
#                 scores = score_head[i](output)
#                 # Lqe does not affect the performance here.
#                 scores = self.lqe_layers[i](scores, pred_corners)
#                 dec_out_logits.append(scores)
#                 dec_out_bboxes.append(inter_ref_bbox)
#                 dec_out_pred_corners.append(pred_corners)
#                 dec_out_refs.append(ref_points_initial)
#
#                 if not self.training:
#                     break
#
#             pred_corners_undetach = pred_corners
#             ref_points_detach = inter_ref_bbox.detach()
#             output_detach = output.detach()
#
#         return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits), \
#                torch.stack(dec_out_pred_corners), torch.stack(dec_out_refs), pre_bboxes, pre_scores
#
# from typing import List, Dict
# @register()
# class DFINETransformer(nn.Module):
#     __share__ = ['num_classes', 'eval_spatial_size']
#
#     def __init__(self,
#                  num_classes=80,
#                  hidden_dim=256,
#                  num_queries=300,
#                  feat_channels=[512, 1024, 2048],
#                  feat_strides=[8, 16, 32],
#                  num_levels=3,
#                  num_points=4,
#                  nhead=8,
#                  num_layers=6,
#                  dim_feedforward=1024,
#                  dropout=0.,
#                  activation="relu",
#                  num_denoising=100,
#                  label_noise_ratio=0.5,
#                  box_noise_scale=1.0,
#                  learn_query_content=False,
#                  eval_spatial_size=None,
#                  eval_idx=-1,
#                  eps=1e-2,
#                  aux_loss=True,
#                  cross_attn_method='default',
#                  query_select_method='default',
#                  reg_max=32,
#                  reg_scale=4.,
#                  layer_scale=1,
#                  **kwargs):
#         super().__init__()
#         assert len(feat_channels) <= num_levels
#         assert len(feat_strides) == len(feat_channels)
#
#         for _ in range(num_levels - len(feat_strides)):
#             feat_strides.append(feat_strides[-1] * 2)
#
#         self.hidden_dim = hidden_dim
#         scaled_dim = round(layer_scale*hidden_dim)
#         self.nhead = nhead
#         self.feat_strides = feat_strides
#         self.num_levels = num_levels
#         self.num_classes = num_classes
#         self.num_queries = num_queries
#         self.eps = eps
#         self.num_layers = num_layers
#         self.eval_spatial_size = eval_spatial_size
#         self.aux_loss = aux_loss
#         self.reg_max = reg_max
#
#         assert query_select_method in ('default', 'one2many', 'agnostic'), ''
#         assert cross_attn_method in ('default', 'discrete'), ''
#         self.cross_attn_method = cross_attn_method
#         self.query_select_method = query_select_method
#
#         # backbone feature projection
#         self._build_input_proj_layer(feat_channels)
#
#         # Transformer module
#         self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
#         self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)
#         decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, \
#             activation, num_levels, num_points, cross_attn_method=cross_attn_method)
#         decoder_layer_wide = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, \
#             activation, num_levels, num_points, cross_attn_method=cross_attn_method, layer_scale=layer_scale)
#         self.decoder = TransformerDecoder(hidden_dim, decoder_layer, decoder_layer_wide, num_layers, nhead,
#                                           reg_max, self.reg_scale, self.up, eval_idx, layer_scale)
#       # denoising
#         self.num_denoising = num_denoising
#         self.label_noise_ratio = label_noise_ratio
#         self.box_noise_scale = box_noise_scale
#         if num_denoising > 0:
#             self.denoising_class_embed = nn.Embedding(num_classes+1, hidden_dim, padding_idx=num_classes)
#             init.normal_(self.denoising_class_embed.weight[:-1])
#
#         # decoder embedding
#         self.learn_query_content = learn_query_content
#         if learn_query_content:
#             self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
#         self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2)
#
#         # if num_select_queries != self.num_queries:
#         #     layer = TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward, activation='gelu')
#         #     self.encoder = TransformerEncoder(layer, 1)
#
#         self.enc_output = nn.Sequential(OrderedDict([
#             ('proj', nn.Linear(hidden_dim, hidden_dim)),
#             ('norm', nn.LayerNorm(hidden_dim,)),
#         ]))
#
#         if query_select_method == 'agnostic':
#             self.enc_score_head = nn.Linear(hidden_dim, 1)
#         else:
#             self.enc_score_head = nn.Linear(hidden_dim, num_classes)
#
#         self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)
#
#         # decoder head
#         self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
#         self.dec_score_head = nn.ModuleList(
#             [nn.Linear(hidden_dim, num_classes) for _ in range(self.eval_idx + 1)]
#           + [nn.Linear(scaled_dim, num_classes) for _ in range(num_layers - self.eval_idx - 1)])
#         self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)
#         self.dec_bbox_head = nn.ModuleList(
#             [MLP(hidden_dim, hidden_dim, 4 * (self.reg_max+1), 3) for _ in range(self.eval_idx + 1)]
#           + [MLP(scaled_dim, scaled_dim, 4 * (self.reg_max+1), 3) for _ in range(num_layers - self.eval_idx - 1)])
#         self.integral = Integral(self.reg_max)
#
#         # init encoder output anchors and valid_mask
#         if self.eval_spatial_size:
#             anchors, valid_mask = self._generate_anchors()
#             self.register_buffer('anchors', anchors)
#             self.register_buffer('valid_mask', valid_mask)
#         # init encoder output anchors and valid_mask
#         if self.eval_spatial_size:
#             self.anchors, self.valid_mask = self._generate_anchors()
#
#         # self._build_heads()
#
#         self._reset_parameters(feat_channels)
#
#     # def _build_heads(self):
#     #     """初始化分类头和回归头"""
#     #     self.class_embed = nn.Linear(self.hidden_dim, self.num_classes)
#     #     self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4 * (self.reg_max + 1), 3)
#     #     self.integral = Integral(self.reg_max)
#
#
#
#
#     def convert_to_deploy(self):
#         self.dec_score_head = nn.ModuleList([nn.Identity()] * (self.eval_idx) + [self.dec_score_head[self.eval_idx]])
#         self.dec_bbox_head = nn.ModuleList(
#             [self.dec_bbox_head[i] if i <= self.eval_idx else nn.Identity() for i in range(len(self.dec_bbox_head))]
#         )
#
#     def _reset_parameters(self, feat_channels):
#         bias = bias_init_with_prob(0.01)
#         init.constant_(self.enc_score_head.bias, bias)
#         init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
#         init.constant_(self.enc_bbox_head.layers[-1].bias, 0)
#
#         init.constant_(self.pre_bbox_head.layers[-1].weight, 0)
#         init.constant_(self.pre_bbox_head.layers[-1].bias, 0)
#
#         for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
#             init.constant_(cls_.bias, bias)
#             if hasattr(reg_, 'layers'):
#                 init.constant_(reg_.layers[-1].weight, 0)
#                 init.constant_(reg_.layers[-1].bias, 0)
#
#         init.xavier_uniform_(self.enc_output[0].weight)
#         if self.learn_query_content:
#             init.xavier_uniform_(self.tgt_embed.weight)
#         init.xavier_uniform_(self.query_pos_head.layers[0].weight)
#         init.xavier_uniform_(self.query_pos_head.layers[1].weight)
#         for m, in_channels in zip(self.input_proj, feat_channels):
#             if in_channels != self.hidden_dim:
#                 init.xavier_uniform_(m[0].weight)
#
#     def _build_input_proj_layer(self, feat_channels):
#         self.input_proj = nn.ModuleList()
#         for in_channels in feat_channels:
#             if in_channels == self.hidden_dim:
#                 self.input_proj.append(nn.Identity())
#             else:
#                 self.input_proj.append(
#                     nn.Sequential(OrderedDict([
#                         ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),
#                         ('norm', nn.BatchNorm2d(self.hidden_dim,))])
#                     )
#                 )
#
#         in_channels = feat_channels[-1]
#
#         for _ in range(self.num_levels - len(feat_channels)):
#             if in_channels == self.hidden_dim:
#                 self.input_proj.append(nn.Identity())
#             else:
#                 self.input_proj.append(
#                     nn.Sequential(OrderedDict([
#                         ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
#                         ('norm', nn.BatchNorm2d(self.hidden_dim))])
#                     )
#                 )
#                 in_channels = self.hidden_dim
#
#     def _get_encoder_input(self, feats: List[torch.Tensor]):
#         # get projection features
#         proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
#         if self.num_levels > len(proj_feats):
#             len_srcs = len(proj_feats)
#             for i in range(len_srcs, self.num_levels):
#                 if i == len_srcs:
#                     proj_feats.append(self.input_proj[i](feats[-1]))
#                 else:
#                     proj_feats.append(self.input_proj[i](proj_feats[-1]))
#
#         # get encoder inputs
#         feat_flatten = []
#         spatial_shapes = []
#         for i, feat in enumerate(proj_feats):
#             _, _, h, w = feat.shape
#             # [b, c, h, w] -> [b, h*w, c]
#             feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
#             # [num_levels, 2]
#             spatial_shapes.append([h, w])
#
#         # [b, l, c]
#         feat_flatten = torch.concat(feat_flatten, 1)
#         return feat_flatten, spatial_shapes
#
#     def _generate_anchors(self,
#                           spatial_shapes=None,
#                           grid_size=0.05,
#                           dtype=torch.float32,
#                           device='cpu'):
#         if spatial_shapes is None:
#             spatial_shapes = []
#             eval_h, eval_w = self.eval_spatial_size
#             for s in self.feat_strides:
#                 spatial_shapes.append([int(eval_h / s), int(eval_w / s)])
#
#         anchors = []
#         for lvl, (h, w) in enumerate(spatial_shapes):
#             grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
#             grid_xy = torch.stack([grid_x, grid_y], dim=-1)
#             grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
#             wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
#             lvl_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)
#             anchors.append(lvl_anchors)
#
#         anchors = torch.concat(anchors, dim=1).to(device)
#         valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
#         anchors = torch.log(anchors / (1 - anchors))
#         anchors = torch.where(valid_mask, anchors, torch.inf)
#
#         return anchors, valid_mask
#
#
#     def _get_decoder_input(self,
#                            memory: torch.Tensor,
#                            spatial_shapes,
#                            denoising_logits=None,
#                            denoising_bbox_unact=None):
#
#         # prepare input for decoder
#         if self.training or self.eval_spatial_size is None:
#             anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
#         else:
#             anchors = self.anchors
#             valid_mask = self.valid_mask
#         if memory.shape[0] > 1:
#             anchors = anchors.repeat(memory.shape[0], 1, 1)
#
#         # memory = torch.where(valid_mask, memory, 0)
#         # TODO fix type error for onnx export
#         memory = valid_mask.to(memory.dtype) * memory
#
#         output_memory :torch.Tensor = self.enc_output(memory)
#         enc_outputs_logits :torch.Tensor = self.enc_score_head(output_memory)
#
#         enc_topk_bboxes_list, enc_topk_logits_list = [], []
#         enc_topk_memory, enc_topk_logits, enc_topk_anchors = \
#             self._select_topk(output_memory, enc_outputs_logits, anchors, self.num_queries)
#
#         enc_topk_bbox_unact :torch.Tensor = self.enc_bbox_head(enc_topk_memory) + enc_topk_anchors
#
#         if self.training:
#             enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)
#             enc_topk_bboxes_list.append(enc_topk_bboxes)
#             enc_topk_logits_list.append(enc_topk_logits)
#
#         # if self.num_select_queries != self.num_queries:
#         #     raise NotImplementedError('')
#
#         if self.learn_query_content:
#             content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
#         else:
#             content = enc_topk_memory.detach()
#
#         enc_topk_bbox_unact = enc_topk_bbox_unact.detach()
#
#         if denoising_bbox_unact is not None:
#             enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
#             content = torch.concat([denoising_logits, content], dim=1)
#
#         return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list
#
#     def _select_topk(self, memory: torch.Tensor, outputs_logits: torch.Tensor, outputs_anchors_unact: torch.Tensor, topk: int):
#         if self.query_select_method == 'default':
#             _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)
#
#         elif self.query_select_method == 'one2many':
#             _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
#             topk_ind = topk_ind // self.num_classes
#
#         elif self.query_select_method == 'agnostic':
#             _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)
#
#         topk_ind: torch.Tensor
#
#         topk_anchors = outputs_anchors_unact.gather(dim=1, \
#             index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_anchors_unact.shape[-1]))
#
#         topk_logits = outputs_logits.gather(dim=1, \
#             index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1])) if self.training else None
#
#         topk_memory = memory.gather(dim=1, \
#             index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))
#
#         return topk_memory, topk_logits, topk_anchors
#
#     def forward(self, feats, targets=None):
#         """
#         Args:
#             feats: List[Tensor] - Backbone输出的多尺度特征图
#             targets: List[Dict] - 每个dict包含:
#                 'boxes': FloatTensor[N, 4] (x1,y1,x2,y2)
#                 'labels': Int64Tensor[N]
#         """
#         # 1. 输入投影和编码
#         memory, spatial_shapes = self._get_encoder_input(feats)
#
#         # 2. 准备去噪训练
#         if self.training and self.num_denoising > 0:
#             denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
#                 get_contrastive_denoising_training_group(
#                     targets,
#                     self.num_classes,
#                     self.num_queries,
#                     self.denoising_class_embed,
#                     num_denoising=self.num_denoising,
#                     label_noise_ratio=self.label_noise_ratio,
#                     box_noise_scale=1.0,
#                 )
#         else:
#             denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None
#
#         # 3. 获取解码器输入
#         init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = \
#             self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact)
#
#         # 4. 解码器前向
#         out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_logits = self.decoder(
#             init_ref_contents,
#             init_ref_points_unact,
#             memory,
#             spatial_shapes,
#             self.dec_bbox_head,
#             self.dec_score_head,
#             self.query_pos_head,
#             self.pre_bbox_head,
#             self.integral,
#             self.up,
#             self.reg_scale,
#             attn_mask=attn_mask,
#             dn_meta=dn_meta)
#
#         # 5. 处理去噪分支
#         if self.training and dn_meta is not None:
#             dn_pre_logits, pre_logits = torch.split(pre_logits, dn_meta['dn_num_split'], dim=1)
#             dn_pre_bboxes, pre_bboxes = torch.split(pre_bboxes, dn_meta['dn_num_split'], dim=1)
#             dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
#             dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
#             dn_out_corners, out_corners = torch.split(out_corners, dn_meta['dn_num_split'], dim=2)
#             dn_out_refs, out_refs = torch.split(out_refs, dn_meta['dn_num_split'], dim=2)
#
#         # 6. 训练时计算损失
#         if self.training:
#             # 1. 准备目标数据
#             assert isinstance(targets, list), "Targets must be a list of dicts"
#
#             # 使用pad_sequence处理变长目标
#             target_boxes = pad_sequence(
#                 [t['boxes'] for t in targets],
#                 batch_first=True,
#                 padding_value=-1
#             )  # [bs, max_objs, 4]
#
#             target_labels = pad_sequence(
#                 [t['labels'] for t in targets],
#                 batch_first=True,
#                 padding_value=-1
#             )  # [bs, max_objs]
#
#             # 2. 确保预测框是连续的
#             pred_boxes = out_bboxes[-1].contiguous()  # [bs, num_queries, 4]
#             pred_logits = out_logits[-1]  # [bs, num_queries, num_classes]
#
#             # 3. 为每个样本计算损失
#             total_cls_loss = 0
#             total_box_loss = 0
#
#             for i in range(len(targets)):
#                 # 获取当前样本的有效目标
#                 valid_mask = target_boxes[i, :, 0] >= 0  # 根据padding值判断
#                 gt_boxes = target_boxes[i][valid_mask]  # [num_gt, 4]
#                 gt_labels = target_labels[i][valid_mask]  # [num_gt]
#
#                 if len(gt_boxes) == 0:
#                     continue  # 如果没有有效目标，跳过当前样本
#
#                 # 4. 使用匈牙利算法匹配预测和真实框
#                 with torch.no_grad():
#                     # 计算代价矩阵
#                     cost_matrix = self.compute_cost_matrix(
#                         pred_boxes[i],
#                         pred_logits[i],
#                         gt_boxes,
#                         gt_labels
#                     )
#
#                     # 调用linear_sum_assignment进行匹配
#                     matched_pred_idx, matched_gt_idx = linear_sum_assignment(cost_matrix.cpu())
#                     matched_pred_idx = torch.as_tensor(matched_pred_idx, device=pred_boxes.device)
#                     matched_gt_idx = torch.as_tensor(matched_gt_idx, device=pred_boxes.device)
#
#                 # 5. 计算分类损失
#                 cls_loss = F.cross_entropy(
#                     pred_logits[i][matched_pred_idx],
#                     gt_labels[matched_gt_idx]
#                 )
#
#                 # 6. 计算CIoU损失
#                 box_loss = self.compute_ciou_loss(
#                     pred_boxes[i][matched_pred_idx],
#                     gt_boxes[matched_gt_idx]
#                 )
#
#                 total_cls_loss += cls_loss
#                 total_box_loss += box_loss
#
#             # 7. 平均损失
#             losses = {
#                 'cls_loss': total_cls_loss / len(targets),
#                 'box_loss': total_box_loss / len(targets),
#             }
#
#             out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1], 'up': self.up, 'reg_scale': self.reg_scale}
#             # 'pred_corners': out_corners[-1],'ref_points': out_refs[-1],
#         else:
#             # 推理输出
#             out = {
#                 'pred_logits': out_logits[-1],
#                 'pred_boxes': out_bboxes[-1],
#             }
#
#         # 7. 处理辅助输出
#         if self.training and self.aux_loss:
#             out['aux_outputs'] = self._set_aux_loss2(
#                 out_logits[:-1], out_bboxes[:-1], out_corners[:-1], out_refs[:-1],
#                 out_corners[-1], out_logits[-1]
#             )
#             out['enc_aux_outputs'] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
#             out['pre_outputs'] = {'pred_logits': pre_logits, 'pred_boxes': pre_bboxes}
#             out['enc_meta'] = {'class_agnostic': self.query_select_method == 'agnostic'}
#
#             if dn_meta is not None:
#                 out['dn_outputs'] = self._set_aux_loss2(
#                     dn_out_logits, dn_out_bboxes, dn_out_corners, dn_out_refs,
#                     dn_out_corners[-1], dn_out_logits[-1]
#                 )
#                 out['dn_pre_outputs'] = {'pred_logits': dn_pre_logits, 'pred_boxes': dn_pre_bboxes}
#                 out['dn_meta'] = dn_meta
#
#         return out
#
#     ############################################################################################
#     # CIoU计算核心逻辑
#
#     def compute_cost_matrix(self, pred_boxes, pred_logits, gt_boxes, gt_labels):
#         """计算匈牙利匹配的代价矩阵"""
#         # 分类代价
#         pred_probs = pred_logits.softmax(dim=-1)  # [num_queries, num_classes]
#         class_cost = -pred_probs[:, gt_labels]  # [num_queries, num_gt]
#
#         # 位置代价
#         iou = self.compute_pairwise_iou(pred_boxes, gt_boxes)  # [num_queries, num_gt]
#         loc_cost = -iou
#
#         # 综合代价
#         return class_cost + loc_cost  # 可以调整权重
#
#     def compute_pairwise_iou(self, boxes1, boxes2):
#         """计算两组框之间的IoU矩阵"""
#         # 计算交集
#         inter_x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
#         inter_y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
#         inter_x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
#         inter_y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
#
#         inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
#
#         # 计算并集
#         area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
#         area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
#         union_area = area1[:, None] + area2 - inter_area
#
#         return inter_area / (union_area + 1e-8)
#
#     def compute_ciou_loss(self, pred_boxes, target_boxes, eps=1e-7):
#         # 确保输入形状一致
#         pred_boxes = pred_boxes.view(-1, 4)
#         target_boxes = target_boxes.view(-1, 4)
#
#         # 计算最小最大坐标
#         inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
#         inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
#         inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
#         inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
#
#         # 计算面积
#         inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
#         pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
#         target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
#
#         # IoU计算
#         union_area = pred_area + target_area - inter_area + eps
#         iou = inter_area / union_area
#
#         # 中心点距离
#         pred_ctr = torch.stack([(pred_boxes[:, 0] + pred_boxes[:, 2]) / 2,
#                                 (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2], dim=1)
#         target_ctr = torch.stack([(target_boxes[:, 0] + target_boxes[:, 2]) / 2,
#                                   (target_boxes[:, 1] + target_boxes[:, 3]) / 2], dim=1)
#         center_dist = torch.sum((pred_ctr - target_ctr) ** 2, dim=1)
#
#         # 封闭框对角线
#         enclose_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
#         enclose_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
#         enclose_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
#         enclose_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
#         enclose_dist = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps
#
#         # 长宽比惩罚
#         with torch.no_grad():
#             arctan_pred = torch.atan((pred_boxes[:, 2] - pred_boxes[:, 0]) /
#                                      (pred_boxes[:, 3] - pred_boxes[:, 1] + eps))
#             arctan_target = torch.atan((target_boxes[:, 2] - target_boxes[:, 0]) /
#                                        (target_boxes[:, 3] - target_boxes[:, 1] + eps))
#             v = (4 / (math.pi ** 2)) * (arctan_pred - arctan_target).pow(2)
#             alpha = v / (1 - iou + v + eps)
#
#         # 最终CIoU
#         ciou = iou - (center_dist / enclose_dist) - (alpha * v)
#         return (1 - ciou).mean()
#     ############################################################################################
#
#     @torch.jit.unused
#     def _set_aux_loss(self, outputs_class, outputs_coord):
#         # this is a workaround to make torchscript happy, as torchscript
#         # doesn't support dictionary with non-homogeneous values, such
#         # as a dict having both a Tensor and a list.
#         return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class, outputs_coord)]
#
#
#     @torch.jit.unused
#     def _set_aux_loss2(self, outputs_class, outputs_coord, outputs_corners, outputs_ref,
#                        teacher_corners=None, teacher_logits=None):
#         # this is a workaround to make torchscript happy, as torchscript
#         # doesn't support dictionary with non-homogeneous values, such
#         # as a dict having both a Tensor and a list.
#         return [{'pred_logits': a, 'pred_boxes': b, 'pred_corners': c, 'ref_points': d,
#                      'teacher_corners': teacher_corners, 'teacher_logits': teacher_logits}
#                 for a, b, c, d in zip(outputs_class, outputs_coord, outputs_corners, outputs_ref)]
