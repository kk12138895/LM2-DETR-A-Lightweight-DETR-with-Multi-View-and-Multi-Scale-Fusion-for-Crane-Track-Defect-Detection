import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F

#     #将输入的特征图划分为多个不重叠的窗口
# def window_partition(x, window_size=8):
#     B, H, W, C = x.shape
#     x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
#     windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

#     return windows

# #将划分后的窗口恢复为原始的特征图。
# def window_reverse(windows, window_size, H, W):
#     B = int(windows.shape[0] / (H * W / window_size / window_size))
#     x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
#     x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
#     return x

# #窗口交叉注意力模块
# class WindowCrossAttention(nn.Module):
#     r""" Window based multi-head self attention (W-MSA) module with relative position bias.
#     It supports both of shifted and non-shifted window.
#     Args:
#         dim (int): Number of input channels.
#         window_size (tuple[int]): The height and width of the window.
#         num_heads (int): Number of attention heads.
#         qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
#         attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
#         proj_drop (float, optional): Dropout ratio of output. Default: 0.0
#     """

#     def __init__(self, dim, window_size, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

#         super().__init__()
#         self.dim = dim
#         self.window_size=window_size
#         # self.window_size = window_size  # Wh, Ww
#         # self.num_heads = [[6,6], [6,6], [6,6], [6,6]]
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5

#         # define a parameter table of relative position bias
#         self.relative_position_bias_table_x = nn.Parameter(
#             torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

#         self.relative_position_bias_table_y = nn.Parameter(
#             torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

#         # get pair-wise relative position index for each token inside the window
#         coords_h = torch.arange(self.window_size[0])
#         coords_w = torch.arange(self.window_size[1])
#         coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
#         coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
#         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
#         relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
#         relative_coords[:, :, 1] += self.window_size[1] - 1
#         relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
#         relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
#         self.register_buffer("relative_position_index", relative_position_index)

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.merge1 = nn.Linear(dim*2, dim)
#         self.merge2 = nn.Linear(dim, dim)
#         self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#         self.proj = nn.Linear(dim, dim)

#         self.proj_drop = nn.Dropout(proj_drop)

#         trunc_normal_(self.relative_position_bias_table_x, std=.02)
#         trunc_normal_(self.relative_position_bias_table_y, std=.02)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, y, mask_x=None, mask_y=None):
#         """
#         Args:
#             x: input features with shape of (num_windows*B, N, C)
#             mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
#         """
#         input_dtype = x.dtype
#         x = x.float()
#         y = y.float()
#         if mask_x is not None:
#             mask_x = mask_x.float()
#         if mask_y is not None:
#             mask_y = mask_y.float()

#         B_, N, C = x.shape
#         # print(C)
#         # print(self.num_heads)
#         with torch.autocast(device_type=x.device.type, enabled=False):
#             qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
#             q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

#             q = q * self.scale
#             attn = (q @ k.transpose(-2, -1).contiguous())

#             relative_position_bias = self.relative_position_bias_table_x[self.relative_position_index.view(-1)].view(
#                 self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
#             relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
#             attn = attn + relative_position_bias.unsqueeze(0)

#             if mask_x is not None:
#                 nW = mask_x.shape[0]
#                 attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask_x.unsqueeze(1).unsqueeze(0)
#                 attn = attn.view(-1, self.num_heads, N, N)
#                 attn = self.softmax(attn)
#             else:
#                 attn = self.softmax(attn)

#             attn = self.attn_drop(attn)

#             x = (attn @ v).transpose(1, 2).reshape(B_, N, C).contiguous()

#             B_, N, C = y.shape
#             kv = self.kv(y).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#             k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)

#             attn = (q @ k.transpose(-2, -1).contiguous())

#             relative_position_bias = self.relative_position_bias_table_y[self.relative_position_index.view(-1)].view(
#                 self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
#             relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
#             attn = attn + relative_position_bias.unsqueeze(0)

#             if mask_y is not None:
#                 nW = mask_y.shape[0]
#                 attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask_y.unsqueeze(1).unsqueeze(0)
#                 attn = attn.view(-1, self.num_heads, N, N)
#                 attn = self.softmax(attn)
#             else:
#                 attn = self.softmax(attn)

#             attn = self.attn_drop(attn)

#             y = (attn @ v).transpose(1, 2).reshape(B_, N, C).contiguous()

#             x = self.merge2(self.act(self.merge1(torch.cat([x, y], dim=-1)))) + x

#             x = self.proj(x)
#             x = self.proj_drop(x)
#         return x.to(input_dtype)
# #基于窗口的交叉注意力块
# class CrossWinBasedAttention(nn.Module):
#     def __init__(self, dim=192, num_heads=8, window_size=8, shift_size=0,
#                  qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
#                  reverse_aux_windows=False):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.window_size = window_size
#         self.shift_size = shift_size
#         self.reverse_aux_windows = reverse_aux_windows

#         assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

#         self.attn = WindowCrossAttention(
#             dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
#             qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def calculate_mask(self, x_size):
#         # calculate attention mask for SW-MSA
#         H, W = x_size
#         img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
#         h_slices = (slice(0, -self.window_size),
#                     slice(-self.window_size, -self.shift_size),
#                     slice(-self.shift_size, None))
#         w_slices = (slice(0, -self.window_size),
#                     slice(-self.window_size, -self.shift_size),
#                     slice(-self.shift_size, None))
#         cnt = 0
#         for h in h_slices:
#             for w in w_slices:
#                 img_mask[:, h, w, :] = cnt
#                 cnt += 1

#         mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
#         mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
#         attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
#         attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

#         return attn_mask


#     def calculate_mask2(self, x_size):
#         # calculate attention mask for SW-MSA
#         H, W = x_size
#         img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
#         h_slices = (slice(0, -self.window_size),
#                     slice(-self.window_size, -self.shift_size),
#                     slice(-self.shift_size, None))
#         w_slices = (slice(0, -self.window_size),
#                     slice(-self.window_size, -self.shift_size),
#                     slice(-self.shift_size, None))
#         cnt = 0
#         for h in h_slices:
#             for w in w_slices:
#                 img_mask[:, h, w, :] = cnt
#                 cnt += 1

#         mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
#         mask_windows = mask_windows.view(-1, self.window_size * self.window_size)

#         ## downscale
#         img_mask_down = F.interpolate(img_mask.permute(0, 3, 1, 2).contiguous(), scale_factor=0.5, mode="bilinear", align_corners=False)
#         img_mask_down = F.pad(img_mask_down, (self.window_size//4, self.window_size//4, self.window_size//4, self.window_size//4), mode='reflect')
#         mask_windows_down = F.unfold(img_mask_down, kernel_size=self.window_size, dilation=1, padding=0, stride=self.window_size//2)
#         mask_windows_down = mask_windows_down.view(self.window_size*self.window_size, -1).permute(1, 0).contiguous()  # nW, window_size*window_size

#         attn_mask = mask_windows_down.unsqueeze(1) - mask_windows.unsqueeze(2)  # nW, window_size*window_size, window_size*window_size
#         attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

#         return attn_mask

#     def forward(self, x, y):
#         B, C, H, W = x.shape
#         if y.shape != x.shape:
#             raise ValueError(f"CrossWinBasedAttention expects paired BCHW features with the same shape, got {x.shape} and {y.shape}")

#         shortcut = x
#         x = x.permute(0, 2, 3, 1)

#         if self.shift_size > 0:
#             # calculate attention mask for SW-MSA
#             img_mask = torch.zeros((1, H, W, 1), device=x.device)
#             h_slices = (slice(0, -self.window_size),
#                         slice(-self.window_size, -self.shift_size),
#                         slice(-self.shift_size, None))
#             w_slices = (slice(0, -self.window_size),
#                         slice(-self.window_size, -self.shift_size),
#                         slice(-self.shift_size, None))
#             cnt = 0
#             for h in h_slices:
#                 for w in w_slices:
#                     img_mask[:, h, w, :] = cnt
#                     cnt += 1

#             mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
#             mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
#             attn_mask_x = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
#             attn_mask_x = attn_mask_x.masked_fill(attn_mask_x != 0, float(-100.0)).masked_fill(attn_mask_x == 0, float(0.0))
#             #Mask2
#             img_mask = torch.zeros((1, H, W, 1), device=x.device)  # 1 H W 1
#             h_slices = (slice(0, -self.window_size),
#                     slice(-self.window_size, -self.shift_size),
#                     slice(-self.shift_size, None))
#             w_slices = (slice(0, -self.window_size),
#                     slice(-self.window_size, -self.shift_size),
#                     slice(-self.shift_size, None))
#             cnt = 0
#             for h in h_slices:
#                 for w in w_slices:
#                     img_mask[:, h, w, :] = cnt
#                     cnt += 1

#             mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
#             mask_windows = mask_windows.view(-1, self.window_size * self.window_size)

#             ## downscale
#             img_mask_down = F.interpolate(img_mask.permute(0, 3, 1, 2).contiguous(), scale_factor=0.5, mode="bilinear", align_corners=False)
#             img_mask_down = F.pad(img_mask_down, (self.window_size//4, self.window_size//4, self.window_size//4, self.window_size//4), mode='reflect')
#             mask_windows_down = F.unfold(img_mask_down, kernel_size=self.window_size, dilation=1, padding=0, stride=self.window_size//2)
#             mask_windows_down = mask_windows_down.view(self.window_size*self.window_size, -1).permute(1, 0).contiguous()  # nW, window_size*window_size

#             attn_mask_y = mask_windows_down.unsqueeze(1) - mask_windows.unsqueeze(2)  # nW, window_size*window_size, window_size*window_size
#             attn_mask_y = attn_mask_y.masked_fill(attn_mask_y != 0, float(-100.0)).masked_fill(attn_mask_y == 0, float(0.0))

#         else:
#             attn_mask_x = None
#             attn_mask_y = None

#         # cyclic shift
#         if self.shift_size > 0:
#             shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
#         else:
#             shifted_x = x

#         # partition windows
#         x_windows = window_partition(shifted_x, self.window_size)
#         x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

#         # W-MSA/SW-MSA
#         shifted_x_down = F.interpolate(y.contiguous(), scale_factor=0.5, mode="bilinear", align_corners=False)
#         shifted_x_down = F.pad(shifted_x_down, (self.window_size//4, self.window_size//4, self.window_size//4, self.window_size//4), mode='reflect')
#         x_windows_down = F.unfold(shifted_x_down, kernel_size=self.window_size, dilation=1, padding=0, stride=self.window_size//2)
#         x_windows_down = x_windows_down.view(B, C, self.window_size*self.window_size, -1)
#         x_windows_down = x_windows_down.permute(0, 3, 2, 1).contiguous()
#         if self.reverse_aux_windows:
#             # Keep the rot180 feature map untouched, but pair each main-view
#             # window with its spatially corresponding window in the rotated view.
#             x_windows_down = x_windows_down.flip(1)
#         x_windows_down = x_windows_down.view(-1, self.window_size*self.window_size, C)  # nW*B, window_size*window_size, C
#         attn_windows = self.attn(x_windows, x_windows_down, mask_x=attn_mask_x, mask_y=attn_mask_y)
        
#         # merge windows
#         attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
#         shifted_x = window_reverse(attn_windows, self.window_size, H, W)

#         # reverse cyclic shift
#         if self.shift_size > 0:
#             x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
#         else:
#             x = shifted_x

#         x = x.permute(0, 3, 1, 2).contiguous()
#         x = shortcut + self.drop_path(x)

#         return x
    
# def conv3x3(in_ch, out_ch, stride=1):
#     """3x3 convolution with padding."""
#     return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)

# def conv1x1(in_ch, out_ch, stride=1):
#     """1x1 convolution."""
#     return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

# class Win_noShift_Attention(nn.Module):
#     """Window-based self-attention module."""

#     def __init__(self, dim, num_heads=8, window_size=8, shift_size=0):
#         super().__init__()
#         N = dim

#         class ResidualUnit(nn.Module):
#             """简单的残差单元类."""

#             def __init__(self):
#                 super().__init__()
#                 self.conv = nn.Sequential(
#                     conv1x1(N, N // 2),
#                     nn.GELU(),
#                     conv3x3(N // 2, N // 2),
#                     nn.GELU(),
#                     conv1x1(N // 2, N),
#                 )
#                 self.relu = nn.GELU()

#             def forward(self, x):
#                 identity = x
#                 out = self.conv(x)
#                 out += identity
#                 out = self.relu(out)
#                 return out
#         #用于对输入特征进行特征提取。
#         self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())

#         self.conv_b = nn.Sequential(
#             CrossWinBasedAttention(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=shift_size),
#             ResidualUnit(),
#             ResidualUnit(),
#             ResidualUnit(),
#             conv1x1(N, N),
#         )

#     def forward(self, x):
#         identity = x
#         a = self.conv_a(x)
#         b = self.conv_b(x)
#         out = a * torch.sigmoid(b)
#         """"将 a 和 b 的输出进行逐元素相乘，并使用 sigmoid 激活函数对 b 的输出进行处理，得到注意力权重。"""
#         out += identity
#         return out


# if __name__ == '__main__':
#     x = torch.rand([2, 192, 64, 64])
#     # attn = WinBasedAttention()
#     attn = CrossWinBasedAttention()
#     # x = window_partition(x)
#     x = attn(x)
#     # print(x.shape)

class CrossWinBasedAttention(nn.Module):
  
    def __init__(self, dim=192, num_heads=8, window_size=8, shift_size=0,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 reverse_aux_windows=False, reduction=64):
        super().__init__()

        hidden_dim = max(dim // reduction, 8)

        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 3, hidden_dim, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

        self.local_refine = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim)
        )

        self.alpha = nn.Parameter(torch.zeros(1))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, y):
        if y.shape != x.shape:
            raise ValueError(
                f"CrossWinBasedAttention expects paired BCHW features with the same shape, "
                f"got {x.shape} and {y.shape}"
            )

        identity = x

        diff = torch.abs(x - y)
        gate_input = torch.cat([x, y, diff], dim=1)

        c_gate = self.channel_gate(gate_input)

        s_avg = torch.mean(diff, dim=1, keepdim=True)
        s_max, _ = torch.max(diff, dim=1, keepdim=True)
        s_gate = self.spatial_gate(torch.cat([s_avg, s_max], dim=1))

        gate = c_gate * s_gate

        fused = x + gate * (y - x)
        fused = self.local_refine(fused)

        out = identity + self.drop_path(self.alpha * fused)
        return out

# class CrossWinBasedAttention(nn.Module):
#     """
#     Lightweight cross-feature fusion for feature-level interaction.
#     This module keeps the original class name and forward(x, y) interface,
#     but replaces heavy window attention with a low-cost gated fusion design.

#     Args:
#         dim: input channel number.
#         reduction: channel reduction ratio for the fusion gate.
#     """
#     def __init__(self, dim=192, num_heads=8, window_size=8, shift_size=0,
#                  qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
#                  reverse_aux_windows=False, reduction=16):
#         super().__init__()
#         hidden_dim = max(dim // reduction, 16)

#         self.gate = nn.Sequential(
#             nn.Conv2d(dim * 3, hidden_dim, kernel_size=1, bias=False),
#             nn.BatchNorm2d(hidden_dim),
#             nn.SiLU(inplace=True),
#             nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False),
#             nn.BatchNorm2d(hidden_dim),
#             nn.SiLU(inplace=True),
#             nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=True),
#             nn.Sigmoid()
#         )

#         self.local = nn.Sequential(
#             nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
#             nn.BatchNorm2d(dim),
#             nn.SiLU(inplace=True),
#             nn.Conv2d(dim, dim, kernel_size=1, bias=False),
#             nn.BatchNorm2d(dim)
#         )

#         self.alpha = nn.Parameter(torch.zeros(1))
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, x, y):
#         if y.shape != x.shape:
#             raise ValueError(
#                 f"CrossWinBasedAttention expects paired BCHW features with the same shape, "
#                 f"got {x.shape} and {y.shape}"
#             )

#         identity = x

#         fusion_input = torch.cat([x, y, torch.abs(x - y)], dim=1)
#         gate = self.gate(fusion_input)

#         fused = x + gate * (y - x)
#         fused = self.local(fused)

#         out = identity + self.drop_path(self.alpha * fused)
#         return out