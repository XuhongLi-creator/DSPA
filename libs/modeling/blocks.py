import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .weight_init import trunc_normal_


class MaskedConv1D(nn.Module):
    """
    Masked 1D convolution. Interface remains the same as Conv1d.
    Only support a sub set of 1d convs
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode='zeros'
    ):
        super().__init__()
        # element must be aligned
        assert (kernel_size % 2 == 1) and (kernel_size // 2 == padding)
        # stride
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode)
        # zero out the bias term if it exists
        if bias:
            torch.nn.init.constant_(self.conv.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()
        # input length must be divisible by stride
        #assert T % self.stride == 0

        # conv
        out_conv = self.conv(x)
        # compute the mask
        '''if self.stride > 1:
            # downsample the mask using nearest neighbor
            out_mask = F.interpolate(
                mask.to(x.dtype),
                size=T // self.stride,
                mode='nearest'
            )
        else:
            # masking out the features
            out_mask = mask.to(x.dtype)'''

        out_mask = F.interpolate(
            mask.to(x.dtype), size=out_conv.size(-1), mode='nearest'
        )

        # masking the output, stop grad to mask
        out_conv = out_conv * out_mask.detach()
        out_mask = out_mask.bool()
        return out_conv, out_mask


class LayerNorm(nn.Module):
    """
    LayerNorm that supports inputs of size B, C, T
    """

    def __init__(
            self,
            num_channels,
            eps=1e-5,
            affine=True,
            device=None,
            dtype=None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels

        # normalization along C channels
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x ** 2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)

        # apply weight and bias
        if self.affine:
            out *= self.weight
            out += self.bias

        return out


# helper functions for Transformer blocks
def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    # return a tensor of size 1 C T
    return torch.FloatTensor(sinusoid_table).unsqueeze(0).transpose(1, 2)


class global_mask_atten(nn.Module):
    """
    Multi Head Conv Attention with mask

    Add a depthwise convolution within a standard MHA
    The extra conv op can be used to
    (1) encode relative position information (relacing position encoding);
    (2) downsample the features if needed;
    (3) match the feature channels

    Note: With current implementation, the downsampled feature will be aligned
    to every s+1 time step, where s is the downsampling stride. This allows us
    to easily interpolate the corresponding positional embeddings.

    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
            self,
            n_embd,  # dimension of the output features
            n_head=4,  # number of heads in multi-head self-attention
            n_qx_stride=1,  # dowsampling stride for query and input
            n_kv_stride=1,  # downsampling stride for key and value
            attn_pdrop=0.0,  # dropout rate for the attention map
            proj_pdrop=0.0,  # dropout rate for projection op
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        # conv/pooling operations
        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        # query conv (depthwise)
        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.query_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.query_norm = LayerNorm(self.n_embd)

        # key, value conv (depthwise)
        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.key_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.key_norm = LayerNorm(self.n_embd)
        self.value_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.value_norm = LayerNorm(self.n_embd)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        x = x.transpose(1, 2)
        # print("333x", x.shape)
        # print("333mask", mask.shape)
        B, C, T = x.size()

        # query conv -> (B, nh * hs, T')
        q, qx_mask = self.query_conv(x, mask)
        q = self.query_norm(q)
        # key, value conv -> (B, nh * hs, T'')
        k, kv_mask = self.key_conv(x, mask)
        k = self.key_norm(k)
        v, _ = self.value_conv(x, mask)
        v = self.value_norm(v)

        # projections
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # move head forward to be the batch dim
        # (B, nh * hs, T'/T'') -> (B, nh, T'/T'', hs)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        # self-attention: (B, nh, T', hs) x (B, nh, hs, T'') -> (B, nh, T', T'')
        att = (q * self.scale) @ k.transpose(-2, -1)
        # prevent q from attending to invalid tokens
        att = att.masked_fill(torch.logical_not(kv_mask[:, :, None, :]), float('-inf'))
        # softmax attn
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        # (B, nh, T', T'') x (B, nh, T'', hs) -> (B, nh, T', hs)
        out = att @ (v * kv_mask[:, :, :, None].to(v.dtype))
        # re-assemble all head outputs side by side
        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * qx_mask.to(out.dtype)
        out = out.transpose(1, 2)
        return out


class Local_cnn(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.TC = nn.Conv1d(hidden_features, hidden_features, 3, 1, 1, bias=True,
                            groups=hidden_features)  # k=3, stride=1, padding=1
        self.act = act_layer()
        self.linear2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.linear1(x)
        x = x.transpose(1, 2)
        x = self.TC(x)
        x = x.transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop(x)
        return x


class connectionBlock(nn.Module):
    """
    Global Local Relational Block
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.global_mask_atten = global_mask_atten(
            dim)

        self.norm2 = LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.Local_cnn = Local_cnn(in_features=dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, mask):
        x = x + self.global_mask_atten(self.norm1(x.transpose(1,2)).transpose(1,2), mask)
        x = x + self.Local_cnn(self.norm2(x.transpose(1,2)).transpose(1,2))
        return x


class Temporal_Merging_Block(nn.Module):
    """
    Temporal_Merging_Block
    """

    def __init__(self, kernel_size=3, stride=1, in_chans=1024, embed_dim=256, with_ln=False):
        super().__init__()
        '''self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride,
                              padding=(kernel_size // 2))'''
        self.proj = MaskedConv1D(
                in_chans, embed_dim, kernel_size=kernel_size,
                stride=stride, padding=(kernel_size // 2), bias=(not with_ln)
            )
        self.norm = LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, mask):
        x, mask = self.proj(x, mask)
        mask = F.interpolate(
            mask.to(x.dtype), size=x.size(-1), mode='nearest'
        )
        x = self.norm(x)
        x = x.transpose(1, 2)
        return x, mask


class TemporalEncoder(nn.Module):
    def __init__(self, in_feat_dim=512, embed_dims=[512,1152],
                 num_head=4, mlp_ratio=8, norm_layer=nn.LayerNorm,
                 num_block=2):
        super().__init__()
        # Stage 1
        self.Temporal_Merging_Block1 = Temporal_Merging_Block(kernel_size=3, stride=1, in_chans=in_feat_dim,
                                                              embed_dim=embed_dims[0])
        self.block1 = nn.ModuleList([connectionBlock(
            dim=embed_dims[0], num_heads=num_head, mlp_ratio=mlp_ratio, norm_layer=norm_layer)
            for i in range(num_block)])
        self.norm1 = LayerNorm(embed_dims[0])

        # Stage 2
        self.Temporal_Merging_Block2 = Temporal_Merging_Block(kernel_size=3, stride=2, in_chans=embed_dims[0],
                                                              embed_dim=embed_dims[1])
        self.block2 = nn.ModuleList([connectionBlock(
            dim=embed_dims[1], num_heads=num_head, mlp_ratio=mlp_ratio, norm_layer=norm_layer)
            for i in range(num_block)])
        self.norm2 = LayerNorm(embed_dims[1])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_init_emb(self):
        self.Temporal_Merging_Block1.requires_grad = False

    def forward(self, x, mask):
        outs = []
        # stage 1
        x, mask = self.Temporal_Merging_Block1(x, mask)
        for i, blk in enumerate(self.block1):
            x = blk(x, mask)
        x = x.permute(0, 2, 1).contiguous()
        x = self.norm1(x)
        outs.append(x)

        # stage 2
        x, mask = self.Temporal_Merging_Block2(x, mask)
        for i, blk in enumerate(self.block2):
            x = blk(x, mask)
        x = x.permute(0, 2, 1).contiguous()
        x = self.norm2(x)
        outs.append(x)
        return outs


class linearProj(nn.Module):
    #
    def __init__(self, input_dim=1024, embed_dim=256):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.proj(x)
        return x


def interpolation(input,
           size=None,
           scale_factor=None,
           mode='nearest'):
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode)


class horizontal_fusion(nn.Module):
    def __init__(self, inter_channels, embedding_dim):
        super().__init__()
        c1_in_channels, c2_in_channels = inter_channels

        self.linear_f2 = linearProj(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_f1 = linearProj(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear1 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1)
        self.linear2 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1)
    def forward(self, x):
        f1, f2 = x
        # Temporal Scale Mixer Module

        _f2 = self.linear_f2(f2).permute(0, 2, 1)
        _f2 = interpolation(_f2, size=f1.size()[2:], mode='nearest')

        _f1 = self.linear_f1(f1).permute(0, 2, 1)

        # Mixer
        _f1_n = self.linear2(_f2) + _f1

        concat_feature = torch.cat([_f2, _f1_n], dim=1)
        # print("concat_feature", concat_feature.shape)
        return concat_feature


'''class Classification_Module(nn.Module):
    def __init__(self,  embedding_dim):
        super().__init__()
        self.linear_fuse = nn.Conv1d(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1)

    def forward(self, concat_feature):
        x = self.linear_fuse(concat_feature)
        return x'''


class HorizontalFPN(nn.Module):
    """
    MS-TCT for action detection
    """

    def __init__(self, inter_channels, num_block, head, mlp_ratio, in_feat_dim, final_embedding_dim):
        super(HorizontalFPN, self).__init__()

        self.dropout = nn.Dropout()

        self.TemporalEncoder = TemporalEncoder(in_feat_dim=in_feat_dim, embed_dims=inter_channels,
                                               num_head=head, mlp_ratio=mlp_ratio, norm_layer=nn.LayerNorm,
                                               num_block=num_block)

        self.horizontal_fusion = horizontal_fusion(inter_channels=inter_channels, embedding_dim=final_embedding_dim)

        # self.Classfication_Module = Classification_Module(embedding_dim=final_embedding_dim)

    def forward(self, inputs, mask):
        inputs = self.dropout(inputs)

        # Temporal Encoder Module
        x = self.TemporalEncoder(inputs, mask)

        # Temporal Scale Mixer Module
        concat_feature = self.horizontal_fusion(x)

        # concat_feature = self.Classfication_Module(concat_feature)

        return concat_feature


class MMACBlock(nn.Module):
    """
    A simple conv block similar to the basic block used in ResNet
    """

    def __init__(
            self,
            n_embd,  # dimension of the input features
            exp_dim, # expand dim for embed_dim
            kernel_size=3,  # conv kernel size
            n_ds_stride=1,  # downsampling stride for the current layer
            group=1,  # group for cnn
            n_out=None,  # output dimension, if None, set to input dim
            n_hidden=None,  # hidden dim for mlp
            path_pdrop=0.0,  # drop path rate
            act_layer=nn.GELU,  # nonlinear activation used after conv, default ReLU,
            downsample_type='max',
            init_conv_vars=1  # init gaussian variance for the weight
    ):
        super().__init__()
        # must use odd sized kernel
        # assert (kernel_size % 2 == 1) and (kernel_size > 1)
        # padding = kernel_size // 2

        self.kernel_size = kernel_size
        self.stride = n_ds_stride

        if n_out is None:
            n_out = n_embd

        self.ln = LayerNorm(n_embd)

        self.gn = nn.GroupNorm(16, n_embd)

        self.attn3 = HorizontalFPN(
            inter_channels=[n_embd, exp_dim],
            num_block=1,
            head=4,
            mlp_ratio=4,
            in_feat_dim=n_embd,
            final_embedding_dim=n_embd // 2
        )
        # input
        if n_ds_stride > 1:
            if downsample_type == 'max':
                kernel_size, stride, padding = \
                    n_ds_stride + 1, n_ds_stride, (n_ds_stride + 1) // 2
                self.downsample = nn.MaxPool1d(
                    kernel_size, stride=stride, padding=padding)
                self.stride = stride
            elif downsample_type == 'avg':
                self.downsample = nn.Sequential(nn.AvgPool1d(n_ds_stride, stride=n_ds_stride, padding=0),
                                                nn.Conv1d(n_embd, n_embd, 1, 1, 0))
                self.stride = n_ds_stride
            else:
                raise NotImplementedError("downsample type error")
        else:
            self.downsample = nn.Identity()
            self.stride = 1

        # two layer mlp
        if n_hidden is None:
            n_hidden = 4 * n_embd  # default
        if n_out is None:
            n_out = n_embd

        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1, groups=group),
            act_layer(),
            nn.Conv1d(n_hidden, n_out, 1, groups=group),
        )

        # drop path
        if path_pdrop > 0.0:
            self.drop_path_out = AffineDropPath(n_embd, drop_prob=path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob=path_pdrop)
        else:
            self.drop_path_out = nn.Identity()
            self.drop_path_mlp = nn.Identity()

        self.act = act_layer()

    def forward(self, x, mask):
        # X shape: B, C, T
        B, C, T = x.shape
        x = self.downsample(x)

        x_in = self.ln(x)

        out2 = self.attn3(x_in, mask)
        out_mask = F.interpolate(
            mask.to(x.dtype),
            size=torch.div(T, self.stride, rounding_mode='trunc'),
            mode='nearest'
        ).detach()

        out = out2 + x_in
        out = x * out_mask + self.drop_path_out(out)
        # FFN
        out = out + self.drop_path_mlp(self.mlp(self.gn(out)))
        return out, out_mask.bool()


# drop path: from https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
class Scale(nn.Module):
    """
    Multiply the output regression range by a learnable constant value
    """

    def __init__(self, init_value=1.0):
        """
        init_value : initial value for the scalar
        """
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32),
            requires_grad=True
        )

    def forward(self, x):
        """
        input -> scale * input
        """
        return x * self.scale


# The follow code is modified from
# https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
def drop_path(x, drop_prob=0.0, training=False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
            x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AffineDropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks) with a per channel scaling factor (and zero init)
    See: https://arxiv.org/pdf/2103.17239.pdf
    """

    def __init__(self, num_dim, drop_prob=0.0, init_scale_value=1e-4):
        super().__init__()
        self.scale = nn.Parameter(
            init_scale_value * torch.ones((1, num_dim, 1)),
            requires_grad=True
        )
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(self.scale * x, self.drop_prob, self.training)
