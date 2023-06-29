import mindspore
import mindspore.nn as nn
#对应于torch的functional
import mindspore.ops as ops
from mindspore import Parameter, Tensor
from functools import partial
from helper import PatchEmbed, trunc_normal_, DropPath


class Mlp(nn.cell):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU(approximate=False), drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Dense(hidden_features, out_features)
        self.drop = nn.Dropout(p=drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.cell):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def forward(self, x, register_hook=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.swapaxes(-2, -1)) * self.scale
        # 参考官方文档
        attn = attn.softmax(axis=-1)
        attn = self.attn_drop(attn)

        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)

        x = (attn @ v).swapaxes(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.cell):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU(approximate=False), norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, register_hook=False):
        x = x + self.drop_path(self.attn(self.norm1(x), register_hook=register_hook))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.cell):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        # 此部分差异见官方文档
        self.cls_token = Parameter(ops.zeros((1, 1, embed_dim)))
        self.pos_embed = Parameter(ops.zeros((1, num_patches + 1, embed_dim)))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in ops.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.CellList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Dense):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Dense) and m.bias is not None:
                constant_init = mindspore.common.initializer.Constant(value=0)
                m.bias = constant_init(m.bias)
                # 具体差异见官方文档
                #nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            constant_init = mindspore.common.initializer.Constant(value=0)
            m.bias = constant_init(m.bias)
            #nn.init.constant_(m.bias, 0)
            constant_init = mindspore.common.initializer.Constant(value=1.0)
            m.weight = constant_init(m.weight)
            #nn.init.constant_(m.weight, 1.0)


    '''
    这部分暂时没用到，不更新
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}
    '''

    def forward(self, x, register_blk=-1):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        #为了能将cls_token与x拼接起来，需要将这两个tensor的第一个维度对齐
        #因此需要先生成一个第一维是B，剩下的维度和cls_token保持一致的张量，然后再用expand_as扩展到相同的张量
        sample = ops.zeros(B, self.cls_token.shape[0], self.cls_token.shape[1])
        cls_tokens = Tensor.expand_as(sample)
        #下面的代码只是在第一维新增了一个维度，并不能够对齐
        #cls_tokens = ops.expand_dims(self.cls_token,0)
        x = ops.cat((cls_tokens, x), axis=1)

        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x, register_blk == i)
        x = self.norm(x)

        return x

