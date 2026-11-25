"""SHARP split for the AX8860: ViT (35 patches + 1 image, run separately) / tail (DPT + Gaussian decoder) / CPU post-process.

The tail is rewritten into an equivalent graph whose tensors are all <= 768x768 ("unshuffle" domain, 2x2 space-to-depth),
because 1536^2 tensors with few channels overflow a DMA field in pulsar2. Used by export_tail.py and make_calib.py.
Needs ml-sharp (`pip install -e ml-sharp`).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from sharp.models import PredictorParams, create_predictor

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"


def load(ckpt=None, device="cpu"):
    """ckpt: path to sharp_2572gikvuh.pt, or None to download it (same as `sharp predict`)."""
    m = create_predictor(PredictorParams())
    sd = torch.load(ckpt, weights_only=True) if ckpt else torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    m.load_state_dict(sd)
    return m.eval().to(device)


def spn(m):
    return m.monodepth_model.monodepth_predictor.encoder


class PatchViT(nn.Module):
    """[1,3,384,384] in [-1,1] -> final [1,1024,24,24], intermediate features (lat0/lat1) [1,1024,24,24]"""
    def __init__(self, vit):
        super().__init__(); self.vit = vit

    def forward(self, x):
        out, inter = self.vit(x)
        ids = self.vit.intermediate_features_ids
        return out, self.vit.reshape_feature(inter[ids[0]]), self.vit.reshape_feature(inter[ids[1]])


class ImageViT(nn.Module):
    def __init__(self, vit):
        super().__init__(); self.vit = vit

    def forward(self, x):
        return self.vit(x)[0]


class _Feed(nn.Module):
    """Replaces a ViT inside the model: returns the features given as tail inputs."""
    def __init__(self, ids, grid):
        super().__init__()
        self.ids = ids; self.val = None; self.num_prefix_tokens = 0

        class PE: pass
        self.patch_embed = PE(); self.patch_embed.grid_size = (grid, grid)

    def reshape_feature(self, t):   # already NCHW
        return t

    def forward(self, x):
        return self.val


class Tail(nn.Module):
    """The model with both ViTs replaced by _Feed."""
    def __init__(self, m):
        super().__init__()
        self.m = m
        enc = spn(m)
        self.pfeed = _Feed(enc.patch_encoder.intermediate_features_ids, 24)
        self.ifeed = _Feed(None, 24)
        enc.patch_encoder = self.pfeed
        enc.image_encoder = self.ifeed


def host_patches(image1536):
    """Same as SPN's normalizer + pyramid + split: [1,3,1536,1536] in [0,1] -> 35 patches [35,3,384,384], 1/4 image."""
    x = image1536 * 2 - 1
    x1 = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
    x2 = F.interpolate(x, scale_factor=0.25, mode="bilinear", align_corners=False)

    def split(img, ov, ps=384):
        st = int(ps * (1 - ov)); n = int(math.ceil((img.shape[-1] - ps) / st)) + 1
        return [img[..., j * st:j * st + ps, i * st:i * st + ps] for j in range(n) for i in range(n)]
    return torch.cat(split(x, 0.25) + split(x1, 0.5) + [x2], 0), x2


class GNDecomp(nn.Module):
    """GroupNorm as per-channel spatial means + small group averages (pulsar2's GroupNorm cannot tile 768^2 inputs)."""
    def __init__(self, gn: nn.GroupNorm):
        super().__init__()
        self.G, self.eps = gn.num_groups, gn.eps
        self.weight, self.bias = gn.weight, gn.bias

    def forward(self, x):
        N, C, H, W = x.shape; G = self.G
        cm = x.mean(dim=(2, 3), keepdim=True)
        gm = cm.reshape(N, G, C // G).mean(-1, keepdim=True).expand(N, G, C // G).reshape(N, C, 1, 1)
        d = x - gm
        cv = (d * d).mean(dim=(2, 3), keepdim=True)
        gv = cv.reshape(N, G, C // G).mean(-1, keepdim=True).expand(N, G, C // G).reshape(N, C, 1, 1)
        y = d / torch.sqrt(gv + self.eps)
        return y * self.weight.reshape(1, C, 1, 1) + self.bias.reshape(1, C, 1, 1)


def replace_groupnorm(mod):
    n = 0
    for name, ch in list(mod.named_children()):
        if isinstance(ch, nn.GroupNorm):
            setattr(mod, name, GNDecomp(ch)); n += 1
        else:
            n += replace_groupnorm(ch)
    return n


# ---- 768^2 rewrite. Sub-pixel s = dy*2+dx; an "unshuffle" tensor has channel c*4+s and shape [N, C*4, H/2, W/2] ----

def unshuffle(x):
    N, C, H, W = x.shape
    return x.reshape(N, C, H // 2, 2, W // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(N, C * 4, H // 2, W // 2)


def shuffle(x):
    N, C4, h, w = x.shape; C = C4 // 4
    return x.reshape(N, C, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(N, C, h * 2, w * 2)


def convT2_as_1x1(ct: nn.ConvTranspose2d) -> nn.Conv2d:
    """ConvTranspose k2 s2 (Cin->Cout) == 1x1 conv Cin -> Cout*4 with unshuffled output."""
    W = ct.weight.data  # [Cin, Cout, 2, 2]
    Cin, Cout = W.shape[:2]
    c = nn.Conv2d(Cin, Cout * 4, 1, bias=True)
    c.weight.data = W.permute(1, 2, 3, 0).reshape(Cout * 4, Cin, 1, 1).clone()
    b = ct.bias.data if ct.bias is not None else torch.zeros(Cout)
    c.bias.data = b.repeat_interleave(4).clone()
    return c


def conv3x3_poly(cv: nn.Conv2d) -> nn.Conv2d:
    """Full-resolution 3x3 conv (pad 1) == 3x3 conv in the unshuffle domain (Cin*4 -> Cout*4)."""
    W = cv.weight.data  # [Cout, Cin, 3, 3]
    Cout, Cin = W.shape[:2]
    Wp = torch.zeros(Cout, 4, Cin, 4, 3, 3)
    for oy in range(2):
        for ox in range(2):
            for ky in range(3):
                for kx in range(3):
                    fy, fx = oy + ky - 1, ox + kx - 1          # offset at full resolution
                    cy, sy = fy // 2, fy % 2                   # coarse offset + sub-pixel
                    cx, sx = fx // 2, fx % 2
                    Wp[:, oy * 2 + ox, :, sy * 2 + sx, cy + 1, cx + 1] += W[:, :, ky, kx]
    c = nn.Conv2d(Cin * 4, Cout * 4, 3, padding=1, bias=True)
    c.weight.data = Wp.reshape(Cout * 4, Cin * 4, 3, 3)
    c.bias.data = cv.bias.data.repeat_interleave(4).clone()
    return c


def conv1x1_unsh(cv: nn.Conv2d) -> nn.Conv2d:
    """Per-pixel 1x1 conv in the unshuffle domain: block diagonal."""
    W = cv.weight.data[:, :, 0, 0]
    Cout, Cin = W.shape
    Wb = torch.zeros(Cout, 4, Cin, 4)
    for s in range(4):
        Wb[:, s, :, s] = W
    c = nn.Conv2d(Cin * 4, Cout * 4, 1, bias=True)
    c.weight.data = Wb.reshape(Cout * 4, Cin * 4, 1, 1)
    c.bias.data = cv.bias.data.repeat_interleave(4).clone()
    return c


def conv_k2s2_as_1x1(cv: nn.Conv2d) -> nn.Conv2d:
    """k2 s2 conv == 1x1 conv on the unshuffled input."""
    W = cv.weight.data  # [Cout, Cin, 2, 2]
    Cout, Cin = W.shape[:2]
    c = nn.Conv2d(Cin * 4, Cout, 1, bias=True)
    c.weight.data = W.reshape(Cout, Cin * 4, 1, 1).clone()
    c.bias.data = cv.bias.data.clone()
    return c


class SplitCinConv(nn.Module):
    """3x3 conv split into 128-input-channel groups and summed (Cin >= 256 at 768^2 overflows a DMA field in pulsar2)."""
    def __init__(self, cv: nn.Conv2d, cg: int = 128):
        super().__init__()
        self.cg = cg
        self.convs = nn.ModuleList()
        for g in range(cv.in_channels // cg):
            c = nn.Conv2d(cg, cv.out_channels, cv.kernel_size, stride=cv.stride, padding=cv.padding, dilation=cv.dilation,
                          bias=(g == 0 and cv.bias is not None))
            c.weight.data = cv.weight.data[:, g * cg:(g + 1) * cg].clone()
            if g == 0 and cv.bias is not None:
                c.bias.data = cv.bias.data.clone()
            self.convs.append(c)

    def forward(self, x):
        y = None
        for g, c in enumerate(self.convs):
            t = c(x[:, g * self.cg:(g + 1) * self.cg]); y = t if y is None else y + t
        return y


class TailNPU768(nn.Module):
    """NPU part of the model, all tensors <= 768^2.

    inputs : image_u [1,12,768,768] (unshuffled 1536^2 image in [0,1]), vit_out [35,1024,24,24],
             lat0 / lat1 [25,1024,24,24], img_vit [1,1024,24,24]
    outputs: disparity_u [1,8,768,768] (channel l*4+s), delta [1,28,768,768]
    disparity_factor is only a global scale that the min-normalisation cancels, so it is not an input;
    the normalised disparity is written as disp / max(disp) to avoid a Reciprocal.
    """
    def __init__(self, m):
        super().__init__()
        self.t = Tail(m)
        head = m.monodepth_model.monodepth_predictor.head
        # head = [conv3x3 256->128, convT 128->128 k2s2, conv3x3 128->32, ReLU, conv1x1 32->2, ReLU]
        self.h0 = SplitCinConv(head[0])
        f0 = m.monodepth_model.monodepth_predictor.decoder.fusions[0]
        for rb in (f0.resnet1, f0.resnet2):
            for i in (1, 3):
                rb.residual[i] = SplitCinConv(rb.residual[i])
        # 512-channel 768^2 tensors overflow too: split into 4 groups of 128 channels, h2 = sum of 4 grouped convs
        h1 = convT2_as_1x1(head[1]); h2 = conv3x3_poly(head[2])
        G = 4; cg = h1.out_channels // G
        self.h1g = nn.ModuleList(); self.h2g = nn.ModuleList()
        for g in range(G):
            a = nn.Conv2d(h1.in_channels, cg, 1, bias=True)
            a.weight.data = h1.weight.data[g * cg:(g + 1) * cg].clone(); a.bias.data = h1.bias.data[g * cg:(g + 1) * cg].clone()
            b = nn.Conv2d(cg, h2.out_channels, 3, padding=1, bias=(g == 0))
            b.weight.data = h2.weight.data[:, g * cg:(g + 1) * cg].clone()
            if g == 0:
                b.bias.data = h2.bias.data.clone()
            self.h1g.append(a); self.h2g.append(b)
        self.h4 = conv1x1_unsh(head[4])
        self.skip = conv_k2s2_as_1x1(m.feature_model.image_encoder.conv)
        # the SPN pyramid / patches only feed the (replaced) ViTs: make them 1x1 placeholders so no big unused tensors remain
        import sharp.models.encoders.spn_encoder as spn_mod
        spn_mod.split = lambda image, overlap_ratio=0.25, patch_size=384: \
            image[..., :1, :1].repeat(25 if overlap_ratio == 0.25 else 9, 1, 1, 1)
        spn(m)._create_pyramid = lambda x: (x[..., :1, :1], x[..., :1, :1], x[..., :1, :1])

    def forward(self, image_u, vit_out, lat0, lat1, img_vit):
        t = self.t; m = t.m; ids = t.pfeed.ids
        t.pfeed.val = (vit_out, {ids[0]: lat0, ids[1]: lat1})
        t.ifeed.val = (img_vit, {})
        mp = m.monodepth_model.monodepth_predictor
        enc = mp.encoder(image_u)          # the image is only a placeholder here (features come from the feeds)
        dec = mp.decoder(enc[:len(mp.encoder.dims_encoder)])
        h = self.h0(dec)
        acc = None
        for a, b in zip(self.h1g, self.h2g):
            y = b(a(h)); acc = y if acc is None else acc + y
        h = torch.relu(acc)
        disp_u = torch.relu(self.h4(h))   # [1,8,768,768]; add_h4_outputs.py moves this layer to the CPU
        d = disp_u.clamp(min=1e-4, max=1e4)
        nd = torch.clamp(d / d.amax(dim=(1, 2, 3), keepdim=True), min=0.01)
        # channel order in the unshuffle domain: image c*4+s (c<3), then disparity l*4+s -> matches the k2s2 conv input
        feat_u = 2.0 * torch.cat([image_u, nd], dim=1) - 1.0
        fm = m.feature_model
        feats = fm.upsample(fm.decoder(enc).contiguous())
        feats = fm.fusion(feats, self.skip(feat_u))
        from sharp.models.gaussian_decoder import ImageFeatures
        f = ImageFeatures(texture_features=fm.texture_head(feats), geometry_features=fm.geometry_head(feats))
        delta = m.prediction_head(f)
        return disp_u, delta.flatten(1, 2)
