"""Calibration set for the tail: run the FP32 ViTs on the GPU for every photo and store the tail inputs.

    python make_calib.py <photo dir> <out dir> [--ckpt sharp_2572gikvuh.pt] [--device cuda]

Writes <out dir>/tail_{image_u,vit_out,lat0,lat1,img_vit}.tar (0.npy, 1.npy, ...) for pulsar2 (calibration_format Numpy).
Photos get the same preparation as the web demo: EXIF rotation, long side <= 4096, width / height cropped to multiples of 16.
Calibrate on many different scenes: MinMax ranges from a few photos clip disparity / delta on others.
The release model used 80 photos (indoor, outdoor, close-ups, landscapes, night scenes, people, animals).
"""
import argparse
import glob
import io
import os
import sys
import tarfile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sharp_split import ImageViT, PatchViT, host_patches, load, spn, unshuffle  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--ckpt", default=None)
ap.add_argument("--device", default="cuda")
a = ap.parse_args()
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
os.makedirs(a.out, exist_ok=True)


def prep(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    s = min(1.0, 4096 / max(im.size))
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    w, h = im.width // 16 * 16, im.height // 16 * 16
    x0, y0 = (im.width - w) // 2, (im.height - h) // 2
    return np.asarray(im.crop((x0, y0, x0 + w, y0 + h)))


dev = a.device
m = load(a.ckpt, dev); enc = spn(m)
pv, iv = PatchViT(enc.patch_encoder), ImageViT(enc.image_encoder)
names = ["image_u", "vit_out", "lat0", "lat1", "img_vit"]
tars = {n: tarfile.open(f"{a.out}/tail_{n}.tar", "w") for n in names}


def put(n, i, arr):
    b = io.BytesIO(); np.save(b, np.ascontiguousarray(arr.astype(np.float32))); b.seek(0)
    ti = tarfile.TarInfo(f"{i}.npy"); ti.size = len(b.getvalue()); tars[n].addfile(ti, b)


exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
files = sorted(p for p in glob.glob(f"{a.src}/*") if p.lower().endswith(exts))
for i, p in enumerate(files):
    img = prep(p)
    x = torch.from_numpy(img.copy()).float().to(dev).permute(2, 0, 1)[None] / 255.0
    x1536 = F.interpolate(x, size=(1536, 1536), mode="bilinear", align_corners=True)
    with torch.no_grad():
        patches, x2 = host_patches(x1536)
        vo, l0, l1 = [], [], []
        for k in range(35):
            o, b, c = pv(patches[k:k + 1]); vo.append(o); l0.append(b); l1.append(c)
        vo = torch.cat(vo); l0 = torch.cat(l0)[:25]; l1 = torch.cat(l1)[:25]
        ivo = iv(x2)
    for n, t in zip(names, [unshuffle(x1536), vo, l0, l1, ivo]):
        put(n, i, t.float().cpu().numpy())
    print(i, p, img.shape, flush=True)
for t in tars.values():
    t.close()
print("done", len(files), "photos ->", a.out)
