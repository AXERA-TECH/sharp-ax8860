"""README 动图: 左边原图, 右边板上生成的 3D 高斯沿 ml-sharp 的相机轨迹渲染 (gsplat, 需要 CUDA)。

    python demo_gif.py photo.jpg result.spz out.gif|out.webp [--f35 30] [--width 360] [--steps 90] [--fps 24] [--traj rotate_forward]

result.spz / .ply 是 sharp_ax 的输出; photo.jpg 是送进 sharp_ax 的同一张图 (用来定内参和左侧画面)。
依赖: ml-sharp (sharp.utils.*), gsplat, torch, plyfile, Pillow。
"""
import argparse
import gzip
import math
import sys

import numpy as np
import torch
from PIL import Image

from sharp.utils import camera, gsplat
from sharp.utils.gaussians import Gaussians3D

SH_C0 = 0.28209479177387814


def read_spz(path):
    """SPZ v2 -> 原始 PLY 语义的数组 (位置, log 尺度, 四元数 wxyz, f_dc, opacity logit)。与 sharp_post.hpp 的 save_spz 对应。"""
    b = gzip.decompress(open(path, "rb").read())
    magic, ver, n = np.frombuffer(b[:12], "<u4")
    assert magic == 0x5053474E and ver == 2, "not an SPZ v2 file"
    fb = b[13]
    o = 16
    P = np.frombuffer(b, np.uint8, n * 9, o).reshape(n, 3, 3).astype(np.int32); o += n * 9
    A = np.frombuffer(b, np.uint8, n, o); o += n
    C = np.frombuffer(b, np.uint8, n * 3, o).reshape(n, 3); o += n * 3
    S = np.frombuffer(b, np.uint8, n * 3, o).reshape(n, 3); o += n * 3
    R = np.frombuffer(b, np.uint8, n * 3, o).reshape(n, 3)
    f = P[:, :, 0] | (P[:, :, 1] << 8) | (P[:, :, 2] << 16)
    f = np.where(f & 0x800000, f - (1 << 24), f)
    xyz = R / 127.5 - 1
    w = np.sqrt(np.clip(1 - (xyz ** 2).sum(1), 0, None))
    q = np.concatenate([w[:, None], xyz], 1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return dict(mean=f / float(1 << fb), log_scale=S / 16.0 - 10, quat=q, f_dc=(C / 255.0 - 0.5) / 0.15,
                opacity=np.log(np.clip(A / 255.0, 1e-6, 1 - 1e-6) / (1 - np.clip(A / 255.0, 1e-6, 1 - 1e-6))))


def read_ply(path):
    from plyfile import PlyData
    v = PlyData.read(path)["vertex"].data
    st = lambda ks: np.stack([v[k] for k in ks], 1).astype(np.float64)
    return dict(mean=st(["x", "y", "z"]), log_scale=st(["scale_0", "scale_1", "scale_2"]), quat=st(["rot_0", "rot_1", "rot_2", "rot_3"]),
                f_dc=st(["f_dc_0", "f_dc_1", "f_dc_2"]), opacity=v["opacity"].astype(np.float64))


def to_gaussians(d):
    """与 ml-sharp load_ply 相同的换算 (颜色 sRGB -> linear)。"""
    from sharp.utils import color_space as cs
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None]
    srgb = torch.from_numpy((d["f_dc"] * SH_C0 + 0.5).astype(np.float32))
    colors = cs.sRGB2linearRGB(srgb)[None]
    return Gaussians3D(mean_vectors=t(d["mean"]), singular_values=t(np.exp(d["log_scale"])), quaternions=t(d["quat"]),
                       colors=colors, opacities=t(1 / (1 + np.exp(-d["opacity"]))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("result"); ap.add_argument("out")
    ap.add_argument("--f35", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=360, help="width of each panel")
    ap.add_argument("--steps", type=int, default=90)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--traj", default="rotate_forward", choices=["swipe", "shake", "rotate", "rotate_forward"])
    ap.add_argument("--quality", type=int, default=80, help="WebP quality")
    ap.add_argument("--colors", type=int, default=255, help="GIF palette size")
    ap.add_argument("--no-dither", action="store_true", help="GIF without dithering (smaller)")
    a = ap.parse_args()

    img = Image.open(a.image).convert("RGB")
    W, H = img.size
    f_px = a.f35 * math.hypot(W, H) / math.hypot(36, 24)
    d = read_spz(a.result) if a.result.endswith(".spz") else read_ply(a.result)
    g = to_gaussians(d)
    dev = torch.device("cuda")

    # 按 panel 宽度缩放内参, 直接渲染小图
    s = a.width / W
    w, h = a.width, int(round(H * s / 2)) * 2
    K = torch.tensor([[f_px * s, 0, (w - 1) / 2.0, 0], [0, f_px * s, (h - 1) / 2.0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32, device=dev)
    params = camera.TrajectoryParams(type=a.traj, num_steps=a.steps)
    cam = camera.create_camera_model(g, K, resolution_px=(w, h))
    traj = camera.create_eye_trajectory(g, params, resolution_px=(W, H), f_px=f_px)
    r = gsplat.GSplatRenderer(color_space="linearRGB")
    gd = g.to(dev)
    left = img.resize((w, h), Image.LANCZOS)
    frames = []
    for eye in traj:
        ci = cam.compute(eye)
        out = r(gd, extrinsics=ci.extrinsics[None].to(dev), intrinsics=ci.intrinsics[None].to(dev), image_width=ci.width, image_height=ci.height)
        right = Image.fromarray((out.color[0].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy())
        f = Image.new("RGB", (w * 2 + 8, h), (17, 17, 17))
        f.paste(left, (0, 0)); f.paste(right, (w + 8, 0))
        frames.append(f)
    if a.out.endswith(".webp"):   # 有损动图 WebP, 比 GIF 小很多
        frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=int(1000 / a.fps), loop=0, quality=a.quality, method=6)
        print(a.out, len(frames), "frames", w * 2 + 8, "x", h)
        return
    # 全局调色板 (取几帧拼起来量化), 避免逐帧闪烁
    pal_src = Image.new("RGB", (frames[0].width, frames[0].height * 4))
    for i, k in enumerate(np.linspace(0, len(frames) - 1, 4).astype(int)):
        pal_src.paste(frames[k], (0, frames[0].height * i))
    pal = pal_src.quantize(colors=a.colors, method=Image.MEDIANCUT)
    q = [fr.quantize(palette=pal, dither=Image.NONE if a.no_dither else Image.FLOYDSTEINBERG) for fr in frames]
    q[0].save(a.out, save_all=True, append_images=q[1:], duration=int(1000 / a.fps), loop=0, optimize=True, disposal=1)
    print(a.out, len(frames), "frames", w * 2 + 8, "x", h)


if __name__ == "__main__":
    sys.exit(main())
