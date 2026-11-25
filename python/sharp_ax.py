"""SHARP (apple/ml-sharp) single image -> 3D Gaussian Splatting on AX8860: Python binding of libsharp_ax.so (ctypes).

    from sharp_ax import Sharp
    with Sharp("models") as s:              # loads the models once
        t = s.run("photo.jpg", "out.spz")   # .spz = SPZ v2 (~1/4 size), anything else = PLY
        print(t["total_ms"], t["gaussians"])
"""
import ctypes
import os

__all__ = ["Sharp", "SharpError"]


class SharpError(RuntimeError):
    pass


class _Timing(ctypes.Structure):
    _fields_ = [("load_ms", ctypes.c_double), ("decode_ms", ctypes.c_double), ("preprocess_ms", ctypes.c_double),
                ("vit_ms", ctypes.c_double), ("tail_ms", ctypes.c_double), ("h4_ms", ctypes.c_double),
                ("post_ms", ctypes.c_double), ("write_ms", ctypes.c_double), ("total_ms", ctypes.c_double),
                ("width", ctypes.c_int), ("height", ctypes.c_int), ("f_px", ctypes.c_double),
                ("gaussians", ctypes.c_ulonglong), ("hw_jpeg", ctypes.c_int)]


def _load_lib(path=None):
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [path] if path else [os.environ.get("SHARP_AX_LIB"), os.path.join(here, "libsharp_ax.so"), "libsharp_ax.so"]
    err = None
    for p in cands:
        if not p:
            continue
        try:
            lib = ctypes.CDLL(p)
            break
        except OSError as e:
            err = e
    else:
        raise SharpError(f"cannot load libsharp_ax.so ({err}); set SHARP_AX_LIB or put it next to sharp_ax.py")
    lib.sharp_create.restype = ctypes.c_void_p
    lib.sharp_create.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.sharp_run.restype = ctypes.c_int
    lib.sharp_run.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_double, ctypes.c_double,
                              ctypes.POINTER(_Timing)]
    lib.sharp_destroy.restype = None
    lib.sharp_destroy.argtypes = [ctypes.c_void_p]
    lib.sharp_last_error.restype = ctypes.c_char_p
    return lib


class Sharp:
    """One instance = models loaded once on the NPU. run() is not thread-safe; serialise calls."""

    def __init__(self, model_dir, threads=0, lib_path=None):
        self._ctx = None
        self._lib = _load_lib(lib_path)
        self._ctx = self._lib.sharp_create(os.fsencode(model_dir), int(threads))
        if not self._ctx:
            raise SharpError(self._lib.sharp_last_error().decode(errors="replace"))

    def run(self, image, out_path, f35=30.0, fpx=0.0):
        """image: path or encoded bytes (JPEG uses the hardware decoder). Returns a dict of timings (ms) and info."""
        if not self._ctx:
            raise SharpError("closed")
        if isinstance(image, (bytes, bytearray)):
            data = bytes(image)
        else:
            with open(image, "rb") as f:
                data = f.read()
        t = _Timing()
        r = self._lib.sharp_run(self._ctx, data, len(data), os.fsencode(out_path), float(f35), float(fpx), ctypes.byref(t))
        if r != 0:
            raise SharpError(self._lib.sharp_last_error().decode(errors="replace"))
        return {k: getattr(t, k) for k, _ in _Timing._fields_}

    def close(self):
        if self._ctx:
            self._lib.sharp_destroy(self._ctx)
            self._ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description="SHARP single image -> 3DGS on AX8860")
    ap.add_argument("-m", "--model-dir", required=True)
    ap.add_argument("-i", "--image", required=True)
    ap.add_argument("-o", "--output", required=True, help=".spz or .ply")
    ap.add_argument("--f35", type=float, default=30.0, help="35mm-equivalent focal length")
    ap.add_argument("--fpx", type=float, default=0.0, help="focal length in pixels (overrides --f35)")
    a = ap.parse_args()
    with Sharp(a.model_dir) as s:
        print(json.dumps(s.run(a.image, a.output, a.f35, a.fpx), indent=1))
