"""Web demo running on the AX8860 board: upload a photo in the browser -> SHARP on the NPU -> view the 3D Gaussians.

    python3 web_demo.py -m ../models --port 8080        # then open http://<board-ip>:8080

Only the Python standard library is needed (plus libsharp_ax.so). The browser rotates the photo by its EXIF orientation,
limits the long side to 4096 and re-encodes it as JPEG, so the board can use its hardware JPEG decoder.
Results (SPZ) are kept in --out-dir; models are loaded once at start-up.
"""
import argparse
import json
import os
import re
import threading
import time
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from sharp_ax import Sharp, SharpError

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description="SHARP AX8860 web demo")
    ap.add_argument("-m", "--model-dir", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--keep", type=int, default=20, help="number of results to keep on disk")
    ap.add_argument("--max-mb", type=float, default=40, help="upload size limit")
    ap.add_argument("--npu-lock", default="", help="optional lock directory shared with other NPU jobs on the board "
                    "(taken with mkdir for each inference; busy -> HTTP 429)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    print("loading models ...", flush=True)
    sharp = Sharp(a.model_dir)
    lock = threading.Lock()     # one inference at a time

    def npu_lock_take():
        if not a.npu_lock:
            return True
        try:
            os.mkdir(a.npu_lock)
        except FileExistsError:
            return False
        with open(os.path.join(a.npu_lock, "owner"), "w") as f:
            with open("/proc/uptime") as u:
                f.write(f"sharp_web {os.getpid()} {u.read().split('.')[0]}\n")
        return True

    def npu_lock_release():
        if a.npu_lock:
            try:
                os.remove(os.path.join(a.npu_lock, "owner"))
                os.rmdir(a.npu_lock)
            except OSError:
                pass

    def history():
        items = []
        for fn in os.listdir(a.out_dir):
            if fn.endswith(".json"):
                try:
                    with open(os.path.join(a.out_dir, fn)) as f:
                        items.append(json.load(f))
                except (OSError, ValueError):
                    pass
        return sorted(items, key=lambda x: -x.get("ts", 0))

    def prune():
        for old in history()[a.keep:]:
            for ext in (".spz", ".json"):
                try:
                    os.remove(os.path.join(a.out_dir, old["id"] + ext))
                except OSError:
                    pass

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kw):
            super().__init__(*args, directory=HERE, **kw)

        def log_message(self, *args):
            pass

        def send_json(self, obj, code=200):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/?"):
                self.path = "/web_demo.html"
            if self.path == "/api/history":
                return self.send_json(history())
            if self.path.startswith("/results/"):
                name = os.path.basename(self.path.split("?")[0])
                p = os.path.join(a.out_dir, name)
                if not re.fullmatch(r"[\w.-]+\.spz", name) or not os.path.isfile(p):
                    return self.send_error(404)
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(os.path.getsize(p)))
                self.end_headers()
                with open(p, "rb") as f:
                    while True:
                        chunk = f.read(1 << 20)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                return
            if self.path not in ("/web_demo.html",):
                return self.send_error(404)
            return super().do_GET()

        def do_POST(self):
            if self.path != "/api/run":
                return self.send_error(404)
            n = int(self.headers.get("Content-Length", 0))
            if n <= 0 or n > a.max_mb * (1 << 20):
                return self.send_json({"error": f"image must be 1 B .. {a.max_mb:g} MB"}, 400)
            data = self.rfile.read(n)
            name = os.path.basename(urllib.parse.unquote(self.headers.get("X-Name", "upload")))[:80]
            try:
                f35 = float(self.headers.get("X-F35", "30") or 30)
            except ValueError:
                f35 = 30.0
            if not lock.acquire(timeout=1):
                return self.send_json({"error": "busy, try again in a few seconds"}, 429)
            if not npu_lock_take():
                lock.release()
                return self.send_json({"error": "the NPU is used by another job, try again later"}, 429)
            try:
                rid = time.strftime("%Y%m%d_%H%M%S_") + os.urandom(2).hex()
                out = os.path.join(a.out_dir, rid + ".spz")
                t0 = time.time()
                t = sharp.run(data, out, f35=f35)
                rec = {"id": rid, "name": name, "ts": time.time(), "url": "results/" + rid + ".spz", "size": os.path.getsize(out),
                       "wall_ms": round((time.time() - t0) * 1000), **{k: (round(v, 1) if isinstance(v, float) else v) for k, v in t.items()}}
                with open(os.path.join(a.out_dir, rid + ".json"), "w") as f:
                    json.dump(rec, f, ensure_ascii=False)
            except SharpError as e:
                return self.send_json({"error": str(e)}, 500)
            finally:
                npu_lock_release()
                lock.release()
            prune()
            return self.send_json(rec)

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"ready: http://{a.host}:{a.port}", flush=True)
    try:
        srv.serve_forever()
    finally:
        sharp.close()


if __name__ == "__main__":
    main()
