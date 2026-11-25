// libsharp_ax.so: C 接口 (给 Python ctypes 等调用)。模型在 sharp_create 时加载一次, sharp_run 可反复调用。
// 同一个句柄不要在多个线程里同时调用 sharp_run。
#include "sharp_pipeline.hpp"
#include "sharp_ax.h"

struct sharp_ctx { SharpPipeline p; };
static thread_local std::string g_err;

extern "C" {

sharp_ctx* sharp_create(const char* model_dir, int threads) {
    auto* c = new sharp_ctx;
    if (!c->p.init(model_dir ? model_dir : ".", threads)) { g_err = c->p.error(); delete c; return nullptr; }
    return c;
}

int sharp_run(sharp_ctx* c, const unsigned char* image, size_t len, const char* out_path, double f35, double fpx, sharp_timing* t) {
    if (!c || !image || !out_path) { g_err = "invalid argument"; return -1; }
    SharpTiming tm;
    if (!c->p.run(image, len, out_path, f35 > 0 ? f35 : 30.0, fpx, &tm)) { g_err = c->p.error(); return -1; }
    if (t) {
        t->load_ms = tm.load; t->decode_ms = tm.decode; t->preprocess_ms = tm.preprocess; t->vit_ms = tm.vit; t->tail_ms = tm.tail;
        t->h4_ms = tm.h4; t->post_ms = tm.post; t->write_ms = tm.write; t->total_ms = tm.total;
        t->width = tm.width; t->height = tm.height; t->f_px = tm.f_px; t->gaussians = (unsigned long long)tm.gaussians; t->hw_jpeg = tm.hw_jpeg;
    }
    return 0;
}

void sharp_destroy(sharp_ctx* c) { delete c; }

const char* sharp_last_error(void) { return g_err.c_str(); }

}  // extern "C"
