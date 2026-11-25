/* libsharp_ax: SHARP single image -> 3D Gaussian Splatting on AX8860, C API */
#ifndef SHARP_AX_H
#define SHARP_AX_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct sharp_ctx sharp_ctx;

typedef struct {
    double load_ms;        /* model loading (once, in sharp_create) */
    double decode_ms, preprocess_ms, vit_ms, tail_ms, h4_ms, post_ms, write_ms;
    double total_ms;       /* decode + inference + post-process + write */
    int width, height;     /* input image size */
    double f_px;           /* focal length used, in pixels */
    unsigned long long gaussians;
    int hw_jpeg;           /* 1 = decoded by the hardware JPEG decoder */
} sharp_timing;

/* Load models from model_dir (fp8vit/, sharp_tail.axmodel, h4_wb.bin). threads <= 0: all cores. NULL on failure. */
sharp_ctx* sharp_create(const char* model_dir, int threads);

/* Run one encoded image (JPEG/PNG/...). out_path ending in .spz writes SPZ v2, otherwise PLY.
   f35: 35mm-equivalent focal length (<= 0: 30); fpx > 0 overrides it with a focal length in pixels.
   Returns 0 on success. Not thread-safe per context. */
int sharp_run(sharp_ctx* ctx, const unsigned char* image, size_t len, const char* out_path, double f35, double fpx, sharp_timing* timing);

void sharp_destroy(sharp_ctx* ctx);

/* Error message of the last failed call in this thread. */
const char* sharp_last_error(void);

#ifdef __cplusplus
}
#endif
#endif
