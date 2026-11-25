// sharp_ax: SHARP (apple/ml-sharp) single image -> 3D Gaussian Splatting on AX8860 (command line)
//   sharp_ax -m <model_dir> -i <image> -o <out.spz|out.ply> [--f35 30] [--fpx F] [-t threads] [--dump <dir>]
//   sharp_ax --post-only <dir with image/disparity/delta.bin> --wh W H --fpx F -o out.ply   (host-side post-process check)
#include "sharp_pipeline.hpp"

static void usage() {
    fprintf(stderr,
            "usage: sharp_ax -m <model_dir> -i <image.jpg> -o <out.spz|out.ply> [options]\n"
            "  -m DIR        model directory (fp8vit/, sharp_tail.axmodel, h4_wb.bin)\n"
            "  -i FILE       input image (JPEG uses the hardware decoder; PNG etc. fall back to stb_image)\n"
            "  -o FILE       output; .spz = SPZ v2 (gzip, ~1/4 size), anything else = PLY (same layout as ml-sharp)\n"
            "  --f35 MM      35mm-equivalent focal length (default 30, same as ml-sharp when EXIF is missing)\n"
            "  --fpx PX      focal length in pixels (overrides --f35)\n"
            "  -t N          CPU threads (default: all cores)\n"
            "  --dump DIR    dump intermediate tensors (image/vit/tail outputs) for debugging\n"
            "  --post-only DIR --wh W H --fpx F   run only the CPU post-process on dumped tensors\n"
            "env: SHARP_PROF=1 per-stage timing, SHARP_SPZ_LEVEL=0..9 (0 = Huffman only, default)\n");
}

int main(int argc, char** argv) {
    std::string mdir, img_path, out_path, dump, post_dir;
    double f35 = 30.0, fpx_arg = 0; int W = 0, H = 0, threads = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&] { if (i + 1 >= argc) { usage(); exit(1); } return std::string(argv[++i]); };
        if (a == "-m") mdir = nx(); else if (a == "-i") img_path = nx(); else if (a == "-o") out_path = nx();
        else if (a == "--f35") f35 = atof(nx().c_str()); else if (a == "--dump") dump = nx();
        else if (a == "--post-only") post_dir = nx(); else if (a == "--fpx") fpx_arg = atof(nx().c_str());
        else if (a == "--wh") { W = atoi(nx().c_str()); H = atoi(nx().c_str()); }
        else if (a == "-t") threads = atoi(nx().c_str());
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); usage(); return 1; }
    }
    if (out_path.empty() || (post_dir.empty() && (mdir.empty() || img_path.empty()))) { usage(); return 1; }

    if (!post_dir.empty()) {   // 只跑后处理 (x86 对拍)
        if (W <= 0 || H <= 0 || fpx_arg <= 0) { fprintf(stderr, "--post-only needs --wh W H and --fpx F\n"); return 1; }
        std::vector<char> im, di, de;
        if (!read_file(post_dir + "/image.bin", im) || !read_file(post_dir + "/disparity.bin", di) || !read_file(post_dir + "/delta.bin", de)) {
            fprintf(stderr, "failed to read %s/{image,disparity,delta}.bin\n", post_dir.c_str()); return 1; }
        double t0 = now_ms();
        Gaussians g = post((float*)im.data(), (float*)di.data(), (float*)de.data(), fpx_arg, W, H, threads);
        double t1 = now_ms();
        if (!save_gaussians(g, fpx_arg, W, H, out_path)) { fprintf(stderr, "failed to write %s\n", out_path.c_str()); return 1; }
        printf("post %.0f ms, write %.0f ms -> %s\n", t1 - t0, now_ms() - t1, out_path.c_str());
        return 0;
    }
#ifdef SHARP_POST_ONLY
    fprintf(stderr, "this build only supports --post-only\n"); return 1;
#else
    std::vector<char> img;
    if (!read_file(img_path, img)) { fprintf(stderr, "failed to read %s\n", img_path.c_str()); return 1; }
    SharpPipeline sp;
    if (!sp.init(mdir, threads)) { fprintf(stderr, "init failed: %s\n", sp.error().c_str()); return 1; }
    printf("load %.0f ms\n", sp.load_ms());
    SharpTiming t;
    if (!sp.run((const unsigned char*)img.data(), img.size(), out_path, f35, fpx_arg, &t, dump)) { fprintf(stderr, "run failed: %s\n", sp.error().c_str()); return 1; }
    printf("decode %.0f ms (%s)\n", t.decode, t.hw_jpeg ? "JDEC" : "stb_image");
    printf("image %dx%d f_px=%.2f\n", t.width, t.height, t.f_px);
    printf("preprocess %.0f ms\n", t.preprocess);
    printf("vit x36 %.0f ms\n", t.vit);
    printf("tail %.0f ms\n", t.tail);
    printf("  h4 (CPU FP32) %.0f ms\n", t.h4);
    printf("postprocess %.0f ms\n", t.post);
    printf("write %.0f ms -> %s (%zu gaussians)\n", t.write, out_path.c_str(), t.gaussians);
    printf("end-to-end %.0f ms (decode + inference + post-process + write, excluding model loading)\n", t.total);
    return 0;
#endif
}
