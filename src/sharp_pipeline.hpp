// SHARP (apple/ml-sharp) single image -> 3D Gaussian Splatting on AX8860 NPU: 完整流程 (头文件实现, 每个可执行文件/so 只 include 一次)
// 流程: JPEG 硬解 -> 1536² -> 金字塔切 35 块 -> FP8 ViT (patch x35 + image x1) -> tail (DPT + 高斯解码, U16)
//       -> CPU FP32 深度头最后一层 (h4) -> CPU FP32 后处理 -> PLY / SPZ
// SharpPipeline::init() 只加载一次模型, run() 可反复调用 (同一对象不要并发调用)。
#pragma once
#include <sys/stat.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "sharp_post.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace sharp;
static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static bool read_file(const std::string& p, std::vector<char>& buf) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    buf.resize(n);
    bool ok = fread(buf.data(), 1, n, f) == (size_t)n;
    fclose(f);
    return ok;
}
static void write_bin(const std::string& p, const void* d, size_t n) {
    FILE* f = fopen(p.c_str(), "wb"); if (f) { fwrite(d, 1, n, f); fclose(f); }
}

// 双线性插值, 对齐 torch F.interpolate。按行并行; 横向下标/权重预先算好。
// 源是 nc 个通道: planar float (src_u8 为空) 或交错 u8 HWC (结果再乘 1/255)
struct Lin { std::vector<int> i0, i1; std::vector<float> w; };
static Lin lin_coords(int sn, int dn, bool align_corners, double scale) {
    Lin L; L.i0.resize(dn); L.i1.resize(dn); L.w.resize(dn);
    for (int d = 0; d < dn; ++d) {
        double f = align_corners ? (dn > 1 ? d * (double)(sn - 1) / (dn - 1) : 0) : std::max<double>(0.0, (d + 0.5) / scale - 0.5);
        int i0 = std::min((int)f, sn - 1); L.i0[d] = i0; L.i1[d] = std::min(i0 + 1, sn - 1); L.w[d] = (float)(f - i0);
    }
    return L;
}
static void resize_bilinear(const float* src, int sw, int sh, float* dst, int dw, int dh, bool align_corners, double scale) {
    Lin X = lin_coords(sw, dw, align_corners, scale), Y = lin_coords(sh, dh, align_corners, scale);
    parallel_for(dh, [&](size_t y) {
        const float* r0 = src + (size_t)Y.i0[y] * sw; const float* r1 = src + (size_t)Y.i1[y] * sw; float ly = Y.w[y];
        float* o = dst + y * dw;
        for (int x = 0; x < dw; ++x) {
            float lx = X.w[x]; int a = X.i0[x], b = X.i1[x];
            float t = r0[a] + lx * (r0[b] - r0[a]), u = r1[a] + lx * (r1[b] - r1[a]);
            o[x] = t + ly * (u - t);
        }
    });
}
static void resize_u8hwc(const unsigned char* src, int sw, int sh, float* dst, int dw, int dh) {   // align_corners=True, 输出 3 个平面 0..1
    Lin X = lin_coords(sw, dw, true, 0), Y = lin_coords(sh, dh, true, 0);
    const size_t P = (size_t)dw * dh;
    parallel_for(dh, [&](size_t y) {
        const unsigned char* r0 = src + (size_t)Y.i0[y] * sw * 3; const unsigned char* r1 = src + (size_t)Y.i1[y] * sw * 3; float ly = Y.w[y];
        for (int x = 0; x < dw; ++x) {
            float lx = X.w[x]; int a = X.i0[x] * 3, b = X.i1[x] * 3;
            for (int c = 0; c < 3; ++c) {
                float t = r0[a + c] + lx * (r0[b + c] - r0[a + c]), u = r1[a + c] + lx * (r1[b + c] - r1[a + c]);
                dst[c * P + y * dw + x] = (t + ly * (u - t)) * (1.0f / 255.0f);
            }
        }
    });
}

#ifndef SHARP_POST_ONLY
#include <ax_engine_api.h>
#include <ax_sys_api.h>

struct Model {
    std::string name;
    AX_ENGINE_HANDLE h = nullptr;
    AX_ENGINE_IO_INFO_T* info = nullptr;
    AX_ENGINE_IO_T io{};
    std::vector<bool> own_in, own_out;  // 是否自己分配(否则借用别的模型的缓冲区)
};

static int alloc_buf(AX_ENGINE_IO_BUFFER_T* b, size_t n) {
    memset(b, 0, sizeof(*b));
    b->nSize = n;
    return AX_SYS_MemAllocCached((AX_U64*)&b->phyAddr, &b->pVirAddr, n, 128, (const AX_S8*)"sharp");
}

static bool load_model(Model& m, const std::string& path, const std::string& name) {
    m.name = name;
    std::vector<char> buf;
    if (!read_file(path, buf)) { fprintf(stderr, "cannot read model %s\n", path.c_str()); return false; }
    int ret = AX_ENGINE_CreateHandle(&m.h, buf.data(), buf.size());
    if (ret) { fprintf(stderr, "%s CreateHandle 0x%x\n", name.c_str(), ret); return false; }
    ret = AX_ENGINE_CreateContext(m.h);
    if (ret) { fprintf(stderr, "%s CreateContext 0x%x\n", name.c_str(), ret); return false; }
    ret = AX_ENGINE_GetIOInfo(m.h, &m.info);
    if (ret) { fprintf(stderr, "%s GetIOInfo 0x%x\n", name.c_str(), ret); return false; }
    m.io.nInputSize = m.info->nInputSize; m.io.nOutputSize = m.info->nOutputSize;
    m.io.pInputs = new AX_ENGINE_IO_BUFFER_T[m.info->nInputSize]();
    m.io.pOutputs = new AX_ENGINE_IO_BUFFER_T[m.info->nOutputSize]();
    m.own_in.assign(m.info->nInputSize, false); m.own_out.assign(m.info->nOutputSize, false);
    if (getenv("SHARP_PROF")) for (uint32_t i = 0; i < m.info->nInputSize; ++i)
        printf("  %s in  %-16s %10u B dtype=%d\n", name.c_str(), m.info->pInputs[i].pName, m.info->pInputs[i].nSize, m.info->pInputs[i].eDataType);
    if (getenv("SHARP_PROF")) for (uint32_t i = 0; i < m.info->nOutputSize; ++i)
        printf("  %s out %-16s %10u B dtype=%d\n", name.c_str(), m.info->pOutputs[i].pName, m.info->pOutputs[i].nSize, m.info->pOutputs[i].eDataType);
    return true;
}
static int find_tensor(const AX_ENGINE_IOMETA_T* metas, uint32_t n, const char* name) {
    for (uint32_t i = 0; i < n; ++i) if (!strcmp(metas[i].pName, name)) return i;
    fprintf(stderr, "tensor %s not found\n", name); return -1;
}
static bool alloc_rest(Model& m) {   // 给还没绑定的输入输出分配缓冲区
    for (uint32_t i = 0; i < m.info->nInputSize; ++i) if (!m.io.pInputs[i].pVirAddr) {
        if (alloc_buf(&m.io.pInputs[i], m.info->pInputs[i].nSize)) return false; m.own_in[i] = true; }
    for (uint32_t i = 0; i < m.info->nOutputSize; ++i) if (!m.io.pOutputs[i].pVirAddr) {
        if (alloc_buf(&m.io.pOutputs[i], m.info->pOutputs[i].nSize)) return false; m.own_out[i] = true; }
    return true;
}
static bool run(Model& m) {
    int ret = AX_ENGINE_RunSync(m.h, &m.io);
    if (ret) fprintf(stderr, "%s RunSync 0x%x\n", m.name.c_str(), ret);
    return ret == 0;
}
static void flush(AX_ENGINE_IO_BUFFER_T& b) { AX_SYS_MflushCache(b.phyAddr, b.pVirAddr, b.nSize); }
static void invalidate(AX_ENGINE_IO_BUFFER_T& b) { AX_SYS_MinvalidateCache(b.phyAddr, b.pVirAddr, b.nSize); }
static void free_model(Model& m) {   // 未加载 (h 为空) 时什么都不做
    if (!m.h) return;
    for (uint32_t i = 0; i < m.io.nInputSize; ++i) if (m.own_in[i]) AX_SYS_MemFree(m.io.pInputs[i].phyAddr, m.io.pInputs[i].pVirAddr);
    for (uint32_t i = 0; i < m.io.nOutputSize; ++i) if (m.own_out[i]) AX_SYS_MemFree(m.io.pOutputs[i].phyAddr, m.io.pOutputs[i].pVirAddr);
    delete[] m.io.pInputs; delete[] m.io.pOutputs;
    AX_ENGINE_DestroyHandle(m.h);
}
#include "fp8vit.hpp"

// 硬件 JPEG 解码 (JDEC, 12MP 约 10 ms; stb_image 在 A55 上要 600 ms) -> NV12 -> RGB u8 HWC。
// 色度上采样和 YCbCr->RGB 按 libjpeg 的整数算法 (h2v2 fancy upsampling + jdcolor 查表), 与 PIL 解码基本一致。
#include <ax_vdec_api.h>
static bool jdec_decode_rgb(const std::vector<char>& js, std::vector<unsigned char>& rgb, int& W, int& H) {
    AX_VDEC_DEC_ONE_FRM_T p; memset(&p, 0, sizeof(p));
    p.stStream.pu8Addr = (AX_U8*)js.data(); p.stStream.u32StreamPackLen = js.size();
    p.enImgFormat = AX_FORMAT_YUV420_SEMIPLANAR; p.enOutputMode = AX_VDEC_OUTPUT_ORIGINAL;
    double t_dec = now_ms();
    int r = AX_VDEC_JpegDecodeOneFrameEx(&p);
    if (getenv("SHARP_PROF")) fprintf(stderr, "  jdec: hw decode %.1f ms\n", now_ms() - t_dec);
    if (r) { fprintf(stderr, "JDEC decode failed 0x%x, falling back to software decode\n", r); return false; }
    AX_VIDEO_FRAME_T& fr = p.stFrame;
    W = fr.u32Width; H = fr.u32Height;
    const unsigned char* Y = (const unsigned char*)fr.u64VirAddr[0];
    // UV 紧跟在 Y 平面后面 (帧大小 = stride*H*1.5)。u64VirAddr[1] 在高度不是 64 的倍数时指向按 64 行对齐的位置, 是错的 (608 高差 32 色度行)
    const unsigned char* UV = Y + (size_t)fr.u32PicStride[0] * H;
    const size_t sy = fr.u32PicStride[0], suv = fr.u32PicStride[1] ? fr.u32PicStride[1] : sy;
    if (!Y || (W & 1) || (H & 1)) { AX_VDEC_JpegReleaseDecodeFrameEx(&fr); return false; }
    // 解码输出在不带 cache 的 CMM 里, 逐像素读极慢: 先按大块并行拷到普通内存
    std::vector<unsigned char> ycopy((size_t)W * H), uvcopy((size_t)W * H / 2);
    parallel_for(H + H / 2, [&](size_t r) {
        if (r < (size_t)H) memcpy(&ycopy[r * W], Y + r * sy, W); else memcpy(&uvcopy[(r - H) * W], UV + (r - H) * suv, W);
    });
    AX_VDEC_JpegReleaseDecodeFrameEx(&fr);
    double t_cp = now_ms();
    Y = ycopy.data(); UV = uvcopy.data();
    const size_t sy2 = W, suv2 = W;
    static int cr_r[256], cb_b[256], cr_g[256], cb_g[256];   // jdcolor.c build_ycc_rgb_table
    for (int i = 0; i < 256; ++i) {
        int x = i - 128; const int SC = 16, HALF = 1 << (SC - 1);
        auto FIX = [](double v) { return (int)(v * (1 << 16) + 0.5); };
        cr_r[i] = (FIX(1.40200) * x + HALF) >> SC; cb_b[i] = (FIX(1.77200) * x + HALF) >> SC;
        cr_g[i] = -FIX(0.71414) * x; cb_g[i] = -FIX(0.34414) * x + HALF;
    }
    rgb.resize((size_t)W * H * 3);
    const int CW = W / 2, CH = H / 2;
    parallel_for(H, [&](size_t yy) {
        int y = yy, cy = y / 2, cy2 = (y & 1) ? std::min(cy + 1, CH - 1) : std::max(cy - 1, 0);   // 最近的色度行 + 上/下邻行
        const unsigned char* c0 = UV + cy * suv2; const unsigned char* c1 = UV + cy2 * suv2;
        std::vector<int> col(CW * 2);   // 竖直方向 3:1 的列和, U/V 交错
        for (int i = 0; i < CW * 2; ++i) col[i] = c0[i] * 3 + c1[i];
        const unsigned char* yr = Y + y * sy2; unsigned char* o = &rgb[(size_t)y * W * 3];
        for (int cx = 0; cx < CW; ++cx) {
            int l = std::max(cx - 1, 0), rr = std::min(cx + 1, CW - 1);
            for (int h = 0; h < 2; ++h) {
                int nb = h ? rr : l, x = cx * 2 + h, cb, cr;
                if ((cx == 0 && h == 0)) { cb = (col[0] * 4 + 8) >> 4; cr = (col[1] * 4 + 8) >> 4; }
                else if (cx == CW - 1 && h == 1) { cb = (col[cx * 2] * 4 + 7) >> 4; cr = (col[cx * 2 + 1] * 4 + 7) >> 4; }
                else { int bias = h ? 7 : 8; cb = (col[cx * 2] * 3 + col[nb * 2] + bias) >> 4; cr = (col[cx * 2 + 1] * 3 + col[nb * 2 + 1] + bias) >> 4; }
                int yv = yr[x];
                int R = yv + cr_r[cr], G = yv + ((cb_g[cb] + cr_g[cr]) >> 16), B = yv + cb_b[cb];
                o[x * 3 + 0] = R < 0 ? 0 : (R > 255 ? 255 : R); o[x * 3 + 1] = G < 0 ? 0 : (G > 255 ? 255 : G); o[x * 3 + 2] = B < 0 ? 0 : (B > 255 ? 255 : B);
            }
        }
    });
    if (getenv("SHARP_PROF")) fprintf(stderr, "  jdec: yuv->rgb %.1f ms\n", now_ms() - t_cp);
    return true;
}
#endif

struct SharpTiming {   // 各阶段耗时 (ms); total = 解码 + 推理 + 后处理 + 写文件, 不含加载模型和 dump
    double load = 0, decode = 0, preprocess = 0, vit = 0, tail = 0, h4 = 0, post = 0, write = 0, total = 0;
    int width = 0, height = 0; double f_px = 0; size_t gaussians = 0; bool hw_jpeg = false;
};

#ifndef SHARP_POST_ONLY
class SharpPipeline {
public:
    ~SharpPipeline() { release(); }
    const std::string& error() const { return err_; }
    double load_ms() const { return load_ms_; }

    // model_dir: fp8vit/, sharp_tail.axmodel, h4_wb.bin; threads <= 0 用全部核
    bool init(const std::string& model_dir, int threads = 0) {
        threads_ = threads;
        const double tl = now_ms();
        if (AX_SYS_Init()) return fail("AX_SYS_Init failed");
        sys_ = true;
        AX_ENGINE_NPU_ATTR_T attr; memset(&attr, 0, sizeof(attr)); attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
        if (AX_ENGINE_Init(&attr)) return fail("AX_ENGINE_Init failed");
        engine_ = true;
        jdec_ = AX_VDEC_Init(NULL) == 0;
        if (!load_model(tail_, model_dir + "/sharp_tail.axmodel", "tail")) return fail("cannot load " + model_dir + "/sharp_tail.axmodel");
        if (!fv_.load(model_dir + "/fp8vit")) return fail("cannot load " + model_dir + "/fp8vit");
        if (!alloc_rest(tail_)) return fail("tail buffer allocation failed");   // ViT 输出直接写进 tail 输入 (零拷贝)
        auto* I = tail_.info;
        t_img_ = find_tensor(I->pInputs, I->nInputSize, "image_u");
        t_vo_ = find_tensor(I->pInputs, I->nInputSize, "vit_out");
        t_l0_ = find_tensor(I->pInputs, I->nInputSize, "lat0");
        t_l1_ = find_tensor(I->pInputs, I->nInputSize, "lat1");
        t_iv_ = find_tensor(I->pInputs, I->nInputSize, "img_vit");
        t_delta_ = find_tensor(I->pOutputs, I->nOutputSize, "delta");
        t_h4x0_ = find_tensor(I->pOutputs, I->nOutputSize, "h4x0");
        t_h4x1_ = find_tensor(I->pOutputs, I->nOutputSize, "h4x1");
        for (int k : {t_img_, t_vo_, t_l0_, t_l1_, t_iv_, t_delta_, t_h4x0_, t_h4x1_}) if (k < 0) return fail("sharp_tail.axmodel lacks an expected input/output");
        auto& T = tail_.io;
        FEAT_ = fv_.tok_bytes() * 2;   // 单块特征 fp32 字节数 (pulsar2 tail 不收 bf16 输入)
        if (T.pInputs[t_vo_].nSize != 35 * FEAT_ || T.pInputs[t_l0_].nSize != 25 * FEAT_ || T.pInputs[t_l1_].nSize != 25 * FEAT_ ||
            T.pInputs[t_iv_].nSize != FEAT_ || T.pInputs[t_img_].nSize != (size_t)12 * 768 * 768 * 4) return fail("tail input sizes do not match the ViT");
        if (T.pOutputs[t_h4x0_].nSize != (size_t)64 * 768 * 768 * 4 || T.pOutputs[t_h4x1_].nSize != (size_t)64 * 768 * 768 * 4) return fail("unexpected h4x output size");
        // 深度头最后的 1x1 卷积 h4 (128->8) 在 CPU 上用 FP32 算: NPU 上 INT8 权重会让远处 (小 disparity) 相对误差到 2%+, 新视角 PSNR 掉到 20 dB
        std::vector<char> wb;
        if (!read_file(model_dir + "/h4_wb.bin", wb) || wb.size() != (8 * 128 + 8) * 4) return fail("missing or bad " + model_dir + "/h4_wb.bin (8x128 weights + 8 biases, fp32)");
        const float* wf = (const float*)wb.data();
        h4rows_.assign(8, H4Row());
        for (int o = 0; o < 8; ++o) {
            h4rows_[o].b = wf[8 * 128 + o];
            for (int c = 0; c < 128; ++c) if (wf[o * 128 + c] != 0.f) { h4rows_[o].c.push_back(c); h4rows_[o].w.push_back(wf[o * 128 + c]); }
        }
        // 输出 o 与 o+4 (两层 l=0/1) 连同一组输入 (c%4 == o%4): 每个任务算一行的两层, 输入只读一次
        for (int o = 0; o < 4; ++o) if (h4rows_[o].c != h4rows_[o + 4].c) return fail("unexpected h4 connectivity");
        TB_ = fv_.tok_bytes(); IB_ = fv_.img_bytes();
        if (alloc_buf(&bfin_, 36 * IB_) || alloc_buf(&stg_, 86 * TB_)) return fail("ViT buffer allocation failed");   // 36 块 bf16 输入 / 86 块 bf16 输出暂存
        load_ms_ = now_ms() - tl;
        ready_ = true;
        return true;
    }

    // img/len: 编码后的图片 (JPEG 走硬解, 其他格式走 stb_image); fpx > 0 时忽略 f35
    // out_path: .spz -> SPZ v2, 其他 -> PLY; dump 非空时写中间张量
    bool run(const unsigned char* img, size_t len, const std::string& out_path, double f35 = 30.0, double fpx = 0, SharpTiming* tm = nullptr,
             const std::string& dump = "") {
        if (!ready_) return fail("pipeline is not initialised");
        SharpTiming t; t.load = load_ms_;
        auto& T = tail_.io;
        const size_t P = (size_t)IN * IN;
        int W = 0, H = 0;
        // ---- 解码 ----
        const double td = now_ms();
        std::vector<unsigned char> rgbv; unsigned char* px = nullptr; bool stb = false;
        {
            std::vector<char> js((const char*)img, (const char*)img + len);
            const bool is_jpeg = len > 3 && img[0] == 0xFF && img[1] == 0xD8;
            if (jdec_ && is_jpeg && jdec_decode_rgb(js, rgbv, W, H)) px = rgbv.data();
            else {
                int ic; px = stbi_load_from_memory(img, (int)len, &W, &H, &ic, 3); stb = true;
                if (!px) return fail("cannot decode the input image");
            }
        }
        t.decode = now_ms() - td; t.hw_jpeg = !stb; t.width = W; t.height = H;
        const double f_px = fpx > 0 ? fpx : f35 * std::sqrt((double)W * W + (double)H * H) / std::sqrt(36.0 * 36 + 24 * 24);
        t.f_px = f_px;

        // ---- 预处理: HWC u8 -> NCHW [0,1] -> 1536² (align_corners=True); tail 吃 2x2 空间转通道后的 image_u [12,768,768] (通道 c*4+dy*2+dx) ----
        const double tp = now_ms();
        std::vector<float> img1536v(3 * P);
        float* img1536 = img1536v.data();
        resize_u8hwc(px, W, H, img1536, IN, IN);
        if (stb) stbi_image_free(px); else std::vector<unsigned char>().swap(rgbv);
        {
            float* iu = (float*)T.pInputs[t_img_].pVirAddr;
            const int G = IN / 2;
            parallel_for(12 * G, [&](size_t r) {   // 平面 (c*4+s) 的第 y 行
                int cs = r / G, y = r % G, c = cs / 4, s = cs % 4, dy = s >> 1, dx = s & 1;
                float* o = iu + (size_t)cs * G * G + (size_t)y * G; const float* q = img1536 + c * P + (size_t)(2 * y + dy) * IN + dx;
                for (int x = 0; x < G; ++x) o[x] = q[2 * x];
            }, threads_);
        }
        flush(T.pInputs[t_img_]);
        // 金字塔 (先归一化到 [-1,1]): 1536² 切 5x5 块, 768² 切 3x3 块, 384² 整图 (各块 384², 步长 288 / 192)
        std::vector<float> x0(3 * P), x1(3 * 768 * 768), x2(3 * 384 * 384);
        parallel_for(3 * IN, [&](size_t r) { const float* a = img1536 + r * IN; float* o = x0.data() + r * IN; for (int i = 0; i < IN; ++i) o[i] = a[i] * 2.f - 1.f; }, threads_);
        for (int c = 0; c < 3; ++c) {
            resize_bilinear(x0.data() + c * P, IN, IN, x1.data() + c * 768 * 768, 768, 768, false, 0.5);
            resize_bilinear(x0.data() + c * P, IN, IN, x2.data() + c * 384 * 384, 384, 384, false, 0.25);
        }
        struct Patch { const float* src; int size; int y, x; };
        std::vector<Patch> patches;
        for (int j = 0; j < 5; ++j) for (int i = 0; i < 5; ++i) patches.push_back({x0.data(), IN, j * 288, i * 288});
        for (int j = 0; j < 3; ++j) for (int i = 0; i < 3; ++i) patches.push_back({x1.data(), 768, j * 192, i * 192});
        patches.push_back({x2.data(), 384, 0, 0});
        // 36 个 ViT 输入 (35 patch + 整图) 先并行转成 bf16, 之后 ViT 循环只剩 NPU 调度
        parallel_for(36 * 3 * 384, [&](size_t r) {   // 第 k 块第 c 通道第 y 行
            int k = r / (3 * 384), c = (r / 384) % 3, y = r % 384;
            const float* src = k < 35 ? patches[k].src + ((size_t)c * patches[k].size + patches[k].y + y) * patches[k].size + patches[k].x
                                      : x2.data() + ((size_t)c * 384 + y) * 384;
            Fp8Vit::to_bf16(src, (uint16_t*)((char*)bfin_.pVirAddr + k * IB_) + ((size_t)c * 384 + y) * 384, 384);
        }, threads_);
        flush(bfin_);
        t.preprocess = now_ms() - tp;

        // ---- FP8 ViT x36 ----
        const double tv = now_ms();
        auto sub = [&](const AX_ENGINE_IO_BUFFER_T& base, size_t off, size_t n) {
            AX_ENGINE_IO_BUFFER_T b; memset(&b, 0, sizeof(b)); b.phyAddr = base.phyAddr + off; b.pVirAddr = (char*)base.pVirAddr + off; b.nSize = n; return b; };
        {
            // ViT 输出 bf16 暂存: [0,35) vit, [35,60) lat0, [60,85) lat1, 85 img; 每块算完由另一个线程转 fp32 写进 tail 输入, 与 NPU 并行
            const size_t TE = TB_ / 2;
            std::vector<std::thread> conv;
            auto widen = [&](size_t slot, const AX_ENGINE_IO_BUFFER_T& dst, size_t k) {
                AX_SYS_MinvalidateCache(stg_.phyAddr + slot * TB_, (char*)stg_.pVirAddr + slot * TB_, TB_);
                const uint16_t* a = (const uint16_t*)((char*)stg_.pVirAddr + slot * TB_); uint32_t* o = (uint32_t*)((char*)dst.pVirAddr + k * FEAT_);
                // [576 token][1024 通道] -> [1024][24x24], 按 16 通道一组分块转置
                const size_t NT = 576, NC = TE / NT;
                for (size_t c0 = 0; c0 < NC; c0 += 16)
                    for (size_t tk = 0; tk < NT; ++tk) { const uint16_t* q = a + tk * NC + c0; for (size_t c = 0; c < 16; ++c) o[(c0 + c) * NT + tk] = (uint32_t)q[c] << 16; }
            };
            bool ok = true;
            for (int k = 0; k < 36 && ok; ++k) {
                AX_ENGINE_IO_BUFFER_T in = sub(bfin_, k * IB_, IB_);
                if (k < 35) {
                    AX_ENGINE_IO_BUFFER_T o = sub(stg_, k * TB_, TB_), a = sub(stg_, (35 + k) * TB_, TB_), b = sub(stg_, (60 + k) * TB_, TB_);
                    ok = fv_.run(0, in, &o, k < 25 ? &a : nullptr, k < 25 ? &b : nullptr);   // patch encoder (lat0/lat1 只要前 25 块)
                    if (ok) conv.emplace_back([&, k] { widen(k, T.pInputs[t_vo_], k); if (k < 25) { widen(35 + k, T.pInputs[t_l0_], k); widen(60 + k, T.pInputs[t_l1_], k); } });
                } else {
                    AX_ENGINE_IO_BUFFER_T o = sub(stg_, 85 * TB_, TB_);
                    ok = fv_.run(1, in, &o, nullptr, nullptr);   // image encoder
                    if (ok) conv.emplace_back([&] { widen(85, T.pInputs[t_iv_], 0); });
                }
            }
            for (auto& th : conv) th.join();
            if (!ok) return fail("ViT inference failed");
            for (int i : {t_vo_, t_l0_, t_l1_, t_iv_}) flush(T.pInputs[i]);
        }
        const double tt = now_ms();
        t.vit = tt - tv;

        // ---- tail (NPU) ----
        if (!::run(tail_)) return fail("tail inference failed");
        invalidate(T.pOutputs[t_delta_]); invalidate(T.pOutputs[t_h4x0_]); invalidate(T.pOutputs[t_h4x1_]);
        const double tq = now_ms();
        t.tail = tq - tt;
        double dump_ms = 0;   // dump 写文件的时间不计入后处理和端到端
        if (!dump.empty()) {
            const double t0 = now_ms();
            mkdir(dump.c_str(), 0755);
            for (int i : {t_vo_, t_l0_, t_l1_, t_iv_}) invalidate(T.pInputs[i]);
            write_bin(dump + "/image.bin", img1536, 3 * P * 4);
            write_bin(dump + "/vit_out.bin", T.pInputs[t_vo_].pVirAddr, T.pInputs[t_vo_].nSize);
            write_bin(dump + "/lat0.bin", T.pInputs[t_l0_].pVirAddr, T.pInputs[t_l0_].nSize);
            write_bin(dump + "/lat1.bin", T.pInputs[t_l1_].pVirAddr, T.pInputs[t_l1_].nSize);
            write_bin(dump + "/img_vit.bin", T.pInputs[t_iv_].pVirAddr, T.pInputs[t_iv_].nSize);
            write_bin(dump + "/delta.bin", T.pOutputs[t_delta_].pVirAddr, T.pOutputs[t_delta_].nSize);
            dump_ms += now_ms() - t0;
        }

        // ---- h4 + ReLU (CPU FP32) -> disparity [2,1536,1536] ----
        std::vector<float> disp(2 * P);
        const double th = now_ms();
        {
            const float* xa = (const float*)T.pOutputs[t_h4x0_].pVirAddr; const float* xb = (const float*)T.pOutputs[t_h4x1_].pVirAddr;
            const int G = IN / 2; const size_t GG = (size_t)G * G;
            parallel_for(IN, [&](size_t Y) {   // 第 Y 行 = 通道 l*4+dy*2+{0,1} 的第 y 行交错
                const int y = Y / 2, dy = Y % 2;
                float acc[4][768];   // [l*2+dx]
                for (int dx = 0; dx < 2; ++dx) {
                    const H4Row& h0 = h4rows_[dy * 2 + dx]; const H4Row& h1 = h4rows_[4 + dy * 2 + dx];
                    float* a0 = acc[dx]; float* a1 = acc[2 + dx];
                    for (int x = 0; x < G; ++x) { a0[x] = h0.b; a1[x] = h1.b; }
                    for (size_t k = 0; k < h0.c.size(); ++k) {
                        const int c = h0.c[k]; const float w0 = h0.w[k], w1 = h1.w[k];
                        const float* q = (c < 64 ? xa + (size_t)c * GG : xb + (size_t)(c - 64) * GG) + (size_t)y * G;
                        for (int x = 0; x < G; ++x) { a0[x] += w0 * q[x]; a1[x] += w1 * q[x]; }
                    }
                }
                for (int l = 0; l < 2; ++l) {
                    float* o = disp.data() + l * P + (size_t)Y * IN;
                    for (int x = 0; x < G; ++x) { o[2 * x] = std::max(acc[l * 2][x], 0.f); o[2 * x + 1] = std::max(acc[l * 2 + 1][x], 0.f); }
                }
            }, threads_);
        }
        t.h4 = now_ms() - th;
        if (!dump.empty()) { const double t0 = now_ms(); write_bin(dump + "/disparity.bin", disp.data(), 2 * P * 4); dump_ms += now_ms() - t0; }

        // ---- 后处理 + 写文件 ----
        Gaussians g = post(img1536, disp.data(), (float*)T.pOutputs[t_delta_].pVirAddr, f_px, W, H, threads_);
        const double tw = now_ms();
        t.post = tw - tq - dump_ms;
        if (!save_gaussians(g, f_px, W, H, out_path, threads_)) return fail("cannot write " + out_path);
        const double t_end = now_ms();
        t.write = t_end - tw; t.gaussians = g.opacity.size();
        t.total = t.decode + t_end - tp - dump_ms;
        if (tm) *tm = t;
        return true;
    }

    void release() {
        if (bfin_.pVirAddr) { AX_SYS_MemFree(bfin_.phyAddr, bfin_.pVirAddr); bfin_.pVirAddr = nullptr; }
        if (stg_.pVirAddr) { AX_SYS_MemFree(stg_.phyAddr, stg_.pVirAddr); stg_.pVirAddr = nullptr; }
        free_model(tail_); tail_.h = nullptr;
        fv_.release();   // 未加载的部分 release 里会跳过
        if (jdec_) { AX_VDEC_Deinit(); jdec_ = false; }
        if (engine_) { AX_ENGINE_Deinit(); engine_ = false; }
        if (sys_) { AX_SYS_Deinit(); sys_ = false; }
        ready_ = false;
    }

private:
    struct H4Row { float b = 0; std::vector<int> c; std::vector<float> w; };
    bool fail(const std::string& e) { err_ = e; return false; }
    std::string err_;
    int threads_ = 0; double load_ms_ = 0;
    bool sys_ = false, engine_ = false, jdec_ = false, ready_ = false;
    Model tail_; Fp8Vit fv_;
    int t_img_ = 0, t_vo_ = 0, t_l0_ = 0, t_l1_ = 0, t_iv_ = 0, t_delta_ = 0, t_h4x0_ = 0, t_h4x1_ = 0;
    size_t FEAT_ = 0, TB_ = 0, IB_ = 0;
    std::vector<H4Row> h4rows_;
    AX_ENGINE_IO_BUFFER_T bfin_{}, stg_{};
};
#endif  // SHARP_POST_ONLY
