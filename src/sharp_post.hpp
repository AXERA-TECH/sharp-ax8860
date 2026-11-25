// SHARP 主机端 FP32 后处理: 初始化基准值 + 组合 + 反投影 + 写 PLY。
// 与 ml-sharp (initializer.py / composer.py / gaussians.py) 一一对应。
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <dlfcn.h>
#include <cstring>
#include <cstdlib>

namespace sharp {

constexpr int IN = 1536;   // 网络内部分辨率
constexpr int GS = 768;    // 高斯网格 (stride 2)
constexpr int NL = 2;      // 高斯层数
constexpr size_t NG = (size_t)NL * GS * GS;

struct Gaussians {
    std::vector<float> mean, scale, quat, color, opacity;  // 每个高斯: 3,3,4,3,1 (已是 PLY 所需的 sRGB 颜色/opacity logit 前的值)
    void resize(size_t n) { mean.resize(n * 3); scale.resize(n * 3); quat.resize(n * 4); color.resize(n * 3); opacity.resize(n); }
};

template <class F>
static void parallel_for(size_t n, F f, int nt = 0) {
    if (nt <= 0) nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> th;
    size_t chunk = (n + nt - 1) / nt;
    for (int t = 0; t < nt; ++t) {
        size_t b = t * chunk, e = std::min(n, b + chunk);
        if (b >= e) break;
        th.emplace_back([=, &f] { for (size_t i = b; i < e; ++i) f(i); });
    }
    for (auto& x : th) x.join();
}


// 数组版 exp/log: aarch64 上调 glibc libmvec 的 NEON 版本 (一次 4 个), 其他平台退回标量
#if defined(__aarch64__)
#include <arm_neon.h>
extern "C" __attribute__((aarch64_vector_pcs)) float32x4_t _ZGVnN4v_expf(float32x4_t);   // glibc>=2.38 libmvec
extern "C" __attribute__((aarch64_vector_pcs)) float32x4_t _ZGVnN4v_logf(float32x4_t);
static inline void vexp(const float* x, float* y, int n) { int i = 0; for (; i + 4 <= n; i += 4) vst1q_f32(y + i, _ZGVnN4v_expf(vld1q_f32(x + i))); for (; i < n; ++i) y[i] = std::exp(x[i]); }
static inline void vlog(const float* x, float* y, int n) { int i = 0; for (; i + 4 <= n; i += 4) vst1q_f32(y + i, _ZGVnN4v_logf(vld1q_f32(x + i))); for (; i < n; ++i) y[i] = std::log(x[i]); }
#else
static inline void vexp(const float* x, float* y, int n) { for (int i = 0; i < n; ++i) y[i] = std::exp(x[i]); }
static inline void vlog(const float* x, float* y, int n) { for (int i = 0; i < n; ++i) y[i] = std::log(x[i]); }
#endif

using real = float;   // A55 上双精度超越函数很慢, 热路径用单精度 (编译加 -fsingle-precision-constant)
static inline real mx(real a, real b) { return a > b ? a : b; }
static inline real mn(real a, real b) { return a < b ? a : b; }
static inline real sigmoid(real x) { return 1.0 / (1.0 + std::exp(-x)); }
static inline real softplus(real x) { return x > 20.0 ? x : std::log1p(std::exp(x)); }
static inline real inverse_softplus(real t) { t = mx(t, 1e-6); return t + std::log(-std::expm1(-t)); }
static inline real logit(real p) { return std::log(p / (1.0 - p)); }

// 对称 3x3 矩阵 Jacobi 特征分解, 输出特征值降序 + 对应特征向量(列)
static void eig_sym3(const real A_in[3][3], real w[3], real V[3][3]) {
    real A[3][3];
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) { A[i][j] = A_in[i][j]; V[i][j] = i == j; }
    for (int sweep = 0; sweep < 12; ++sweep) {
        real off = std::fabs(A[0][1]) + std::fabs(A[0][2]) + std::fabs(A[1][2]);
        real diag = std::fabs(A[0][0]) + std::fabs(A[1][1]) + std::fabs(A[2][2]);
        if (off <= 1e-30f || off < 1e-7 * diag) break;   // float 精度够了; 阈值太严会跑满所有轮
        for (int p = 0; p < 2; ++p) for (int q = p + 1; q < 3; ++q) {
            if (A[p][q] == 0.0) continue;
            real theta = (A[q][q] - A[p][p]) / (2.0 * A[p][q]);
            real t = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
            real c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
            for (int k = 0; k < 3; ++k) {   // A = J^T A J
                real akp = A[k][p], akq = A[k][q];
                A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq;
            }
            for (int k = 0; k < 3; ++k) {
                real apk = A[p][k], aqk = A[q][k];
                A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk;
            }
            for (int k = 0; k < 3; ++k) {
                real vkp = V[k][p], vkq = V[k][q];
                V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq;
            }
        }
    }
    int idx[3] = {0, 1, 2};
    std::sort(idx, idx + 3, [&](int a, int b) { return A[a][a] > A[b][b]; });
    real Vs[3][3];
    for (int c = 0; c < 3; ++c) { w[c] = A[idx[c]][idx[c]]; for (int r = 0; r < 3; ++r) Vs[r][c] = V[r][idx[c]]; }
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) V[r][c] = Vs[r][c];
}

static void quat_from_rot(const real R[3][3], real q[4]) {  // q = (w,x,y,z), w>=0
    real tr = R[0][0] + R[1][1] + R[2][2];
    real w, x, y, z;
    if (tr > 0) {
        real s = std::sqrt(tr + 1.0) * 2; w = 0.25 * s;
        x = (R[2][1] - R[1][2]) / s; y = (R[0][2] - R[2][0]) / s; z = (R[1][0] - R[0][1]) / s;
    } else if (R[0][0] > R[1][1] && R[0][0] > R[2][2]) {
        real s = std::sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2; w = (R[2][1] - R[1][2]) / s;
        x = 0.25 * s; y = (R[0][1] + R[1][0]) / s; z = (R[0][2] + R[2][0]) / s;
    } else if (R[1][1] > R[2][2]) {
        real s = std::sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2; w = (R[0][2] - R[2][0]) / s;
        x = (R[0][1] + R[1][0]) / s; y = 0.25 * s; z = (R[1][2] + R[2][1]) / s;
    } else {
        real s = std::sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2; w = (R[1][0] - R[0][1]) / s;
        x = (R[0][2] + R[2][0]) / s; y = (R[1][2] + R[2][1]) / s; z = 0.25 * s;
    }
    real n = std::sqrt(w * w + x * x + y * y + z * z);
    if (w < 0) n = -n;
    q[0] = w / n; q[1] = x / n; q[2] = y / n; q[3] = z / n;
}

// image: [3,1536,1536] 0..1 (NCHW)；disparity: [2,1536,1536]；delta: [28,768,768] (通道 c*2+层)
// f_px/W/H: 原图焦距(像素)与尺寸；返回度量空间高斯 (OpenCV 坐标)
static Gaussians post(const float* image, const float* disparity, const float* delta,
                      real f_px, int W, int H, int nthreads = 0) {
    const size_t P = (size_t)IN * IN, Q = (size_t)GS * GS;
    const real disparity_factor = f_px / W;
    // depth = disparity_factor / clamp(disp); 全局最小值归一化 (_rescale_depth)
    // min(depth) = disparity_factor / max(clamp(disp)); 并行求 max
    auto T0 = std::chrono::steady_clock::now();
    auto ms = [&] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - T0).count(); };
    const bool prof = getenv("SHARP_PROF") != nullptr;
    const int NT = nthreads > 0 ? nthreads : (int)std::max(1u, std::thread::hardware_concurrency());
    std::vector<float> pmax(NT, 1e-4f);
    parallel_for(NT, [&](size_t t) {
        size_t n = NL * P, b = n * t / NT, e = n * (t + 1) / NT; float m = 1e-4f;
        for (size_t i = b; i < e; ++i) m = disparity[i] > m ? disparity[i] : m;
        pmax[t] = m;
    }, NT);
    real dmin = disparity_factor / mn((real)*std::max_element(pmax.begin(), pmax.end()), 1e4);
    const real depth_factor = 1.0 / (dmin + 1e-6);
    const real global_scale = 1.0 / depth_factor;
    const real dsf = 2.0 * 1.0 * 2.0 / IN;                       // disparity_scale_factor
    const real sa = 10.0 / 1.0 / 9.0, sb = logit(0.1);           // _get_scale_activation_constant(10, 0)
    const real Ax = W / (2.0 * f_px), Ay = H / (2.0 * f_px);     // 反投影: NDC -> 度量 (对角阵)

    if (prof) fprintf(stderr, "  post: dmin %.1f ms\n", ms());
    Gaussians g; g.resize(NG);
    if (prof) fprintf(stderr, "  post: alloc %.1f ms\n", ms());
    // 按 CH 个像素分块: 逐元素部分写成 SoA 简单循环 (gcc -O3 -ffast-math 用 libmvec 向量化 exp/log),
    // 只有四元数/特征分解留在标量循环里。
    constexpr int CH = 256;
    const real kd = 1.0 / (disparity_factor * depth_factor);   // 1/depth_norm = disp * kd
    const size_t nchunk = (Q + CH - 1) / CH;
    parallel_for(nchunk, [&](size_t ci) {
        const size_t p0 = ci * CH; const int n = (int)std::min((size_t)CH, Q - p0);
        alignas(16) float lbc[3][CH], dl[CH], zz[CH], sv[3][CH], col[3][CH], ta[CH], tb[CH];
        for (int c = 0; c < 3; ++c) {
            const float* im = image + c * P;
            for (int i = 0; i < n; ++i) {
                size_t pix = p0 + i; int h = pix / GS, w = pix % GS;
                const float* r0 = im + (2 * h) * IN + 2 * w; const float* r1 = r0 + IN;
                float s = (r0[0] + r0[1] + r1[0] + r1[1]) * 0.25f;
                s = s < 0.01f ? 0.01f : (s > 0.99f ? 0.99f : s);
                ta[i] = s / (1.0f - s);
            }
            vlog(ta, lbc[c], n);   // logit(base color)
        }
        for (int l = 0; l < NL; ++l) {
            const float* dsp = disparity + l * P;
            auto D = [&](int c) { return delta + (size_t)(c * NL + l) * Q + p0; };
            for (int i = 0; i < n; ++i) {
                size_t pix = p0 + i; int h = pix / GS, w = pix % GS;
                const float* r0 = dsp + (2 * h) * IN + 2 * w; const float* r1 = r0 + IN;
                float m = std::max(std::max(r0[0], r0[1]), std::max(r1[0], r1[1]));
                m = m < 1e-4f ? 1e-4f : (m > 1e4f ? 1e4f : m);
                float d = m * kd; d = d < 0.01f ? 0.01f : d;   // surface_min = max-pool(1/min(depth_norm,100))
                dl[i] = d < 1e-6f ? 1e-6f : d; ta[i] = -dl[i];
            }
            // inverse_softplus(t) = t + log(1 - exp(-t)); 再加 delta 后 softplus(y) = log(1 + exp(y))
            const float* d2 = D(2);
            vexp(ta, tb, n);
            for (int i = 0; i < n; ++i) tb[i] = 1.0f - tb[i];
            vlog(tb, ta, n);
            for (int i = 0; i < n; ++i) { ta[i] = dl[i] + ta[i] + 0.001f * d2[i]; dl[i] = ta[i]; }
            vexp(ta, tb, n);
            for (int i = 0; i < n; ++i) tb[i] = 1.0f + tb[i];
            vlog(tb, ta, n);
            for (int i = 0; i < n; ++i) { float iz = dl[i] > 20.0f ? dl[i] : ta[i]; zz[i] = 1.0f / (iz + 1e-3f); }
            for (int k = 0; k < 3; ++k) {
                const float* dk = D(3 + k);
                for (int i = 0; i < n; ++i) ta[i] = -(sa * dk[i] + sb);
                vexp(ta, tb, n);
                for (int i = 0; i < n; ++i) sv[k][i] = dsf * zz[i] * 10.0f / (1.0f + tb[i]) * global_scale;
                const float* dc = D(10 + k);
                for (int i = 0; i < n; ++i) ta[i] = -(lbc[k][i] + 0.1f * dc[i]);
                vexp(ta, tb, n);
                for (int i = 0; i < n; ++i) col[k][i] = 1.0f / (1.0f + tb[i]);
            }
            const float *d0 = D(0), *d1 = D(1), *d6 = D(6), *d7 = D(7), *d8 = D(8), *d9 = D(9), *d13 = D(13);
            for (int i = 0; i < n; ++i) {
                size_t pix = p0 + i; int h = pix / GS, w = pix % GS;
                size_t gi = (size_t)l * Q + pix;
                real bx = 2.0 * (1.0 + 2.0 * w) / IN - 1.0, by = 2.0 * (1.0 + 2.0 * h) / IN - 1.0;
                real xx = bx + 0.001 * d0[i], yy = by + 0.001 * d1[i], z = zz[i];
                g.mean[gi * 3 + 0] = (float)(Ax * z * xx * global_scale);
                g.mean[gi * 3 + 1] = (float)(Ay * z * yy * global_scale);
                g.mean[gi * 3 + 2] = (float)(z * global_scale);
                for (int k = 0; k < 3; ++k) g.color[gi * 3 + k] = col[k][i];   // 原版 linearRGB 往返抵消
                g.opacity[gi] = d13[i];                                          // opacity logit (sigmoid/logit 抵消)
                real q[4] = {1.0 + d6[i], d7[i], d8[i], d9[i]};
                real qn = 1.0 / std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
                real w0 = q[0] * qn, x = q[1] * qn, y = q[2] * qn, zq = q[3] * qn;
                real R[3][3] = {{1 - 2 * (y * y + zq * zq), 2 * (x * y - w0 * zq), 2 * (x * zq + w0 * y)},
                                {2 * (x * y + w0 * zq), 1 - 2 * (x * x + zq * zq), 2 * (y * zq - w0 * x)},
                                {2 * (x * zq - w0 * y), 2 * (y * zq + w0 * x), 1 - 2 * (x * x + y * y)}};
                // C = A R S² Rᵀ A。转到 R 的坐标系: B = Rᵀ C R = M S² M, M = Rᵀ A R
                real a[3] = {Ax, Ay, 1.0}, s2[3] = {sv[0][i] * sv[0][i], sv[1][i] * sv[1][i], sv[2][i] * sv[2][i]}, M[3][3], B[3][3];
                for (int r = 0; r < 3; ++r) for (int c = r; c < 3; ++c) {
                    real s = 0;
                    for (int k = 0; k < 3; ++k) s += R[k][r] * a[k] * R[k][c];
                    M[r][c] = M[c][r] = s;
                }
                for (int r = 0; r < 3; ++r) for (int c = r; c < 3; ++c) {
                    real s = 0;
                    for (int k = 0; k < 3; ++k) s += M[r][k] * s2[k] * M[k][c];
                    B[r][c] = B[c][r] = s;
                }
                real ev[3], Wv[3][3], U[3][3];
                eig_sym3(B, ev, Wv);
                for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c)
                    U[r][c] = R[r][0] * Wv[0][c] + R[r][1] * Wv[1][c] + R[r][2] * Wv[2][c];
                real det = U[0][0] * (U[1][1] * U[2][2] - U[1][2] * U[2][1]) - U[0][1] * (U[1][0] * U[2][2] - U[1][2] * U[2][0]) + U[0][2] * (U[1][0] * U[2][1] - U[1][1] * U[2][0]);
                if (det < 0) for (int r = 0; r < 3; ++r) U[r][2] = -U[r][2];
                real qq[4]; quat_from_rot(U, qq);
                for (int k = 0; k < 4; ++k) g.quat[gi * 4 + k] = (float)qq[k];
                for (int k = 0; k < 3; ++k) g.scale[gi * 3 + k] = (float)std::sqrt(mx(ev[k], 0.0));
            }
        }
    }, nthreads);
    if (prof) fprintf(stderr, "  post: main %.1f ms\n", ms());
    return g;
}

// 与 ml-sharp save_ply 相同的二进制 PLY (含 extrinsic/intrinsic/image_size/frame/disparity/color_space/version)
static bool save_ply(const Gaussians& g, double f_px, int W, int H, const std::string& path, int nthreads = 0) {
    auto T0 = std::chrono::steady_clock::now();
    auto ms = [&] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - T0).count(); };
    const bool prof = getenv("SHARP_PROF") != nullptr;
    const int NT = nthreads > 0 ? nthreads : (int)std::max(1u, std::thread::hardware_concurrency());
    const size_t n = g.opacity.size();
    // disparity 分位数 (torch.quantile 线性插值): 只拷一份, 两次 nth_element
    std::vector<float> v(n);
    parallel_for(NT, [&](size_t t) { for (size_t i = n * t / NT, e = n * (t + 1) / NT; i < e; ++i) v[i] = 1.0f / g.mean[i * 3 + 2]; }, NT);
    float dq[2];
    {
        double pos1 = 0.1 * (n - 1), pos2 = 0.9 * (n - 1); size_t lo1 = (size_t)pos1, lo2 = (size_t)pos2;
        std::nth_element(v.begin(), v.begin() + lo1, v.end());
        double a1 = v[lo1];
        std::nth_element(v.begin() + lo1 + 1, v.begin() + lo2, v.end());   // lo2 > lo1
        double b1 = *std::min_element(v.begin() + lo1 + 1, v.begin() + lo2 + 1);
        double a2 = v[lo2], b2 = lo2 + 1 < n ? *std::min_element(v.begin() + lo2 + 1, v.end()) : a2;
        dq[0] = (float)(a1 + (b1 - a1) * (pos1 - lo1)); dq[1] = (float)(a2 + (b2 - a2) * (pos2 - lo2));
    }
    if (prof) fprintf(stderr, "  ply: quantile %.1f ms\n", ms());
    std::string hdr;
    {
        char buf[64]; snprintf(buf, sizeof buf, "element vertex %zu\n", n);
        hdr = std::string("ply\nformat binary_little_endian 1.0\n") + buf;
        const char* props[] = {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"};
        for (auto p : props) hdr += std::string("property float ") + p + "\n";
        hdr += "element extrinsic 16\nproperty float extrinsic\nelement intrinsic 9\nproperty float intrinsic\n"
               "element image_size 2\nproperty uint image_size\nelement frame 2\nproperty int frame\n"
               "element disparity 2\nproperty float disparity\nelement color_space 1\nproperty uchar color_space\n"
               "element version 3\nproperty uchar version\nend_header\n";
    }
    std::string tail;
    {
        float ext[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        float in[9] = {(float)f_px, 0, W * 0.5f, 0, (float)f_px, H * 0.5f, 0, 0, 1};
        uint32_t isz[2] = {(uint32_t)W, (uint32_t)H};
        int32_t fr[2] = {1, (int32_t)n};
        uint8_t cs = 0 /* sRGB, 见 color_space.encode_color_space */, ver[3] = {1, 5, 0};
        tail.append((char*)ext, 64).append((char*)in, 36).append((char*)isz, 8).append((char*)fr, 8).append((char*)dq, 8);
        tail.append((char*)&cs, 1).append((char*)ver, 3);
    }
    // ftruncate + mmap, 各线程直接打包进映射区 (pwrite 同一文件会被 inode 锁串行化)
    int fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return false;
    const size_t body = n * 14 * 4, total = hdr.size() + body + tail.size();
    if (ftruncate(fd, (off_t)total) != 0) { close(fd); return false; }
    char* map = (char*)mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) { close(fd); return false; }
    memcpy(map, hdr.data(), hdr.size());
    memcpy(map + hdr.size() + body, tail.data(), tail.size());
    const float C0 = std::sqrt(1.0 / (4.0 * M_PI)), iC0 = 1.0f / C0;
    float* out = (float*)(map + hdr.size());   // 头长度不一定 4 字节对齐, 逐 float memcpy 由编译器处理
    parallel_for(NT, [&](size_t t) {
        constexpr size_t B = 4096;
        std::vector<float> rec(B * 14), lg(B * 3);
        for (size_t i0 = n * t / NT, e = n * (t + 1) / NT; i0 < e; i0 += B) {
            size_t m = std::min(B, e - i0);
            vlog(&g.scale[i0 * 3], lg.data(), (int)(m * 3));
            for (size_t j = 0; j < m; ++j) {
                size_t i = i0 + j; float* r = &rec[j * 14];
                for (int k = 0; k < 3; ++k) r[k] = g.mean[i * 3 + k];
                for (int k = 0; k < 3; ++k) r[3 + k] = (g.color[i * 3 + k] - 0.5f) * iC0;
                r[6] = g.opacity[i];
                for (int k = 0; k < 3; ++k) r[7 + k] = lg[j * 3 + k];
                for (int k = 0; k < 4; ++k) r[10 + k] = g.quat[i * 4 + k];
            }
            memcpy((char*)out + i0 * 14 * 4, rec.data(), m * 14 * 4);
        }
    }, NT);
    bool ok = munmap(map, total) == 0;
    ok = close(fd) == 0 && ok;
    if (prof) fprintf(stderr, "  ply: write %.1f ms\n", ms());
    return ok;
}

// ---------------------------------------------------------------------------------------------
// SPZ v2 (Niantic, gzip 压缩的量化高斯): 体积约为 PLY 的 1/5, 渲染与原 PLY 对比 47–51 dB。
// 位置 24 位定点 (fractionalBits), 不透明度/颜色/尺度/旋转各 8 位; 按 Morton 码排序让 gzip 压得更好。
// zlib 运行时 dlopen 板上系统的 libz.so.1 (BSP 没带头文件, 接口稳定, 这里自己声明)。
// 多线程 deflate: 每段带上一段末尾 32 KB 作字典, 非末段 Z_SYNC_FLUSH 收尾, 拼起来是一个合法的 gzip 流 (同 pigz)。
struct ZStream {
    const unsigned char* next_in; unsigned avail_in; unsigned long total_in;
    unsigned char* next_out; unsigned avail_out; unsigned long total_out;
    const char* msg; void* state; void* zalloc; void* zfree; void* opaque;
    int data_type; unsigned long adler; unsigned long reserved;
};
struct Zlib {
    int (*deflateInit2_)(ZStream*, int, int, int, int, int, const char*, int) = nullptr;
    int (*deflate)(ZStream*, int) = nullptr;
    int (*deflateEnd)(ZStream*) = nullptr;
    int (*deflateSetDictionary)(ZStream*, const unsigned char*, unsigned) = nullptr;
    unsigned long (*deflateBound)(ZStream*, unsigned long) = nullptr;
    unsigned long (*crc32)(unsigned long, const unsigned char*, unsigned) = nullptr;
    unsigned long (*crc32_combine)(unsigned long, unsigned long, long) = nullptr;
    bool ok = false;
    Zlib() {
        void* h = dlopen("libz.so.1", RTLD_NOW);
        if (!h) h = dlopen("libz.so", RTLD_NOW);
        if (!h) return;
        *(void**)&deflateInit2_ = dlsym(h, "deflateInit2_"); *(void**)&deflate = dlsym(h, "deflate");
        *(void**)&deflateEnd = dlsym(h, "deflateEnd"); *(void**)&deflateSetDictionary = dlsym(h, "deflateSetDictionary");
        *(void**)&deflateBound = dlsym(h, "deflateBound"); *(void**)&crc32 = dlsym(h, "crc32"); *(void**)&crc32_combine = dlsym(h, "crc32_combine");
        ok = deflateInit2_ && deflate && deflateEnd && deflateSetDictionary && deflateBound && crc32 && crc32_combine;
    }
};

// data -> 单个 gzip 流 (level 1, nt 段并行)。失败返回空
static std::vector<unsigned char> gzip_parallel(const Zlib& z, const unsigned char* data, size_t len, int nt, int level = 1, int strategy = 0) {
    const size_t DICT = 32768, seg = std::max<size_t>((len + nt - 1) / nt, 1 << 20);
    const int ns = (int)((len + seg - 1) / seg);
    std::vector<std::vector<unsigned char>> out(ns); std::vector<unsigned long> crc(ns); std::vector<int> err(ns, 0);
    parallel_for(ns, [&](size_t s) {
        const size_t b = s * seg, e = std::min(len, b + seg);
        ZStream st; memset(&st, 0, sizeof st);
        if (z.deflateInit2_(&st, level, 8, -15, 8, strategy, "1.3.1", (int)sizeof(ZStream)) != 0) { err[s] = 1; return; }
        if (s > 0) z.deflateSetDictionary(&st, data + b - DICT, (unsigned)DICT);
        out[s].resize(z.deflateBound(&st, e - b) + 64);
        st.next_in = data + b; st.avail_in = (unsigned)(e - b); st.next_out = out[s].data(); st.avail_out = (unsigned)out[s].size();
        const int r = z.deflate(&st, s + 1 == (size_t)ns ? 4 /* Z_FINISH */ : 2 /* Z_SYNC_FLUSH */);
        if (r < 0 || st.avail_in != 0 || (s + 1 == (size_t)ns && r != 1 /* Z_STREAM_END */)) err[s] = 1;
        out[s].resize(out[s].size() - st.avail_out);
        z.deflateEnd(&st);
        crc[s] = z.crc32(0, data + b, (unsigned)(e - b));
    }, nt);
    for (int s = 0; s < ns; ++s) if (err[s]) return {};
    unsigned long c = crc[0];
    for (int s = 1; s < ns; ++s) c = z.crc32_combine(c, crc[s], (long)(std::min(len, (s + 1) * seg) - s * seg));
    std::vector<unsigned char> gz = {0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 3};
    for (auto& o : out) gz.insert(gz.end(), o.begin(), o.end());
    for (int k = 0; k < 4; ++k) gz.push_back((unsigned char)(c >> (8 * k)));
    for (int k = 0; k < 4; ++k) gz.push_back((unsigned char)((uint32_t)len >> (8 * k)));
    return gz;
}

static bool save_spz(const Gaussians& g, const std::string& path, int nthreads = 0) {
    auto T0 = std::chrono::steady_clock::now();
    auto ms = [&] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - T0).count(); };
    const bool prof = getenv("SHARP_PROF") != nullptr;
    static Zlib z;
    if (!z.ok) { fprintf(stderr, "libz.so.1 not found, cannot write SPZ\n"); return false; }
    const int NT = nthreads > 0 ? nthreads : (int)std::max(1u, std::thread::hardware_concurrency());
    const size_t n = g.opacity.size();
    // 位置定点小数位: 默认 12 位, 场景太大时减少, 保证 24 位不溢出
    std::vector<float> tmax(NT, 0.0f);
    parallel_for(NT, [&](size_t t) { float m = 0; for (size_t i = n * 3 * t / NT, e = n * 3 * (t + 1) / NT; i < e; ++i) m = std::max(m, std::fabs(g.mean[i])); tmax[t] = m; }, NT);
    const float amax = *std::max_element(tmax.begin(), tmax.end());
    int fb = 12; while (fb > 0 && amax * (float)(1 << fb) >= (float)((1 << 23) - 1)) --fb;
    // 输出顺序 (固定, 不依赖数据): 768² 网格切 16x16 块, 块按行优先, 块内 Morton 序, 两层交错。
    // 第 j 个输出 -> 源下标 src(j); 压缩率与按 3D 位置 Morton 排序基本一样 (test1: 16.31 vs 16.27 MB)
    const size_t Q = (size_t)GS * GS; const bool grid = n == (size_t)NL * Q && GS % 16 == 0;
    auto compact = [](uint32_t v) { v &= 0x55; v = (v | (v >> 1)) & 0x33; return (v | (v >> 2)) & 0x0F; };
    auto src = [&](size_t j) -> size_t {
        if (!grid) return j;
        const size_t l = j & 1, t = j >> 1, tile = t >> 8; const uint32_t m = (uint32_t)(t & 255);
        const size_t tx = tile % (GS / 16), ty = tile / (GS / 16);
        return l * Q + (ty * 16 + compact(m >> 1)) * GS + tx * 16 + compact(m);
    };
    if (prof) fprintf(stderr, "  spz: order %.1f ms\n", ms());
    // 打包: 头 16 B + positions(9n) alphas(n) colors(3n) scales(3n) rotations(3n)
    std::vector<unsigned char> buf(16 + n * 19);
    uint32_t hdr[3] = {0x5053474e /* "NGSP" */, 2, (uint32_t)n};
    memcpy(buf.data(), hdr, 12); buf[12] = 0; buf[13] = (unsigned char)fb; buf[14] = 0; buf[15] = 0;
    unsigned char *P = buf.data() + 16, *A = P + n * 9, *C = A + n, *S = C + n * 3, *R = S + n * 3;
    const float C0 = std::sqrt(1.0 / (4.0 * M_PI)), iC0 = 1.0f / C0, fscale = (float)(1 << fb);
    auto u8 = [](float v) { v = std::nearbyint(v); return (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v)); };
    parallel_for(NT, [&](size_t t) {
        constexpr size_t B = 4096; std::vector<float> sc(B * 3), lg(B * 3);
        for (size_t j0 = n * t / NT, e = n * (t + 1) / NT; j0 < e; j0 += B) {
            const size_t m = std::min(B, e - j0);
            for (size_t j = 0; j < m; ++j) { const size_t i = src(j0 + j); for (int k = 0; k < 3; ++k) sc[j * 3 + k] = g.scale[i * 3 + k]; }
            vlog(sc.data(), lg.data(), (int)(m * 3));
            for (size_t jj = 0; jj < m; ++jj) {
                const size_t j = j0 + jj, i = src(j);
                for (int k = 0; k < 3; ++k) {
                    int32_t f = (int32_t)std::nearbyint(g.mean[i * 3 + k] * fscale);
                    P[j * 9 + k * 3] = (unsigned char)f; P[j * 9 + k * 3 + 1] = (unsigned char)(f >> 8); P[j * 9 + k * 3 + 2] = (unsigned char)(f >> 16);
                    C[j * 3 + k] = u8(((g.color[i * 3 + k] - 0.5f) * iC0 * 0.15f + 0.5f) * 255.0f);
                    S[j * 3 + k] = u8((lg[jj * 3 + k] + 10.0f) * 16.0f);
                }
                A[j] = u8(255.0f / (1.0f + std::exp(-g.opacity[i])));
                const float* q = &g.quat[i * 4];   // w x y z
                float nq = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]); nq = (q[0] < 0 ? -1.0f : 1.0f) / std::max(nq, 1e-20f);
                for (int k = 0; k < 3; ++k) R[j * 3 + k] = u8((q[k + 1] * nq + 1.0f) * 127.5f);
            }
        }
    }, NT);
    if (prof) fprintf(stderr, "  spz: pack %.1f ms\n", ms());
    // SHARP_SPZ_LEVEL: 0 = 只做 Huffman 编码 (默认, 板上快约 3 倍); 1..9 = zlib 等级 (文件小 5–15%)
    const char* lv = getenv("SHARP_SPZ_LEVEL"); const int level = lv ? atoi(lv) : 0;
    std::vector<unsigned char> gz = level > 0 ? gzip_parallel(z, buf.data(), buf.size(), NT, level, 0) : gzip_parallel(z, buf.data(), buf.size(), NT, 1, 2 /* Z_HUFFMAN_ONLY */);
    if (gz.empty()) { fprintf(stderr, "SPZ gzip failed\n"); return false; }
    if (prof) fprintf(stderr, "  spz: gzip %.1f ms (%zu -> %zu B)\n", ms(), buf.size(), gz.size());
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    bool ok = fwrite(gz.data(), 1, gz.size(), f) == gz.size();
    ok = fclose(f) == 0 && ok;
    if (prof) fprintf(stderr, "  spz: write %.1f ms\n", ms());
    return ok;
}

// 按扩展名选格式: .spz -> SPZ, 其他 -> 与 ml-sharp 相同的 PLY
static bool save_gaussians(const Gaussians& g, double f_px, int W, int H, const std::string& path, int nthreads = 0) {
    if (path.size() > 4 && path.compare(path.size() - 4, 4, ".spz") == 0) return save_spz(g, path, nthreads);
    return save_ply(g, f_px, W, H, path, nthreads);
}

}  // namespace sharp
