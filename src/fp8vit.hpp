// DINOv2-L FP8 ViT (AXDSL 4 核 kernel) 的板端驱动: vit_embed -> vit_layer x24 -> vit_final_ln。
// 权重库 <enc>.bank/.idx 由 tools/bank_export.py 生成, 整块读进一段 CMM, 每次调用把输入指到库里的偏移 (零拷贝)。
// 需要在 sharp_ax.cpp 里 Model/load_model/alloc_buf/run/flush 定义之后 include。
#pragma once
#include <map>

struct Bank {
    AX_U64 phy = 0; void* vir = nullptr; size_t size = 0;
    std::map<std::string, std::pair<size_t, size_t>> idx;   // 名字 -> (偏移, 字节数)
    bool load(const std::string& base) {
        FILE* fi = fopen((base + ".idx").c_str(), "r");
        if (!fi) { fprintf(stderr, "cannot read %s.idx\n", base.c_str()); return false; }
        char name[128], dt[32], shp[64]; size_t off, n; size_t end = 0;
        while (fscanf(fi, "%127s %zu %zu %31s %63s", name, &off, &n, dt, shp) == 5) { idx[name] = {off, n}; end = std::max(end, off + n); }
        fclose(fi);
        size = end;
        if (AX_SYS_MemAllocCached(&phy, &vir, size, 128, (const AX_S8*)"sharp_bank")) { fprintf(stderr, "weight bank CMM allocation failed (%zu B)\n", size); return false; }
        FILE* fb = fopen((base + ".bank").c_str(), "rb");
        if (!fb) { fprintf(stderr, "cannot read %s.bank\n", base.c_str()); return false; }
        bool ok = fread(vir, 1, size, fb) == size;
        fclose(fb);
        AX_SYS_MflushCache(phy, vir, size);
        return ok;
    }
    mutable bool bad = false;   // buf() 找不到或大小不符时置位
    AX_ENGINE_IO_BUFFER_T buf(const std::string& k, size_t expect) const {
        AX_ENGINE_IO_BUFFER_T b; memset(&b, 0, sizeof(b));
        auto it = idx.find(k);
        if (it == idx.end()) { fprintf(stderr, "weight bank is missing %s\n", k.c_str()); bad = true; return b; }
        if (it->second.second != expect) { fprintf(stderr, "weight %s has %zu B, kernel expects %zu B\n", k.c_str(), it->second.second, expect); bad = true; return b; }
        b.phyAddr = phy + it->second.first; b.pVirAddr = (char*)vir + it->second.first; b.nSize = expect;
        return b;
    }
    void release() { if (vir) AX_SYS_MemFree(phy, vir); vir = nullptr; }
};

struct Fp8Vit {
    Model emb, lay, fin;
    Bank bank[2];                       // 0 = patch encoder, 1 = image encoder
    AX_ENGINE_IO_BUFFER_T x[2]{}, qkv[3]{}, tap_scratch{}, fin_scratch{};
    // 每层的权重绑定预先算好 (24 层 x 19 个输入)
    std::vector<AX_ENGINE_IO_BUFFER_T> lw[2][24], ew[2], fw[2];
    static constexpr const char* LNAMES[19] = {"qkv", "qkv_s", "qkv_b", "o", "o_s", "o_b", "up", "up_s", "up_b", "down", "down_s", "down_b",
                                              "ln1_w", "ln1_b", "ls1", "ln2_w", "ln2_b", "ls2", "key_mask"};
    bool load(const std::string& dir) {
        if (!load_model(emb, dir + "/vit_embed.axmodel", "vit_embed") || !load_model(lay, dir + "/vit_layer.axmodel", "vit_layer") ||
            !load_model(fin, dir + "/vit_final_ln.axmodel", "vit_final_ln")) return false;
        if (lay.info->nInputSize != 20 || lay.info->nOutputSize != 5 || emb.info->nInputSize != 5 || fin.info->nInputSize != 3) {
            fprintf(stderr, "FP8 ViT kernel IO count does not match the expected ABI\n"); return false; }
        if (!bank[0].load(dir + "/patch") || !bank[1].load(dir + "/image")) return false;
        for (int e = 0; e < 2; ++e) {
            const Bank& B = bank[e];
            for (int i = 0; i < 24; ++i)
                for (int j = 0; j < 19; ++j) {
                    std::string k = j == 18 ? "key_mask" : "b" + std::to_string(i) + "_" + LNAMES[j];
                    lw[e][i].push_back(B.buf(k, lay.info->pInputs[j + 1].nSize));
                }
            const char* en[4] = {"patch_w", "patch_b", "cls", "pos"};
            for (int j = 0; j < 4; ++j) ew[e].push_back(B.buf(en[j], emb.info->pInputs[j + 1].nSize));
            fw[e].push_back(B.buf("norm_w", fin.info->pInputs[1].nSize)); fw[e].push_back(B.buf("norm_b", fin.info->pInputs[2].nSize));
            if (B.bad) return false;
        }
        size_t xs = lay.info->pInputs[0].nSize;
        if (alloc_buf(&x[0], xs) || alloc_buf(&x[1], xs) ||
            alloc_buf(&tap_scratch, lay.info->pOutputs[4].nSize) || alloc_buf(&fin_scratch, fin.info->pOutputs[0].nSize)) return false;
        for (int j = 0; j < 3; ++j) if (alloc_buf(&qkv[j], lay.info->pOutputs[j].nSize)) return false;
        return true;
    }
    size_t tok_bytes() const { return fin.info->pOutputs[0].nSize; }   // [576,1024] bf16
    size_t img_bytes() const { return emb.info->pInputs[0].nSize; }   // [1,3,384,384] bf16
    // float -> bf16 (就近舍入到偶数), 写进 vit_embed 的输入格式
    static void to_bf16(const float* im, uint16_t* d, size_t n) {
        for (size_t i = 0; i < n; ++i) { uint32_t u; memcpy(&u, &im[i], 4); u += 0x7FFF + ((u >> 16) & 1); d[i] = (uint16_t)(u >> 16); }
    }
    // in_img: 已是 bf16 且已 flush 的输入; out_final/out_tap5/out_tap11: [576,1024] bf16 目标缓冲 (空 = 丢弃)
    bool run(int e, const AX_ENGINE_IO_BUFFER_T& in_img, const AX_ENGINE_IO_BUFFER_T* out_final, const AX_ENGINE_IO_BUFFER_T* out_tap5,
             const AX_ENGINE_IO_BUFFER_T* out_tap11) {
        AX_ENGINE_IO_BUFFER_T img = in_img;
        emb.io.pInputs[0] = img;
        for (int j = 0; j < 4; ++j) emb.io.pInputs[j + 1] = ew[e][j];
        emb.io.pOutputs[0] = x[0];
        if (!run_model(emb)) return false;
        int cur = 0;
        for (int i = 0; i < 24; ++i) {
            lay.io.pInputs[0] = x[cur];
            for (int j = 0; j < 19; ++j) lay.io.pInputs[j + 1] = lw[e][i][j];
            for (int j = 0; j < 3; ++j) lay.io.pOutputs[j] = qkv[j];
            lay.io.pOutputs[3] = x[1 - cur];
            const AX_ENGINE_IO_BUFFER_T* t = i == 5 ? out_tap5 : (i == 11 ? out_tap11 : nullptr);
            lay.io.pOutputs[4] = t ? *t : tap_scratch;
            if (!run_model(lay)) return false;
            cur = 1 - cur;
        }
        fin.io.pInputs[0] = x[cur]; fin.io.pInputs[1] = fw[e][0]; fin.io.pInputs[2] = fw[e][1];
        fin.io.pOutputs[0] = out_final ? *out_final : fin_scratch;
        return run_model(fin);
    }
    static bool run_model(Model& m) { return ::run(m); }
    void release() {
        for (auto* b : {&x[0], &x[1], &qkv[0], &qkv[1], &qkv[2], &tap_scratch, &fin_scratch}) if (b->pVirAddr) AX_SYS_MemFree(b->phyAddr, b->pVirAddr);
        // 绑定的都是借用的缓冲区, free_model 只释放 own_* 为 true 的 (这里全是 false)
        free_model(emb); free_model(lay); free_model(fin);
        bank[0].release(); bank[1].release();
    }
};
