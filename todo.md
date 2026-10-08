# Qwen3.8-27B block-FP8：寄存器反量化 + HMMA 重写（已否证收口）+ fold 移除纯 fp16 累加（已兑现）+ fp8 KV cache（进行中）

> 2026-10-07 起的新方向。旧 todo 已整体并入 `history.md`（见文末「todo.md 存档」）。
> 权威证据链仍在 `history.md`；本文只保留当前口径、目标与进行中的工作。

## 一句话现状

- **性能**（32K/512 同会话对拍 §80）：prefill 1176.78 t/s（对手 1402.73，−16.1%）、
  decode 65.70 tok/s（MTP5 最优，对手 86.08，−24%）；**差距全在基座轮**（50.12 vs 33.33 ms，1.504×），
  投机机制不落后（我们草稿边际 11.79 ms 优于对手 13.14）。
- **§98 fold 移除（纯 fp16 累加，对齐对手 Marlin use_fp16_accum）已兑现**：prefill GEMM
  去掉每 K=64 折叠进 fp32 的 160 指令（27% 循环体）并消除 swiglu 叶 104 B 栈溢出；
  同会话交替 A/B **prefill +10.3%**（1081→1191 t/s），swiglu 叶 44.9→54.7 TFLOPS，
  decode 与 spec acc 不变，质量门 argmax 99/99 与旧引擎全同。线性叶 41.8 不变（小 N 形状，
  非 fold 限制），仍是对 1402 的未闭合一块。
- **质量**（§86/§93）：同权重 teacher-forcing 逐步对照，四种体制（AIME 推理 / 短文本 / 代码 /
  数字推理）合计 **116 position 里 115 argmax 一致（0.9914）**，top-8 集合重叠 ~7.83/8，
  唯一分歧是 FP16/BF16 精确并列翻转。
- **七条性能战线已收口**（合计 22 个变体全负/中性），见文末「已收口」表。
- **「寄存器反量化 + HMMA」重写也已全线否证收口**：swiglu 两条 tile 路线（§90/§91）+
  linear broadcast（§94）+ packed 4×2/2×4（§95）+ 对手 Marlin 源码级复现（ALU decode +
  指数偏置 + 64×256 tile，§96）全负/持平，45（swiglu）/46.5（linear bench）TFLOPS 是本
  tile 组织近似最优；里程碑 1/2/3 全部收口。
- **§104 修正：「65 TFLOPS」这个靶子本身是时钟归一化产物，不是结构。** 直接在他们环境里调
  他们的 `ops.marlin_gemm`，我们所有形状上是 50.2–59.5 TFLOPS（均值约 56）；§68.3 的
  5.60 ms 就是同一 kernel 在 ~1.75 GHz 升频下的读数（= 50.3 TFLOPS 换时钟表达）。同会话交替
  下扣掉时钟后，孤立 kernel 差距是 **1.09×**（他们 50.3 / 我们 46.0），不是 1.46×。
  真正有 2.00× 结构的是 MMA 累加模式（f16 vs f32），§98 已吃掉。
- **长上下文现状（§105/§106，当前主线）**：85,070 占用 prefill **1030.22 t/s**、decode 47.21 tok/s、
  acceptance 78.76%；138,933 占用 prefill 808.64 t/s、decode 43.42 tok/s、acceptance 82.57%。
  85k 同 corpus 比 §97.6 的 878.1 t/s 高 17.3%。
- **§106 同内容对拍（`" the"` filler，FP16 KV，我们 MTP3 / 他们 MTP4）**：
  **prefill 我们掉得少**（85k→139k：我们 −19.2%，他们 −21.3%；绝对比 0.888 → 0.911 随长度收窄）；
  **decode 我们掉得多**（每轮时间他们 47.34 → 47.05 ms 基本平，我们 71.77 → 79.69 ms，+11.0%，
  其中基座轮 57.40 → 64.06 ms）。我们的长度相关成本 ≈0.125 ns/context-token，是 FP16 KV
  纯字节下限（34.8 KB/token/卡 ÷ 616 GB/s ≈ 0.055）的 2.2×，多出的 ~3.7 ms/轮不在 KV 字节。
  ⇒ **下一个靶子：长上下文 decode 的长度相关段（attention / draft 侧）**。
- **§110 decode 归因（已完成，量化到内核）**：长度相关成本**100% 是 decode attention 一个内核**
  （85k→139k：attention 14.54→22.30 ms/轮 = +7.76，占窗口增长 +7.63 的 102%；fp8 linear/swiglu
  完全平：23.97→24.03 / 16.06→16.07）。而它**只跑到 KV 字节下限的 5.7×**：达成带宽
  101.8→108.4 GB/s = 峰值的 17%，**且不随长度变化** ⇒ 不是带宽受限，是一个固定速率的引擎。
  发射几何实测 `grid=(2,136)`、regs=182、static smem=25,832 B ⇒ 2 blocks/SM = 8 warp/SM = 25%
  理论占用，`Br=32` 被 mma.m8n8k4 的 quadpair 钉死。量化上限：139k 时 18.4 ms/轮 ≈ 轮长 23%。
- **§110 跨引擎基座轮逐项对拍（已完成，85k、双方无投机 + fp16 KV）**：窗口/步 我们 57.95 vs
  他们 38.50 = **1.51×**；增长 19.45 ms 的构成 = **权重 GEMM +13.31（68%，1.57×）** +
  **attention +5.53（28%，1.97×）** + 空隙 +4.14 − 其它内核 3.53；LM head 两边都是 2.17 ms。
  ⇒ **孤立叶子只差 1.09× 而在位权重 GEMM 是 1.57×** —— 悬案从"叶子速率"转到"64 层链的每次
  发射/结构开销"；attention 在位 1.97× 是第二个独立靶子。
  ⇒ **下一批动作（按证据）：①在位权重 GEMM 的 13.3 ms 分解（内核级分开"字节地板/发射次数/
  依赖链"）；②attention 的 3 blocks/SM 候选（`Bc` 8→4 + 削 pad 到 ≤20,821 B，或 warp 间 split-K）。**
- **§110 项 1 收口：`DecodeSplitScale = 2` 不需要 A/B。** 实测 grid=(2,136) = 272 CTAs，而驻留
  136 块 ⇒ 恰好两个满波、零尾波浪费，与该头文件自己的推导一致；可动项只剩 reduce（1.03 ms/轮
  = attention 的 7%），splits 减半最多省 ~0.7%。**从"几个百分点"降级为 <1%。**
- **§111 修正 + 定位 + 一条已否证的修复**：
  ① **§104 的"孤立叶子 1.09×"是在大 T（计算受限）形状上量的，而 decode 权重 GEMM 是 T=1
  纯带宽受限（2 FLOP/byte），两者不可比** —— 这是本轮修正的一个体制错误。
  ② **在位带宽：我们 385 GB/s = 峰值的 62.5%，他们 576 GB/s = 93.5%**（他们权重只少 4.4%，
  取自其 server 日志 14.28 GiB）⇒ 权重 GEMM 的 1.57× 就是 DRAM 利用率差。
  ③ **ncu 定位：叶子卡在 L1/TEX（74.5%）而不是 DRAM（51.6%）**，L1 命中率 79.7% 说明请求量
  约为 DRAM 流量的 5 倍；机制是**激活地址不含 row**，每个 warp / 每个 row-block 重读同一份激活
  （每 K-tile：权重 4 sectors vs 激活 32 sectors）。
  ④ **修复"每 warp 多行、共享激活读"已实测否证并回退**：内核级完全按设计生效（L1 74.6→65.4%、
  命中 79.7→67.8%、同形状 Duration −3.7%、数值逐位不变），但引擎级 **R=2 慢 2.3%**，因为
  `n=5120` 形状退 **+15.8%**，吃掉其余形状全部收益。选择性启用净值仅 −0.64%（在 ±3% 噪声内），
  **不值得加形状条件分支**。⚠️ 下一步必须先解释 `n=5120` 为何退 15.8%，再碰这个叶的粒度。
- **§107 q4（groupwise-int）控制组复验：未受影响。** FP8 的改动只在
  `src/ops/linear/fp8_block/`（未跟踪新文件）与文档里，所有 `NINFER_FP8_BLOCK_*` 旋钮都是
  fp8_block 私有；Q4/Q5/W8 端到端 32,066 token prefill **1221.45 t/s**（控制值 1,220.70）、
  轮时间 44.7 ms（控制值 44.2），q4/q5/w8/q6/bf16/nvfp4-a16 的 Op oracle 全过。
- **新方向：fp8 KV cache**（E4M3 存储，容量减半 + 质量损失最小化，进行中）：见待办。
- **§108 bug 普查（未修，待定）**：
  1. `ninfer_gdn_input_proj_conv_snapshot_test` 的 block-FP8 快照判据没跟 §98 重标
     （`NOFOLD=0` 时 0 失败、默认 2 失败）⇒ 按 fp16 累加 profile 重标，或把该用例钉在 fold 路由。
  2. `bf16_linear_add_gemm_mma.cu` 的 `UpTo32` schedule = **98,304 B** 动态 shared > sm_75 的
     65,536 B ⇒ `cudaFuncSetAttribute` 失败后 abort（**5..32 token 必崩**）。需像
     `gqa_attention_prefill_common.cuh` 那样加 `NINFER_SM75` 变体。本机磁盘 artifact 走不到
     （groupwise-int 绑 Q5），但是 `nvfp4` 身份的潜在崩溃点。
  3. 另有 23 项失败落在工作树其它已改区域（三个协议 schema 测试、int8 attention 判据、
     chat template、UTF-8 decoder、swiglu split 的 NVFP4 T=17 断言），未逐条定位。
     （原 24 项里的 `ninfer_bench_support_test` 已由 §109 证实是**过期断言**并清除。）
- **§109 YaRN 核查与扩展窗口实测（已交付）**：YaRN 早已完整实现（独立 kernel + 家族运行时
  建表 + fp16/int8/fp8 三条 decode KV 的 split bound 重定标），FP8 block128 身份本来就准入
  `--rope yarn`，**模型侧零改动**。本目标真实缺的是接口：`--kv-dtype fp8` 被 CLI 拒绝，
  而 README/docs 都把它记为已交付契约、`ninfer_bench`/`ninfer-serve` 都接受 ⇒ 已修 CLI、
  修正 fp8 的汇报串（原会误报 `int8-group64`）、清掉一条过期 bench 断言（§108 表减一条）、
  并补上 `--yarn-origin × --yarn-factor` 必须落在整数 token 的文档。
  **容量（实测，投机后端必须分开列）**：fp16 KV 上限 153,984 **低于**原生 262,144 ⇒ YaRN 在该
  配置下无用；fp8 KV **无投机** 300,000（320,000 需 7.13 GB、只有 6.89 GB 可用）；fp8 KV +
  **MTP3 + `--lm-head-draft`** 只剩 **254,000**（开 MTP3 先吃 417 MB 运行时预算）。
  ⇒ **254,000 < 原生 262,144：在本项目的验收 lane 上 `--rope yarn` 一点窗口都买不到**，因为是
  显存够不到原生窗口，不是 YaRN 扩展不足；+14.4% 只属于无投机配置。254,000 是贴边值（剩
  172 MiB），实用取 250,000。
  ⇒ **瓶颈是 22 GB 显存而不是 RoPE 上限**；32 GB 卡上的 1M 是显存差异，不是能力差异。
  ⚠️ 顶端失败模式会换手：259,200/260,000 被规划器的预留检查拒绝，而 **256,000 通过检查后死在
  `cudaMalloc`** ⇒ 显式 `--max-context` 的预留估算比真实分配乐观几 MB。
  **检索（实测）**：native 261,000 / yarn×1.125 272,000 / yarn×4 272,000 / **MTP3 lane 244,000
  （容量 254,000）** 四点全 PASS。证据边界：单 needle、单深度、粗门 ⇒ 只证明「跨过原生上限后
  检索未被破坏」，**不能**分辨因子好坏（×4 也过）。
  ⚠️ 探针陷阱已修：prompt 占满窗口时引擎 `finish reason context-capacity` 只出 1 token，那是
  构造错误不是检索失败；`.scratch/yarn_needle.sh` 现在判 `NOWINDOW`。
  **下一靶子：因子—窗口匹配的质量判据（多深度/多 needle 或困惑度），以及 YaRN 的同会话
  A/B prefill 开销。**

## 为什么推翻：sm_75 没有 cp.async，staging 模型是错的

现有 prefill GEMM（row-major 与 Marlin 两套布局**都一样**）走：

```
LDG 权重 → LUT 反量化 → STS 进 shared → ldmatrix → HMMA
```

这是 cp.async 时代（sm_80+）的模式：现代架构用异步拷贝引擎把 staging 藏掉，sm_120 还有
TMA/wgmma。**sm_75 没有 cp.async**——staging 是同步 LSU 操作，与 `ldmatrix` **争用同一条 LSU
通道**。§69.1 实测撞到这面墙："staging 与 mma 的 LDSM 共用同一条 LSU 通道、两个相位完全相加、
ring buffer 零重叠"。逐指令对拍（§71）：**我们 4.76 指令/HMMA，对手 3.08**——因为对手 Marlin
**在寄存器里按 mma 片段直接反量化，根本不经过 shared**。

| 内核 | 反量化路径 | 实测 |
|---|---|---|
| 我们 prefill（row-major 与 Marlin 两套） | LUT → STS → LDSM | **44.7 TFLOPS** |
| 我们 Marlin decode 叶 | 寄存器反量化 + **SIMT 点积** | 89% 带宽（仅 T ≤ 4/8） |
| 对手 Marlin prefill | **寄存器反量化 + HMMA** | 50.2–59.5 TFLOPS（§104 直接调用实测；§68.3 的 65 是升频读数） |

**缺口**：我们的 decode 叶证明了寄存器反量化在 sm_75 可行，但它用 SIMT 点积（只适合小 T）；
**「寄存器反量化 + HMMA」的 prefill 路径从未真正实现过**。§62 试过一版「直接消费 fragment」，
慢 74%——因为当时没有 fragment 预排布、寄存器压爆（255 regs + spill）。

## 目标 —— 已否证（2026-10-07），靶子「65 TFLOPS」已由 §104 撤回（2026-10-08）

原目标：把 prefill GEMM 从「shared staging」改成「**寄存器反量化 + HMMA**」，靶子 **65 TFLOPS**。
**判定：全线否证。** swiglu 256-token tile 否证（4× warp_col 权重冗余，§90）、swiglu 64×64 小 tile
否证（block 间权重 DRAM 4×，§91）、linear 128×128 broadcast 持平（§94）；对手 Marlin 的持久
一维网格 + 小 tile 组织已由 §69 否证。**45/46.5 TFLOPS 是本 tile 组织近似最优。**

**靶子本身也要撤回**（§104）：65 TFLOPS 是 §68.3 拿他们升频读数与我们降频读数直接相减得出的，
扣掉时钟后他们的真实孤立速率是 50.2–59.5 TFLOPS，对我们是 **1.09×**。所以「追 65」这个命题
从一开始就不成立——上面那些否证只说明「照他们的 tile 组织重做拿不到收益」，不说明存在一个
我们够不到的 65 TFLOPS 结构。

## 技术确认（2026-10-07，已钉死）

- 当前 Marlin 持久布局 `marlin_fp8_block_word` 的地址公式解出的 word = **1 行 × 4 列**，
  每个 thread 的 8 个连续 word 覆盖 **4 行（4 个 warp 组）× 8 列（2 个 result 组）**——
  这是为 decode 叶的 **SIMT 点积**排布的（每线程连续 32 B、独立累加 2 行），
  **不是**为 HMMA B 片段（m16n8k8 的 8×8、每线程 2 个 f16）排布的。
- ⇒ 要做「寄存器反量化 + HMMA」，必须**新定义一个 HMMA 适配的持久布局**（converter 改动），
  使每个线程的 `LDG 16B → decode_marlin_quad → 寄存器` 恰好等于 HMMA B 片段；
  否则只能靠 PRMT 重排反量化结果，会重蹈 §71 的 XU（+96M 指令）开销。
- 参考实现：我们自己的 decode 叶（`marlin_decode_accumulate` 的寄存器反量化机制）+
  对手 Marlin 的「寄存器直接反量化」内核。

## 待办（按依赖排序）

### 里程碑 1：swiglu 叶原型（一叶定成败）—— 已判否证（§90）

- [x] 机制全部验证正确、无 spill：A/B 片段 lane 映射（probe 钉死）、16×32 packed 布局、
      LUT 反量化、LDG.128 消费（§88/§89）
- [x] 四个变体全部实测否证（§90.1）：regdequant 30.48 / packed 39.85 / broadcast 44.12 /
      broadcast+constant LUT 43.14，均 ≤ staged 45.27
- [x] 根本原因（§90.2）：256-token tile 的 **4× warp_col 权重冗余**——staged 靠 shared 做
      block 级 1× 权重共享，寄存器反量化去掉 staging 后要么 4× LDG（DRAM）要么 4× LUT（shared），
      都无法超越 staged。**在 256-token tile 上「寄存器反量化 + HMMA」判为不可行**
- [x] 改 tile 组织（A 方案）也已否证（§91.2）：64×64 小 tile 消除 warp_col 冗余，但 block 间
      token 维权重共享损失 → **权重 DRAM 流量 4×，23.70 TFLOPS，慢 47%**。两条 tile 路线都否证
- [x] 顺带修复两个真 bug（§91.1）：row0 缺失、`build_fp8_code_table` 128-thread 内核 LUT
      高半未填（加强测试 kCodes 4→17 质数后暴露）
- [x] **里程碑 1 最终判定：swiglu 叶 45 TFLOPS 是本 tile 组织近似最优，追 60+ 不可行**；
      对手 Marlin 65 是 linear 口径（非 swiglu 融合），不可直接比

### 里程碑 2：推广到五个生产形状 —— 已判否证（§94）

- [x] linear 叶 `fp8_block_linear_hmma_broadcast_kernel`（shared-broadcast 寄存器反量化，
      复用里程碑 1 全部机制，单份权重，128×128 tile）已实现，oracle 4/5/8/9/192 全过、
      REG 158–159 无 spill
- [x] 四形状同会话交替 A/B（T=4096）：down/GDN-in/QKV/attn-out 全部 **±0.5% 噪声内持平**
- [x] 根本原因（§94.3）：linear 叶同样有 warp_col=4，broadcast 用 LDS4×+LUT4× 换 LDSM4×，
      但 linear 权重面比 swiglu 小一半 ⇒ 两者抵消，净 ≈ 0（swiglu 是 −2.5%）
- [x] **里程碑 2 判定：linear 叶「寄存器反量化 + HMMA」否证（持平）**

### 里程碑 3：端到端验证 —— 已实测否证（§95/§96）

- [x] 对手 Marlin 的组织「小 tile + 持久 CTA + 寄存器反量化」做掉可判定部分：
      packed 16×32 持久布局 + 寄存器反量化（`fp8_block_linear_hmma_packed_kernel`，
      4×2 与 2×4 两种 warp 网格），oracle 4/5/8/9/192 全过、REG 158–159 无 spill
- [x] 两种网格四形状同会话交替 A/B（T=4096）全部 **±0.5% 噪声内持平**（§95.3）
- [x] 读对手源码（`marlin_template.h`/`dequant.h`）拼出反量化路径（ALU 位变换 + 指数偏置
      融合 + packed 布局 + block 级 shared）与 tile 方向（64 token × 256 N），实测仍持平（§96）
- [x] **里程碑 3 判定：六条寄存器反量化路线（packed 4×2/2×4、broadcast ALU、packed256、
      §89、§91、§94）全否证/持平，45/46.5 TFLOPS 是本 tile 组织近似最优**；§104 进一步
      证明靶子本身（65）是时钟读数差异，真实孤立差 1.09×

### 质量门（与重写并行，§86/§93）

- [x] 短文本/代码/推理 workload 的逐步 argmax agreement 分布（§93）：
      shorttext 32/33、code 33/33、reasoning 33/33（合计 **98/99**，唯一分歧是
      FP16/BF16 精确并列翻转、top-8 集合 8/8 相同）；合并 §86 的 AIME 17/17 ⇒
      **116 position 里 115 argmax 一致（0.9914）**。
      （材料：`.scratch/quality_{shorttext,code,reasoning}.*`、`make_quality_prompts.py`、
      `compare_quality.py`）
- [ ] （可选）source 侧 layer-boundary hook，补上「层边界」这一环（§N 缺口）

### 新方向：fp8 KV cache（E4M3 存储，容量减半 + 质量损失最小化）

> §70.7 已证 int8 KV 减半后 decode 轮长不变（KV 非带宽瓶颈），但掉 **4.3 接受率点**、
> prefill 退 **10–15%**。fp8 的目标**不是 decode 提速**，而是「容量翻倍 + 接受率损失
> 远小于 int8」——E4M3 浮点比 int8 整数更贴 KV 激活（大动态范围 + 小值多，指数位更合适）。

- [x] 枚举 `KvCacheStorage::Fp8E4M3` + 容量/字节计算（1 byte/元素，同 int8 减半，
      无 scale 平面，`quant_group == 0`）
- [x] 写路径：bf16 → E4M3 量化（**直接 cast，无 per-group scale**——E4M3 指数位覆盖
      KV 激活范围；`gqa_attention_kv_quant.cuh` 加 `gqa_kv_quant_fp8` /
      `gqa_kv_quant_fp8x8_bf16` codec）
- [x] 读路径：fp8 → fp16 反量化（ALU 位变换 `gqa_kv_dequant_fp8x8_f16_from`，
      attention decode TC-volta / prefill 两处读 K/V 的路径）
- [x] oracle：KV 量化往返 vs bf16 的数值判据（fixture 加 E4M3FN 独立 host codec +
      `kAttentionFp8Criterion`；fp8 用例 0 失败，实测 relative-L2 ~2.0–2.4e-3，
      **优于 int8 的 3.15e-3**）
- [x] 判据：8K/16K/32K 三档上下文，同会话交替 `--kv-dtype fp8` vs fp16 vs int8
      （fp8 artifact、devices 0,1、codechat corpus）。结果：
      **KV payload 553.6 MiB vs fp16 1.08 GiB（−50%）**；prefill **−2.0~−4.7%**
      （int8 为 −7.3~−13.6%）；acceptance **与上下文非单调**——8K fp8 0.7231 vs fp16
      0.6026（**+12 点**）、16K +3.9 点、32K −2.87 点；decode 跟随（8K +11.9%、32K −3.1%）。
      `--capture-generation` 显示 8K/128 输出**逐 token 相同**、8K/512 与 32K 轨迹分叉
      ⇒ **acceptance 是轨迹量，不能直接当质量损失相减**。
- [x] 长上下文判据（fp8 的主场）：**85k** 与 **140k** 占用，同会话单轮。
      结果：**acceptance 逐位相同**（85k 0.8182、140k 0.7333）；**KV payload 精确减半**
      （85k 1.380 vs 2.760 GiB、140k 2.271 vs 4.543 GiB；140k 每卡省 2.27 GiB）；
      decode 略快（+1.2% / +0.7%）；prefill −5.7% / −8.7%（int8 @85k 为 −16.8%）。
      ⇒ **长上下文下 fp8 对 MTP 无损，容量减半是纯收益**；32K 的 −2.87 点确认为轨迹效应。
- [x] 容量边界（`--max-ctx` 扫描 + 各自上限的真实 prompt 端到端跑通）：
      **fp16 上限 153,984**（页组粒度；153,950 OK / 153,990 报
      `requires 6475811584 bytes, but only 6475399424`，差 0.4 MB），
      **fp8 上限 262,144 = 模型原生上限**（显存不是瓶颈）⇒ **1.70×**。
      端到端：fp16 @153,984 prefill 703.0 / decode 50.89 / KV 4.993 GiB；
      fp8 @262,144 prefill 467.1 / decode 28.62 / **KV 4.250 GiB**
      （比 fp16 在 154k 的还少 15%）。262k 分配后 GPU 占用 21,155 / 22,528 MiB。
      固定开销：权重 15.296 GiB、workspace 0.753 GiB、graph 0.094 GiB，
      运行时预算 6,475,399,424 bytes（6.03 GiB）。
- [ ] 开放（质量，非容量）：固定轨迹或独立质量评估，把 8K/32K 的 acceptance 差与
      轨迹结构彻底分离。现状证据只到：8K/128 输出逐 token 相同（差全是轮结构）、
      32K 从生成第 11 个 token 起轨迹分叉（该点上不可分离）、85k/140k acceptance
      逐位相同。需要一个 teacher-forcing 固定序列下的 acceptance 对比，或独立质量
      oracle（如长文 PPL / 下游题），才能对中等上下文给出无损结论。
      已写入 `docs/performance.md` 的 “FP8-E4M3 KV cache on this target”。


## 已收口的战线（不再有动作，详情见 history）

| 战线 | 结论 | history |
|---|---|---|
| prefill GEMM staging 变体 | 7 变体 + 2CTA + 照对手组织重做，全负 | §81、§82.6、§69 |
| decode 权重流叶 | 五个结构解释排除，418 GB/s 无机制可改；最后落点（激活 shuffle 广播）机制不成立 | §75–77、§92 |
| small-T attention | 16 warp 寄存器墙，DimSplit8 中性 | §82.7 |
| k=4/5 投机异常 | 已修复，draft=5 反超成最优 | §82.8 |
| 布局切换 | 混合布局净 −0.75%、全量 +1.6%，关闭 | §83 |
| row-parallel allreduce 同步 | 编排层 ≤0.5%、graph 只值 1.6% | §84 |
| prefill 墙钟 | GEMM 组织已按对手重做并否证；attention 顶满 64 KB shared | §85 |
| linear 叶（ncu 定案） | latency-bound、25% 占用、2-CTA 双堵死（136 regs / 32.77 KB shared），不追 | §99 |
| head 路由 | 已优化，decode +15.7% | §57 |
| 基座轮 T=1 归因 | 四构成（权重流 80.3%/attention 10.9%/head 4.8%/allreduce 8%）全否证，无可执行动作 | §82.9、§92 |

## 纪律（沿用，不重写）

- **同会话交替 A/B**（本机漂移大），median 口径；叶子级收益必须在引擎级复现才算数（§82.5）。
- **bench 排序不能外推到引擎级**（§83 教训：5 个形状 3 个符号相反）。
- 数值变换需 oracle：bit-identical，或独立 FP64 数学 oracle（§N 协议）。
- 构建：`export CPATH=/data/deps/root/usr/include/x86_64-linux-gnu LIBRARY_PATH=/data/deps/root/usr/lib/x86_64-linux-gnu PKG_CONFIG_PATH=/data/deps/root/usr/lib/x86_64-linux-gnu/pkgconfig; ninja -C build -j4 <target>`。
- 测量（32K/512 MTP3）：`./build/bench/ninfer_bench --weights /data/models/qwen3.8-27b/qwen3_8_27b_fp8_block128.ninfer --corpus .scratch/synth_the_32768.ids -pg 32768,512 --tp 2 --devices 0,1 --prefill-chunk 4096 --kv-dtype fp16 --mtp-draft-tokens 3 --lm-head-draft --repetitions 2 --warmup 1 -o table`。
- vLLM 0.26 环境：`/home/zhaomh/miniconda3/envs/vllm/bin/python`；source checkpoint `/data/models/qwen3.8-27b/Qwen3.8-27B-FP8`。
