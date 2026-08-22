# 扫描件 OCR 自部署（SCAN_OCR）

把扫描书籍的整页 OCR 从云端通用 VLM 换成自部署的端到端文档解析模型，
成本从"按图像 token 计费"变成"按电费计费"。

## 为什么需要它

扫描书籍是整个入库流程里最贵的一环。现有链路是：

1. `docreader/parser/pdf_parser.py` 按图像面积占比判定整页是扫描件，
   渲染成 JPEG，打上 `image_source_type=scanned_pdf`（docreader 自己不做 OCR）；
2. Go 侧 `internal/application/service/image_multimodal.go` 对每张图调 VLM
   **两次**——一次 OCR，一次 caption。

一本 300 页的扫描书 = 600 次整页图像请求。账单主体是图像 token，文本 token 是零头。

除了贵，通用 VLM 做整页密集转写还有个更麻烦的问题：它**不会报错**。
漏行、跳段之后，它会用语言模型能力"顺"出一段读起来通顺的假文本。
这些内容会直接进 RAG 索引，事后极难发现。专门训练的文档解析模型在这件事上
明显更可靠，而且小到可以自己养。

## 选型

| | OvisOCR2 | Unlimited-OCR |
|---|---|---|
| 参数 | 0.8B（Qwen3.5-0.8B 后训练） | 3B 总参 / 500M 激活（MoE，DeepSeek-OCR 改） |
| 许可 | Apache-2.0 | MIT |
| OmniDocBench v1.6 | 96.58 | 93.92 |
| 特长 | 单页精度最高 | KV cache 恒定，一次前向吃数十页 |
| 输出 | Markdown + HTML 表格 + LaTeX 公式 | Markdown + 版面标签 |

默认选 **OvisOCR2**，理由：

- 同版本榜单上单页精度更高；
- 0.8B 在共享 GPU 上只需要很小一块显存；
- **Unlimited-OCR 的多页一次前向在当前架构里用不上**——图片是逐张进
  asynq 队列的，没有把几十页塞进一次请求的入口。要吃到那个红利需要改
  docreader 的分页与任务调度，是另一个量级的改动。

> 榜单分数只在 OmniDocBench 的语料上成立（论文、财报、教材、网页截图）。
> 竖排、繁体、影印模糊件、民国铅字在该基准里基本没有，几分的差距在这类
> 语料上翻转是正常的。**上生产前请用自己的真实书页做盲测，看漏行率和
> 幻觉率，不要看编辑距离。**

## 部署

目标机需要 NVIDIA GPU。以下命令基于本项目使用的 A100 开发/线上机，
所有文件都放在 `/data` 下（根分区还要装 docker 镜像，空间紧张）。

### 1. 安装 vLLM

**先看驱动版本**，这一步决定后面所有命令：

```bash
nvidia-smi --query-gpu=driver_version --format=csv
```

vLLM 自 **v0.20.0** 起 PyPI 默认 wheel 改成了 **CUDA 13** 构建，要求驱动
**≥ 580**。驱动低于 580 时 `pip install vllm` 装得上、一跑就废
（`CUDA driver version is insufficient`）。

vLLM 同时为每个版本发布 `+cu129` 变体，CUDA 12.x 靠 minor version
compatibility 可以在 ≥525 的驱动上运行。本项目那台 A100 驱动是 550.107.02
（CUDA 12.4），因此走 cu129 路线：

**Python 必须是 3.12 或更高**。vLLM 依赖的 flashinfer 在模块顶层写了
`array.array[int]` 这类注解，而 `array.array` 到 Python 3.12 才支持下标，
3.10/3.11 上引擎启动会以 `TypeError: 'type' object is not subscriptable` 失败。
Ubuntu 22.04 自带 3.10，需要额外装：

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
```

`-dev` 不能省：triton 会在运行时 JIT 编译 `cuda_utils.c`，缺 `Python.h`
时报的是 `Model architectures [...] failed to be inspected`，看起来像模型
不支持，实际只是缺头文件。

```bash
mkdir -p /data/ocr && python3.12 -m venv /data/ocr/venv
PY=/data/ocr/venv/bin/python

# 1) 先装 cu129 的 torch 全家桶（download.pytorch.org 国内直连速度尚可）
"$PY" -m pip install --index-url https://download.pytorch.org/whl/cu129 \
  torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0
"$PY" -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
# 期望 2.13.0+cu129 True —— 这一行是整条路线的成败点，False 就别往下走

# 2) 再装 vLLM 的 cu129 wheel（GitHub release 资产，不是 PyPI 上那个）
V=0.27.1
W="vllm-${V}+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
curl -L -C - -o "/data/ocr/$W" \
  "https://github.com/vllm-project/vllm/releases/download/v${V}/vllm-${V}%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
"$PY" -m pip install "/data/ocr/$W" \
  -i https://mirrors.aliyun.com/pypi/simple \
  --extra-index-url https://download.pytorch.org/whl/cu129
```

**驱动 ≥580 的机器**不需要这些，直接 `pip install vllm huggingface_hub`。

> **wheel 文件名不能改**：pip 靠文件名解析版本与平台标签，存成
> `vllm-cu129.whl` 会直接报 `Invalid wheel filename`。

**国内网络注意事项**（那台机器实测）：

- GitHub 直连不通，release 资产加 `https://gh-proxy.com/` 前缀；
- PyPI 用 `https://mirrors.aliyun.com/pypi/simple`；
- HuggingFace 用 `HF_ENDPOINT=https://hf-mirror.com`，且**必须设
  `HF_HUB_DISABLE_XET=1`**——hf-mirror 不代理 Xet CAS 后端，否则下载会以
  `401 Unauthorized ... cas-server.xethub.hf.co` 失败。

### 2. 起服务

```bash
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1   # 国内需要
scripts/scan_ocr_server.sh start     # 首次会自动下载权重（约 2GB）
scripts/scan_ocr_server.sh status    # 探活 + 列出已加载模型
scripts/scan_ocr_server.sh logs
```

可用环境变量覆盖：`SCAN_OCR_PORT`（默认 9800）、`SCAN_OCR_GPU_FRAC`
（默认 0.25，共享卡上刻意只占四分之一）、`SCAN_OCR_MODEL`（HF repo id）。

### 3. 配置 WeKnora

在 `.env` 里加：

```bash
SCAN_OCR_BASE_URL=http://<GPU 机 IP>:9800/v1
```

**后端跑在 Docker 里时不能填 `localhost`**——那是容器自己。填 compose 网络的
网关地址：

```bash
docker inspect WeKnora-app -f '{{range .NetworkSettings.Networks}}{{.Gateway}}{{end}}'
# 本项目那台机器上是 172.20.0.1，因此 SCAN_OCR_BASE_URL=http://172.20.0.1:9800/v1
docker exec WeKnora-app curl -s http://172.20.0.1:9800/v1/models   # 先验证连通
```

其余参数都有默认值，完整说明见 `.env.example` 的「扫描件专用 OCR 后端」一节。

**`SCAN_OCR_BASE_URL` 不配置时本功能完全关闭**，扫描页继续走原来的 VLM 路径。

重启后端后，日志里会出现：

```
[ImageMultimodal] Scanned-page OCR backend enabled: model=OvisOCR2 skip_caption=true
```

## 生效范围

| 图片来源 | 走哪条路 |
|---|---|
| `image_source_type=scanned_pdf`（docreader 判定的扫描页） | 自部署文档解析模型 |
| 其它图片（文档内插图、直接上传的图片） | 知识库配置的 VLM |

插图仍然走 VLM 是有意的：**"描述这张图画的是什么器物"是文档解析模型做不了的事**，
它只会把图周围的文字抄出来。

扫描页默认还会跳过 caption 调用（`SCAN_OCR_SKIP_CAPTION=true`）。
"一页扫描的中文文字"这种描述没有检索价值，却要多花一次整图请求，
还会在索引里塞进一条与 OCR 文本近似重复的 chunk。

## 容错

- 本地服务挂了、超时了，且知识库配了 VLM → **自动回退到 VLM**，不会因为
  一台机器下线就在书里丢页。回退会记进 trace 的 `ocr_fallback` 字段。
- 知识库**没配** VLM，但扫描页由本地后端处理且跳过 caption → 照常入库。
  这让"完全本地、零云端调用"地导入扫描书成为可能。

## 排查

trace / `image_multimodal` 输出里的相关字段：

| 字段 | 含义 |
|---|---|
| `ocr_backend` | `vlm` 或本地模型名 |
| `ocr_fallback` | 出现即表示本地后端失败并回退了 |
| `caption_skipped` | `scanned_page`（按设计跳过）/ `no_vlm` |
| `vlm_model_id` | `none` 表示这一页没用到任何 VLM |

常见问题：

- **整页只出来一半就断了**：`SCAN_OCR_MAX_TOKENS` 调大（默认 16384）。
- **请求超时**：`VLM_HTTP_TIMEOUT_SECONDS` 调大，本后端共用这个超时。
- **输出里有奇怪的 `<img src="images/bbox_...">`**：正常，会被
  `cleanScanOCRText` 剥掉；如果泄漏到 chunk 里说明模型输出格式变了。

## 生产接入实录（2026-08-22）

线上（GPU 机 docker compose 栈）已按以下步骤接入，可作为其他环境的模板：

1. OCR 服务交给 systemd 托管（开机自启、崩溃重拉）：
   `/etc/systemd/system/scan-ocr.service` → `ExecStart=/data/ocr/scan_ocr_server.sh start`，
   `Type=forking` + `PIDFile=/data/ocr/server.pid`。
2. `.env` 追加两行后 `docker compose build app && docker compose up -d app`：
   - `SCAN_OCR_BASE_URL=http://172.20.0.1:9800/v1`（compose 网关，容器内可达）
   - `DUCKDB_SKIP_EXTENSION_LOAD=1`——**这台机器没这行起不来**：DuckDB 启动时
     INSTALL 扩展要访问公网，无出口时整个后端无超时挂死，容器停在
     `health: starting`，最后一条日志是 "Connected to docreader"。
3. 启动日志确认两行：`[DuckDB] Skipping ... extension` 与
   `[ImageMultimodal] Scanned-page OCR backend enabled: model=OvisOCR2`。

端到端验证（知识库**未配 VLM**，五页扫描期刊 test1.pdf）：

- docreader 判定 `5 pages (5 scanned, 0 text)` → 渲染 5 张整页图
- OvisOCR2 逐页 OCR → 5 个 `image_ocr` chunk（223–3405 字）
- 向量化落库：8 条 × 1024 维（text-embedding-v4），摘要自动生成
- 检索「鄂君启金节是在哪里出土的？」Top1 命中含答案的 OCR chunk
- 上传到 completed 全程约 40 秒

注意：通过 API 建知识库时必须显式传 `embedding_model_id`
（如 `builtin-embedding-default`），漏传时文档会停在 processing，
日志报 `processChunks get embedding model failed`。

## 实测

A100-SXM4-40GB（`--gpu-memory-utilization 0.25`，KV cache 6.02 GiB /
494,592 tokens），一页 1654×2339 的中文双栏论文扫描件（含三线表、公式、
上下标，加了旋转和噪点模拟扫描）：

- **2.3 秒/页**，prompt 3922 tokens，输出 709 tokens
- 双栏阅读顺序正确（摘要→引言→实验方法→结果→结论，没有横穿两栏）
- 表格转成 HTML `<table>`，四行五列数据逐格正确
- 公式转 LaTeX：`$$\mathrm{LOD} = 3\sigma / \mathrm{S} \cdot \sqrt{\mathrm{t}}$$`
- 上下标正确：`SiO₂` → `$\mathrm{SiO}_{2}$`，作者角标 → `$^{1}$`
- **忠实转写**：测试图里混进的一处错字（中文句子里夹了个英文单词）被
  原样保留，没有被"顺"成通顺中文——这正是它相对通用 VLM 的核心价值

按 2.3 秒/页算，一本 300 页的扫描书约 12 分钟，成本只有电费。

## 一个容易被忽略的质量杠杆：渲染分辨率

模型再好也只能看到 docreader 渲染出来的那张图。扫描页的渲染上限由
`DOCREADER_PDF_RENDER_MAX_EDGE` 控制，**默认 2000px 长边**（A4 约 170 DPI），
JPEG 质量 `DOCREADER_PDF_JPEG_QUALITY` 默认 85。

对普通印刷体够用，但小字、竖排、密排的古籍/影印件在 2000px 下笔画会糊，
这时模型的错字率跟提高分辨率的收益比换模型大得多。OvisOCR2 的输入上限是
2880×2880，所以可以往上调：

```bash
DOCREADER_PDF_RENDER_MAX_EDGE=2880
DOCREADER_PDF_JPEG_QUALITY=92
```

代价是 gRPC 消息更大、渲染更慢。默认值没有改动，因为它影响所有已有部署。
**盲测时如果发现错字集中在小字区域，先调这两个值再考虑换模型。**

## 换成别的模型

集成层是模型无关的，任何 OpenAI 兼容的文档解析模型都能接。换成
Unlimited-OCR：

```bash
SCAN_OCR_MODEL_NAME=Unlimited-OCR
SCAN_OCR_PROMPT=<image>document parsing.
```

**提示词必须跟着换**：文档解析模型只认训练时那一句，改写会明显掉点。

注意不同模型的输出标记不同（例如 DeepSeek-OCR 系的 `<|det|>` 版面标签），
`cleanScanOCRText` 目前只处理 OvisOCR2 的 bbox 占位图，换模型时可能需要补。
