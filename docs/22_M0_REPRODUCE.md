# M0 2026-11-02：复现与审核

本轮为冻结模型推理，不训练。先读 [预注册](20_M0_PREREGISTRATION.md)、[来源和授权](21_M0_DATA_AND_ASSETS.md)及 [研究报告](M0_REPORT.md)。已发布结果在 `results/m0-20261102/`；图片、完整注释、知识实例和权重留在本地。

## 1. 不下载模型即可复算

```bash
python -m pip install numpy==1.26.4 Pillow==12.0.0
python -m m0.validate results/m0-20261102
python -m m0.summarize \
  results/m0-20261102/siglip-predictions.jsonl \
  results/m0-20261102/clip-predictions.jsonl \
  results/m0-20261102/qwen-predictions.jsonl \
  --out summary.recomputed.json
python -m unittest discover -s tests -p 'test_m0*.py' -v
```

`validate` 检查清单、权重指纹、逐样本文本/图像身份、官方标签、划分、排除项及完整覆盖；缺行、重复行和未完成运行不能通过。`summarize` 从原始分数重新计算指标，不信任预测文件内预存的 `metrics`。不要将 smoke 和 full 一起输入：重复 ID 会报错。CPU 测试与模型语义能力是两类证据。

## 2. 获取固定来源并重建样本

```bash
git clone https://github.com/Sri-Harsha/scpp external/scpp
git -C external/scpp checkout c083bbf039a5ca215acaf25e7ee736d2ea6f3864
git clone https://github.com/maximek3/e-ViL external/e-ViL
git -C external/e-ViL checkout 94f17f5c189573c9d1087f82eb27db37690fc810
# 从官方 VWSD 页面取得 test v1.1 查询/答案，解压到 external/vwsd_queries。
python -m m0.prepare --scpp external/scpp/data \
  --esnli external/e-ViL/data/esnlive_test.csv \
  --vwsd external/vwsd_queries --out data/local/m0-reproduction
cp results/m0-20261102/exclusions.json data/local/m0-reproduction/exclusions.json
cmp data/local/m0-reproduction/manifest.jsonl results/m0-20261102/manifest.jsonl
```

目标私有清单 SHA256 为 `9a2452230030e0851bbdbca480d49d0cca3c5190e600fd5efd43823b797aad98`，公开清单为 `6a78fb30fe13551fa8717702cedb66882833c418979cc5cb66e36c554317b156`。文本不改大小写、标点或拼写。代码用 LF 输出；不同 Git 换行设置导致原始来源字节散列不同，应比较 `normalized_source_sha256` 和最终清单，不应偷偷替换清单。原先冻结的 `exclusions.json` 保留实跑的原始 CRLF 字节（Git 标为 `-text`），其 SHA256 为 `a3dde1f8df57100abd873c93fe7f2e98c2ffe903735bd8f71751ecc35d115177`，请直接复制，勿重排或重格式化。输出目录必须不存在，以避免覆盖既有冻结文件。

按官方数据许可获取图片。可复用已有 Flickr30K 原图目录，也可使用 SNLI-VE 官方脚本指向的公开归档。VWSD 使用官方 resized test image ZIP。Google Drive 文件 ID 和许可在来源文档中；不将本仓库当作原图分发站。

```bash
python -m m0.download images \
  --manifest data/local/m0-reproduction/manifest.private.jsonl \
  --out data/local/m0-reproduction/images \
  --flickr-root /path/to/flickr30k-images \
  --vwsd-archive /path/to/test_images_resized.zip
# 没有原图目录时可改为 --flickr-archive /path/to/flickr30k_images.tar.gz。
python -m m0.audit --manifest data/local/m0-reproduction/manifest.private.jsonl \
  --images data/local/m0-reproduction/images --out image-audit.recomputed.json
```

必须无 missing、无 active byte duplicates；本轮为 `passed_with_quarantine`，保留 4 个原始 VWSD 排除单元，禁止补抽。核对公开 image audit 的 640 个文件指纹；不同压缩/分辨率的同名图片不是本轮输入。

## 3. 推理环境与模型

建议新建隔离环境。实跑 Python 3.11.15、PyTorch 2.6.0+cu124、torchvision 0.21.0+cu124；完整相关依赖见 `requirements/m0-runtime.txt`。Qwen 的 AutoProcessor 即使只输入图片也需要 torchvision。首次小试因缺此依赖失败，未产生任何模型答案；修复后使用新运行目录重试，未改提示或样本。

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements/m0-runtime.txt
python -m pip install -e . --no-deps
python -m m0.download model siglip --out models/m0-siglip
python -m m0.download model clip --out models/m0-clip
python -m m0.download model qwen --out models/m0-qwen
python -m m0.download wordnet --out data/local/nltk_data
```

所有模型 revision 固定在下载代码中。SigLIP/Qwen 实跑权重通过 ModelScope 中转，逐文件 SHA256 与固定 Hugging Face LFS 元数据一致；CLIP 是同一固定 HF 原文件经本地中转。镜像不构成换模型。预期权重散列见 `m0.validate.WEIGHTS`；WordNet 存档散列见 `qwen-run.json`。复现时应核对它，NLTK 的移动下载分支不是内容版本保证。

本服务器只授权物理 GPU 4/5。脚本先用 `nvidia-smi` 解析 UUID，再导入 torch，并在 `cuda:0` 加载，禁止 auto 分配或 CPU 静默回退。其他机器需按其实际授权设备显式修改此安全边界；原运行 UUID 保存在记录中。

## 4. 先小试，再全清单

以下为普通 Linux 环境命令。实跑使用独立依赖目录，等效地设置 `PYTHONPATH=<isolated-deps>:.`；未修改旧实验环境。

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m m0.run_dual --manifest data/local/m0-reproduction/manifest.private.jsonl \
  --images data/local/m0-reproduction/images --model siglip --model-dir models/m0-siglip \
  --gpu 4 --split exploration --limit 20 --out runs/m0-siglip-smoke
# 小试仅检查加载、身份、有限分数和有效分母；不得根据成绩改提示。
python -m m0.run_dual --manifest data/local/m0-reproduction/manifest.private.jsonl \
  --images data/local/m0-reproduction/images --model siglip --model-dir models/m0-siglip \
  --gpu 4 --out runs/m0-siglip-full
# CLIP 同样先跑 --split exploration --limit 20，再去掉这两个参数跑 full；
# 将 --model 改为 clip、--model-dir 改为 models/m0-clip，使用全新输出目录。
python -m m0.run_joint --manifest data/local/m0-reproduction/manifest.private.jsonl \
  --images data/local/m0-reproduction/images --model-dir models/m0-qwen \
  --wordnet data/local/nltk_data --gpu 5 --split exploration --limit 20 --arms K0 \
  --out runs/m0-qwen-smoke
python -m m0.run_joint --manifest data/local/m0-reproduction/manifest.private.jsonl \
  --images data/local/m0-reproduction/images --model-dir models/m0-qwen \
  --wordnet data/local/nltk_data --gpu 5 --out runs/m0-qwen-full
```

正式 Qwen 默认 7 条件 × 120 单元 = 840 次；成功小试另 20 次，总计 860 次，无付费推理 API。运行目录不覆盖；失败目录保留。只有写出 `run.json` 才视作完成，不能把部分 JSONL 当全量结果。不同硬件/库的数值可能略变，因此比较指标与逐项分数，并保留新的运行身份。

## 5. 语义复核规则

公开 `ai-review.jsonl` 为单个 Codex 对实际图片的诊断记录，不是人类金标。取确认集 SugarCrepe++ 的 6 个成功和 6 个失败、VWSD 的 2+2 个、e-SNLI-VE 每类 4 个；各池按 ID 排序后用 seed=2026 抽样，再打乱展示。前两项的选择有结果条件，不能用这 28 项估计全数据集噪声率。复核时不同时展示模型分数，但并非独立双盲实验；原 ID 也可能透露标签。

之后追加全部 3 个两模型共享确认 XOR 单元，以及尚未在上述队列中的 1 个 K2/K0 改变单元，总计 32 项。追加记录有单独 `review_type`，属于已知结果的事后案例分析。

人类后续审核应先看原图和完整文本，记录：正例是否成立、改写是否等义、负例是否也可成立、否定是否有直接视觉证据、是否需要隐藏身份/意图/未来事件、候选义项是否真的匹配。单独填写 `human_review`、理由与分歧，不覆盖 `gold`。困难项应由第二位真实评审独立判断。

人工 Oracle 尚未执行：本轮没有可声称为人类核验的正确义项/局部证据。后续上限实验须由人类在不看预测的情况下选择相关 WordNet sense/edge，固定后作为独立条件；不得用本轮 AI 意见伪装 Oracle，也不得根据确认集成绩重新选择知识。
