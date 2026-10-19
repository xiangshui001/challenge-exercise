# 10｜基础版 CLI 与 A100 小试

实现了 Issue #1 的基础评测代码，CPU 合成测试已通过。用户服务器依赖与四卡 BF16 小矩阵检查通过，2026-10-19 真实图片审计通过；**CLIP/SigLIP 2 推理尚待执行**。以下是交互式执行步骤；安装或导入不会自动启动任务。每次使用新的输出目录，程序拒绝覆盖旧运行。

## 1. 获取与安装

```bash
git clone https://github.com/xiangshui001/challenge-exercise.git
cd challenge-exercise
python -m pip install -e '.[test]'
python -m unittest discover -s tests -v
python -m exp5 --help
```

基础代码已在 `main`，不再要求切换旧功能分支。建议使用独立 Python 3.11 环境。CPU 路径只需 NumPy/Pillow；模型命令才导入 torch/transformers。用户 2026-10-11 提供的终端记录已确认下面依赖安装、`pip check`、原有 24 项测试及四卡 BF16 小矩阵检查通过；真实模型适配仍待验收：

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e '.[models]'
```

安装入口见 [PyTorch 官方版本表](https://pytorch.org/get-started/previous-versions/)。可优先复用稳定环境，不修改驱动。首次成功编码会写实际 `requirements-lock.txt`，本 PR 不提供冒称通过 GPU 验证的环境锁。

## 2. 本地数据审计

从[来源说明](../references/SOURCES.md)取得经允许使用的 Karpathy JSON 和图片；具体下载、解压及原始 1014 验证图运行步骤见 [12](12_DATA_PREPARATION.md)。当前不提供自动下载器或镜像转换器。

```bash
python -m exp5 prepare \
  --annotations /你的数据目录/dataset_flickr30k.json \
  --image-root /你的数据目录/flickr30k-images \
  --expected-val-images 1014 \
  --source-revision '实际数据来源和版本' --out data/local/manifests
```

路径为 `image-root / filepath / filename`，不要重复加目录。保留 `imgid`、`sentid`，无 sentid 时生成 `image_id:caption_index`。要求 val/test 数量与声明匹配：作者原始包为 1014/1000 图、每图五描述。默认验证数量仍为 1000，使用作者包时必须显式传 `--expected-val-images 1014`；不得删图凑数。坏图、重复 ID、空文本、缺正例、跨划分相同图片字节、sentids 错序和路径越界都会停止。输出 manifests、按图分的 val_tune/val_confirm、审计及数据 SHA256。两验证组查询面对完整验证图库。

同划分重复图片字节与重复描述会记录，不按模型效果删样本。字节哈希不是感知重复检测。使用已确认的 Karpathy 文件，数量吻合本身不能证明来源正确；记录来源版本与原 JSON 哈希。`--small` 仅用于构造样例，无法进入冻结/正式测试。

## 3. 固定模型 revision

2026-10-19 已读取两个官方模型仓库的 Hub API 元数据，固定以下版本，不需执行者手工查询或计算文件哈希：

```bash
export EXP5_B0_REV='57c216476eefef5ab752ec549e440a49ae4ae5f3'
export EXP5_B1_REV='e8e487298228002f3d8a82e0cd5c8ea9c567f57f'
```

CLI 拒绝空 revision 和 `main`。默认只用已有 Hugging Face 缓存，只有显式 `--allow-download` 才能获取权重。两个模型分别用自己的 processor；SigLIP 2 小写、64 token、max_length padding，原始描述不添加分类模板。依据固定版本的 [CLIP 文档](https://huggingface.co/docs/transformers/v4.57.3/en/model_doc/clip) 和 [SigLIP 2 文档](https://huggingface.co/docs/transformers/v4.57.3/en/model_doc/siglip2)。

服务器通过已验证的电脑代理隧道下载模型时，在同一终端设置以下变量；仅影响当前 shell 及其子进程，保持电脑 SSH 隧道窗口打开：

```bash
export http_proxy=http://127.0.0.1:17890
export https_proxy=http://127.0.0.1:17890
export no_proxy=localhost,127.0.0.1
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_ETAG_TIMEOUT=30
export HF_HUB_DOWNLOAD_TIMEOUT=120
```

## 4. GPU 1/2：20 图、100 描述小试

编号为 `nvidia-smi` 中的物理索引，从 0 起数。用户 2026-10-19 审计快照显示 GPU 0/3 已分别占用 59770/30625MiB，GPU 1/2 各约 10MiB；因此安排 B0 在 GPU 1、B1 在 GPU 2，执行前重新读取占用。程序不清理其他进程。启动器先将编号解析为 UUID，再设 `CUDA_VISIBLE_DEVICES`，每个进程中选定的卡均为 `cuda:0`。以下命令按顺序执行，编码成功后再评测缓存。

```bash
nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,memory.total,memory.used,utilization.gpu --format=csv

python scripts/on_gpu.py --index 1 -- \
  --data data/local/manifests --method B0 --revision "$EXP5_B0_REV" \
  --split val --pilot-images 20 --image-batch 16 --text-batch 64 \
  --out cache/B0-pilot --allow-download &&
python -m exp5 pilot-evaluate --data data/local/manifests \
  --cache cache/B0-pilot --out runs/B0-pilot

python scripts/on_gpu.py --index 2 -- \
  --data data/local/manifests --method B1 --revision "$EXP5_B1_REV" \
  --split val --pilot-images 20 --image-batch 16 --text-batch 64 \
  --out cache/B1-pilot --allow-download &&
python -m exp5 pilot-evaluate --data data/local/manifests \
  --cache cache/B1-pilot --out runs/B1-pilot
```

已有权重时删除 `--allow-download`。OOM 时降低 batch、换新输出目录，不静默丢样本。检查成功输出、shape、有限性、范数、ID 顺序、截断计数、显存及依赖锁。小试固定选前 20 张验证图，候选库缩小；结果标为 `pilot_only`，不能用它声称正式 R@10 达标。

## 5. 全量验证、选择与冻结

```bash
python scripts/on_gpu.py --index 2 -- \
  --data data/local/manifests --method B0 --revision "$EXP5_B0_REV" \
  --split val --image-batch 16 --text-batch 64 --out cache/B0-val
python scripts/on_gpu.py --index 2 -- \
  --data data/local/manifests --method B1 --revision "$EXP5_B1_REV" \
  --split val --image-batch 16 --text-batch 64 --out cache/B1-val
python -m exp5 select --data data/local/manifests \
  --b0 cache/B0-val --b1 cache/B1-val --fusion --out runs/validation
```

删除 `--fusion` 则只比较 B0/B1；开启则加入预注册的 RRF 权重 `{0,.25,.5,.75,1}`。按 val_tune T2I R@10、NDCG@10 选择；同质量优先单模型，最后固定优先 B1。只报告一次 val_confirm，不据此返调。两方向 RRF 独立重算。`frozen.json` 锁定数据、模型/processor/依赖/代码指纹、方法、指标和统计口径，带校验和。冻结后不要改代码或依赖。

## 6. 独立测试与复算

```bash
python scripts/on_gpu.py --index 2 -- \
  --data data/local/manifests --method B0 --revision "$EXP5_B0_REV" \
  --split test --frozen runs/validation/frozen.json \
  --image-batch 16 --text-batch 64 --out cache/B0-test
python scripts/on_gpu.py --index 2 -- \
  --data data/local/manifests --method B1 --revision "$EXP5_B1_REV" \
  --split test --frozen runs/validation/frozen.json \
  --image-batch 16 --text-batch 64 --out cache/B1-test
python -m exp5 test --data data/local/manifests \
  --b0 cache/B0-test --b1 cache/B1-test \
  --frozen runs/validation/frozen.json --out runs/test
python -m exp5 recompute runs/test/B1.t2i.queries.jsonl
```

输出 B0/B1 和被选中的 F1（若有），双向 R@1/5/10/20/50、二值 NDCG@10、集合召回、逐查询正例排名和 Top-50、2000 次按图分组 bootstrap 与配对差值、修复和退化案例。主目标只看冻结方案的 T2I R@10≥.85，不达标也如实保留。人工语义错误分类仍需执行者复核，机器变化清单不等于人工分析完成。

墙钟编码时间包含 processor/传输，不含加载/下载；加载另记。记录 PyTorch 峰值 allocated/reserved。GPU event 时间未测，保持 null；不把缓存评测时间冒称模型推理时间。

## 7. 支持范围

| 项目 | 状态 |
|---|---|
| prepare / encode / pilot-evaluate / select / test / recompute | 已实现 |
| 数据审计、FP32 归一化缓存、B0/B1、双向指标、RRF、冻结和统计 | 代码已实现，CPU 路径已测 |
| configs/experiment5.plan.yaml | 仍为方案，不作为运行配置；CLI 不接受 --config |
| Qwen 重排、门控、收费 API、训练、ANN、前后端 | 未实现，不自动运行 |
| 真实数据 | 2026-10-19 服务器审计通过：1014/1000 图、5070/5000 描述 |
| 真实模型小试和正式指标 | 等服务器实际执行 |

成功查 `metadata.json`、`requirements-lock.txt`，失败查 `failed.json`。数据/权重/缓存/逐查询日志放 gitignore 本地目录；审查后才上传允许公开的小型汇总。CPU 验证证据见 [11](11_CPU_VALIDATION.md)。
