# 16｜SigLIP-base 原始验证集零微调基线（P2）

本次实现基于 SigLIP-base（独立代号 **S0**）而非历史评测的 B1/SigLIP 2。所有 Flickr30K 图文检索指标复用 `exp5.metrics`，不另写评测公式，不运行测试集。

## 数据和边界

2026-10-16，用户服务器已完成原作者 Karpathy 标注和 Flickr30K 图像解压及正式验证/测试图片审计。实际审计记录位于服务器被 Git 忽略的 `data/local/manifests-20261016-084017`，状态 `passed`，val=1014/5070，test=1000/5000；图片无跨 val/test 字节重复，4 组重复 caption 文本已记录，**训练集尚未全量审计**。

固定标注 SHA256：`db779ee2c5df40c25ff0eebeca464f6124b620cc4a9398650aea71fd1b53d4af`；审计数据指纹：`d148fa170c3cc68f37deee629d8abef4ef2340171a578b4907184d5d7675f290`。不要使用历史 1000-val 假设截断原数据。

固定 SigLIP-base 模型：`google/siglip-base-patch16-224@7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`，P1 smoke 通过（合成图片验证特征接口，不代表实际 Recall）。

## 在服务器使用独立 Git worktree

现有 `repo` 训练分支和 `repo-audit` 不动。执行：

```bash
conda activate challenge-exp5
MAIN=/workspace/challenge-exp5/repo
BASE=/workspace/challenge-exp5/repo-baseline
git -C "$MAIN" fetch origin
git -C "$MAIN" worktree add --detach "$BASE" origin/feature/siglip-base-val-baseline
cd "$BASE"
python -m unittest discover -s tests -v
python -m exp5 encode --help
python -m exp5 baseline-evaluate --help
```

若目标已存在，先检查 `git worktree list` 和目录，不强制覆盖。完成 CPU 测试前不启动 GPU。

## S0 20 图 / 100 描述接口小试

先使用 `nvidia-smi` 检查所选 GPU 是否允许共享、剩余显存及利用率；命令的 `--index 2` 仅为示例，不表示该卡始终空闲或自动获得授权。

```bash
cd /workspace/challenge-exp5/repo-baseline
# P2 真实 pilot 发现 AutoProcessor 仍可能对 main/tokenizer_config.json 发起 HEAD。
# 权重已被 P1 和 pilot 从本机缓存成功加载，后续固定 revision 全部离线。
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
MAIN=/workspace/challenge-exp5/repo
DATA="$MAIN/data/local/manifests-20261016-084017"
REV=7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed
PILOT="$MAIN/cache/S0-pilot-$(date +%Y%m%d-%H%M%S)"
python scripts/on_gpu.py --index 2 -- \
  --data "$DATA" --method S0 --revision "$REV" \
  --split val --pilot-images 20 --dtype float32 \
  --image-batch 8 --text-batch 32 --out "$PILOT"
python -m exp5 pilot-evaluate --data "$DATA" --cache "$PILOT" \
  --out "$MAIN/runs/S0-pilot-$(date +%Y%m%d-%H%M%S)"
```

严格检查 PILOT `metadata.json`：`method=S0`，正确模型/revision，特征长度图像 20 / caption 100，feature 为有限、L2 归一的 FP32。这里的 Recall 仅是缩小候选库的接口指标，不是正式 Flickr30K 指标。失败优先查看 `failed.json`，不在原输出目录重试。

## 完整验证集（小试成功后）

```bash
cd /workspace/challenge-exp5/repo-baseline
# P2 真实 pilot 发现 AutoProcessor 仍可能对 main/tokenizer_config.json 发起 HEAD。
# 权重已被 P1 和 pilot 从本机缓存成功加载，后续固定 revision 全部离线。
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
MAIN=/workspace/challenge-exp5/repo
DATA="$MAIN/data/local/manifests-20261016-084017"
REV=7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed
CACHE="$MAIN/cache/S0-val-$(date +%Y%m%d-%H%M%S)"
RUN="$MAIN/runs/S0-val-$(date +%Y%m%d-%H%M%S)"
python scripts/on_gpu.py --index 2 -- \
  --data "$DATA" --method S0 --revision "$REV" \
  --split val --dtype float32 --image-batch 8 --text-batch 32 \
  --out "$CACHE"
python -m exp5 baseline-evaluate --data "$DATA" --cache "$CACHE" --out "$RUN"
cat "$RUN/metrics.csv"
```

完整指标是 **5,070 文本查询、1,014 图像图库** 的 T2I，以及 **1,014 图像查询、5,070 描述图库** 的 I2T。输出 `S0.t2i.queries.jsonl`、`S0.i2t.queries.jsonl`、双向 bootstrap、`metrics.csv` 和 `run_record.json`。不要把该验证结果当独立测试成绩，也不要把查询图库减到少数图片来宣称达到 R@10≥85%。

## 真实 P2 pilot 验收记录（2026-10-16）

服务器 `repo-baseline`（commit `65e1d38`）已完成 34/34 CPU 测试；20 图 / 100 描述、GPU 2、FP32 的真实 S0 编码输出 `status=encoded`，并成功运行 `pilot-evaluate`：T2I R@1=0.95、R@10=1，I2T R@1=1、R@10=1，**仅是缩小候选库的 pilot_only 结果**。缓存 `cache/S0-pilot-20261016-132449`，评测 `runs/S0-pilot-eval-20261016`。模型编码耗时 1.22 秒（不含加载/联网重试），CUDA allocated 峰值 897,119,744 B，截断 0。下一步是全部验证集，尚无正式性能数据。

## 当前支持边界

- **已实现代码**：审计 1014-val 的整合、S0 编码/cache、20 图小试和全量 S0 验证评价（待服务器执行验证）。
- **尚未实现**：训练集 Dataset、部分微调、checkpoint 评估、微调配置冻结与独立 test 的 S0 支持。
- 不使用 `exp5 select` / `exp5 test` 旧 B0/B1 流程来冒充 S0 测试；该流程需要在后续阶段扩展正式锁定支持。
- 保留所有失败日志；若本分支的 CPU 测试或模型加载失败，停止并反馈，不修改已通过 P1 的旧代码/权重。
