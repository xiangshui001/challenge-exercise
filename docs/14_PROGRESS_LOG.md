# 14｜阶段进度与同步日志

不把规划、代码完成、CPU 测试、真实模型运行混为同一个状态。每完成一个可复现阶段，就追加或更新记录并提交 GitHub。

| 日期 | 阶段 | 真实完成事项 | 服务器状态/下一步 |
|---|---|---|---|
| 2026-09-30 | 原评测基础版 | PR #2 已合并（7932413）；审计、指标、缓存、RRF、冻结/测试/统计代码；24 项 CPU 测试通过 | 真实 Flickr30K/CLIP/SigLIP 2/A100 未运行 |
| 2026-09-30 | 主路线调整与框架 | 用户确认 SigLIP-base + 成熟数据 + 简单部分微调；建立统一目录、计划和同步约定 | PR #3 工作分支，服务器 checkout 尚未同步确认 |
| 2026-09-30 13:56（上海） | P0 核心环境验证 | 用户服务器日志：pip check 无冲突；Python 3.11.16；torch 2.8.0+cu126；Transformers 4.57.3；CUDA runtime 12.6；CUDA 可用；可见卡数 1；A100-SXM4-80GB；2×2 全一矩阵乘积为全 2 | 环境检查通过；尚未加载模型、下载数据或训练。完整依赖锁和所选 UUID 待 P1 |
| 2026-09-30 | GPU 使用约定 | 用户澄清可按空闲情况和任务需要使用多张卡，按需使用、用完释放；卡 2 不是长期固定限制 | 当前小试仍用单卡，运行参数可选择其他卡；分布式训练尚未实现 |
| 2026-09-30 | P1 小试代码 | 基于工作分支 364fcf5 接入原生 SigLIP-base 封装、短小试、固定模型 SHA、参数名与成本输出；CPU 检查见 [15](15_SIGLIP_SMOKE.md) | 真实权重小试待用户服务器运行；合成图片结果不能算检索成绩 |
| 2026-10-01 | 模型下载 | 用户服务器直连 huggingface.co 返回 Network is unreachable；通过 HF-Mirror 成功下载 SigLIP-base 的 7 个文件，model.safetensors 约 813 MB，revision `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`，保存于默认 HF 缓存对应 snapshots 目录 | 已下载不等于加载验证；下一步显式使用该 SHA、local_files_only 和所选 GPU 运行短小试。外部聊天提供的独立 test_siglip.py 尚无执行日志，不计为已完成 |
| 2026-10-16 | P1 真实模型小试 | 用户服务器输出 `runs/p1-smoke-20261016/smoke.json`：`status=smoke_passed`，固定 `google/siglip-base-patch16-224@7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`，代码 `9348b443`；单张 A100-SXM4-80GB（物理索引 2，进程内 cuda:0）FP32 推理完成。输入 1 张合成红色方块及 2 条文本；输出 1×768/2×768，`cosine(red)=0.109220 > cosine(blue)=0.040971`；参数 203,155,970；推理墙钟 2.158 s（含预处理与传输），GPU 峰值 allocated 829,237,760 B、reserved 926,941,184 B；CPU 阶段模型加载记录 407.845 s | **P1 已通过（接口验收）**；不代表真实 Flickr30K 基线或训练完成。原始完整日志、参数名与依赖锁留服务器 `runs/`，不提交仓库。下一步审计真实 Flickr30K；PR #4 数据数量修复尚未合并，不能直接用当前分支默认 1000 val 模式审计原作者 1014 val |
| 2026-10-16 | P2 数据解压与抽样定位 | 用户服务器输出：作者 `caption_datasets.zip` 已解压，Karpathy JSON 分组 `train=29000, val=1014, test=1000`；`flickr30k-images.zip` 经 HF 镜像下载并已解压。ZIP 列出 63,566 个图片后缀条目（包括 `__MACOSX` 元数据条目，不能当作真实图片数）。分别抽查 train/val/test 各一张，实际文件均位于 `data/local/raw/flickr30k-images/` | **仅下载、解压和三图路径抽查成功**；全部 2,014 张验证/测试图片尚未运行审计，训练集 29,000 张也尚未审计。接下来在 PR #4 的 1014-val 修复代码中运行 `prepare`，保留原训练分支与已有 P1 日志 |
| 2026-10-16 | P2 真实数据审计通过 | 用户服务器 `/workspace/challenge-exp5/repo-audit` (`9f4d080`) 调用 `exp5 prepare --expected-val-images 1014`，输出 `data/local/manifests-20261016-084017/data_audit.json` 的 `status=passed`、`standard_protocol=true`；val=1,014 图/5,070 描述，test=1,000 图/5,000 描述，跨两集合无字节重复图片、4 组重复文本；审计数据指纹 `d148fa170c3cc68f37deee629d8abef4ef2340171a578b4907184d5d7675f290` | **P2 评测数据验收通过**；训练集 29,000 图尚未全审计。新分支 `feature/siglip-base-val-baseline` 已移植 1014-val 修复并加入 S0 编码和独立验证评估；**代码及真实 S0 验证成绩仍待服务器测试**，不将其视为已运行 |
| 2026-10-16 | P2 集成分支服务器 CPU 验证 | 用户在服务器 worktree `/workspace/challenge-exp5/repo-baseline` 检出 commit `65e1d38`，在 `challenge-exp5` 环境执行 `python -m unittest discover -s tests -v`；实际结果 **Ran 34 tests in 52.899s / OK**。覆盖 1014-val 原始计数、数据缓存/审计、评价统计、历史合成冻结流程、S0 独立身份和命令行，以及 P1 小试接口测试。原始日志 `runs/s0-cpu-tests.log` 留在服务器 | **34/34 CPU 测试通过**；尚未在真实 Flickr30K 上运行 S0 GPU pilot 或全量验证，不得将 CPU 测试当真实检索指标。下一步先确认可共享 GPU，再做 20 图 100 描述 pilot |

## 下一节点

P1 真实 SigLIP-base 小试已由服务器日志确认通过。Flickr30K 原作者标注及图片已就位，1014-val/1000-test 正式审计已通过，指纹如上。新集成分支加入 S0 SigLIP-base 编码/完整验证指标入口，下一步在服务器运行 CPU 测试、20 图小试和全量验证评估，实测后方可填入 Recall。训练 Dataset、微调循环和 checkpoint 评估仍未实现。

## 后续每次记录

```text
日期与阶段：
起始 commit/分支：
改动文件：
执行位置：云端 CPU / 用户服务器
实际命令与配置：
数据/模型/依赖版本：
真实输出与结果文件相对路径：
检查结果、失败或未运行项：
下一步：
```

记录本次变更以起始 commit 和提交消息定位；本次 commit SHA 在提交后由 Git 产生，不在提交内容中虚构自引用 SHA。
