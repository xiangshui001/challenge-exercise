# 04｜实施指引：从方案到可运行实验

本节保留原实施方案；表中的独立脚本名仍是规划。当前实际实现为 exp5 包的 CLI，使用[基础版运行说明](10_RUNNING_BASELINE.md)中的命令，CPU 证据见[验证记录](11_CPU_VALIDATION.md)。真实模型与用户 A100 尚未验证。实施前阅读[评测协议](03_EXPERIMENT_PROTOCOL.md)。

## 1. 环境策略

建议在 A100 所在的 Linux 服务器建立独立 Python 3.11 环境。先运行 `nvidia-smi` 确认 GPU 型号、显存、驱动和其他占用进程。不要仅根据系统安装的 CUDA toolkit 版本决定 wheel，也不要直接修改共享服务器驱动。

PyTorch 官方列出了 torch 2.8.0 与 torchvision 0.23.0 的 CUDA 12.6 安装组合；Qwen 官方示例依赖 torch 2.8.0、Transformers 4.57 系列或以上版本。本方案先采用下面的保守起点，而不是追逐最新主版本。[S3][T2]

```bash
conda create -n challenge-exp5 python=3.11 -y
conda activate challenge-exp5

# 仅在服务器驱动支持该 CUDA wheel 时采用此组合。
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu126

python -m pip install transformers==4.57.3 accelerate \
  numpy pandas scipy scikit-learn matplotlib pillow tqdm \
  huggingface_hub datasets sentencepiece protobuf pyyaml

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

已有稳定环境可以优先复用，不必机械重装。首次通过模型小样本检查后执行 `python -m pip freeze > requirements-lock.txt`，并记录驱动、GPU 显存、模型 revision。版本锁应由成功的小试生成；本仓库不提供冒称已经验证的环境锁文件。

重排是可选依赖。需要时从官方仓库取得封装并固定源码版本：

```bash
mkdir -p external
# 在项目根目录执行；external 已加入 .gitignore。
git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git external/Qwen3-VL-Embedding
python -m pip install 'qwen-vl-utils>=0.0.14'
```

按所固定版本的官方 README 导入 `Qwen3VLReranker`；不能只安装 transformers 后就假设 `scripts.qwen3_vl_reranker` 已存在于本项目。适配层负责添加明确的上游路径或采用上游推荐的包安装方式。[T2]

默认使用 PyTorch 原生 attention/SDPA，暂不编译 FlashAttention，不引入 vLLM 服务、量化和 TensorRT。若依赖排查超过约 2 小时，先分离重排环境或回退基础版，而不是把环境重构变成新的主任务。

## 2. 数据获取与整理

官方数据说明和 Karpathy 项目链接在 [D1/D2](../references/SOURCES.md#d1)。优先获取图像及既有划分 JSON。为节省下载整理，也可以使用 [D3 镜像](../references/SOURCES.md#d3)，但必须保留来源说明并核对内部划分。

`nlphuji/flickr30k` 的外层 viewer 标为 `test`，约 31k 行，记录内部另有 `split` 字段。不能把外层整个分区当作标准 1000 张测试集。按内部划分筛选后，还需与选定 Karpathy 样本 ID 核对；镜像内容和原始图像分辨率可能有差异，应写入数据记录。[D3]

数据层输出三份文件：`images.jsonl`（image_id、path、split）、`captions.jsonl`（caption_id、text、image_id、split）和 `data_audit.json`。审计记录样本数量、损坏文件、空文本、重复 ID、跨划分重叠和正例映射错误。未通过审计不得运行正式指标。

建议按 image_id 固定排序，caption 使用原始 sentid；没有稳定 sentid 时显式生成 `{image_id}:{caption_index}`，保存映射规则，不依赖 DataLoader 的随机顺序。

## 3. 最小模块划分与接口

| 待实现模块 | 输入 | 输出 | 首要测试 |
|---|---|---|---|
| `prepare_data.py` | 数据路径与划分 JSON | manifests、data_audit.json | ID 唯一、划分不交叉、正例存在 |
| `encode.py` | manifest、模型与 processor 配置 | 两种特征、ID 顺序、编码日志 | 形状、有限数值、范数与顺序 |
| `evaluate.py` | 分数/排名、正例映射 | 汇总表与逐查询表 | 单正例、多正例、并列分数 |
| `fuse.py` | 两模型排名、权重 | F1 排名 | w=0/1 回到各自单模型 |
| `rerank.py` | Top-M、查询与图像路径 | 缓存的相关性得分 | 不传入真值、不丢候选 |
| `selective.py` | B1 分数、阈值与重排缓存 | 门控/随机结果、成本记录 | 不读取标签、预算计数正确 |
| `analyze.py` | 各方法逐查询结果 | bootstrap、差异与案例清单 | 配对分组一致、图库不变 |

基础版只需要前三个模块和结果分析；F1 及重排按需添加。项目不需要复杂服务分层。任何脚本成功运行后，记录其实际命令到 run_record，而不是把设计中的示意命令当作运行证据。

## 4. 编码实现要点

CLIP 使用 CLIP 对应 AutoProcessor/CLIPModel；SigLIP 2 使用自身 AutoProcessor/AutoModel 或所固定文档支持的对应类。两个模型分别加载、编码、保存、卸载。不要直接把 CLIP 的处理器交给 SigLIP 2。

通用流程示意如下，具体模型输入交由适配器处理：

```python
model.eval()
with torch.inference_mode():
    # adapter 必须调用模型原生图像/文本特征接口。
    raw = adapter.encode(batch)
    features = torch.nn.functional.normalize(raw.float(), dim=-1)
    # 保存 CPU 特征和与之严格同序的 item_ids。
```

建议初始图像 batch：CLIP=64、SigLIP 2=32；文本 batch=128。BF16 可作为 A100 推理起点，特征保存为 FP32。批量大小不是保证，先用 20 张验证图像和 100 条描述做小试，再据峰值显存调节。OOM 时先减 batch，不更改正式模型输入分辨率来偷偷改变方法配置。

SigLIP 2 文本小写、固定 64 token padding 与截断处理见官方文档。[S2] 统计截断数量。图像只按处理器默认配置处理，不自行加随机裁剪。初始化、输入转换和模型处理器 revision 全部记录。

## 5. 指标先行，避免“模型跑完才发现算错”

使用小型人工构造排名测试指标：正例第一名时 R@1=1；正例第 7 名时 R@5=0、R@10=1；I2T 五个正例只找回一个时，命中式 R@K=1，但集合召回为 0.2；理想排序 NDCG=1；缺正例或重复候选 ID 应报错。

NDCG 的 IDCG 依据完整正例数计算。若 sklearn 默认并列得分处理与本项目固定 ID 次序不同，应先生成确定性排名得分再交叉核对，而非让两个实现各自采用不同 tie 规则。参考检索指标实现和 NDCG 接口分别见 [S1/S4](../references/SOURCES.md#s1)。

每次评测保存每条查询的正例排名、hit@1/5/10、NDCG@10。报告指标必须可由该文件重新计算，而不必再次加载大模型。

## 6. 重排适配与失败处理

从官方示例确认固定版本的输入结构。查询提供文本，候选提供图片；不把候选人工描述作为额外模型输入，不将 image_id 文本化为可能泄漏答案的提示。固定一个用于图文相关性匹配的英文任务 instruction，并记录原文。先采用官方默认图像像素限制，记录配置，不在测试时逐查询调分辨率。

100 条 val_tune 查询 × 50 个候选 = 5000 图文对的小试。记录每批耗时、总配对数、峰值显存、重试数及输出有限性。若受资源限制缩短小试，记录实际样本与抽样方式，不混入正式全量结果。

本地失败批次可减 batch 后有限重试。某查询最终不完整时，不对部分候选进行有偏重排，整条查询回退 B1 并标注失败；该查询仍计入指标分母。API 超时或预算耗尽使用同一回退规则。

缓存键至少包含模型及源码 revision、查询原文、图片内容指纹/版本、processor 配置、instruction、输入像素限制。候选次序改变不能导致分数错配。缓存保存 query_id 和 candidate_id 的显式键，禁止只靠数组位置关联。

## 7. 推荐执行顺序

首先实现数据审计及指标单元测试；通过后用小样本跑通 B0/B1；再完成验证集特征、指标和错误清单。选择基础版配置后可以直接冻结、评测试集并写报告。

研究版在验证阶段新增 R1 小试，收益和预算允许时生成全查询重排分数，再计算 G/RAND。最后冻结全部方案，在同一正式测试集运行已注册对照。若模拟使用全量缓存，另测最终选定策略的一次真实路径，才可给出实际时间收益。

## 8. 故障排查

| 故障 | 优先处理 |
|---|---|
| CUDA 不可用 | 查看 nvidia-smi、torch wheel、驱动与容器 GPU 映射；不要先改模型 |
| 图片编码 OOM | 检查 inference_mode、batch、多模型常驻；减 batch |
| Qwen 导入封装失败 | 检查官方 repo 是否已取得及 Python 路径；它不是本仓库已有文件 |
| 新库更新导致接口变化 | 固定已通过小试的版本；查相同版本文档，不混用新旧示例 |
| 数据或模型下载慢 | 使用用户授权的镜像或预下载目录；保留来源，不把包提交 Git |
| 指标远低于参考 | 查图库是否误为 31k、五描述映射、预处理与特征顺序 |
| R@10 异常满分 | 查候选库是否只有 10 项、输入是否混入标准答案 |
| 重排没有改善 | 查 R@50 上界和修复/退化数，保留负结果，不无止境增加模型 |
| 中文查询效果变化 | 主实验保持原始英文；中文可单列外部扩展，不与英文主表混算 |
