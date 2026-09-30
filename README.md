# challenge-exercise

## 实验五：图文语义匹配评估与优化

**拟定题目：预训练视觉语言模型驱动的图文语义匹配评估与预算约束重排优化。**

本仓库只服务于实验五，不开展实验一、二、三、四、六。当前包含**方案包与基础版 CLI**：数据审计、CLIP/SigLIP 2 编码适配、双向指标、可选 RRF、验证选择/冻结/独立测试与统计分析。CPU 合成测试已通过；真实数据与 GPU 实验尚未运行，不包含真实模型成绩，也没有调用付费 API。用户提供的 GPU 2 为 A100 80GB，实际可用容量以启动时占用为准。

### 开始运行基础版

```bash
python -m pip install -e '.[test]'
python -m unittest discover -s tests -v
python -m exp5 --help
```

服务器流程见 [基础版运行说明](docs/10_RUNNING_BASELINE.md)，验证证据见 [CPU 测试记录](docs/11_CPU_VALIDATION.md)。模型命令默认离线，不因安装/导入/提交自动启动任务。

### 建议路线

```text
Flickr30K 固定验证集 / 测试集
       ↓
CLIP 基线 → SigLIP 2 强基线 → 双向 Recall / NDCG / 错误分析
                               ↓ 可选研究扩展
                 Top-50 候选 → Qwen3-VL 重排
                               ↓
                 全量重排 vs 困难查询选择性重排
                               ↓
                 质量—计算量对比 + 改写鲁棒性诊断
```

**基础版先完成“模型评估—错误分析—优化—复测”；研究版再回答“能否只给部分查询分配昂贵的图文重排计算”。** 不默认训练模型，不建向量数据库，不做前后端和部署。达到 85% 是待实测目标，不是预先填入的结论。

### 阅读顺序

| 文档 | 用途 |
|---|---|
| [00 范围、依据与当前状态](docs/00_SCOPE_AND_STATUS.md) | 区分材料要求、用户约束、研究建议和未完成事项 |
| [01 近期论文与方案取舍](docs/01_LITERATURE_REVIEW.md) | 五篇 2025–2026 顶会论文与两项技术报告如何影响设计 |
| [02 课程实验开题报告](docs/02_PROJECT_PROPOSAL.md) | 可直接修改姓名、日期等信息的完整开题正文 |
| [03 实验与评测协议](docs/03_EXPERIMENT_PROTOCOL.md) | 数据划分、指标、方法定义、消融、统计分析 |
| [04 实施指引](docs/04_IMPLEMENTATION_PLAN.md) | 环境、数据准备、模块接口、运行顺序和排障 |
| [05 日程与资源预算](docs/05_TIMELINE_AND_BUDGET.md) | 基础版与研究版里程碑、工时估计和停止条件 |
| [06 错误分析与 API 使用](docs/06_ERROR_ANALYSIS_AND_API.md) | 人工复核、改写诊断、API 成本与数据保护 |
| [07 风险与决策记录](docs/07_RISKS_AND_DECISIONS.md) | 数据泄漏、指标误读、重排失败及应对 |
| [08 结题报告与答辩模板](docs/08_REPORT_AND_DEFENSE_TEMPLATE.md) | 结果表、证据清单、答辩提纲 |
| [09 后续实现交接](docs/09_HANDOFF.md) | 给本地开发者或交互式编码助手的任务边界 |
| [来源与官方入口](references/SOURCES.md) | 论文发表状态、官方代码、模型与数据链接 |
| [BibTeX](references/references.bib) | 文献管理与报告引用 |

### 配置与记录

[实验计划 YAML](configs/experiment5.plan.yaml) 仍是设计配置，不由基础版 CLI 执行；实际参数见 `python -m exp5 --help` 与[运行说明](docs/10_RUNNING_BASELINE.md)。`templates/` 中提供 [运行记录](templates/run_record.json)、[汇总指标](templates/metrics.csv)、[逐查询结果](templates/query_results.csv)、[错误复核](templates/error_review.csv)、[消融记录](templates/ablation.csv)。空值表示待实测，不能解释为 0 分或已完成。

### 本次调研的关键区别

LamRA 为 CVPR 2025；MM-Embed、VLM2Vec 为 ICLR 2025；WISER、PinPoint 为 CVPR 2026。SigLIP 2 和 Qwen3-VL-Embedding/Reranker 在本方案中按技术报告引用，不冒称已核实的顶会论文。WISER、PinPoint 的原任务是组合图像检索，本项目仅借鉴其计算分配和诊断思路，不照搬任务或成绩。详见[文献调研](docs/01_LITERATURE_REVIEW.md)。

### 状态与公开范围

研究整理与首批代码日期：2026-09-30。文档中的方法改进均为待验证假设。CPU 人工样例/合成流程已通过；当前没有下载真实实验数据或模型、没有训练、没有运行 A100 推理、没有真实模型 Recall/NDCG 成绩，也没有配置自动执行任务。

仓库为公开仓库，仅上传本次生成的方案、来源链接、配置草案和空白模板。原课程 PDF、论文全文、Flickr 图像、模型权重、API 密钥和未审查的原始 API 输出不在上传范围内。数据与上游代码分别遵守各自使用条件。
