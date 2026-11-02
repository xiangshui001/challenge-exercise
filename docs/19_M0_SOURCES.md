# M0｜论文、数据与模型官方入口

仅列与“问题存在性验证”直接有关的材料。Astra开始时应核对可访问性、具体版本、使用条款、引用方式与评测定义；本清单不代表数据已经下载、论文结果已经复现。

## 数据集

- **SugarCrepe++**：NeurIPS 2024 Datasets & Benchmarks，图像、两条语义等价正描述与一条困难负描述。https://github.com/Sri-Harsha/scpp ；https://papers.nips.cc/paper_files/paper/2024/hash/200661bf8f4993b7828a45a2a90f2ecf-Abstract-Datasets_and_Benchmarks_Track.html
- **SNLI-VE**：视觉蕴含 entailment/neutral/contradiction；原始标签有噪声，重点人工核验 neutral。https://github.com/necla-ml/SNLI-VE
- **e-SNLI-VE**：校正版和解释数据；核验其划分和许可，不把“校正版本”自动当绝对真值。https://github.com/virginie-do/e-SNLI-VE
- **Visual-WSD**：SemEval 2023 Task 1，词和上下文选择候选图片。https://raganato.github.io/vwsd/ ；https://aclanthology.org/2023.semeval-1.308/
- **Winoground**：组合顺序/关系与图片-文本匹配。https://huggingface.co/datasets/facebook/winoground

## 直接相关工作：必须在报告里比较贡献边界

- **K-LITE**（NeurIPS 2022），利用外部词典知识进行视觉语言训练：https://papers.nips.cc/paper_files/paper/2022/hash/63fef0802863f47775c3563e18cbba17-Abstract-Conference.html
- **Structure-CLIP**（AAAI 2024），场景图与结构化负例：https://ojs.aaai.org/index.php/AAAI/article/view/28017
- **CECE**（ICLR 2025），蕴含/矛盾扩展与视觉语言模型验证：https://proceedings.iclr.cc/paper_files/paper/2025/hash/7eeec4dffcfe4912482ffbf2ab8ddb41-Abstract-Conference.html
- **READ-CLIP**（NeurIPS 2025），改写一致性与视觉语言表示：https://proceedings.neurips.cc/paper_files/paper/2025/hash/1487b44589123633117fb27d10bebcda-Abstract-Conference.html
- **SememeLRM**（ACL 2026），结构化义原及词汇关系挖掘：https://aclanthology.org/2026.acl-long.224/
- **Concept Bottleneck Models**（ICML 2020），可解释中间概念与决策：https://proceedings.mlr.press/v119/koh20a.html
- **BLIP-2**（ICML 2023），视觉编码器与语言模型连接模块：https://proceedings.mlr.press/v202/li23q.html

**当前假设的增量**：不是“词典知识增强CLIP”或“两个模型拼接”，而是验证有类型、带方向、有适用条件的义项关系在视觉证据支持/冲突/不可判断三种情况下，是否超越等知识量普通释义。还没有证据支持时，不得写成已证实创新。

## 历史资产

- [本仓库 SigLIP-base S0 验证集记录（PR #5）](https://github.com/xiangshui001/challenge-exercise/blob/feature/siglip-base-val-baseline/docs/14_PROGRESS_LOG.md)；仅是历史验证集，不是M0基准结果。
- [原文献与模型资料](../references/SOURCES.md)。