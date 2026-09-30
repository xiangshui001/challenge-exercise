# 来源登记与官方入口

核验日期：2026-09-30。这里只登记直接用于方案的来源，不声称穷尽近期文献。会议归属优先依据官方 proceedings/CVF 页面；方法补充参考作者 arXiv；工程用法参考官方仓库、模型卡和文档。网页当前默认分支可能更新，实际运行时仍需固定模型及源码 revision。

原课程材料：戴波，《跨模态检索实验设计与实现——从理论算法到工程落地的完整实践路径》，封面 2026 年 4 月，22 页，用户提供。实验五见第 14 页，指标汇总见第 21 页，预训练模型拓展见第 17 页。原 PDF 不在公开仓库转载。

<a id="p0"></a>
## P0｜CLIP，ICML 2021（基础文献）

Radford et al. **Learning Transferable Visual Models From Natural Language Supervision**. ICML 2021，PMLR 139。

[正式论文](https://proceedings.mlr.press/v139/radford21a.html) · [官方代码](https://github.com/openai/CLIP) · [本项目基线权重](https://huggingface.co/openai/clip-vit-base-patch16)

用途：提供经典图文表示基线。复用其预训练权重，而不是复现大规模训练。不是近期论文，单列为基础文献。

<a id="p1"></a>
## P1｜MM-Embed，ICLR 2025 主会

Sheng-Chieh Lin, Chankyu Lee, Mohammad Shoeybi, Jimmy Lin, Bryan Catanzaro, Wei Ping. **MM-EMBED: UNIVERSAL MULTIMODAL RETRIEVAL WITH MULTIMODAL LLMS**.

[正式会议页](https://proceedings.iclr.cc/paper_files/paper/2025/hash/6d5d6afa9957cfc9142ba60e78a467e9-Abstract-Conference.html) · [作者预印本](https://arxiv.org/abs/2411.02571) · [官方权重](https://huggingface.co/nvidia/MM-Embed)

用途：支持对初检和交互重排进行区别选型，提醒大模型的检索模态偏置。其训练与多任务平均结果不直接迁移为本项目分数。

<a id="p2"></a>
## P2｜VLM2Vec，ICLR 2025 主会

Ziyan Jiang, Rui Meng, Xinyi Yang, Semih Yavuz, Yingbo Zhou, Wenhu Chen. **VLM2Vec: Training Vision-Language Models for Massive Multimodal Embedding Tasks**.

[正式会议页](https://proceedings.iclr.cc/paper_files/paper/2025/hash/04261fce1705c4f02f062866717d592a-Abstract-Conference.html) · [官方仓库](https://github.com/TIGER-AI-Lab/VLM2Vec) · [V1 分支](https://github.com/TIGER-AI-Lab/VLM2Vec/tree/v1)

用途：指令条件化嵌入的备选方向，不纳入基础版。官方仓库已更新到后续版本，重现 ICLR 2025 工作时不能无说明地使用当前主分支并沿用 V1 名称。

<a id="p3"></a>
## P3｜LamRA，CVPR 2025 主会

Yikun Liu, Yajie Zhang, Jiayin Cai, Xiaolong Jiang, Yao Hu, Jiangchao Yao, Yanfeng Wang, Weidi Xie. **LamRA: Large Multimodal Model as Your Advanced Retrieval Assistant**. pp. 4015–4025。

[正式会议页](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_LamRA_Large_Multimodal_Model_as_Your_Advanced_Retrieval_Assistant_CVPR_2025_paper.html) · [作者预印本](https://arxiv.org/abs/2412.01720) · [官方仓库](https://github.com/Code-kunkun/LamRA)

用途：借鉴快速检索与 pointwise/listwise 重排分层。原方法包含训练，本项目不重现其训练流程。早期 arXiv v1 与会议终稿作者信息不同，BibTeX 使用正式会议页作者列表。

<a id="p4"></a>
## P4｜WISER，CVPR 2026 主会

Tianyue Wang, Leigang Qu, Tianyu Yang, Xiangzhao Hao, Yifan Xu, Haiyun Guo, Jinqiao Wang. **WISER: Wider Search, Deeper Thinking, and Adaptive Fusion for Training-Free Zero-Shot Composed Image Retrieval**. pp. 16865–16875。

[正式会议页](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_WISER_Wider_Search_Deeper_Thinking_and_Adaptive_Fusion_for_Training-Free_CVPR_2026_paper.html) · [作者预印本](https://arxiv.org/abs/2602.23029) · [官方仓库](https://github.com/Physicsmile/WISER)

用途：按不确定性分配额外计算的启发。其输入为参考图像加修改文本，本项目不做其双路生成或多轮细化，不引用它的 CIR 指标作为 Flickr30K 预期收益。

<a id="p5"></a>
## P5｜PinPoint，CVPR 2026 主会

Rohan Mahadev, Joyce Yuan, Patrick Poirson, David Xue, Hao-Yu Wu, Dmitry Kislyuk. **PinPoint: Evaluation of Composed Image Retrieval with Explicit Negatives, Multi-Image Queries, and Paraphrase Testing**. pp. 9742–9751。

[正式会议页](https://openaccess.thecvf.com/content/CVPR2026/html/Mahadev_PinPoint_Evaluation_of_Composed_Image_Retrieval_with_Explicit_Negatives_Multi-Image_CVPR_2026_paper.html) · [作者预印本](https://arxiv.org/abs/2603.04598) · [官方代码与数据](https://github.com/pinterest/pinpoint-dataset)

用途：为查询改写、显式负例和多正例诊断提供依据。本项目采用小规模补充诊断，不声称完整复现 PinPoint。这里不是同名的其他视觉理解方法。

<a id="p6"></a>
## P6｜RRF，SIGIR 2009（经典方法）

Gordon V. Cormack, Charles L. A. Clarke, Stefan Buettcher. **Reciprocal rank fusion outperforms condorcet and individual rank learning methods**. pp. 758–759。

[ACM 原论文 DOI](https://doi.org/10.1145/1571941.1572114) · [PyTerrier 实现说明](https://pyterrier.readthedocs.io/en/latest/ext/pyterrier-alpha/fusion.html)

用途：低成本排名融合。协议中的不同模型加权、验证权重网格是本项目配置，不把特定数据集上的收益写成方法保证。

<a id="t1"></a>
## T1｜SigLIP 2，2025 技术报告

Michael Tschannen et al. **SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features**. arXiv:2502.14786。

[报告](https://arxiv.org/abs/2502.14786) · [用于本次表 1 核对的 v1 全文](https://arxiv.org/html/2502.14786v1) · [官方权重](https://huggingface.co/google/siglip2-so400m-patch14-384) · [官方实现项目](https://github.com/google-research/big_vision)

用途：基础版强编码器，省去训练。文献调研中的 Flickr 数字为该表的 Recall@1，只供选型参考，不是项目结果。未将该报告归入未经核实的会议。

<a id="t2"></a>
## T2｜Qwen3-VL-Embedding/Reranker，2026 技术报告与官方模型

Mingxin Li et al. **Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for State-of-the-Art Multimodal Retrieval and Ranking**. arXiv:2601.04720。

[报告](https://arxiv.org/abs/2601.04720) · [官方重排模型](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) · [官方代码与依赖示例](https://github.com/QwenLM/Qwen3-VL-Embedding)

用途：复用已训练的 pointwise 图文相关性评分，避免自建重排训练集。官方通用任务指标不替代本地 Flickr 评测；不声称已核实其顶会录用。

<a id="d1"></a>
## D1｜Flickr30K 原始数据说明

[作者数据页：From image descriptions to visual denotations](https://shannon.cs.illinois.edu/DenotationGraph/)

用途：图像与人工描述来源、数据背景。下载与使用遵循来源说明；本仓库不重发图片。

<a id="d2"></a>
## D2｜Karpathy 图像描述项目与划分入口

[作者项目页：Deep Visual-Semantic Alignments for Generating Image Descriptions](https://cs.stanford.edu/people/karpathy/deepimagesent/)

用途：取得既有图文对应与划分资料，避免自行随机划分后仍称标准评测。实际落地以下载文件、样本 ID 和审计结果为准。

<a id="d3"></a>
## D3｜Flickr30K 下载镜像（便利来源，非原始作者来源）

[Hugging Face nlphuji/flickr30k](https://huggingface.co/datasets/nlphuji/flickr30k)

用途：可选的图像、描述与 split 字段打包。外层分区为 test、约 31k 行，不等于标准 1000 张测试集；必须检查每行内部 split 并比对 ID。镜像的数据授权声明不能替代原图使用条件，图像处理差异需记录。

<a id="s1"></a>
## S1｜CLIP Benchmark 检索评估

[官方仓库](https://github.com/LAION-AI/CLIP_benchmark) · [检索指标代码](https://raw.githubusercontent.com/LAION-AI/CLIP_benchmark/main/clip_benchmark/metrics/zeroshot_retrieval.py)

用途：核对图文检索 hit-style R@K、多描述正例处理及现成加载流程。仅复用必要逻辑，引用并遵守上游许可，不搬入完整无关基准。

<a id="s2"></a>
## S2｜SigLIP 2 的 Transformers 版本化文档

[Transformers v4.57.1 SigLIP2 文档](https://huggingface.co/docs/transformers/v4.57.1/model_doc/siglip2)

用途：小写、固定文本长度与 padding、自身 processor、模型类型。文档有 FixRes/NaFlex 两种变体，代码必须与实际权重对应；不要机械混用其中不同权重的片段。

<a id="s3"></a>
## S3｜PyTorch 官方版本组合

[Previous PyTorch Versions](https://pytorch.org/get-started/previous-versions/)

用途：核对 torch 2.8.0 / torchvision 0.23.0 / CUDA 12.6 wheel 组合。具体驱动与运行兼容性仍需用户机器验证，不宣称为本机已经通过的环境锁。

<a id="s4"></a>
## S4｜NDCG 接口

[scikit-learn ndcg_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html)

用途：标准指标实现与单元测试参考；特别注意多正例的 IDCG 和并列分数处理。正式运行应记录实际 sklearn 版本。

## 来源使用边界

官方会议页用于确认论文身份，模型卡用于确认接口和权重，不把自报 benchmark 视作已在本地复现。部分网站抓取受限时，以可访问的官方索引和作者文本交叉核验，不凭第三方博客补全结论。没有下载或公开转载论文全文，也没有访问需要绕过权限的内容。
