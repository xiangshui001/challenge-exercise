# models：模型封装代码

当前训练路线的首个模型：`google/siglip-base-patch16-224`，见 [训练路线](../docs/12_TRAINING_ROADMAP.md)。实际模型 revision 与环境在服务器小试后固定。

- 已实现 `models/siglip_model.py`：原生 processor、固定 revision 的预训练权重、FP32 归一化图像/文本特征。小试入口见 [15](../docs/15_SIGLIP_SMOKE.md)。可训练参数选择和训练 checkpoint 适配待 P3 实现。
- 已有 `exp5/models.py` 支持原方案 B0=CLIP 与 B1=SigLIP 2，它不支持新 SigLIP-base；不要直接把 B1 命令解释为新模型。
- 本目录跟踪 README 和未来 Python 代码。下载权重使用 Hugging Face 缓存或 `models/pretrained/`，不提交 Git。
- 输出 checkpoint 统一放 `checkpoints/`，不混进模型代码。

先在 CPU 完成文件准备，再移到当前显式选择的 GPU。小试记录真实参数名称；尚未指定解冻层，也尚未运行真实模型小试。
