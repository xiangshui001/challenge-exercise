# models：模型封装代码

当前训练路线的首个模型：`google/siglip-base-patch16-224`，见 [训练路线](../docs/12_TRAINING_ROADMAP.md)。实际模型 revision 与环境在服务器小试后固定。

- 计划实现 `models/siglip_model.py`：原生 processor、图像/文本特征、可训练参数选择、保存/加载适配器或 checkpoint。
- 已有 `exp5/models.py` 支持原方案 B0=CLIP 与 B1=SigLIP 2，它不支持新 SigLIP-base；不要直接把 B1 命令解释为新模型。
- 本目录跟踪 README 和未来 Python 代码。下载权重使用 Hugging Face 缓存或 `models/pretrained/`，不提交 Git。
- 输出 checkpoint 统一放 `checkpoints/`，不混进模型代码。

SigLIP-base 封装尚未实现，创建目录不表示真实推理已验证。
