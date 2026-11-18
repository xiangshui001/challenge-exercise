# M0 追加实验的逐项证据

[研究报告](../../docs/M0_FOLLOWUP_REPORT.md) · [冻结方案](../../docs/23_FOLLOWUP_PROTOCOL.md) · [复算方法](../../docs/24_FOLLOWUP_REPRODUCE.md)

| 文件或目录 | 内容 |
|---|---|
| audit.json | 780 张新图的注册 ID、图像字节散列、文本散列和官方标签 |
| siglip-natural、clip-natural | 各 500 张图的实际余弦分数、对应身份和模型元数据 |
| qwen-natural | 180 个新 e-SNLI-VE 样本的三个提示条件 |
| qwen-spatial | 100 个 CLEVR 场景，每个方向分别进行有图和去图推理 |
| qwen-reference | 13 个旧难例中 AI 选义后的四条件探索 |
| ai-reference.jsonl、selected-senses.json | AI 在匿名顺序下给出的判断、证据、信心、位置映射及选义 |
| review-comparison.json | 将锁定的 AI 答案映射回原始顺序，与官方标注逐项对照 |
| summary.json | 从原始预测复算的统计量，包括分类型结果、配对差和区间 |
| verification.json | 完整性、提示重建和跨平台汇总核对记录 |

所有模型目录中 full 运行的原始分数或答案位于 predictions.jsonl；run.json 是正式运行元数据，smoke-run.json 单列两项接口检查。AI 和人类答案以审核题号关联，保留所有歧义和分歧。人类原图工作包由 followup.reference 在私有目录重建。
