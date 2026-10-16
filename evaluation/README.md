# evaluation：评估与错误分析入口

复用 `exp5/metrics.py`、`exp5/stats.py` 与逐查询预测接口，不复制另一套 Recall/NDCG。

- 已实现：命中式双向 R@1/5/10/20/50、二值 NDCG@10、集合召回、确定性并列、按图分组 bootstrap、修复/退化变化。
- 下一步为同一 SigLIP-base 训练前/后增加评估入口；两个模型使用相同完整候选库、正例、原始 processor 和数据划分。
- 验证集选学习率、epoch/checkpoint；固定后才评测试集。测试结果不能指导返调。
- 每次保存逐查询排名、汇总、运行配置与模型版本；人工错误分析包含修复和退化。

现有 `exp5` CLI 只接受其既有 B0/B1 缓存，不能直接评估尚未接入的 SigLIP-base checkpoint。主路线评估适配器待实现。
